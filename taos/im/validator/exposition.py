# SPDX-License-Identifier: MIT
"""One scrape, one cycle: the Prometheus exposition served from a frozen per-cycle snapshot.

Until 30 September 2026 every scrape rendered the live registries on demand while the reporting child was
still applying the next cycle, so a read could carry a family half rebuilt (three of 206 mainnet board reads
in the last week lost some or all coldkeys) and families from two cycles, and each scrape paid the whole
render (about 26 s at mainnet size with the stock client) while holding the GIL against the apply loop.

Here the child publishes one CycleSnapshot at the end of each report cycle: the snapshot-backed families
contribute their frozen dicts (replaced, never mutated, so the reference is enough), the small eager
families are copied. The first scrape of a cycle renders the body once, in the scrape thread, single-flight;
later scrapes of the same cycle get the same bytes, and every per-registry endpoint is a slice of them. The
renderer writes the same text the client's generate_latest writes (the test asserts byte identity), about
five times faster.

MVTRX_METRICS_SNAPSHOT=0 restores the per-scrape live render. MVTRX_METRICS_MAIN=board serves /metrics
without the two bulk registries (agent, trades), which keep their own endpoints; the default, all, keeps
/metrics complete so a scraper that has not been told about the endpoints sees no gap.
"""
import os
import threading
import time

import prometheus_client.exposition as _pe

REGISTRY_ORDER = ('validator', 'simulation', 'miner', 'agent', 'books', 'trades', 'gentrx')
BOARD_REGISTRIES = ('validator', 'simulation', 'miner', 'books', 'gentrx')
SNAPSHOT_ENABLED = os.environ.get("MVTRX_METRICS_SNAPSHOT", "1").strip().lower() not in ("0", "false", "no", "off")
MAIN_MODE = os.environ.get("MVTRX_METRICS_MAIN", "all").strip().lower() or "all"


def _sample_line(name, labels, value):
    """One sample line, as generate_latest writes it (legacy names, UTF-8 label values)."""
    om = _pe.openmetrics
    if labels:
        labelstr = '{' + ','.join(
            '{}="{}"'.format(om.escape_label_name(k, om.UNDERSCORES), om._escape(v, om.ALLOWUTF8, False))
            for k, v in sorted(labels.items())) + '}'
    else:
        labelstr = ''
    return f'{om.escape_metric_name(name, om.UNDERSCORES)}{labelstr} {_pe.floatToGoString(value)}\n'


def _header(name, mtype, documentation):
    om = _pe.openmetrics
    mname = name
    if mtype == 'counter':
        mname = name + '_total'
    elif mtype == 'info':
        mname = name + '_info'
        mtype = 'gauge'
    elif mtype == 'stateset':
        mtype = 'gauge'
    elif mtype == 'gaugehistogram':
        mtype = 'histogram'
    elif mtype == 'unknown':
        mtype = 'untyped'
    doc = documentation.replace('\\', r'\\').replace('\n', r'\n')
    return (f'# HELP {om.escape_metric_name(mname, om.UNDERSCORES)} {doc}\n'
            f'# TYPE {om.escape_metric_name(mname, om.UNDERSCORES)} {mtype}\n')


def render_metric(metric, out):
    """Append one materialised metric family (a client Metric with its samples) in text format."""
    out.append(_header(metric.name, metric.type, metric.documentation))
    om_samples = {}
    for s in metric.samples:
        for suffix in ('_created', '_gsum', '_gcount'):
            if s.name == metric.name + suffix:
                om_samples.setdefault(suffix, []).append(_sample_line_ts(s))
                break
        else:
            out.append(_sample_line_ts(s))
    for suffix, lines in sorted(om_samples.items()):
        out.append(_header(metric.name + suffix, 'gauge', metric.documentation))
        out.extend(lines)


def _sample_line_ts(s):
    line = _sample_line(s.name, s.labels, s.value)
    if s.timestamp is not None:
        line = f'{line[:-1]} {int(float(s.timestamp) * 1000):d}\n'
    return line


def render_frozen(name, documentation, labelnames, frozen, out):
    """Append a snapshot-backed family straight from its frozen dict, as its collect() would expose it:
    unusable samples (None or non-numeric) skipped, labels stringified."""
    out.append(_header(name, 'gauge', documentation))
    for labels, value in frozen.items():
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        out.append(_sample_line(name, dict(zip(labelnames, (str(label) for label in labels))), value))


def _collectors_of(registry):
    names = getattr(registry, '_collector_to_names', None)
    return list(names) if names is not None else []


class CycleSnapshot:
    """Everything one scrape may see, captured at the end of a report cycle and never mutated."""

    def __init__(self, registries, order=REGISTRY_ORDER, **meta):
        self.meta = dict(meta, published_at=time.time())
        self.parts = []
        for name in order:
            registry = registries.get(name)
            if registry is None:
                continue
            captured = []
            for collector in _collectors_of(registry):
                frozen = getattr(collector, '_snapshot', None)
                if frozen is not None and hasattr(collector, '_labelnames'):
                    captured.append(('frozen', collector, frozen))
                else:
                    captured.append(('metrics', collector, list(collector.collect())))
            self.parts.append((name, captured))
        self._body = None
        self._ranges = None
        self._lock = threading.Lock()
        self._render_seconds = None
        self.renders = 0

    def render(self):
        """The body and the per-registry byte ranges, rendered once (single-flight) per snapshot."""
        if self._body is not None:
            return self._body, self._ranges
        with self._lock:
            if self._body is None:
                started = time.perf_counter()
                out = [self.comment()]
                ranges = {}
                for name, captured in self.parts:
                    start = sum(len(x) for x in out)
                    for kind, collector, payload in captured:
                        if kind == 'frozen':
                            render_frozen(collector._name, collector._documentation, collector._labelnames,
                                          payload, out)
                        else:
                            for metric in payload:
                                render_metric(metric, out)
                    ranges[name] = (start, sum(len(x) for x in out))
                self._render_seconds = time.perf_counter() - started
                self.renders += 1
                self._ranges = ranges
                self._body = ''.join(out).encode('utf-8')
        return self._body, self._ranges

    def comment(self):
        """The cycle stamp every reader can check across families: one line, ignored by scrapers."""
        m = self.meta
        return (f"# cycle step={m.get('step')} sim_timestamp={m.get('sim_timestamp')} sim_id={m.get('sim_id')} "
                f"published_at={m['published_at']:.3f}\n")

    @property
    def render_seconds(self):
        return self._render_seconds


class Exposition:
    """Holds the current snapshot and serves bytes from it."""

    def __init__(self, registries, order=REGISTRY_ORDER, main_mode=None):
        self._registries = registries
        self._order = tuple(order)
        self._snapshot = None
        self.main_mode = (main_mode or MAIN_MODE)

    def publish(self, **meta):
        """Freeze the registries as they stand: called by the child at the end of every report cycle."""
        self._snapshot = CycleSnapshot(self._registries, self._order, **meta)
        return self._snapshot

    @property
    def current(self):
        return self._snapshot

    def main_registries(self):
        return BOARD_REGISTRIES if self.main_mode == 'board' else self._order

    def serve(self, name=None):
        """The bytes for /metrics (name None) or /metrics/<registry>; None before the first publish."""
        snap = self._snapshot
        if snap is None:
            return None
        body, ranges = snap.render()
        if name is None:
            if self.main_mode != 'board':
                return body
            head = body[:len(snap.comment().encode('utf-8'))]
            return head + b''.join(body[ranges[n][0]:ranges[n][1]] for n in self.main_registries() if n in ranges)
        if name not in ranges:
            return b''
        start, end = ranges[name]
        return body[start:end]
