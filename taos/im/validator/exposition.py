# SPDX-License-Identifier: MIT
"""One scrape, one cycle: the Prometheus exposition served from a frozen per-cycle snapshot.

A scrape that renders the live registries on demand while the reporting child is still applying the next cycle
can carry a family half rebuilt (some or all coldkeys missing) and families from two cycles, and each such scrape
pays the whole render while holding the GIL against the apply loop.

Here the child publishes one CycleSnapshot at the end of each report cycle: the snapshot-backed families
contribute their frozen dicts (replaced, never mutated, so the reference is enough), the small eager
families are copied. The first scrape of a cycle renders the body once, in the scrape thread, single-flight;
later scrapes of the same cycle get the same bytes, and every per-registry endpoint is a slice of them. The
renderer writes the same text the client's generate_latest writes (the test asserts byte identity), in less
time.

MVTRX_METRICS_SNAPSHOT=0 restores the per-scrape live render. /metrics serves the board registries without the two
bulk registries (agent, trades), which keep their own endpoints; MVTRX_METRICS_MAIN=all serves every registry on
/metrics.
"""
import functools
import os
import re
import threading
import time

import prometheus_client.exposition as _pe

REGISTRY_ORDER = ('validator', 'simulation', 'miner', 'agent', 'books', 'trades', 'gentrx')
BOARD_REGISTRIES = ('validator', 'simulation', 'miner', 'books', 'gentrx')
SNAPSHOT_ENABLED = os.environ.get("MVTRX_METRICS_SNAPSHOT", "1").strip().lower() not in ("0", "false", "no", "off")


def main_mode_from(raw):
    """What the combined /metrics serves: `board` (the default) leaves the agent and trades registries to their own
    paths, `all` serves every registry."""
    mode = (raw or "").strip().lower()
    return mode or "board"


MAIN_MODE = main_mode_from(os.environ.get("MVTRX_METRICS_MAIN"))


# Name and value escaping as the text exposition writes them: a legacy-valid name goes through unchanged, any other
# rune becomes an underscore, and a label value escapes backslash, newline and double quote. Written here rather
# than taken from prometheus_client.openmetrics.exposition, whose escaping functions exist only from 0.22: a host
# with an older package would serve HTTP 500 on every metrics family.
#
# Speed matters here as much as correctness: a cycle renders a few hundred megabytes, tens of millions of label
# escapes, in a thread that shares the interpreter with the validator. A per-character check costs several times
# the client's regex, enough to stall the metrics server.
# Names come from a small fixed set and label values repeat heavily (uids, book ids, wallets), so both are cached;
# the regex check runs once per distinct name, and a value without a special character is returned as it is.
_LEGACY_METRIC_NAME = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*$')
_LEGACY_LABEL_NAME = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _legacy_rune(ch, i, colon):
    return ('a' <= ch <= 'z') or ('A' <= ch <= 'Z') or ch == '_' or (colon and ch == ':') or ('0' <= ch <= '9' and i > 0)


def _escape_name(name, colon):
    return ''.join(ch if _legacy_rune(ch, i, colon) else '_' for i, ch in enumerate(name))


@functools.lru_cache(maxsize=None)
def _escape_metric_name(name):
    return name if _LEGACY_METRIC_NAME.match(name) else _escape_name(name, colon=True)


@functools.lru_cache(maxsize=None)
def _escape_label_name(name):
    return name if _LEGACY_LABEL_NAME.match(name) else _escape_name(name, colon=False)


@functools.lru_cache(maxsize=1 << 16)
def _escape_label_value_str(value):
    if '\\' in value or '\n' in value or '"' in value:
        return value.replace('\\', r'\\').replace('\n', r'\n').replace('"', r'\"')
    return value


def _escape_label_value(value):
    return _escape_label_value_str(value if isinstance(value, str) else str(value))


def _sample_line(name, labels, value):
    """One sample line, as generate_latest writes it (legacy names, UTF-8 label values)."""
    if labels:
        labelstr = '{' + ','.join(
            '{}="{}"'.format(_escape_label_name(k), _escape_label_value(v))
            for k, v in sorted(labels.items())) + '}'
    else:
        labelstr = ''
    return f'{_escape_metric_name(name)}{labelstr} {_pe.floatToGoString(value)}\n'


def _header(name, mtype, documentation):
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
    return (f'# HELP {_escape_metric_name(mname)} {doc}\n'
            f'# TYPE {_escape_metric_name(mname)} {mtype}\n')


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
        # EACH REGISTRY ON ITS OWN. Each registry renders under its own lock, so /metrics/<registry> never waits
        # behind the large agent and trade registries.
        self._part_bytes = {}
        self._part_locks = {name: threading.Lock() for name, _ in self.parts}
        self._render_seconds = None
        self.renders = 0

    def part(self, name):
        """One registry's bytes, rendered once (single-flight) per snapshot without touching any other registry."""
        got = self._part_bytes.get(name)
        if got is not None:
            return got
        lock = self._part_locks.get(name)
        if lock is None:
            return b''
        with lock:
            if name not in self._part_bytes:
                out = []
                for kind, collector, payload in dict(self.parts)[name]:
                    if kind == 'frozen':
                        render_frozen(collector._name, collector._documentation, collector._labelnames, payload, out)
                    else:
                        for metric in payload:
                            render_metric(metric, out)
                self._part_bytes[name] = ''.join(out).encode('utf-8')
        return self._part_bytes[name]

    def render(self):
        """The body and the per-registry byte ranges, rendered once (single-flight) per snapshot."""
        if self._body is not None:
            return self._body, self._ranges
        with self._lock:
            if self._body is None:
                started = time.perf_counter()
                out = [self.comment().encode('utf-8')]
                ranges = {}
                # Ranges are byte offsets into the UTF-8 body.
                at = len(out[0])
                for name, _ in self.parts:
                    chunk = self.part(name)
                    ranges[name] = (at, at + len(chunk))
                    at += len(chunk)
                    out.append(chunk)
                self._render_seconds = time.perf_counter() - started
                self.renders += 1
                self._ranges = ranges
                self._body = b''.join(out)
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
        """The bytes for /metrics (name None) or /metrics/<registry>; None before the first publish. A registry's
        endpoint, and the board's /metrics, render only the registries they serve."""
        snap = self._snapshot
        if snap is None:
            return None
        if name is not None:
            return snap.part(name)
        if self.main_mode != 'board':
            return snap.render()[0]
        return snap.comment().encode('utf-8') + b''.join(snap.part(n) for n in self.main_registries())
