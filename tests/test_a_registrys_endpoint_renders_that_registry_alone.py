# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A per-registry metrics endpoint renders that registry alone, and the board's /metrics renders no bulk registry.

Each registry renders on its own, once per cycle, under its own lock, so a small registry's endpoint never waits behind
the bulk agent and trade registries. Each body is that registry's own UTF-8 encoding, so a non-ASCII label cannot shift
another registry's bytes.
"""
import os
import sys
import threading

from prometheus_client import CollectorRegistry, Gauge, generate_latest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taos.im.validator import exposition as ex  # noqa: E402
from taos.im.validator.report import _SnapshotCollector  # noqa: E402

ORDER = ('validator', 'miner', 'agent', 'books', 'trades')


def _registries():
    validator = CollectorRegistry()
    Gauge('validator_gauges', 'Validator gauges.', ['wallet', 'name'], registry=validator).labels('w', 'ram').set(1.0)
    miner = CollectorRegistry()
    identity = _SnapshotCollector('miner_identity', 'Miner identity.', ['wallet', 'uid', 'name'])
    miner.register(identity)
    identity.update({('w', 1, 'Zürich desk ✓'): 1.0})
    agent = CollectorRegistry()
    agent_gauges = _SnapshotCollector('agent_gauges', 'Agent gauges.', ['wallet', 'book_id', 'agent_id', 'name'])
    agent.register(agent_gauges)
    agent_gauges.update({('w', 0, 1, 'pnl'): -1.5e-7})
    books = CollectorRegistry()
    Gauge('book_gauges', 'Book gauges.', ['wallet', 'book_id', 'name'], registry=books).labels('w', 0, 'mid').set(2.0)
    trades = CollectorRegistry()
    miner_trades = _SnapshotCollector('miner_trades', 'Trades.', ['wallet', 'book_id', 'uid', 'slot', 'name'],
                                      carry_forward=False)
    trades.register(miner_trades)
    miner_trades.update({('w', 0, 1, 0, 'price'): 300.5})
    return {'validator': validator, 'miner': miner, 'agent': agent, 'books': books, 'trades': trades}


def _recording(monkeypatch):
    rendered = []
    frozen, metric = ex.render_frozen, ex.render_metric

    def render_frozen(name, *args, **kwargs):
        rendered.append(name)
        return frozen(name, *args, **kwargs)

    def render_metric(m, *args, **kwargs):
        rendered.append(m.name)
        return metric(m, *args, **kwargs)

    monkeypatch.setattr(ex, 'render_frozen', render_frozen)
    monkeypatch.setattr(ex, 'render_metric', render_metric)
    return rendered


def test_every_registry_is_served_exactly_as_the_client_writes_it_after_a_non_ascii_label():
    regs = _registries()
    e = ex.Exposition(regs, ORDER, main_mode='all')
    snap = e.publish(step=1, sim_timestamp=5, sim_id='s')
    for name in ORDER:
        assert e.serve(name) == generate_latest(regs[name]), name
    body, ranges = snap.render()
    for name in ORDER:
        assert body[ranges[name][0]:ranges[name][1]] == generate_latest(regs[name]), name


def test_a_registrys_endpoint_renders_that_registry_alone(monkeypatch):
    rendered = _recording(monkeypatch)
    e = ex.Exposition(_registries(), ORDER, main_mode='board')
    e.publish(step=1, sim_timestamp=5, sim_id='s')
    assert e.serve('validator').startswith(b'# HELP validator_gauges')
    assert rendered == ['validator_gauges']


def test_the_boards_metrics_render_no_bulk_registry(monkeypatch):
    rendered = _recording(monkeypatch)
    e = ex.Exposition(_registries(), ORDER, main_mode='board')
    e.publish(step=1, sim_timestamp=5, sim_id='s')
    body = e.serve()
    assert body.startswith(b'# cycle step=1') and b'book_gauges' in body and b'miner_identity' in body
    assert not {'agent_gauges', 'miner_trades'} & set(rendered)


def test_a_slow_bulk_render_does_not_hold_up_another_registry(monkeypatch):
    started, release = threading.Event(), threading.Event()
    frozen = ex.render_frozen

    def render_frozen(name, *args, **kwargs):
        if name == 'miner_trades':
            started.set()
            release.wait(10)
        return frozen(name, *args, **kwargs)

    monkeypatch.setattr(ex, 'render_frozen', render_frozen)
    e = ex.Exposition(_registries(), ORDER, main_mode='board')
    e.publish(step=1, sim_timestamp=5, sim_id='s')
    bulk = threading.Thread(target=e.serve, args=('trades',))
    bulk.start()
    assert started.wait(5)
    got = []
    reader = threading.Thread(target=lambda: got.append(e.serve('validator')))
    reader.start()
    reader.join(2)
    finished = not reader.is_alive()
    release.set()
    bulk.join(10)
    reader.join(10)
    assert finished and got and got[0].startswith(b'# HELP validator_gauges')
