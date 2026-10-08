# SPDX-License-Identifier: MIT
"""A scrape of the validator's metrics is one cycle, whole, and costs one render per cycle however many
readers ask. Until 30 September 2026 every scrape rendered the live registries while the next cycle was
being applied, so a read could carry a family half rebuilt (three of 206 mainnet board reads lost some or
all coldkeys in one week) and two cycles mixed, and each scrape paid the whole render. Now the reporting
child publishes a frozen snapshot at the end of each cycle; the first scrape renders it once, single-flight,
and every per-registry endpoint is a slice of the same bytes. The text is byte-identical to the client's
own renderer, and the two bulk registries can be left off /metrics without losing their own endpoints."""
import os
import sys
import threading
import types

from prometheus_client import CollectorRegistry, Counter, Gauge, Info, generate_latest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taos.im.validator import exposition as ex  # noqa: E402
from taos.im.validator.report import _SnapshotCollector  # noqa: E402


def _registries():
    validator = CollectorRegistry()
    Counter('counters', 'Counter summaries.', ['wallet', 'name'], registry=validator).labels('w', 'a').inc(3)
    Gauge('validator_gauges', 'Validator gauges, with a\\backslash and a\nnewline.', ['wallet', 'name'],
          registry=validator).labels('w', 'ram "quoted"').set(float('inf'))
    Info('neuron_info', 'Info summaries.', ['wallet'], registry=validator).labels('w').info({'v': '0.6.3'})
    miner = CollectorRegistry()
    miner_gauges = _SnapshotCollector('miner_gauges', 'Miner gauges.', ['wallet', 'agent_id', 'name'])
    miner.register(miner_gauges)
    miner_gauges.update({('w', 1, 'score'): 0.25, ('w', 2, 'score'): 1, ('w', 3, 'score'): None, ('w', 4, 'score'): 'x'})
    agent = CollectorRegistry()
    agent_gauges = _SnapshotCollector('agent_gauges', 'Agent gauges.', ['wallet', 'book_id', 'agent_id', 'name'])
    agent.register(agent_gauges)
    agent_gauges.update({('w', 0, 1, 'pnl'): -1.5e-7, ('w', 0, 2, 'pnl'): 123456789.0})
    trades = CollectorRegistry()
    miner_trades = _SnapshotCollector('miner_trades', 'Trades.', ['wallet', 'book_id', 'uid', 'slot', 'name'],
                                      carry_forward=False)
    trades.register(miner_trades)
    miner_trades.update({('w', 0, 1, 0, 'price'): 300.5})
    return {'validator': validator, 'miner': miner, 'agent': agent, 'trades': trades}, miner_gauges, agent_gauges


ORDER = ('validator', 'miner', 'agent', 'trades')


def test_the_renderer_writes_exactly_what_the_client_writes():
    regs, _, _ = _registries()
    snap = ex.CycleSnapshot(regs, ORDER, step=1, sim_timestamp=5, sim_id='s')
    body, ranges = snap.render()
    expected = b''.join(generate_latest(regs[n]) for n in ORDER)
    assert body[len(snap.comment().encode()):] == expected
    for n in ORDER:
        assert body[ranges[n][0]:ranges[n][1]] == generate_latest(regs[n])


def test_a_snapshot_is_not_moved_by_the_next_cycles_updates():
    regs, miner_gauges, agent_gauges = _registries()
    e = ex.Exposition(regs, ORDER)
    first = e.publish(step=1, sim_timestamp=5, sim_id='s')
    miner_gauges.update({('w', 1, 'score'): 0.75})
    agent_gauges.update({('w', 0, 1, 'pnl'): 9.0})
    body, _ = first.render()
    assert b'miner_gauges{agent_id="1",name="score",wallet="w"} 0.25' in body
    assert b'agent_gauges{agent_id="1",book_id="0",name="pnl",wallet="w"} -1.5e-07' in body
    second = e.publish(step=2, sim_timestamp=6, sim_id='s')
    body2, _ = second.render()
    assert b'miner_gauges{agent_id="1",name="score",wallet="w"} 0.75' in body2
    assert e.serve() is body2 and e.serve('miner').startswith(b'# HELP miner_gauges')


def test_the_body_is_rendered_once_per_cycle_whatever_the_number_of_readers():
    regs, _, _ = _registries()
    snap = ex.CycleSnapshot(regs, ORDER, step=1, sim_timestamp=5, sim_id='s')
    bodies = []

    def reader():
        bodies.append(snap.render()[0])

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert snap.renders == 1 and len({id(b) for b in bodies}) == 1


def test_the_board_mode_leaves_the_bulk_registries_to_their_own_endpoints():
    regs, _, _ = _registries()
    e = ex.Exposition(regs, ORDER, main_mode='board')
    e.publish(step=1, sim_timestamp=5, sim_id='s')
    main = e.serve()
    assert b'miner_gauges' in main and b'validator_gauges' in main
    assert b'agent_gauges' not in main and b'miner_trades' not in main
    assert e.serve('agent').startswith(b'# HELP agent_gauges') and b'miner_trades' in e.serve('trades')
    assert main.startswith(b'# cycle step=1 sim_timestamp=5 sim_id=s')
    everything = ex.Exposition(regs, ORDER, main_mode='all')
    everything.publish(step=1, sim_timestamp=5, sim_id='s')
    assert b'agent_gauges' in everything.serve()


def test_rows_written_by_keyword_labels_reach_the_cycle_whole():
    from taos.im.validator import report as r   # the live module: another test may have reloaded it

    miners = r._SnapshotCollector('miners', 'Miners.', ['wallet', 'agent_id', 'coldkey'], carry_forward=False)
    miners.clear()
    r._set_if_changed_metric(miners, 1.0, wallet='w', agent_id=1, coldkey='5A')
    r._set_if_changed_metric(miners, 1.0, wallet='w', agent_id=2, coldkey='5B')
    assert list(miners.collect())[0].samples == []          # nothing exposed until the cycle swaps
    cycle = {}
    cycle.update(miners.take_pending())
    miners.update(cycle)
    labels = sorted(s.labels['coldkey'] for s in list(miners.collect())[0].samples)
    assert labels == ['5A', '5B'] and miners.take_pending() == {}


def test_nothing_is_served_before_the_first_cycle():
    regs, _, _ = _registries()
    assert ex.Exposition(regs, ORDER).serve() is None


def test_the_bulk_families_of_the_reporter_are_all_snapshot_backed():
    import inspect
    from taos.im.validator import report

    src = inspect.getsource(report.ReportingService._init_prometheus)
    for family in ('simulation_gauges', 'miner_gauges', 'book_gauges', 'miners', 'miner_identity',
                   'agent_gauges', 'trades', 'miner_trades', 'books'):
        assert f"_SnapshotCollector('{family}'" in src, f"{family} is still an eager Gauge"
    assert "self.exposition.publish(" in inspect.getsource(report.report)


def test_the_renderer_needs_no_escaping_api_from_the_client(monkeypatch):
    """The 6 October testnet deploy served HTTP 500 on every metrics family: the renderer reached for
    prometheus_client's metric-name escaping API (escape_metric_name, UNDERSCORES, added in 0.22), which the
    host's older package did not have, and nothing pinned the version the launcher installs. The renderer
    carries its own escaping, so the same bytes come out whatever the installed client offers."""
    import prometheus_client.openmetrics.exposition as om
    regs, _, _ = _registries()
    expected = b''.join(generate_latest(regs[n]) for n in ORDER)   # the client's own render, taken while it still works
    for name in ("UNDERSCORES", "ALLOWUTF8", "escape_metric_name", "escape_label_name", "_escape"):
        if hasattr(om, name):
            monkeypatch.delattr(om, name)
    snap = ex.CycleSnapshot(regs, ORDER, step=1, sim_timestamp=5, sim_id='s')
    body, ranges = snap.render()
    assert body[len(snap.comment().encode()):] == expected
