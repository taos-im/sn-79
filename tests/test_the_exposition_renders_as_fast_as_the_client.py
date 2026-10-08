# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The exposition renders a few hundred megabytes per cycle, tens of millions of label escapes. When the renderer took
over the client's escaping on 6 October 2026 it checked every name character by character in Python, four times the
client's cost per name, and the testnet validator's metrics server stopped answering within minutes of the deploy:
the render thread held the interpreter and the families trickled out at kilobytes per second. The renderer must be
no slower than the client's generate_latest on the same registry, byte for byte the same."""
import time

from prometheus_client import CollectorRegistry, Gauge, generate_latest

from taos.im.validator import exposition as ex

SAMPLES = 60_000


def _registry():
    reg = CollectorRegistry()
    g = Gauge('miner_gauges', 'Per-miner gauges.', ['wallet', 'netuid', 'sim_id', 'agent_id', 'book_id', 'miner_gauge_name'], registry=reg)
    names = ('debeta_making_captured', 'debeta_making_realized', 'total_daily_maker_volume', 'score')
    for i in range(SAMPLES):
        g.labels('5E7RbxVyLz681r9tF9a9bfB9ZuBQa4ZARbJ8HApjRruGaNtY', '366', '20261006_1603', str(i % 257), str(i % 128), names[i % 4]).set(i * 0.5)
    return reg


def _best_of(fn, n=3):
    best = None
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


def test_the_render_is_no_slower_than_the_client_and_byte_identical():
    reg = _registry()
    client = _best_of(lambda: generate_latest(reg))
    snaps = []

    def ours():
        s = ex.CycleSnapshot({'miner': reg}, ('miner',), step=1, sim_timestamp=5, sim_id='s')
        s.render()
        snaps.append(s)
    mine = _best_of(ours)
    body, ranges = snaps[-1].render()
    assert body[ranges['miner'][0]:ranges['miner'][1]] == generate_latest(reg)
    assert mine <= max(1.1 * client, 0.05), f"render {mine:.3f}s against the client's {client:.3f}s on {SAMPLES} samples"
