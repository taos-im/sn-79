# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Each miner's limit and market placements the engine answered are counted, accepted and refused, beside its request
stats for the same publish block, and published with them; a refusal for CONTRACT_VIOLATION (post-only, time in force)
is also counted apart.

The engine's answers reached each miner as notices and nowhere else, so nothing outside the miner could see how many
of its orders the engine refused. The published gauges make each uid's refusals visible.
"""
import inspect
import types

from taos.im.validator import forward, report


def _validator():
    return types.SimpleNamespace(miner_stats={
        3: {'requests': 0, 'timeouts': 0, 'failures': 0, 'rejections': 0, 'call_time': []}})


def _n(y, u=True, m=""):
    return {"y": y, "u": u, "m": m, "b": 5}


def test_accepted_and_refused_placements_are_counted_per_uid():
    v = _validator()
    forward.update_placement_stats(v, {3: [
        _n("RDPOL"), _n("RDPOM"), _n("RDPOL"),
        _n("ERDPOL", False, "BOOK 5 : FAILED TO PLACE SELL LIMIT ORDER FOR 1.0@29.55 : CONTRACT_VIOLATION"),
        _n("ERDPOM", False, "BOOK 5 : FAILED TO PLACE BUY MARKET ORDER : INSUFFICIENT_BALANCE"),
        _n("ET"), _n("RDCO"),
    ]})
    st = v.miner_stats[3]
    assert (st["placed"], st["refused"], st["refused_contract"]) == (3, 2, 1)


def test_counts_add_up_across_updates_and_a_new_uid_gets_a_row():
    v = _validator()
    forward.update_placement_stats(v, {3: [_n("RDPOL")]})
    forward.update_placement_stats(v, {3: [_n("ERDPOL", False, "x : CONTRACT_VIOLATION")], 7: [_n("RDPOL")]})
    assert (v.miner_stats[3]["placed"], v.miner_stats[3]["refused"]) == (1, 1)
    assert v.miner_stats[7]["placed"] == 1 and v.miner_stats[7]["requests"] == 0


def test_a_validator_without_notices_counts_nothing():
    v = _validator()
    forward.update_placement_stats(v, None)
    forward.update_placement_stats(v, {})
    assert "placed" not in v.miner_stats[3]


def test_the_block_publish_carries_and_resets_the_counts():
    src = inspect.getsource(report)
    for g in ('"placed"', '"refused"', '"refused_contract"'):
        assert g in src, g
    block = src[src.index("_ms = self.miner_stats.get(agentId)"):]
    block = block[:block.index("_set_if_changed_metric")]
    assert "'placed': 0" in block and "'refused': 0" in block and "'refused_contract': 0" in block


def test_the_validator_counts_each_state_once():
    from taos.im.neurons import validator as vmod
    src = inspect.getsource(vmod)
    i = src.index("self.process_resets(state)")
    assert "update_placement_stats(self, state.notices)" in src[i:i + 400]
