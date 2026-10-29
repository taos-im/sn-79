# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Each miner's block of rounds also counts the engine's view of its instructions: placements and cancel instructions
the validator sent, the orders the cancels named by outcome, fills, and instructions the caps dropped, published with
the request stats and reset with them.

The placements the engine accepted and refused were already counted from its answers; the rest of what an agent asks
and gets was visible only in logs. These gauges let the agent page read the whole exchange per block.
"""
from types import SimpleNamespace

from bittensor.core.synapse import TerminalInfo

from taos.im.protocol import MarketSimulationStateUpdate
from taos.im.protocol.models import OrderDirection
from taos.im.protocol.response import FinanceAgentResponse
from taos.im.validator import block_stats, forward, report
from taos.im.validator.query import QueryService


def _validator():
    return SimpleNamespace(miner_stats={})


def _n(y, m="", c=None):
    return {"y": y, "u": y[0] != "E", "m": m, "b": 5, "c": c or []}


def test_cancel_answers_are_counted_per_order_beside_the_placements():
    v = _validator()
    forward.update_placement_stats(v, {3: [_n("RDCO", c=[{"o": 1}]), _n("RDCO", c=[{"o": 2}]),
                                           _n("ERDCO", "x : ORDER NOT FOUND", c=[{"o": 3}]), _n("RDPOL"), _n("ET")]})
    st = v.miner_stats[3]
    assert (st["orders_cancelled"], st["orders_not_found"], st["placed"], st["fills"]) == (2, 1, 1, 1)


def _fill(book, tid):
    return {"y": "ET", "b": book, "i": tid, "Ma": 3, "Ta": 8}


def test_a_fill_the_exchange_redelivers_is_counted_once_and_a_simulation_fill_each_time():
    xch = SimpleNamespace(miner_stats={}, engine=SimpleNamespace(mode="exchange"))
    forward.update_placement_stats(xch, {3: [_fill(0, 7), _fill(0, 7), _fill(1, 7)]})
    forward.update_placement_stats(xch, {3: [_fill(0, 7)], 8: [_fill(0, 7)]})
    assert (xch.miner_stats[3]["fills"], xch.miner_stats[8]["fills"]) == (2, 1)
    sim = SimpleNamespace(miner_stats={}, engine=SimpleNamespace(mode="simulation"))
    forward.update_placement_stats(sim, {3: [_fill(0, 7), _fill(0, 7)]})
    assert sim.miner_stats[3]["fills"] == 2


def test_the_instructions_sent_are_counted_by_kind_per_agent():
    v = _validator()
    r = FinanceAgentResponse(agent_id=4)
    r.limit_order(book_id=0, direction=OrderDirection.BUY, quantity=1.0, price=299.0)
    r.limit_order(book_id=1, direction=OrderDirection.SELL, quantity=1.0, price=301.0)
    r.cancel_orders(book_id=0, order_ids=[11, 12])
    forward.update_sent_stats(v, [r])
    assert (v.miner_stats[4]["placements_sent"], v.miner_stats[4]["cancels_sent"]) == (2, 1)


def test_a_row_restored_from_an_older_state_takes_the_new_counters_from_zero():
    v = SimpleNamespace(miner_stats={9: {"requests": 5, "timeouts": 0, "failures": 0, "rejections": 0, "call_time": []}})
    forward.update_placement_stats(v, {9: [_n("ERDCO", c=[{"o": 4}])]})
    block_stats.bump(v.miner_stats, 9, "fills")
    names = dict(report.round_block_gauges(v.miner_stats[9]))
    assert (names["orders_not_found"], names["fills"], names["placements_sent"], names["dropped"], names["requests"]) == (1, 1, 0, 0, 5)


def _reply(uid, orders):
    synapse = MarketSimulationStateUpdate(timestamp=1, version=64, compression_engine="lz4")
    synapse.response = FinanceAgentResponse(agent_id=uid)
    for book in orders:
        synapse.response.limit_order(book_id=book, direction=OrderDirection.BUY, quantity=1.0, price=299.0)
    synapse = synapse.clear_inputs().compress()
    synapse.dendrite = TerminalInfo(status_code=200)
    synapse.axon = TerminalInfo(status_code=200)
    return synapse


def test_instructions_over_the_per_book_cap_are_counted_as_dropped_for_their_uid():
    service = SimpleNamespace()
    synapses = {1: _reply(1, [0, 0, 0]), 2: _reply(2, [0])}
    request = {"version": 64, "books": {}, "book_count": 1, "book_ids": [0], "volume_sums": {},
               "max_instructions_per_book": 2}
    stats = QueryService.validate_responses(service, synapses, request, set())
    assert stats[1] == 3                       # two of uid 1's three kept, uid 2's one kept
    assert service._dropped_instructions == {1: 1}


def test_the_dropped_counts_reach_each_miners_block_from_the_query_result():
    v = _validator()
    forward.update_dropped_stats(v, {"validation_stats": {"dropped": {1: 2, 6: 1}}})
    forward.update_dropped_stats(v, {"validation_stats": {}})
    assert (v.miner_stats[1]["dropped"], v.miner_stats[6]["dropped"]) == (2, 1)


def test_the_block_publishes_the_engine_counters_and_the_reset_row_holds_them_at_zero():
    row = block_stats.empty_block_stats()
    names = [n for n, _ in report.round_block_gauges(row)]
    for name in ("placements_sent", "cancels_sent", "cancel_orders_sent", "orders_cancelled", "orders_not_found",
                 "stp_cancels", "fills", "dropped"):
        assert name in names and row[name] == 0
    assert row["call_time"] == []
