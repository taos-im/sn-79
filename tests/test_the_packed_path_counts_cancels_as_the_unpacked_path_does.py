# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Under neuron.orders_packed (the default) a miner's cancels are counted as they are without it.

The query process packs the orders, so main never sees them and takes the counts the query process returns. Both paths
count through one function, the placements and cancel instructions sent and the orders those cancels name, and main
keeps the named orders pending until their answer, so a cancel success for a named order is booked as a cancel and one
for an order no instruction named as a self-trade-prevention cancel. Statistics only: nothing in the scoring reads these
counters.
"""
import contextlib
import random
from types import SimpleNamespace

from taos.im.protocol import instructions as I
from taos.im.protocol.models import OrderDirection
from taos.im.protocol.response import FinanceAgentResponse
from taos.im.validator import forward
from taos.im.validator.query import QueryService

KEYS = ("placements_sent", "cancels_sent", "cancel_orders_sent", "orders_cancelled", "orders_not_found", "stp_cancels")


def _validator():
    return SimpleNamespace(
        config=SimpleNamespace(
            neuron=SimpleNamespace(timeout=3.0),
            scoring=SimpleNamespace(min_delay=10_000_000, max_delay=1_000_000_000,
                                    min_instruction_delay=5_000_000, max_instruction_delay=25_000_000)),
        simulation=SimpleNamespace(time_unit="ns"), miner_stats={}, engine=SimpleNamespace(mode="simulation"))


def _replies():
    def cancel(uid, book, ids):
        return I.CancelOrdersInstruction(agentId=uid, bookId=book,
                                         cancellations=[I.CancelOrderInstruction(orderId=i, volume=None) for i in ids])
    limit = I.PlaceLimitOrderInstruction(agentId=3, bookId=0, direction=OrderDirection.BUY, quantity=1.0, price=100.0,
                                         clientOrderId=None)
    return {
        3: SimpleNamespace(response=FinanceAgentResponse(agent_id=3, instructions=[limit, cancel(3, 0, [11, 12, 13])]),
                           dendrite=SimpleNamespace(process_time=0.4)),
        5: SimpleNamespace(response=FinanceAgentResponse(agent_id=5, instructions=[cancel(5, 1, [7])]),
                           dendrite=SimpleNamespace(process_time=1.7)),
    }


def _answers():
    def answer(y, book, ids):
        return {"y": y, "b": book, "c": [{"o": i, "b": book, "u": y[0] != "E"} for i in ids]}
    # uid 3: two of its three cancelled, one not found, and a self-trade cancel of order 40 nobody asked for
    return {3: [answer("RDCO", 0, [11, 13]), answer("ERDCO", 0, [12]), answer("RDCO", 0, [40])],
            5: [answer("RDCO", 1, [7])]}


def _counts(v):
    return {uid: tuple(row.get(k, 0) for k in KEYS) for uid, row in sorted(v.miner_stats.items())}


@contextlib.contextmanager
def _seeded(seed):
    """The delays draw from the global random; both paths draw the same sequence, and the state is put back."""
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def _unpacked():
    v = _validator()
    with _seeded(3):
        forward.update_sent_stats(v, forward.set_delays(v, _replies()))
    forward.update_placement_stats(v, _answers())
    return v


def _packed():
    v = _validator()
    with _seeded(3):
        result = QueryService._pack_orders(None, _replies(), forward.delay_settings(v))
    forward.take_packed_sent(v, result)
    forward.update_placement_stats(v, _answers())
    return v


def test_the_packed_path_counts_each_cancel_outcome_as_the_unpacked_path_does():
    unpacked, packed = _counts(_unpacked()), _counts(_packed())
    assert unpacked == {3: (1, 1, 3, 2, 1, 1), 5: (0, 1, 1, 1, 0, 0)}
    assert packed == unpacked


def test_the_packed_path_hands_main_the_orders_its_cancels_name():
    v = _validator()
    result = QueryService._pack_orders(None, _replies(), forward.delay_settings(v))
    assert sorted(map(tuple, result["cancel_named"])) == [(3, 0, 11), (3, 0, 12), (3, 0, 13), (5, 1, 7)]
    forward.take_packed_sent(v, result)
    assert set(v._cancel_pending) == {(3, 0, 11), (3, 0, 12), (3, 0, 13), (5, 1, 7)}


def test_a_result_from_a_query_process_without_the_names_still_counts_what_it_sent():
    v = _validator()
    forward.take_packed_sent(v, {"sent": {3: {"placements_sent": 1, "cancels_sent": 1}}})
    assert _counts(v)[3][:3] == (1, 1, 0)
