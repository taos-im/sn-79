# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner's cancels are counted per order: the orders its cancel instructions name, the orders the engine cancelled
and the orders it could not find (already filled or gone). The engine's answer to one instruction can be split
across both kinds, and a self-trade-prevention cancel arrives as a success answer for an order no instruction asked
to cancel, so it is counted on its own.
"""
from types import SimpleNamespace

from taos.im.protocol.response import FinanceAgentResponse
from taos.im.validator import block_stats, forward, report


def _validator(mode="simulation"):
    return SimpleNamespace(miner_stats={}, engine=SimpleNamespace(mode=mode))


def _cancel(uid, book, ids):
    r = FinanceAgentResponse(agent_id=uid)
    r.cancel_orders(book_id=book, order_ids=ids)
    return r


def _answer(y, book, ids):
    return {"y": y, "b": book, "c": [{"o": i, "b": book, "u": y[0] != "E"} for i in ids]}


def _counts(v, uid):
    st = v.miner_stats[uid]
    return tuple(st.get(k, 0) for k in ("cancels_sent", "cancel_orders_sent", "orders_cancelled", "orders_not_found",
                                        "stp_cancels"))


def test_a_split_answer_counts_each_order_once_by_its_outcome():
    v = _validator()
    forward.update_sent_stats(v, [_cancel(3, 0, [11, 12, 13])])
    forward.update_placement_stats(v, {3: [_answer("RDCO", 0, [11, 13]), _answer("ERDCO", 0, [12])]})
    assert _counts(v, 3) == (1, 3, 2, 1, 0)


def test_a_success_answer_for_an_order_no_instruction_named_is_a_self_trade_cancel():
    v = _validator()
    forward.update_sent_stats(v, [_cancel(3, 0, [11]), _cancel(4, 1, [11])])
    forward.update_placement_stats(v, {3: [_answer("RDCO", 0, [11]), _answer("RDCO", 0, [40])],
                                       4: [_answer("RDCO", 0, [11])]})
    assert _counts(v, 3) == (1, 1, 1, 0, 1)
    assert _counts(v, 4) == (1, 1, 0, 0, 1), "the asked order was on book 1; book 0's answer is another order"


def test_an_order_is_matched_once_so_a_second_success_for_it_is_not_its_cancel_again():
    v = _validator()
    forward.update_sent_stats(v, [_cancel(3, 0, [11])])
    forward.update_placement_stats(v, {3: [_answer("RDCO", 0, [11])]})
    forward.update_placement_stats(v, {3: [_answer("RDCO", 0, [11])]})
    assert _counts(v, 3) == (1, 1, 1, 0, 1)


def test_before_the_first_round_is_sent_an_unmatched_success_is_a_cancel_not_a_self_trade():
    v = _validator()
    forward.update_placement_stats(v, {3: [_answer("RDCO", 0, [11]), _answer("ERDCO", 0, [12])]})
    assert _counts(v, 3) == (0, 0, 1, 1, 0), "after a restart the first answers belong to cancels sent before it"


def test_the_exchange_redelivering_an_answer_counts_it_once():
    v = _validator("exchange")
    forward.update_sent_stats(v, [_cancel(3, 0, [11])])
    forward.update_placement_stats(v, {3: [_answer("RDCO", 0, [11])]})
    forward.update_placement_stats(v, {3: [_answer("RDCO", 0, [11])]})
    assert _counts(v, 3) == (1, 1, 1, 0, 0)


def test_answers_read_from_events_as_well_as_from_dicts():
    from taos.im.protocol.events import OrderCancellationEvent, OrderCancellationsEvent
    v = _validator()
    forward.update_sent_stats(v, [_cancel(3, 0, [11])])
    ev = OrderCancellationsEvent.model_construct(y="RDCO", b=0, c=[OrderCancellationEvent.model_construct(o=11, b=0)])
    forward.update_placement_stats(v, {3: [ev]})
    assert _counts(v, 3) == (1, 1, 1, 0, 0)


def test_the_block_publishes_the_per_order_series_and_no_longer_the_answer_counts():
    row = block_stats.empty_block_stats()
    names = [n for n, _ in report.round_block_gauges(row)]
    for name in ("cancels_sent", "cancel_orders_sent", "orders_cancelled", "orders_not_found", "stp_cancels"):
        assert name in names and row[name] == 0
    assert "cancels_ok" not in names and "cancels_failed" not in names
