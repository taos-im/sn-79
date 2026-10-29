# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A fill waits in the capture queue until enough prints follow it to mark it (up to the flush time), so at a reset
the departed miner's last fills are still queued. The reset clears the uid's settled sums but used to leave the queue,
and the next miner on that slot then settled the departed miner's capture. The reset now drops the uid's side of every
queued fill on both queues; the counterparty's side stays queued and settles as before."""
import inspect

from taos.im.validator import debeta, trade


def _fills():
    # side 1: the maker (Ma) buys, the taker (Ta) sells; uid 7 makes against uid 9, then uid 9 against uid 3
    return [{"p": 100.0, "q": 1.0, "s": 1, "Ma": 7, "Ta": 9, "ts": 1},
            {"p": 100.2, "q": 1.0, "s": 0, "Ma": 3, "Ta": 9, "ts": 2},
            {"p": 99.8, "q": 1.0, "s": 1, "Ma": 3, "Ta": 7, "ts": 3}]


def _queued():
    """A live capture state with the three fills still queued (W far beyond the prints seen)."""
    state, buy, sell = {}, {}, {}
    from collections import defaultdict
    buy, sell = defaultdict(dict), defaultdict(dict)
    rbuy, rsell = defaultdict(dict), defaultdict(dict)
    debeta.accumulate_book_capture(buy, sell, 0, _fills(), 50, mid_state=state, ts=10, now_ns=10,
                                   real_buy_sums=rbuy, real_sell_sums=rsell)
    return state, buy, sell, rbuy, rsell


def _settle(state, buy, sell, rbuy, rsell):
    debeta.flush_capture_state(state, buy, sell, 50, ts=11, now_ns=11, force=True,
                               real_buy_sums=rbuy, real_sell_sums=rsell)


def test_the_fills_are_queued_before_the_reset():
    state, buy, sell, *_ = _queued()
    assert len(state[0]["pend"]) == 3 and not buy and not sell, "nothing settles until the window fills"


def test_without_the_drop_the_queued_fills_settle_to_the_slot():
    state, buy, sell, rbuy, rsell = _queued()
    _settle(state, buy, sell, rbuy, rsell)
    assert 7 in buy or 7 in sell, "the departed uid's queued fills settle to its slot"


def test_the_drop_leaves_the_uid_nothing_and_the_counterparty_its_side():
    state, buy, sell, rbuy, rsell = _queued()
    ref = _queued()
    _settle(*ref)
    debeta.drop_uid_from_pending(state, 7)
    _settle(state, buy, sell, rbuy, rsell)
    for sums in (buy, sell, rbuy, rsell):
        assert 7 not in sums or not any(sums[7].values()), "the reset uid books nothing from its old fills"
    assert sell[9] == ref[2][9], "uid 9's taker side of its fill against uid 7 still settles, unchanged"
    assert buy[3] == ref[1][3] and sell.get(3) == ref[2].get(3), "fills not involving uid 7 are untouched"


def test_the_drop_copies_the_fill_rather_than_editing_a_shared_one():
    state, *_ = _queued()
    shared = state[0]["pend"][0][2]
    debeta.drop_uid_from_pending(state, 7)
    assert shared["Ma"] == 7, "a fill object other queues or callers hold is not edited in place"
    assert state[0]["pend"][0][2]["Ma"] == -1 and state[0]["pend"][0][2]["Ta"] == 9


def test_the_reset_calls_it():
    src = inspect.getsource(trade.reset_agent_histories)
    assert "drop_uid_from_pending(getattr(self, 'debeta_capture_mid', None), uid)" in src
