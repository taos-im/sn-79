# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A pending capture fill waits up to 300 simulated seconds for its forward side, and every settled fill is counted
as settled full or settled cut, per book, then per asset class for the report.

The counts make the share of fills settled cut visible: the validator publishes them per class (class_gauges
capture_full, capture_cut, capture_cut_share).
"""
from collections import defaultdict

from taos.im.validator import debeta
from taos.im.validator.debeta import (CAPTURE_FLUSH_NS, accumulate_book_capture, capture_settle_by_class,
                                      capture_settle_counts, flush_capture_state)

S = 1_000_000_000
W = 15


def _fill(t_s, p=100.0, q=1.0):
    return {"p": p, "q": q, "s": 0, "Ma": -1, "Ta": 5, "t": int(t_s * S)}


def _bg(t_s, p=100.0, q=1.0):
    return {"p": p, "q": q, "s": 1, "Ma": -1, "Ta": -2, "t": int(t_s * S)}


def _sums():
    return defaultdict(dict), defaultdict(dict)


def test_the_flush_is_300_simulated_seconds():
    assert CAPTURE_FLUSH_NS == 300 * S


def test_a_quiet_book_holds_its_fill_past_60s_and_settles_it_cut_at_300s():
    buy, sell = _sums()
    state = {}
    accumulate_book_capture(buy, sell, 0, [_fill(0), _bg(10), _bg(20), _bg(30)], W, mid_state=state, ts=0,
                            now_ns=70 * S)
    st = state[0]
    assert len(st["pend"]) == 4, "nothing settles at 70 s: no forward side is complete and none is stale"
    assert st.get("n_cut", 0) == 0 and st.get("n_full", 0) == 0
    flush_capture_state(state, buy, sell, W, ts=0, now_ns=301 * S)
    assert len(st["pend"]) == 3 and st["n_cut"] == 1, "the fill that has waited 300 s settles, cut"
    flush_capture_state(state, buy, sell, W, ts=0, now_ns=331 * S)
    assert not st["pend"]
    assert st["n_cut"] == 4 and st.get("n_full", 0) == 0


def test_a_busy_book_settles_full():
    buy, sell = _sums()
    state = {}
    prints = [_fill(0)] + [_bg(0.01 * k) for k in range(1, 41)]
    accumulate_book_capture(buy, sell, 0, prints, W, mid_state=state, ts=0, now_ns=1 * S)
    st = state[0]
    assert st["n_full"] == len(prints) - W and st.get("n_cut", 0) == 0, "every fill with W later prints settles full"


def test_a_forced_drain_counts_the_incomplete_fills_as_cut():
    buy, sell = _sums()
    state = {}
    prints = [_fill(0)] + [_bg(0.01 * k) for k in range(1, 21)]
    accumulate_book_capture(buy, sell, 0, prints, W, mid_state=state, ts=0, now_ns=1 * S)
    flush_capture_state(state, buy, sell, W, ts=0, now_ns=1 * S, force=True)
    st = state[0]
    assert st["n_full"] == len(prints) - W and st["n_cut"] == W


def test_the_counts_roll_up_per_book_and_per_class():
    state = {0: {"n_full": 90, "n_cut": 10}, 1: {"n_full": 50, "n_cut": 0}, 2: {"n_full": 30, "n_cut": 20},
             3: {"pend": []}}
    assert capture_settle_counts(state) == {0: {"full": 90, "cut": 10}, 1: {"full": 50, "cut": 0},
                                            2: {"full": 30, "cut": 20}}
    by_class = capture_settle_by_class(capture_settle_counts(state), {0: 0, 1: 0, 2: 1})
    assert by_class == {0: {"full": 140, "cut": 10, "cut_share": 10 / 150},
                        1: {"full": 30, "cut": 20, "cut_share": 20 / 50}}
    assert capture_settle_by_class(capture_settle_counts(state), {}) == {
        0: {"full": 170, "cut": 30, "cut_share": 30 / 200}}, "a single market is one class"


def test_the_validator_ships_and_the_report_publishes_the_class_counts():
    import inspect
    from taos.im.neurons import validator
    from taos.im.validator import report
    assert "'capture_settle_by_class': capture_settle_by_class(" in inspect.getsource(validator)
    src = inspect.getsource(report)
    for gauge in ("capture_full", "capture_cut", "capture_cut_share"):
        assert f'"{gauge}"' in src
    assert "publish_capture_settle_gauges(self, updates, wallet_addr, netuid, simid)" in src


def test_the_queue_format_is_unchanged():
    assert debeta._finalize_ready.__code__.co_varnames[:11] == (
        "st", "book_id", "buy_sums", "sell_sums", "W", "buy_hist", "sell_hist", "ts", "now_ns", "flush_ns", "force")
