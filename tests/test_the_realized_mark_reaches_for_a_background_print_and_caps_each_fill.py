# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The two halves of the 7 October hardening of the realized mark. (1) No plain-average fallback: when the centred
window at a fill's horizon holds no background-sided print, the mark reaches outward to the nearest ones, and a fill
with none in reach books nothing. (2) Each fill's realized spread is clamped to plus or minus its own captured spread
before the sums, so one fill moves a uid's markout quality by at most its own weight. Both on the batch path and the
live path, which must agree."""
import collections

from taos.im.validator.debeta import (MARK_REACH_WINDOWS, _capped_mark, accumulate_book_capture, centered_mark,
                                      markout_quality)

S, H, W = 1_000_000_000, 20_000_000_000, 15


def _dd():
    return collections.defaultdict(lambda: collections.defaultdict(float))


def _run(trades, live, uid=1):
    cb, cs, rb, rs = _dd(), _dd(), _dd(), _dd()
    if live:
        st = {}
        for t in sorted(trades, key=lambda x: x["ts"]):
            accumulate_book_capture(cb, cs, 0, [t], W, ts=t["ts"], mid_state=st, real_buy_sums=rb, real_sell_sums=rs, horizon_ns=H)
    else:
        accumulate_book_capture(cb, cs, 0, sorted(trades, key=lambda x: x["ts"]), W, real_buy_sums=rb, real_sell_sums=rs, horizon_ns=H)
    captured = cb[uid].get(0, 0.0) + cs[uid].get(0, 0.0)
    realized = rb[uid].get(0, 0.0) + rs[uid].get(0, 0.0)
    return captured, realized, markout_quality(cb, cs, rb, rs, [uid])[uid]


def _background(start_s, end_s, price=100.0):
    return [{"p": price, "q": 1.0, "s": k % 2, "Ma": -3, "Ta": -4, "ts": k * S} for k in range(start_s, end_s)]


def test_the_mark_reaches_past_a_dust_filled_window_to_the_nearest_background_prints():
    # the fill's 20.5 s horizon lands inside a dust burst of 40 prints; the background printed at 100 one second
    # before and after; the mark reaches to those prints and the fill is graded as on a clean tape
    fill = {"p": 99.9, "q": 1.0, "s": 1, "Ma": 1, "Ta": -5, "ts": S // 2}
    dust = [{"p": 120.0, "q": 0.001, "s": j % 2, "Ma": 2, "Ta": 3, "ts": 20 * S + 200_000_000 + j * 15_000_000} for j in range(40)]
    for live in (False, True):
        c_clean, r_clean, q_clean = _run([fill] + _background(1, 60), live)
        c_pump, r_pump, q_pump = _run([fill] + _background(1, 60) + dust, live)
        assert abs(r_pump - r_clean) < 1e-6, (live, r_clean, r_pump)
        assert abs(q_pump - q_clean) < 1e-9, (live, q_clean, q_pump)


def test_a_fill_with_no_background_print_in_reach_books_nothing():
    prices = [100.0] * 10 + [120.0] * (2 * MARK_REACH_WINDOWS * W + 10) + [100.0] * 10
    weights = [1.0] * 10 + [0.0] * (2 * MARK_REACH_WINDOWS * W + 10) + [1.0] * 10
    marks = centered_mark(prices, weights, W)
    middle = len(prices) // 2
    assert marks[middle] is None, "no background-sided print within reach: no mark"
    assert marks[5] == 100.0 and marks[-5] == 100.0
    t = {"p": 99.9, "q": 1.0}
    assert _capped_mark(t, None, 100.0) == 99.9, "an unmarked fill books zero realized spread"


def test_one_fill_cannot_lift_the_quality_beyond_its_own_weight():
    # fill A captured 0.1 at a 100 mid; the honest market then runs to 150 before A's horizon, which would book a
    # realized spread of 50.1 on a fill whose captured spread is 0.1. Capped, A books 0.1 and the uid's quality is
    # its weighted average over fills, here 1.0 from a single fill and no more
    fill = {"p": 99.9, "q": 1.0, "s": 1, "Ma": 1, "Ta": -5, "ts": S // 2}
    tape = [fill] + _background(1, 15) + [{"p": 150.0, "q": 1.0, "s": k % 2, "Ma": -3, "Ta": -4, "ts": k * S} for k in range(15, 60)]
    for live in (False, True):
        captured, realized, q = _run(tape, live)
        assert abs(realized) <= abs(captured) + 1e-9, (live, captured, realized)
        assert q <= 1.0 + 1e-12


def test_capping_leaves_an_honest_fill_alone_and_clamps_an_excessive_one():
    # a buy at 99.9 for 2 against a 100 mid captured 0.2; a mark within that leaves the realized spread alone
    t = {"p": 99.9, "q": 2.0}
    assert _capped_mark(t, 99.95, 100.0) == 99.95
    # a mark 0.6 above the fill would book 1.2, six times the captured spread: clamped to 0.2, i.e. the mid
    assert abs(_capped_mark(t, 100.5, 100.0) - 100.0) < 1e-12
    # and symmetrically below: -1.8 clamped to -0.2, a mark 0.1 under the fill
    assert abs(_capped_mark(t, 99.0, 100.0) - 99.8) < 1e-12
