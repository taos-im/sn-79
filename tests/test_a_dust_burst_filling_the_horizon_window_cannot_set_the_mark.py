# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Emil's report of 7 October on the realized basis: the horizon mark weights prints by size with the background
market on at least one side, but falls back to the plain average when the 31-print window holds no such print. A
burst of 31 or more dust prints between two cooperating uids, straddling the horizon of a third uid's genuine fill,
fills that window and sets the mark; since the markout quality sums over all of a uid's fills before the ratio, one
such fill lifts the uid's quality from 0 to 1 and pays its full captured credit. 0.26% of real windows reach the
fallback with no attack at all. The rule under test: a window without a background-sided print does not mark a fill
at a price the two uids chose, and one fill cannot lift a uid's quality beyond its own weight."""
import collections

from taos.im.validator.debeta import accumulate_book_capture, markout_quality

S, H, W = 1_000_000_000, 20_000_000_000, 15


def _dd():
    return collections.defaultdict(lambda: collections.defaultdict(float))


def _tape(pump, burst=40):
    # uid 1 sells a genuine fill to the background at 99.9 half a second in (its captured spread against a 100 mid is
    # positive); the background prints at 100 once a second; with the pump on, uids 2 and 3 print `burst` dust fills
    # to each other at 120 between 20.2 s and 20.8 s, so the fill's 20.5 s horizon lands in the middle of the burst
    # and the 31-print window around the horizon print holds no background-sided print at all
    tr = [{"p": 99.9, "q": 1.0, "s": 1, "Ma": 1, "Ta": -5, "ts": S // 2}]
    for k in range(1, 60):
        tr.append({"p": 100.0, "q": 1.0, "s": k % 2, "Ma": -3, "Ta": -4, "ts": k * S})
    if pump:
        for j in range(burst):
            tr.append({"p": 120.0, "q": 0.001, "s": j % 2, "Ma": 2, "Ta": 3, "ts": 20 * S + 200_000_000 + j * 15_000_000})
    return sorted(tr, key=lambda t: t["ts"])


def _quality(trades, live):
    cb, cs, rb, rs = _dd(), _dd(), _dd(), _dd()
    if live:
        st = {}
        for t in trades:
            accumulate_book_capture(cb, cs, 0, [t], W, ts=t["ts"], mid_state=st, real_buy_sums=rb, real_sell_sums=rs, horizon_ns=H)
    else:
        accumulate_book_capture(cb, cs, 0, trades, W, real_buy_sums=rb, real_sell_sums=rs, horizon_ns=H)
    return markout_quality(cb, cs, rb, rs, [1])[1], rb[1].get(0, 0.0)


def test_a_burst_that_fills_the_window_does_not_move_the_mark_on_either_path():
    for live in (False, True):
        q_clean, r_clean = _quality(_tape(False), live)
        q_pump, r_pump = _quality(_tape(True), live)
        assert abs(r_pump - r_clean) < 1e-6, f"live={live}: realized credit {r_clean} -> {r_pump} under the burst"
        assert abs(q_pump - q_clean) < 1e-9, f"live={live}: quality {q_clean} -> {q_pump} under the burst"
