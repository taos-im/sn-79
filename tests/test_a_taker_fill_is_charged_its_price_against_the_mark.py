# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""scoring.debeta.fill_charge charges a taker fill its price against the mark at the fill.

Under the mid, vwap and median marks every fill of a batch is valued at the mark and the fill price enters the
notional only, so a taker that buys at 319 with the mark at 289.44 books no loss and, when the mark moves to 309.2,
its inventory shows +19.76 a unit: a sweep through a thin book is scored as if it traded at the mark. With
fill_charge=taker the buyer's mtm takes
(mark - fill) x q at the fill and the seller's (fill - mark) x q, so the same buy books -29.56 at once and -9.80 after
the move, which is what it made. The resting side stays marked (its spread is the making leg's); fill_charge=both
credits it too, and none keeps the path as it was.
"""
from collections import defaultdict
from pathlib import Path

from taos.im.validator.debeta import accumulate_book_mtm

ROOT = Path(__file__).resolve().parents[1]


def _t(p, q, s, ma, ta):
    return {"p": p, "q": q, "s": s, "Ma": ma, "Ta": ta}


def _fresh():
    return (defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float)), {},
            defaultdict(lambda: defaultdict(float)), {}, {}, {})


def _run(charge, mode="mid", mids=(289.44, 309.2), window=0):
    """A taker (uid 7) buys 1.0 at 319 from a resting maker (uid 5) with the mark at the first mid, then a background
    print at the second mid moves the mark. Returns (taker mtm, maker mtm, mtm history)."""
    mtm, invsum, invn, inv, pf, pl, drift = _fresh()
    st, hist = {}, {}
    kw = dict(drift=drift, mark_mode=mode, mark_window=window, mark_state=st, mtm_hist=hist, ts=1, fill_charge=charge)
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(319.0, 1.0, 0, 5, 7)], mark_mid=mids[0], **kw)
    first = (mtm[7][0], mtm[5][0])
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(mids[1], 1.0, 1, -1, -2)], mark_mid=mids[1], **kw)
    return first, (mtm[7][0], mtm[5][0]), hist


def test_a_sweep_buy_is_charged_what_it_paid_over_the_mark_and_keeps_only_what_it_made():
    first, after, _ = _run("taker")
    assert abs(first[0] + 29.56) < 1e-9, "the buyer at 319 with the mark at 289.44 is charged 29.56 at the fill"
    assert abs(after[0] + 9.80) < 1e-9, "after the mark moves to 309.2 the buy shows what it made: 309.2 - 319"
    assert first[1] == 0.0 and abs(after[1] + 19.76) < 1e-9, "the resting seller is marked, not credited the spread"


def test_without_the_charge_the_same_buy_scores_the_marks_move_as_alpha():
    first, after, _ = _run("none")
    assert first[0] == 0.0 and abs(after[0] - 19.76) < 1e-9, "without the charge the fill price drops out"


def test_both_credits_the_resting_side_at_its_fill_price():
    first, after, _ = _run("both")
    assert abs(first[1] - 29.56) < 1e-9 and abs(after[1] - 9.80) < 1e-9


def test_under_the_last_mark_the_charge_is_nothing_because_the_mark_is_the_fill():
    first, after, _ = _run("taker", mode="last")
    assert first[0] == 0.0 and first[1] == 0.0, "the mark is the fill, so there is nothing to charge"
    assert abs(after[0] + 9.80) < 1e-9, "the next print at 309.2 marks the buy down, as the last mark always did"


def test_under_the_median_mark_the_charge_is_against_the_window_median():
    mtm, invsum, invn, inv, pf, pl, drift = _fresh()
    st, hist = {}, {}
    kw = dict(drift=drift, mark_mode="median", mark_window=5, mark_state=st, mtm_hist=hist, ts=1, fill_charge="taker")
    warm = [_t(300.0, 1.0, 1, -1, -2), _t(300.0, 1.0, 1, -1, -2), _t(300.0, 1.0, 1, -1, -2), _t(300.0, 1.0, 1, -1, -2)]
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, warm, **kw)
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(330.0, 2.0, 0, 5, 7)], **kw)
    assert abs(mtm[7][0] + 60.0) < 1e-9, "the window's median stays 300 against one print at 330: charged 30 x 2"


def test_the_charge_is_in_the_mtm_history_so_a_boundary_shift_and_a_prune_carry_it():
    _, after, hist = _run("taker")
    assert abs(sum(hist[7][0].values()) - after[0]) < 1e-9
    assert abs(sum(hist[5][0].values()) - after[1]) < 1e-9


def test_the_trade_path_passes_the_dial_to_both_marks_and_the_dial_defaults_to_taker():
    trade = (ROOT / "taos" / "im" / "validator" / "trade.py").read_text()
    assert trade.count("fill_charge=_fill_charge") >= 2, "the paid mark and the alternative mark take the same charge"
    cfg = (ROOT / "taos" / "im" / "config" / "__init__.py").read_text()
    i = cfg.index('"--scoring.debeta.fill_charge"')
    block = cfg[i:i + 1200]
    assert 'choices=["none", "taker", "both"]' in block and 'default="taker"' in block
    reward = (ROOT / "taos" / "im" / "validator" / "reward.py").read_text()
    assert "'fill_charge': str(debeta_dial(_dcfg, 'fill_charge'))" in reward, "the dial is published with the others"
