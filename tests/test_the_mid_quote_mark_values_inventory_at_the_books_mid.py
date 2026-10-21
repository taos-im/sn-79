# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""scoring.debeta.mark_mode=mid values inventory at the book's mid-quote from each state, not at the last trade.

The mid of the best bid and ask the state carries has no bid-ask bounce by construction, one state of lag and no
window. The trade path passes each book's mid from the state's levels; a state without both sides keeps the previous
mark.
"""
from collections import defaultdict

from taos.im.validator import trade
from taos.im.validator.debeta import accumulate_book_mtm


def _t(p, q, s, ma, ta):
    return {"p": p, "q": q, "s": s, "Ma": ma, "Ta": ta}


def _fresh():
    return (defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float)), {},
            defaultdict(lambda: defaultdict(float)), {}, {}, {})


def test_the_bounce_books_nothing_under_the_mid_mark():
    """A maker buys at the bid and sells at the ask while the mid stands still: mtm 0 under mid, nonzero under last."""
    for mode, mid in (("last", None), ("mid", 100.0)):
        mtm, invsum, invn, inv, pf, pl, drift = _fresh()
        st = {}
        batches = [[_t(99.9, 1.0, 1, 5, -1)], [_t(100.1, 1.0, 0, 5, -1)], [_t(99.9, 1.0, 1, 5, -1)], [_t(100.1, 1.0, 0, 5, -1)]]
        for b in batches:
            accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, b, drift=drift, mark_mode=mode, mark_window=0,
                                mark_state=st, mark_mid=mid)
        if mode == "last":
            assert abs(mtm[5][0]) > 0.0
        else:
            assert mtm[5][0] == 0.0 and drift.get(0, 0.0) == 0.0


def test_a_moving_mid_marks_inventory_and_the_drift():
    mtm, invsum, invn, inv, pf, pl, drift = _fresh()
    st = {}
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(100.0, 2.0, 1, 5, -1)], drift=drift, mark_mode="mid",
                        mark_window=0, mark_state=st, mark_mid=100.0)
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(100.9, 1.0, 1, -1, 7)], drift=drift, mark_mode="mid",
                        mark_window=0, mark_state=st, mark_mid=101.0)
    assert abs(mtm[5][0] - 2.0) < 1e-9, "a holder of 2 marked from 100 to 101"
    assert abs(drift[0] - 1.0) < 1e-9 and pl[0] == 101.0


def test_a_state_without_both_sides_keeps_the_previous_mark():
    mtm, invsum, invn, inv, pf, pl, drift = _fresh()
    st = {}
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(100.0, 1.0, 1, 5, -1)], drift=drift, mark_mode="mid",
                        mark_window=0, mark_state=st, mark_mid=100.0)
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(103.0, 1.0, 1, -1, 7)], drift=drift, mark_mode="mid",
                        mark_window=0, mark_state=st, mark_mid=None)
    assert mtm[5][0] == 0.0 and pl[0] == 100.0


def test_the_trade_path_reads_the_mid_from_the_states_levels():
    assert trade.book_mid({"b": [{"p": 99.0, "q": 1}], "a": [{"p": 101.0, "q": 1}]}) == 100.0
    assert trade.book_mid({"b": [], "a": [{"p": 101.0, "q": 1}]}) is None
    assert trade.book_mid({"b": [{"p": 102.0, "q": 1}], "a": [{"p": 101.0, "q": 1}]}) is None, "a crossed stand-in is no mark"
    import inspect
    src = inspect.getsource(trade.update_trade_volumes)
    assert "mark_mid=book_mid(book) if _mark_mode == 'mid' else None" in src


def test_the_dial_offers_mid():
    import argparse
    from taos.im.config import add_im_validator_args
    p = argparse.ArgumentParser(add_help=False)
    add_im_validator_args(None, p)
    assert vars(p.parse_args(["--scoring.debeta.mark_mode", "mid"]))["scoring.debeta.mark_mode"] == "mid"
