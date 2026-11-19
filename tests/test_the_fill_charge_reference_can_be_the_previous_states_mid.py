"""scoring.debeta.charge_ref sets what a taker fill is charged against.

Under 'mark' (0.6.7's) a taker fill is charged against the state's mark at the fill: under the mid mark, the book's
mid after the interval's trades, so a momentum taker whose own batch moved the mid is charged only the part of its
price above the moved mid and is credited the move inside the interval. On mainnet's storm night that residual kept
the sweeping family 13 to 24 per cent of the skill pool. Under 'prev' the reference is the book's mid as it stood at
the end of the previous state, before any of the interval's trades, the quote the taker crossed: its own move is
charged to it in full. The resting side stays marked either way."""
from collections import defaultdict

from taos.im.validator.debeta import accumulate_book_mtm


def _t(p, q, s, ma, ta):
    return {"p": p, "q": q, "s": s, "Ma": ma, "Ta": ta}


def _run(charge_ref):
    """The mid stands at 289.44 after the previous state. In this state the taker (uid 7) buys 1.0 at 319 from the
    resting maker (uid 5) and the batch leaves the mid at 309.2. Returns (taker mtm, maker mtm) after the batch."""
    mtm, invsum, invn, inv, pf, pl, drift = (defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float)), {},
                                             defaultdict(lambda: defaultdict(float)), {}, {}, {})
    st, hist = {}, {}
    kw = dict(drift=drift, mark_mode="mid", mark_window=0, mark_state=st, mtm_hist=hist, ts=1, fill_charge="taker",
              charge_ref=charge_ref)
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(289.44, 1.0, 1, -1, -2)], mark_mid=289.44, **kw)
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(319.0, 1.0, 0, 5, 7)], mark_mid=309.2, **kw)
    return mtm[7][0], mtm[5][0]


def test_under_the_mark_the_taker_is_charged_only_above_the_mid_its_own_batch_moved():
    taker, maker = _run("mark")
    assert abs(taker + 9.80) < 1e-9, "319 against the post-batch mid 309.2: the move 289.44 to 309.2 is credited"
    assert maker == 0.0


def test_under_prev_the_taker_is_charged_against_the_quote_it_crossed():
    taker, maker = _run("prev")
    assert abs(taker + 29.56) < 1e-9, "319 against the previous state's mid 289.44: its own move is charged in full"
    assert maker == 0.0, "the resting side stays marked"


def test_the_first_batch_of_a_book_has_no_previous_mid_and_charges_against_the_mark():
    mtm, invsum, invn, inv, pf, pl, drift = (defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float)), {},
                                             defaultdict(lambda: defaultdict(float)), {}, {}, {})
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(319.0, 1.0, 0, 5, 7)], drift=drift, mark_mode="mid",
                        mark_state={}, ts=1, fill_charge="taker", charge_ref="prev", mark_mid=309.2)
    assert abs(mtm[7][0] + 9.80) < 1e-9


def test_the_dial_reaches_the_accumulator_from_the_trade_path():
    import inspect
    from taos.im.validator import trade
    src = inspect.getsource(trade.update_trade_volumes)
    assert "_charge_ref = str(debeta_dial(_debeta_cfg, 'charge_ref'))" in src
    assert src.count("charge_ref=_charge_ref") == 2, "the paid leg and the alternative mark's leg"
