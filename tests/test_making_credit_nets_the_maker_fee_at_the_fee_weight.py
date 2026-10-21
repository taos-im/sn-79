"""Fee-aware making credit (0.6.4): a maker's captured and realized spread are booked net
of ``scoring.debeta.making_fee_weight`` times the maker fee it paid on the fill (a rebate, a negative fee, adds), so
the making pool pays the spread a maker earns after the venue's charge, the economics of a real market maker. The
weight is stepped: 0.005 first, raised at later boundaries as maker fees fall. Only the maker's side of a fill is
charged; the taker's side is booked as before. At weight 0 every sum is unchanged."""
import argparse
import inspect
from collections import defaultdict
from pathlib import Path

from taos.im.config import add_im_validator_args
from taos.im.validator import debeta

DEV = Path(__file__).resolve().parents[1]


def _sums():
    return defaultdict(dict), defaultdict(dict)


def _fill(s, p=100.0, q=2.0, ma=5, ta=-3, mf=0.4):
    # s 0: the taker bought, so the maker (uid 5) sold
    return {"y": "t", "s": s, "p": p, "q": q, "Ma": ma, "Ta": ta, "Mf": mf, "Tf": 0.0, "t": 1}


def test_the_maker_side_is_booked_net_of_the_weighted_fee_and_the_taker_side_is_not():
    b, s = _sums()
    debeta._attribute_capture(b, s, 0, _fill(0, ta=7), 99.5, None, None, 0, fee_weight=0.5)
    assert abs(s[5][0] - (1.0 - 0.2)) < 1e-12, s[5][0]      # maker sold 2 at 100 against a 99.5 mid: +1.0, fee 0.5 x 0.4
    assert abs(b[7][0] - (-1.0)) < 1e-12, b[7][0]          # the taker's side is unchanged


def test_a_rebate_adds_to_the_maker_side():
    b, s = _sums()
    debeta._attribute_capture(b, s, 0, _fill(1, p=100.0, mf=-0.4, ma=5, ta=-3), 99.5, None, None, 0, fee_weight=1.0)
    assert abs(b[5][0] - (-1.0 + 0.4)) < 1e-12, b[5][0]


def test_the_realized_pair_is_net_of_the_fee_too():
    b, s = _sums()
    debeta._attribute_capture(b, s, 0, _fill(0, ta=7), 99.5, None, None, 0, maker_only=True, fee_weight=1.0)
    assert abs(s[5][0] - (1.0 - 0.4)) < 1e-12 and 7 not in b


def test_weight_zero_books_exactly_what_it_did():
    b0, s0 = _sums()
    b1, s1 = _sums()
    debeta._attribute_capture(b0, s0, 0, _fill(0, ta=7), 100.5, None, None, 0)
    debeta._attribute_capture(b1, s1, 0, _fill(0, ta=7), 100.5, None, None, 0, fee_weight=0.0)
    assert dict(b0) == dict(b1) and dict(s0) == dict(s1)


def _stream(n=60):
    out = []
    for i in range(n):
        out.append({"y": "t", "s": i % 2, "p": 100.0 + (0.02 if i % 2 == 0 else -0.02), "q": 1.0,
                    "Ma": 5, "Ta": -1 - (i % 3), "Mf": 0.01, "Tf": 0.0, "t": i * 1_000_000_000, "i": i})
    return out


def test_the_live_path_nets_the_same_fees_as_the_batch_path():
    trades = _stream()
    totals = {}
    for live in (False, True):
        for w in (0.0, 1.0):
            b, s = _sums()
            kw = {"mid_state": {}} if live else {}
            debeta.accumulate_book_capture(b, s, 0, trades, 15, ts=0, now_ns=10 ** 15, fee_weight=w, **kw)
            if live:
                debeta.flush_capture_state(kw["mid_state"], b, s, 15, ts=0, force=True, fee_weight=w)
            totals[(live, w)] = sum(b[5].values()) + sum(s[5].values())
    fees = sum(t["Mf"] for t in trades)
    for live in (False, True):
        assert abs(totals[(live, 0.0)] - totals[(live, 1.0)] - fees) < 1e-9, totals
    assert abs(totals[(False, 1.0)] - totals[(True, 1.0)]) < 1e-9, totals


def test_the_dial_ships_at_the_first_step():
    p = argparse.ArgumentParser(add_help=False)
    add_im_validator_args(None, p)
    assert p.get_default("scoring.debeta.making_fee_weight") == 0.005


def test_main_and_the_scoring_child_read_the_same_weight_at_every_capture_call():
    trade = (DEV / "taos/im/validator/trade.py").read_text()
    assert trade.count("fee_weight=") >= 3, "the live accumulator, the per-update flush and the boundary flush"
    from types import SimpleNamespace
    from taos.im.validator.scoring_shadow import _debeta_duck
    assert _debeta_duck(SimpleNamespace(making_fee_weight=0.01)).making_fee_weight == 0.01, (
        "the child's config carries the dial, or it nets nothing while main does")
    assert "fee_weight" in inspect.signature(debeta.flush_capture_state).parameters
