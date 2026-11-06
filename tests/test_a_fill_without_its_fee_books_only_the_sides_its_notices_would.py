# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""On the live path the de-beta accumulators book a fill's side only when the field its notice needs is there, as the
notice path does.

A maker's notice needs the maker fee (Mf) and a taker's the taker fee (Tf); the notice path skips a notice without it
(notice_fee raises MissingNoticeField) rather than reading 0, so that side's volume and PnL are not booked.
update_trade_volumes hands the de-beta accumulators each fill through fee_complete_sides: a fill without Mf has no
maker side (Ma -1), one without Tf no taker side (Ta -1), one without either is dropped; a fill with both is passed
unchanged."""
import inspect
from collections import defaultdict

from taos.im.validator import debeta, trade

_DROP = object()


def _fill(**over):
    t = {"y": "t", "p": 99.0, "q": 2.0, "s": 0, "Ma": 7, "Ta": 9, "Mf": 0.05, "Tf": 0.1}
    t.update(over)
    for k in [k for k, v in over.items() if v is _DROP]:
        del t[k]
    return t


def _book(t):
    buy, sell = defaultdict(dict), defaultdict(dict)
    debeta._attribute_capture(buy, sell, 0, t, 100.0, None, None, 1, fee_weight=1.0)
    return {u: b[0] for u, b in buy.items()}, {u: b[0] for u, b in sell.items()}


def test_a_fill_with_both_fees_passes_unchanged():
    t = _fill()
    assert trade.fee_complete_sides(t) is t


def test_a_fill_without_the_maker_fee_keeps_only_its_taker_side():
    for t in (_fill(Mf=_DROP), _fill(Mf=None), _fill(Mf="x")):
        out = trade.fee_complete_sides(t)
        assert out["Ma"] == -1 and out["Ta"] == 9 and t["Ma"] == 7, "a copy, the caller's fill untouched"
        assert _book(out) == ({9: 2.0}, {})


def test_a_fill_without_the_taker_fee_keeps_only_its_maker_side():
    out = trade.fee_complete_sides(_fill(Tf=_DROP))
    assert out["Ta"] == -1 and out["Ma"] == 7
    assert _book(out) == ({}, {7: -2.05})


def test_a_fill_without_either_fee_is_dropped():
    assert trade.fee_complete_sides(_fill(Mf=_DROP, Tf=None)) is None


def test_the_live_path_hands_the_accumulators_the_checked_fills():
    src = inspect.getsource(trade.update_trade_volumes)
    assert "de_trades = [x for x in map(fee_complete_sides, de_trades) if x is not None]" in src
