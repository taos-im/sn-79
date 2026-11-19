# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A trade record carries the settlement legs the engine executed, or says it does not know them.

Reported live: a resting BUY with a positive maker fee was credited LESS base than the trade said.
The books were right. ClearingManager::handleTrade computes the maker fee in quote, but a buyer has
no quote left to charge (its reservation is consumed by the trade), so the fee is converted to base,
rounded UP to the base grid, and the round-up overshoot is returned as a quote deposit. The wire
carried only the quote fee `Mf`, so the base actually docked and the quote returned were invisible,
and on a full fill the quote that moved is the reservation remainder, not price*quantity.

The engine now states them per side: `Mfb`/`Tfb` (base docked), `Mfr`/`Tfr` (quote returned) and
`qs` (quote that changed hands). Two rules, the same ones `Mf`/`Tf` follow:

  * a producer that knows a leg states it, zero included -- in exchange mode nothing is charged and
    the fee legs are TRUE zeros, so the canonical notice builder emits them;
  * a consumer never substitutes a default. A record written before the fields existed reads back
    as None, which is distinguishable from a genuine zero leg.
"""
from taos.im.protocol.exchange.models import TradeInfo as ExchangeTradeInfo
from taos.im.protocol.models import TradeInfo as SimTradeInfo
from taos.im.validator.engines import NormalizedTradeEvent


WIRE = {"i": 7, "t": 1, "q": 2.0, "s": 1, "p": 300.0,
        "Ta": -3, "Ti": 9, "Ma": 171, "Mi": 8, "Mf": 0.06, "Tf": 0.09}
# The resting BUY above: 0.06 quote of maker fee at 300 is 0.0002 base exactly, so no overshoot.
LEGS = {"Mfb": 0.0002, "Mfr": 0.0, "Tfb": 0.0, "Tfr": 0.0, "qs": 600.0}
EVENT = {"tradeId": 7, "timestamp": 1, "volume": 2.0, "direction": 1, "price": 300.0,
         "aggressingAgentId": -3, "aggressingOrderId": 9, "restingAgentId": 171, "restingOrderId": 8,
         "fees": {"maker": 0.06, "taker": 0.09}}
SETTLEMENT = {"makerFeeBase": 0.0002, "makerRefund": 0.0, "takerFeeBase": 0.0, "takerRefund": 0.0,
              "quoteSettled": 600.0}


def _legs(t):
    return (t.maker_fee_base, t.maker_fee_refund, t.taker_fee_base, t.taker_fee_refund, t.quote_settled)


def test_both_models_carry_the_legs_from_the_wire():
    for model in (SimTradeInfo, ExchangeTradeInfo):
        assert _legs(model.from_json({**WIRE, **LEGS})) == (0.0002, 0.0, 0.0, 0.0, 600.0)


def test_both_models_carry_the_legs_from_the_event():
    for model in (SimTradeInfo, ExchangeTradeInfo):
        assert _legs(model.from_event({**EVENT, "settlement": SETTLEMENT})) == (0.0002, 0.0, 0.0, 0.0, 600.0)


def test_absent_legs_read_as_unknown_never_as_zero():
    """A pre-upgrade record: the legs are unknown, and unknown must not look like a zero-fee fill."""
    for model in (SimTradeInfo, ExchangeTradeInfo):
        assert _legs(model.from_json(WIRE)) == (None,) * 5
        assert _legs(model.from_event(EVENT)) == (None,) * 5


def test_canonical_notice_states_fee_legs_and_omits_an_unknown_quote_leg():
    kw = dict(book_id=5, quantity=1.0, price=0.5, side=0, maker_uid=None, taker_uid=171,
              maker_fee=0.0, taker_fee=0.0)
    d = NormalizedTradeEvent(**kw).to_notice_dict()
    assert (d["Mfb"], d["Mfr"], d["Tfb"], d["Tfr"]) == (0.0, 0.0, 0.0, 0.0)
    assert "qs" not in d
    assert NormalizedTradeEvent(**kw, quote_settled=0.5).to_notice_dict()["qs"] == 0.5
