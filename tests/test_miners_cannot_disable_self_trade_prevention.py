# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""MINERS MUST NOT BE ABLE TO SELF-TRADE. Prevention is forced on, deliberately and explicitly.

A self-trade moves no position and manufactures volume, and volume feeds scoring, so the flag that
switches prevention off is not a miner's to set.

HOW IT IS ENFORCED. Both the simulation and the exchange instruction classes accept NO_STP on the
wire and neither ever expresses it: the field validator on PlaceOrderInstruction coerces it to
CANCEL_OLDEST before the instruction is constructed. Accepting it keeps the miner's batch intact,
since rejecting one disallowed flag would cost it every order sent alongside; coercing it keeps
prevention on. It is done on BOTH classes because the rule is about miners, not about mechanisms.

Coercion sits on the base class rather than at a call site so that no code path can bypass it: market
and limit orders both inherit it, and so does anything added later.
"""

import pytest

from taos.im.protocol.exchange.instructions import (
    PlaceLimitOrderInstruction as ExchangeLimitOrder,
    PlaceMarketOrderInstruction as ExchangeMarketOrder,
)
from taos.im.protocol.instructions import (
    PlaceLimitOrderInstruction as SimLimitOrder,
    PlaceMarketOrderInstruction as SimMarketOrder,
)
from taos.im.protocol.models import STP

_COMMON = dict(agentId=171, bookId=5, direction=0, quantity=1.0, price=0.01, clientOrderId=None)
_MARKET = dict(agentId=171, bookId=5, direction=0, quantity=1.0, clientOrderId=None)
# The exchange instruction additionally requires a delegate: the proxy ss58 that signs the on-chain leg.
# It has no bearing on the STP question but the model will not construct without it.
_DELEGATE = "5EfGWygxcnCZQiFjD3fi9A15MmFzNXeqwdirEUzPyUcDRKdz"
_EX = dict(_COMMON, delegate=_DELEGATE)
_EX_MARKET = dict(_MARKET, delegate=_DELEGATE)


def test_stp_is_on_by_default_and_that_is_deliberate():
    """Every order carries a prevention policy unless it names another. Matches simulation."""
    assert ExchangeLimitOrder(**_EX).stp == STP.CANCEL_OLDEST
    assert SimLimitOrder(**_COMMON).stp == STP.CANCEL_OLDEST


@pytest.mark.parametrize("model,kwargs", [
    ("SimLimitOrder", _COMMON),
    ("SimMarketOrder", _MARKET),
    ("ExchangeLimitOrder", _EX),
    ("ExchangeMarketOrder", _EX_MARKET),
])
def test_no_stp_is_accepted_and_then_coerced_on_every_model(model, kwargs):
    """THE PIN. A miner may ASK for NO_STP; it must never GET it.

    If this starts failing, self-trade prevention has been turned off for any miner that asks. That is
    a product decision and must be a deliberate one, so it breaks a test rather than slipping through.

    Both halves matter and they pull in opposite directions:
      - accepted, so one disallowed flag does not cost the miner its whole batch;
      - coerced, so the engine never sees NO_STP and prevention stays on.
    """
    cls = {"SimLimitOrder": SimLimitOrder, "SimMarketOrder": SimMarketOrder,
           "ExchangeLimitOrder": ExchangeLimitOrder, "ExchangeMarketOrder": ExchangeMarketOrder}[model]
    order = cls(**kwargs, stp=STP.NO_STP)
    assert order.stp == STP.CANCEL_OLDEST, (
        f"{model} expressed NO_STP; a miner can disable self-trade prevention"
    )


@pytest.mark.parametrize("raw", [0, "0", STP.NO_STP])
def test_coercion_is_on_the_value_not_the_spelling(raw):
    """The flag arrives off the wire, so it can be the int, the string or the enum member.

    A validator that only matched the enum member would be bypassed by the very path that matters --
    a miner's JSON, where it is 0.
    """
    assert SimLimitOrder(**_COMMON, stp=raw).stp == STP.CANCEL_OLDEST
    assert ExchangeLimitOrder(**_EX, stp=raw).stp == STP.CANCEL_OLDEST


def test_the_wire_alias_is_coerced_too():
    """Orders re-validated from a serialised payload come back through the ALIAS, not the field name.

    payload() emits "stpFlag", so a round trip validates {"stpFlag": 0}. If coercion only covered the
    field name, every re-validated order would carry NO_STP straight past it.
    """
    assert SimLimitOrder.model_validate(dict(_COMMON, stpFlag=0)).stp == STP.CANCEL_OLDEST
    assert ExchangeLimitOrder.model_validate(dict(_EX, stpFlag=0)).stp == STP.CANCEL_OLDEST


def test_no_stp_never_reaches_the_wire():
    """The engine reads the msgpack key "stp". Whatever the miner asked for, that key must not be 0."""
    for order in (SimLimitOrder(**_COMMON, stp=STP.NO_STP),
                  ExchangeLimitOrder(**_EX, stp=STP.NO_STP)):
        payload = order.payload()
        key = "stp" if "stp" in payload else "stpFlag"
        assert int(payload[key]) == int(STP.CANCEL_OLDEST), (
            f"{type(order).__name__} put NO_STP on the wire as {payload[key]!r}"
        )


@pytest.mark.parametrize("flag", [STP.CANCEL_OLDEST, STP.CANCEL_NEWEST, STP.CANCEL_BOTH,
                                  STP.DECREASE_CANCEL])
def test_the_four_usable_policies_survive_both_models(flag):
    """The flags a miner CAN choose must actually travel, or the choice is decorative.

    This is the assertion whose absence let vacuous passes stand: the old scenario only checked that an
    order was accepted, and an order carrying the wrong policy is accepted just as readily. It also
    guards the coercion itself -- a validator that flattened everything to CANCEL_OLDEST would satisfy
    the pin above and break every other policy.
    """
    assert ExchangeLimitOrder(**_EX, stp=flag).stp == flag
    assert SimLimitOrder(**_COMMON, stp=flag).stp == flag


def test_the_two_enums_agree_numerically():
    """A mismatch here would silently convert one policy into another across the language boundary.

    The C++ side is `enum class STPFlag : uint32_t { NONE, CO, CN, CB, DC }` in util/Flags.hpp, i.e.
    0..4 in that order. Python must match member for member or a miner asking for CANCEL_NEWEST gets
    CANCEL_BOTH and nothing anywhere reports a problem.
    """
    assert [int(m) for m in STP] == [0, 1, 2, 3, 4]
    assert [m.name for m in STP] == [
        "NO_STP", "CANCEL_OLDEST", "CANCEL_NEWEST", "CANCEL_BOTH", "DECREASE_CANCEL",
    ]


# ---------------------------------------------------------------------------------------------------
# MERGED FROM tests/test_no_stp_is_not_overridden.py, which pinned the OPPOSITE contract.
#
# That file existed because a previous coercion, in validate_responses, was harmful:
#
#     if stp_value == 'NO_STP' or stp_value == 0:
#         instruction.stp = STP.CANCEL_OLDEST
#
# It broke marketable limit orders outright -- "nothing filled, nothing rested, no
# notice, in BOTH directions" -- because the sweep the engine creates at submission looked like the same
# agent as the resting order, so STP cancelled the order the sweep existed to fill. NO_STP was the only
# way out, and that line discarded it.
#
# WHY THAT DOES NOT APPLY TO THE COERCION ABOVE. The breakage was a comparison bug in Book.cpp, which
# compared the SWEEP'S OWNER against the resting owner; it has since been fixed to compare the ACTING
# agent. The proof is that CANCEL_OLDEST is, and always was, the model DEFAULT: every order that does
# not name a policy already gets exactly what the coercion now forces. If forcing it still broke
# marketable limits, every default order would be broken too. It is not:
#
#     simulation  [PASS] BUY marketable fills in full within limit
#                                   [PASS] SELL marketable fills in full within limit
#                                   [PASS] a marketable GTT fills immediately, resting only the shortfall
#     exchange    the same three, passing
#
# Both mechanisms, under the default CANCEL_OLDEST, filling correctly. The escape hatch is no longer
# needed because the thing it was escaping is fixed.
#
# The two facts below are what made the OLD override provably redundant, and they are equally load
# bearing for the new one: if the model default ever became NO_STP, an unset flag would be
# indistinguishable from an explicit NONE and coercing would start overriding silence rather than a
# choice.


def test_no_stp_is_zero_and_the_model_default_is_cancel_oldest():
    """The two facts the coercion rests on, pinned so they cannot drift."""
    assert STP.NO_STP.value == 0
    unset = ExchangeLimitOrder(
        agentId=1, bookId=5, direction=0, quantity=1.0, price=0.01,
        clientOrderId=None, delegate="",
    )
    assert unset.stp == STP.CANCEL_OLDEST, (
        "if the model default changes to NO_STP, an unset flag becomes indistinguishable from an "
        "explicit NONE, and the coercion above would be overriding silence rather than a choice"
    )
