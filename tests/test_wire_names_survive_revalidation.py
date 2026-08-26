# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""An instruction re-validated from its own wire form must not lose fields.

THE DEFECT (found 2026-08-06, root-caused after two wrong attempts). `payload()` serialises three fields
under names the model does not answer to:

    wire "volume"       <- field quantity
    wire "stpFlag"      <- field stp
    wire "allowPartial" <- field allow_partial   (exchange limit order only)

Pydantic ignores unknown keys without raising, so anything that re-validates an instruction from its own
serialised form silently drops them and applies the field DEFAULT instead. Measured on a live stack:

    598 of 598 forwarded orders reached the engine with "stpFlag":"CO", whatever the miner asked for.

and reduced to two lines:

    {"stpFlag": 2} -> model.stp == CANCEL_OLDEST   (the default; the key was ignored)
    {"stp": 2}     -> model.stp == CANCEL_NEWEST   (survives)

WHY IT HID SO LONG. Nothing raises. The engine defaults its own payload field to CO as well, so a dropped
flag and an explicit CO are indistinguishable on the wire. The acceptance scenario asserted only that the
order was ACCEPTED, and an order carrying the wrong policy is accepted exactly as readily as one carrying
the right one. It took an assertion that read the flag back off the engine's own forwarded instruction.

`quantity` is the alarming one. It is the ORDER SIZE, and it has the same mismatch. It has not bitten
because every current call site passes `quantity=` by name, but any path that re-validates from the wire
would have produced a differently-sized order with no error anywhere.

THE FIX is additive: an alias for the wire spelling plus populate_by_name, so BOTH spellings work and no
existing call site changes.
"""

import pytest

from taos.im.protocol.exchange.instructions import (
    PlaceLimitOrderInstruction as ExchangeLimitOrder,
)
from taos.im.protocol.instructions import PlaceLimitOrderInstruction as SimLimitOrder
from taos.im.protocol.models import STP

_BASE = dict(agentId=171, bookId=5, direction=0, price=0.01, clientOrderId=None)
_EX = dict(_BASE, delegate="5EfGWygxcnCZQiFjD3fi9A15MmFzNXeqwdirEUzPyUcDRKdz")

_CLASSES = [
    pytest.param(SimLimitOrder, _BASE, id="simulation"),
    pytest.param(ExchangeLimitOrder, _EX, id="exchange"),
]


@pytest.mark.parametrize("cls,base", _CLASSES)
def test_the_python_spelling_still_works(cls, base):
    """Every existing call site passes quantity= and stp=. The fix must be additive, not a rename."""
    o = cls(**base, quantity=2.5, stp=STP.CANCEL_NEWEST)
    assert o.quantity == 2.5
    assert o.stp == STP.CANCEL_NEWEST


@pytest.mark.parametrize("cls,base", _CLASSES)
def test_the_wire_spelling_survives(cls, base):
    """THE REGRESSION. These are the keys payload() emits, so these are the keys that must round-trip."""
    o = cls.model_validate({**base, "volume": 2.5, "stpFlag": int(STP.CANCEL_NEWEST)})
    assert o.quantity == 2.5, "order SIZE was dropped and defaulted"
    assert o.stp == STP.CANCEL_NEWEST, "self-trade policy was dropped and defaulted to CANCEL_OLDEST"


@pytest.mark.parametrize("cls,base", _CLASSES)
def test_a_full_payload_round_trip_loses_nothing(cls, base):
    """Serialise with the real serialiser, re-validate, and compare. This is the actual failing path."""
    original = cls(**base, quantity=2.5, stp=STP.CANCEL_BOTH)
    wire = original.payload()
    # Keys taken from the serialiser itself, not hardcoded. The exchange class now emits "stp" rather
    # than "stpFlag", because "stp" is the key the C++ actually reads
    # (MSGPACK_NVP("stp", stpFlag), ExchangeAgentMessagePayloads.hpp:139); emitting "stpFlag" meant the
    # engine never found the field and used its struct default CO for every order ever placed. A test
    # that hardcodes wire names cannot survive fixing one, which is the point of deriving them.
    carried = {k: v for k, v in wire.items() if k not in base}
    back = cls.model_validate({**base, **carried})
    assert back.quantity == original.quantity
    assert back.stp == original.stp


def test_allow_partial_survives_too():
    """Third field with the same mismatch, and it decides whether a remainder rests."""
    o = ExchangeLimitOrder.model_validate({**_EX, "volume": 1.0, "allowPartial": False})
    assert o.allow_partial is False, (
        "allowPartial was dropped, so a miner asking for all-or-nothing silently got partial fills"
    )


@pytest.mark.parametrize("cls,base", _CLASSES)
def test_every_payload_key_is_answerable(cls, base):
    """The general property, so a NEW field with a mismatched wire name fails here rather than in prod.

    This is the check whose absence allowed three fields to drift. It compares what the serialiser emits
    against what the model will accept, rather than testing the three known cases and calling it done.
    """
    o = cls(**base, quantity=1.0)
    answerable = set(cls.model_fields) | {
        f.alias for f in cls.model_fields.values() if f.alias
    }
    unanswerable = sorted(k for k in o.payload() if k not in answerable)
    assert not unanswerable, (
        f"{cls.__module__}.{cls.__name__} serialises {unanswerable} under names it cannot read back. "
        f"Add an alias, or these are dropped and defaulted with no error anywhere."
    )
