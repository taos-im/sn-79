# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Self-trade prevention is deliberately FORCED ON, and this pins that plus the route to turning it off.

OPERATOR DECISION (2026-08-06): "this matches simulation where STP was forced on, can keep it this way
for now, but tests confirm that we can enable it later." So these tests do NOT make NO_STP work. They
record that it does not, why, and exactly what would change if we decide to allow it.

WHAT IS ACTUALLY BROKEN, as opposed to chosen. Two instruction classes exist with different types:

    taos/im/protocol/exchange/instructions.py:65
        stp: Literal[STP.NO_STP, CANCEL_OLDEST, CANCEL_NEWEST, CANCEL_BOTH, DECREASE_CANCEL]  <- allows it
    taos/im/protocol/instructions.py:66
        stp: Literal[CANCEL_OLDEST, CANCEL_NEWEST, CANCEL_BOTH, DECREASE_CANCEL]              <- does not

The miner builds with the EXCHANGE class, which accepts NO_STP. The response envelope
(taos/im/protocol/response.py:14) is typed with the SIMULATION class, so the instruction is re-validated
on arrival against the stricter Literal and NO_STP falls back to the default, CANCEL_OLDEST.

HOW IT WAS FOUND. s_stp iterated NONE, CO, CN, CB, DC and asserted only that each was "accepted". The
engine log said otherwise:

    598 of 598 forwarded orders carried "stpFlag":"CO". Not one NONE, CN, CB or DC.

Both enums agree numerically (NO_STP/NONE=0, CANCEL_OLDEST/CO=1, ...), so this is not a mapping error:
the value really is being replaced with the default. The engine defaults its own payload field to CO as
well, which is why a dropped flag and an explicit CO are indistinguishable on the wire, and why five
"flag accepted" passes were five vacuous acceptances of the same flag.

TO ENABLE NO_STP LATER, one change: point response.py at the exchange instruction classes (or widen the
simulation Literal to match). The rest of the path is already in place and is asserted below.
"""

import pytest
from pydantic import ValidationError

from taos.im.protocol.exchange.instructions import (
    PlaceLimitOrderInstruction as ExchangeLimitOrder,
)
from taos.im.protocol.instructions import PlaceLimitOrderInstruction as SimLimitOrder
from taos.im.protocol.models import STP

_COMMON = dict(agentId=171, bookId=5, direction=0, quantity=1.0, price=0.01, clientOrderId=None)
# The exchange instruction additionally requires a delegate: the proxy ss58 that signs the on-chain leg.
# It has no bearing on the STP question but the model will not construct without it.
_EX = dict(_COMMON, delegate="5EfGWygxcnCZQiFjD3fi9A15MmFzNXeqwdirEUzPyUcDRKdz")


def test_stp_is_on_by_default_and_that_is_deliberate():
    """Every order carries a prevention policy unless it names another. Matches simulation."""
    assert ExchangeLimitOrder(**_EX).stp == STP.CANCEL_OLDEST
    assert SimLimitOrder(**_COMMON).stp == STP.CANCEL_OLDEST


def test_no_stp_is_currently_unreachable_through_the_response_envelope():
    """THE PIN. This asserts the CURRENT, CHOSEN state, not a bug to be fixed silently.

    If this test starts failing, someone has widened the simulation Literal or repointed response.py,
    which turns self-trade prevention OFF for any miner that asks. That is a product decision and must
    be a deliberate one, so it should break a test rather than slip through.
    """
    with pytest.raises(ValidationError):
        SimLimitOrder(**_COMMON, stp=STP.NO_STP)


def test_the_exchange_class_is_already_ready_for_it():
    """Half the route already exists: the exchange-side model accepts NO_STP today."""
    assert ExchangeLimitOrder(**_EX, stp=STP.NO_STP).stp == STP.NO_STP


def test_the_response_envelope_is_the_single_blocker():
    """Name the exact line, so enabling it later is a known one-line change and not an investigation."""
    import taos.im.protocol.response as response

    src = response.__file__
    with open(src) as fh:
        head = fh.read(4000)
    assert "from taos.im.protocol.instructions import" in head, (
        "response.py no longer imports the simulation instruction classes. If it now imports the "
        "exchange ones, NO_STP has become reachable and self-trade prevention is no longer forced on"
    )


@pytest.mark.parametrize("flag", [STP.CANCEL_OLDEST, STP.CANCEL_NEWEST, STP.CANCEL_BOTH,
                                  STP.DECREASE_CANCEL])
def test_the_four_usable_policies_survive_both_models(flag):
    """The flags a miner CAN choose must actually travel, or the choice is decorative.

    This is the assertion whose absence let the vacuous passes stand: the old scenario only checked that
    an order was accepted, and an order carrying the wrong policy is accepted just as readily.
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
