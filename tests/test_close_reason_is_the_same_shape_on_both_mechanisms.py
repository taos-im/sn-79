# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner must be able to tell WHY a position closed, the same way on both mechanisms.

The engine sets closeReason on the closing order (MultiBookExchangeAgent.cpp: 1=SL, 2=TP) and packs it
onto the simulation notice wire as the key "cr" (NoticePack.hpp). The exchange half of the validator
surfaces the same idea as a STRING: engines/__init__.py sets d['cr'] = self.close_reason, which is
'SL' | 'TP' | None.

So the two mechanisms disagreed about the TYPE of the same field. Simulation delivered the raw integer
-- it survives only because TradeEvent has extra='allow', so it was never a declared field at all --
while the exchange delivered 'SL'/'TP'. A miner writing `if notice['cr'] == 'TP'` works on the exchange
and silently never fires in simulation. Nothing raises; the branch simply never runs.

Found in one run. The engine did everything right:

    2785549203426 | SLTP DISPATCH TP | AGENT #4 BOOK 125 : closingSide=1 volume=1 observedPrice=334.81
    BOOK 125 : SELL TRADE #1301 : YOUR AGGRESSIVE ORDER #85378 (AGENT 4) ... 1.0@334.37 (T=2785549203426)

Same nanosecond, matching side and volume: the take-profit fired and closed the position. The
acceptance scenario still failed with "0 close notice(s) after the move", because it looked for the
string the exchange sends.

NORMALISED AT from_json, NOT WITH A FIELD VALIDATOR. The abbreviated wire codes are built with
model_construct, which skips validation entirely, so a field_validator would never run on the path
that matters. The "RDPOM" branch already mutates json['r'] the same way for the same reason.
"""

import pytest

from taos.im.protocol.events import FinanceEvent, TradeEvent
from taos.im.protocol.models import STP  # noqa: F401  (import parity with the sibling STP pin)

_WIRE = {"y": "ET", "b": 125, "i": 1301, "Ta": 4, "Ti": 1, "Tf": 0.0,
         "Ma": -103, "Mi": 2, "Mf": 0.0, "s": 1, "p": 334.37, "q": 1.0}


@pytest.mark.parametrize("raw,want", [(1, "SL"), (2, "TP")])
def test_the_engines_integer_close_reason_is_surfaced_as_the_string_the_exchange_sends(raw, want):
    event = FinanceEvent.from_json(dict(_WIRE, cr=raw))
    assert event.cr == want, (
        f"the engine's closeReason {raw} reached the miner as {event.cr!r}; the exchange sends {want!r} "
        f"for the same event, so a miner cannot write one check that works on both mechanisms"
    )
    assert event.closeReason == want, "the readable accessor must agree with the wire field"


def test_an_ordinary_trade_has_no_close_reason():
    """Only an SL/TP dispatch sets it. 0 is the engine's "none" and must not become a truthy string."""
    event = FinanceEvent.from_json(dict(_WIRE, cr=0))
    assert event.cr is None
    assert event.closeReason is None


def test_a_trade_notice_without_the_field_at_all_still_parses():
    """Archived runs and any engine build predating the field send no "cr"."""
    event = FinanceEvent.from_json(dict(_WIRE))
    assert event.cr is None


def test_close_reason_is_a_declared_field_not_an_accident_of_extra_allow():
    """It survived only because TradeEvent allows extras, so nothing documented or typed it.

    A field nobody declared is a field nobody can rely on: it does not appear in the model, it has no
    accessor, and the next model_dump filter that drops unknown keys removes it silently.
    """
    assert "cr" in TradeEvent.model_fields, "cr must be declared on TradeEvent, not carried as an extra"


def test_the_string_form_survives_the_dump_the_miner_publishes():
    """The acceptance miner records notices with model_dump(), which is what the harness reads."""
    dumped = FinanceEvent.from_json(dict(_WIRE, cr=2)).model_dump()
    assert dumped.get("cr") == "TP", f"model_dump lost or altered the close reason: {dumped.get('cr')!r}"
