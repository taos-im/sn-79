# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Every builder on the mode-aware response must produce a VALID instruction, in BOTH modes.

WHY THIS EXISTS. UnifiedAgentResponse constructs pydantic instruction models from keyword arguments
written by hand, twice over -- once for the simulation instruction set and once for the exchange one. A
wrong or missing field name is not a static error there; it is a ValidationError raised inside an agent
at runtime, in one mode only, at whichever moment the miner first calls that builder. Two of the seven
builders shipped with exactly that bug (close_positions with `positions` instead of `closes`, fixed
2026-08-09; close_position with the same mistake, still present until 2026-08-19 and found only by
reading the note left on its sibling).

The pattern is what makes it dangerous: each was noticed as an agent losing an update in production
rather than as a test failure. So the test is not "close_position works" -- it is every builder, both
modes, called the way a miner calls it.
"""

import pytest

from taos.im.agents import UnifiedAgentResponse


def _resp(exchange: bool) -> UnifiedAgentResponse:
    return UnifiedAgentResponse(agent_id=8, exchange_mode=exchange, delegate="5Delegate")


# (builder name, kwargs) -- the miner-facing call, not an internal one.
CALLS = [
    ("limit_order", dict(book_id=79, direction=0, quantity=1.5, price=100.0)),
    ("market_order", dict(book_id=79, direction=0, quantity=1.5)),
    ("cancel_order", dict(book_id=79, order_id=555)),
    ("cancel_order", dict(book_id=79, order_id=555, quantity=0.5)),
    ("cancel_orders", dict(book_id=79, order_ids=[555, 556])),
    ("close_position", dict(book_id=79, order_id=555)),
    ("close_position", dict(book_id=79, order_id=555, quantity=1.5)),
    ("close_positions", dict(book_id=79, order_ids=[555, 556])),
]


@pytest.mark.parametrize("mode", [False, True], ids=["simulation", "exchange"])
@pytest.mark.parametrize("name,kwargs", CALLS, ids=[f"{n}{sorted(k)}" for n, k in CALLS])
def test_builder_produces_a_valid_instruction(mode, name, kwargs):
    r = _resp(mode)
    getattr(r, name)(**kwargs)

    if mode and name.startswith("close_position"):
        # Closing a position is a margin operation and the exchange mechanism has no leverage: the
        # exchange instruction protocol contains no close-position instruction, and the engine answers
        # any unknown type with INVALID_INSTRUCTION_TYPE. So the builder deliberately drops the call
        # with a warning rather than emitting something the engine would refuse. Asserted here so the
        # asymmetry stays a decision on the record instead of looking like a gap.
        assert r.instructions == []
        return

    assert len(r.instructions) == 1, f"{name} emitted {len(r.instructions)} instructions"
    instr = r.instructions[0]
    # Construction already validated it; re-validating catches a builder that bypassed the model.
    instr.model_validate(instr.model_dump())
    assert instr.payload(), f"{name} produced an empty payload"


@pytest.mark.parametrize("mode", [False, True], ids=["simulation", "exchange"])
def test_finalize_accepts_every_builder_together(mode):
    """One update calling several builders must serialize as a whole, not just per instruction."""
    r = _resp(mode)
    r.limit_order(book_id=79, direction=0, quantity=1.5, price=100.0)
    r.market_order(book_id=79, direction=1, quantity=0.5)
    r.cancel_orders(book_id=79, order_ids=[555])
    if not mode:
        r.close_position(book_id=79, order_id=556)
    final = r.finalize()
    assert final.instructions, "finalize dropped every instruction"
