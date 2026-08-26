# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner's pre-0.6.0 agent must keep working in simulation when 0.6.0 deploys.

WHY THIS EXISTS. 0.6.0 adds a second mechanism and renamed the agent base class: `FinanceSimulationAgent`
became `FinanceAgentBase`, because its body has no mode branch at all and the name read as though it were
the simulation-mode class. Deployed miner agents subclass the OLD name and were written before the exchange
existed. Miners' deployed agents cannot be edited retroactively. So every element of the pre-0.6.0 authoring surface is pinned
here, and the pins are about SIMULATION: the exchange is new and opt-in, but an existing agent that only
ever spoke to the simulation must not notice the upgrade.

The pre-0.6.0 shape, as taught by the examples and documentation at the time:

    from taos.im.agents import FinanceSimulationAgent
    from taos.im.protocol.response import FinanceAgentResponse

    class MyAgent(FinanceSimulationAgent):
        def initialize(self): ...
        def respond(self, state) -> FinanceAgentResponse:
            response = FinanceAgentResponse(agent_id=self.uid)
            response.limit_order(...)
            return response

Three separate things in that snippet can break, and each has its own test below: the import, the base
class, and the raw response object crossing handle() without being coerced.
"""

import pytest


def test_the_old_base_class_name_still_imports():
    """`from taos.im.agents import FinanceSimulationAgent` is in every deployed miner agent's header."""
    from taos.im.agents import FinanceSimulationAgent

    assert FinanceSimulationAgent is not None


def test_the_old_name_is_the_same_class_not_a_copy():
    """An alias, not a duplicate.

    A second class object would make `issubclass(FinanceAgent, FinanceSimulationAgent)` false and break
    the composer's own validation, which asserts exactly that.
    """
    from taos.im.agents import FinanceAgentBase, FinanceAgent, FinanceSimulationAgent

    assert FinanceSimulationAgent is FinanceAgentBase, (
        "the old name must BE the new class, not a subclass or a copy of it"
    )
    assert issubclass(FinanceAgent, FinanceSimulationAgent), (
        "mvtrx/service tests/test_composer.py asserts issubclass(ComposedAgent, FinanceSimulationAgent); "
        "breaking this breaks the composer"
    )


def test_an_agent_can_still_be_declared_against_the_old_name():
    """Subclassing the old name must still produce a working class object."""
    from taos.im.agents import FinanceSimulationAgent, FinanceAgentBase

    class LegacyMinerAgent(FinanceSimulationAgent):
        def initialize(self):
            pass

        def respond(self, state):
            return None

    assert issubclass(LegacyMinerAgent, FinanceAgentBase)
    assert LegacyMinerAgent.__mro__[1] is FinanceAgentBase


def test_a_raw_finance_agent_response_crosses_handle_unchanged():
    """Pre-0.6.0 agents build `FinanceAgentResponse` directly; 0.6.0 must not coerce it.

    `handle()` gained a `finalize()` step for the mode-aware `UnifiedAgentResponse` that
    `make_response()` returns. If that step were applied unconditionally, every deployed agent -- all of
    which construct the response type directly -- would have its response mangled or would raise on the
    missing `finalize`. The branch must be conditional on the type.

    update() and report() are stubbed deliberately: the assertion is about what type crosses handle(),
    not about history bookkeeping, and stubbing them keeps the test from depending on a live state
    object. A simulation state is represented by `None` here, which is simply not an
    `ExchangeStateUpdate`, so handle() takes the simulation branch -- which is the branch under test.
    """
    from taos.im.agents import FinanceAgent
    from taos.im.protocol.response import FinanceAgentResponse

    made = FinanceAgentResponse(agent_id=7)

    class LegacyShapedAgent(FinanceAgent):
        def initialize(self):
            pass

        def _apply_pending_live_config(self):
            pass

        def update(self, state):
            pass

        def report(self, state, response):
            pass

        def respond(self, state):
            return made

    agent = LegacyShapedAgent.__new__(LegacyShapedAgent)
    got = agent.handle(None)

    assert got is made, (
        f"handle() returned {type(got).__name__} rather than the agent's own FinanceAgentResponse. "
        "Every deployed miner agent constructs this type directly, so coercing it here breaks all of "
        "them in simulation."
    )


def test_both_names_are_exported_from_the_package():
    """A miner may import either name; the old one must not vanish from the public surface."""
    import taos.im.agents as A

    for name in ("FinanceSimulationAgent", "FinanceAgentBase", "FinanceAgent"):
        assert hasattr(A, name), f"taos.im.agents no longer exports {name}"


def test_close_position_singular_builds_a_valid_instruction():
    """A miner calling the SINGULAR form must not get a ValidationError.

    UnifiedAgentResponse.close_position passed `positions=[...]` to ClosePositionsInstruction, whose
    field is `closes` -- there is no `positions` field. So the singular form raised "Field required" at
    runtime, in simulation mode, on the release whose whole promise is that existing simulation miners
    keep working. Its plural sibling carries a comment saying exactly this about the same mistake, which
    is how it was noticed: the note was written for close_positions and the fix applied only there.
    """
    from taos.im.agents import UnifiedAgentResponse

    r = UnifiedAgentResponse.__new__(UnifiedAgentResponse)
    r.instructions = []
    r.agent_id = 8
    r._exchange_mode = False

    r.close_position(book_id=79, order_id=555, quantity=1.5)
    assert len(r.instructions) == 1
    instr = r.instructions[0]
    assert instr.type == "CLOSE_POSITIONS"
    assert [c.orderId for c in instr.closes] == [555]
    assert instr.payload()["closes"][0]["orderId"] == 555
