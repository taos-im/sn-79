# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""THE LIFECYCLE EVENTS TRAVEL THE STATE UPDATE, SO THE NOTICE CHANNEL MUST NOT SEND THEM TWICE.

The engine announces a simulation's start and end twice over, by design, to two different consumers:

  * the lifecycle HTTP post (postGeneralMessage -> the validator's /account route), which the
    VALIDATOR consumes for its own engine.on_start / engine.on_end, and
  * the proxy's notice queue, which rides the state update out to every AGENT.

Because the engine delivers the second of those, re-broadcasting the HTTP copy over the dendrite is
redundant: forwarding the end event as well would call every agent's onEnd twice, once from
FinanceAgentBase.update and once from FinanceAgentBase.process.

So the rule this file pins: a message the validator consumes for itself is not also pushed to miners.
The channel stays live for anything else -- an unrecognised type still falls through and is
forwarded, which is what keeps it available for whatever we put on it next.
"""

import pytest

validator = pytest.importorskip("taos.im.neurons.validator")


def test_the_lifecycle_events_are_consumed_rather_than_forwarded():
    consumed = validator._NOTICES_CONSUMED_NOT_FORWARDED
    assert "EVENT_SIMULATION_START" in consumed
    assert "EVENT_SIMULATION_END" in consumed, (
        "the end event is still forwarded to miners. It now reaches agents in the state update, so "
        "forwarding it as well calls onEnd twice -- once via update(), once via process()."
    )


def test_anything_else_is_still_forwarded():
    """The channel is inert, not disabled. A new notice type must reach miners without a code change."""
    assert validator._notice_is_forwarded_to_miners("SOME_FUTURE_NOTICE")
    assert validator._notice_is_forwarded_to_miners("EVENT_TRADE")


def test_the_consumed_set_is_exactly_the_lifecycle_pair():
    """A set that grows silently would turn 'inert' into 'deaf'. Changing it should fail this test."""
    assert set(validator._NOTICES_CONSUMED_NOT_FORWARDED) == {
        "EVENT_SIMULATION_START",
        "EVENT_SIMULATION_END",
    }


@pytest.mark.parametrize("mtype", ["EVENT_SIMULATION_START", "EVENT_SIMULATION_END"])
def test_the_predicate_and_the_set_agree(mtype):
    assert not validator._notice_is_forwarded_to_miners(mtype)
