# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Importing the exchange protocol tree must not silently change which class an agent already has.

`taos.im.protocol` and `taos.im.protocol.exchange` are parallel wire types for the two modes, and both
are star-imported in practice -- our own simulation examples use `from taos.im.protocol.models import *`,
so a miner writing the exchange equivalent naturally does the same. Where the two trees export DIFFERENT
classes under the SAME name, the last import silently wins and an agent's behaviour depends on the order
of two lines at the top of its file.

That was not hypothetical. Both trees defined `AgentEventHistory`; the copies were identical except that
one tolerated dict-shaped notices and the other read `.type` directly. An agent that imported the agents
package and then the exchange tree got the intolerant copy, which raised AttributeError inside the call
that builds its response -- losing the whole response, not one notice, on every exchange tick.

Parallel classes are fine and expected: `Book`, `Account`, `FeePolicy` and the instruction types differ
genuinely between modes, and an agent works in one mode and imports one tree. What is not fine is a name
that resolves to different behaviour depending on import order, in a class agents are handed by the base
package.
"""

import importlib
import inspect

import pytest

# Names an agent gets from `taos.im.agents` and could then have silently rebound by a later star-import.
# AgentEventHistory is the one that bit; the guard is written for the class of problem.
SHARED_WITH_AGENTS = ("AgentEventHistory",)


def test_the_agents_package_and_the_exchange_tree_agree():
    agents = importlib.import_module("taos.im.agents")
    exchange = importlib.import_module("taos.im.protocol.exchange")

    for name in SHARED_WITH_AGENTS:
        a = getattr(agents, name, None)
        x = getattr(exchange, name, None)
        if a is None or x is None:
            continue
        assert a is x, (
            f"{name} resolves to two different objects "
            f"({inspect.getsourcefile(a)} vs {inspect.getsourcefile(x)}). A miner doing "
            f"`from taos.im.agents import *` then `from taos.im.protocol.exchange import *` gets "
            f"whichever came last, so its behaviour depends on import order."
        )


def test_import_order_does_not_decide_which_class_an_agent_uses():
    """The concrete two-line sequence, exercised rather than reasoned about."""
    ns: dict = {}
    exec("from taos.im.agents import *", ns)
    first = ns.get("AgentEventHistory")
    exec("from taos.im.protocol.exchange import *", ns)
    second = ns.get("AgentEventHistory")

    if first is None or second is None:
        pytest.skip("AgentEventHistory not exported by one of the two packages")
    assert first is second, (
        "importing the exchange tree rebound AgentEventHistory to a different class; an agent written "
        "against the base package would change behaviour on a line it did not write"
    )


def test_the_shared_history_reads_notices_without_asking_their_shape():
    """Pins WHY the two copies must not diverge again.

    A future copy that reintroduces a dict fallback would pass the identity checks above only if it
    replaced both -- this states the property that made divergence dangerous in the first place.
    """
    agents = importlib.import_module("taos.im.agents")
    history = getattr(agents, "AgentEventHistory", None)
    if history is None:
        pytest.skip("AgentEventHistory not exported")

    src = inspect.getsource(history.append)
    assert "isinstance(e, dict)" not in src, (
        "notices are parsed to models on both paths; a dict fallback here means something upstream "
        "stopped parsing and the shape asymmetry is back"
    )
