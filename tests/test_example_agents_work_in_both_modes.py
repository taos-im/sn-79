# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner's existing agent must run on the exchange without being rewritten.

WHY THIS EXISTS. The public carve ships the exchange wire types and the agent base class so a miner can
point an existing strategy at the exchange. Two things broke that on 2026-08-17, and both were silent:

  1. FinanceAgent.respond_exchange() delegated to respond(). An agent implementing respond_simulation()
     -- which is what RandomMakerAgent, RandomTakerAgent and RevengAgent demonstrate, so it is the
     pattern a miner copies -- never implements respond(); the abstract stub returned None, handle()
     passed that None to report(), and report() raised
     `AttributeError: 'NoneType' object has no attribute 'instructions'`.
     The exchange path crashed for exactly the agents the public examples teach.

  2. Three examples requested leverage unconditionally. Exchange mode runs with maxLeverage=0, so those
     orders are REFUSED at placement (s_leverage_is_inert asserts this). The agent appeared to work and
     placed nothing -- and the `quantity() * (1 + leverage())` sizing inflated the request too.

Neither failure announces itself in simulation, which is where an example gets tested.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[1]
AGENTS = DEV / "agents"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_respond_exchange_reaches_a_simulation_only_agents_logic():
    """The dispatch chain must cover all three single-method shapes an agent can be written in."""
    from taos.im.agents import FinanceAgent

    class RespondOnly(FinanceAgent):
        def initialize(self):
            pass

        def respond(self, state):
            return "RESPOND"

    class SimOnly(FinanceAgent):
        def initialize(self):
            pass

        def respond_simulation(self, state):
            return "SIM"

    class ExchangeOwn(FinanceAgent):
        def initialize(self):
            pass

        def respond_exchange(self, state):
            return "EXCH"

    for cls, expected in ((RespondOnly, "RESPOND"), (SimOnly, "SIM"), (ExchangeOwn, "EXCH")):
        agent = cls.__new__(cls)
        got = agent.respond_exchange(None)
        assert got == expected, (
            f"{cls.__name__}: respond_exchange returned {got!r}, expected {expected!r}. A None here "
            "reaches report() and raises AttributeError on .instructions, so the miner's agent crashes "
            "on the exchange while working in simulation."
        )


def test_the_agent_exposes_a_supported_mode_accessor():
    """Examples need to ask the mode without reaching for a private attribute.

    Request-scoped, not instance-scoped: one agent instance serves both validators concurrently, so an
    instance flag can report the other request's mode.
    """
    import taos.im.agents as A
    from taos.im.agents import FinanceAgent

    assert isinstance(getattr(FinanceAgent, "exchange_mode", None), property), (
        "FinanceAgent.exchange_mode must be a property; examples branch on it"
    )

    class Probe(FinanceAgent):
        def initialize(self):
            pass

        def respond(self, state):
            return None

    agent = Probe.__new__(Probe)
    agent._exchange_mode = False
    assert agent.exchange_mode is False
    token = A._REQUEST_EXCHANGE_MODE.set(True)
    try:
        assert agent.exchange_mode is True, (
            "the request-scoped mode must win over the instance attribute, or a concurrent simulation "
            "request can make an exchange request read as simulation"
        )
    finally:
        A._REQUEST_EXCHANGE_MODE.reset(token)


@pytest.mark.parametrize(
    "filename,attrs",
    [
        ("RandomTakerAgent.py", {"min_leverage": 1.0, "max_leverage": 2.0}),
        ("RandomMakerAgent.py", {"min_leverage": 1.0, "max_leverage": 2.0}),
        ("OrderOptionAgent.py", {}),
    ],
)
def test_examples_stand_down_leverage_on_the_exchange(filename, attrs):
    """maxLeverage is 0 there, so a leveraged order is refused rather than executed unleveraged."""
    import taos.im.agents as A

    path = AGENTS / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")
    name = filename[:-3]
    mod = _load(path, f"_probe_{name}")
    cls = getattr(mod, name)
    agent = cls.__new__(cls)
    for k, v in attrs.items():
        setattr(agent, k, v)
    agent._exchange_mode = False

    sim = agent.leverage()
    token = A._REQUEST_EXCHANGE_MODE.set(True)
    try:
        exch = agent.leverage()
    finally:
        A._REQUEST_EXCHANGE_MODE.reset(token)

    assert exch == 0.0, (
        f"{filename} requests leverage={exch} on the exchange, where maxLeverage=0 refuses it at "
        "placement. Every order this agent sends would be rejected."
    )
    assert sim > 0.0, (
        f"{filename} no longer requests leverage in SIMULATION either -- the guard is too broad and the "
        "example has stopped demonstrating the parameter at all."
    )


def test_no_example_agent_is_left_with_a_bare_leverage_literal():
    """A literal cannot be mode-aware, so a new one reintroduces the same defect.

    Scoped to what an agent SENDS: a `leverage=` keyword on an order. Reading a leverage value off an
    account or logging one is not a request and is unaffected.
    """
    offenders = []
    for path in sorted(AGENTS.glob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "leverage":
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)):
                    if kw.value.value != 0:
                        offenders.append(f"{path.name}:{kw.value.lineno} leverage={kw.value.value}")
    assert not offenders, (
        "example agent(s) request a hardcoded non-zero leverage, which the exchange refuses: "
        f"{offenders}. Route it through a leverage() helper that returns 0.0 when self.exchange_mode."
    )
