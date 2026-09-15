"""A published agent's mode-dependent decisions must come from the RESPONSE, not from the agent.

WHY THIS EXISTS. One agent instance serves both validators concurrently. `self.exchange_mode` on the
AGENT therefore describes whichever request most recently ran update(), and it READS as "this agent
is in exchange mode" -- which is not a thing an agent can be. An example agent is the thing miners
copy, so the shape it teaches matters more than whether it happens to be correct today.

The mode is a property of the response being built: fixed at construction, one response per request.
A decision taken from it is unambiguous about what it applies to.

The agent's own attribute is kept for internal use -- gentrx derives its bucket shard from it outside
any request, where there is no state to ask -- but nothing in a respond path should read it.
"""
import importlib.util as ilu
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_AGENTS = _ROOT / "agents"

# The agents whose strategy genuinely varies by mechanism: exchange runs with maxLeverage=0 and
# REFUSES a leveraged order at placement, so a leverage-using agent has to stand down there.
_MODE_DEPENDENT = ["RandomMakerAgent", "RandomTakerAgent", "OrderOptionAgent"]


def _load(name):
    sys.path.insert(0, str(_AGENTS))
    spec = ilu.spec_from_file_location(name, _AGENTS / f"{name}.py")
    mod = ilu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        pytest.skip(f"{name} needs the taos runtime: {exc!r}")
    return getattr(mod, name)


def _response(exchange: bool):
    try:
        from taos.im.agents import UnifiedAgentResponse
    except Exception as exc:
        pytest.skip(f"taos.im.agents not importable: {exc!r}")
    return UnifiedAgentResponse(1, exchange)


def test_the_response_exposes_its_own_mechanism():
    assert _response(True).exchange_mode is True
    assert _response(False).exchange_mode is False


def test_no_published_agent_reads_the_agents_own_mode_flag():
    """The shape example agents teach is the point: a miner copies it."""
    offenders = {}
    for f in sorted(_AGENTS.glob("*.py")):
        code = "\n".join(ln for ln in f.read_text().splitlines() if not ln.lstrip().startswith("#"))
        hits = [ln.strip() for ln in code.splitlines()
                if "self.exchange_mode" in ln or "self._exchange_mode" in ln]
        if hits:
            offenders[f.name] = hits
    assert not offenders, (
        "these agents decide from the AGENT's mode, which describes whichever request last ran "
        f"update() rather than the response being built: {offenders}"
    )


@pytest.mark.parametrize("name", _MODE_DEPENDENT)
def test_the_mode_dependent_decision_follows_the_response(name):
    cls = _load(name)
    agent = object.__new__(cls)
    agent.min_leverage, agent.max_leverage = 1.0, 2.0

    # Exchange must stand down to zero; simulation must not. Compared on that PROPERTY rather than
    # on equality of two calls: the simulation branch is random in two of these three, and an
    # earlier version of this test asserted two random draws were equal and failed for the wrong
    # reason -- the exact class of check this repo spends its time removing.
    assert cls.leverage(agent, _response(True)) == 0.0, (
        f"{name} requests leverage on an exchange response, which the engine refuses at placement"
    )
    assert cls.leverage(agent, _response(False)) != 0.0, (
        f"{name} stands leverage down on simulation, where it is the behaviour being demonstrated"
    )


@pytest.mark.parametrize("name", _MODE_DEPENDENT)
def test_the_agents_own_flag_cannot_change_the_answer(name):
    """The race made concrete: set the agent's flag against the response and the answer must hold."""
    cls = _load(name)
    agent = object.__new__(cls)
    agent.min_leverage, agent.max_leverage = 1.0, 2.0

    agent._exchange_mode = False          # as if a simulation request had just run update()
    assert cls.leverage(agent, _response(True)) == 0.0
    agent._exchange_mode = True           # as if an exchange request had just run update()
    assert cls.leverage(agent, _response(False)) != 0.0
