# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""`simulation_output_dir` must not crash when `simulation_id` is absent.

WHY THIS EXISTS. `simulation_id` is optional on the model (`simulation_id : str | None = None` in
taos/im/protocol/models.py), and this method joined it into a path unguarded. Every state update then
raised

    TypeError: join() argument must be str, bytes, or os.PathLike object, not 'NoneType'

inside respond(), which the miner framework CATCHES -- so the process stayed up, logged a bare ERROR
with an empty message, and the acceptance stage recorded "no simulation update observed in 90s" for an
agent that was receiving updates and dying on every one of them.

That is the shape worth guarding: a crash in a caught path is indistinguishable from silence. The
first test reproduces it, the second proves the normal path is unchanged, and the third pins the
behaviour that makes the next occurrence diagnosable instead of anonymous.
"""
import os
import types

from taos.im.agents import FinanceSimulationAgent


def _state(sim_id, hotkey="5FakeHotkey", netuid=5, cls_name="ExchangeStateUpdate"):
    """The smallest object the method actually reads: config.simulation_id, dendrite.hotkey, netuid."""
    state_cls = type(cls_name, (), {})
    st = state_cls()
    st.config = types.SimpleNamespace(simulation_id=sim_id)
    st.dendrite = types.SimpleNamespace(hotkey=hotkey)
    st.netuid = netuid
    return st


def _agent(tmp_path):
    """Bind the method to a stub: it uses only self.output_dir and the warn-once flag."""
    stub = types.SimpleNamespace(output_dir=str(tmp_path))
    stub.simulation_output_dir = types.MethodType(
        FinanceSimulationAgent.simulation_output_dir, stub)
    return stub


def test_missing_simulation_id_does_not_raise(tmp_path):
    agent = _agent(tmp_path)
    out = agent.simulation_output_dir(_state(None))
    assert os.path.isdir(out), "the directory must be created, not merely named"
    assert str(tmp_path) in out


def test_present_simulation_id_is_used_verbatim(tmp_path):
    agent = _agent(tmp_path)
    out = agent.simulation_output_dir(_state("sim-1234"))
    assert out.endswith(os.path.join("5FakeHotkey", "sim-1234"))
    assert os.path.isdir(out)


def test_fallback_names_its_own_source(tmp_path):
    """A missing id is a real gap. The fallback must say WHICH state lacked it and on which netuid,
    so the next occurrence is diagnosable from the directory name alone."""
    agent = _agent(tmp_path)
    out = agent.simulation_output_dir(_state(None, netuid=79, cls_name="MarketSimulationStateUpdate"))
    leaf = os.path.basename(out)
    assert "MarketSimulationStateUpdate" in leaf, leaf
    assert "netuid79" in leaf, leaf


def test_empty_string_id_is_treated_as_missing(tmp_path):
    """An empty string joins without raising, so it would silently pool every run's data under the
    hotkey directory -- a quieter version of the same bug."""
    agent = _agent(tmp_path)
    out = agent.simulation_output_dir(_state(""))
    assert os.path.basename(out) != "5FakeHotkey"
    assert "unidentified" in os.path.basename(out)
