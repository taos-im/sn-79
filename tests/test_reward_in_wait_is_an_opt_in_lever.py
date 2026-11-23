# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""reward_in_wait is an opt-in lever: off until named in the levers-on file or TAOS_LEVERS_ON, and off again when named
in the levers-off file as well. The four round-time levers keep their default, on; the report publishes the opt-in
states beside them under the same lever_<name> gauges; and the levers-off file's state line names an opt-in lever
named off as off, not as unknown.
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROUND_TIME = ("push_in_wait", "pack_shared", "account_deltas", "prereq_in_wait")


@pytest.fixture
def levers(tmp_path, monkeypatch):
    off, on = tmp_path / "levers_off", tmp_path / "levers_on"
    monkeypatch.setenv("TAOS_LEVERS_FILE", str(off))
    monkeypatch.setenv("TAOS_LEVERS_ON_FILE", str(on))
    monkeypatch.delenv("TAOS_LEVERS_OFF", raising=False)
    monkeypatch.delenv("TAOS_LEVERS_ON", raising=False)
    import taos.im.validator.levers as mod
    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "CHECK_S", 0.0)
    monkeypatch.setattr(mod, "_say", lambda msg: said.append(msg))
    said = []
    mod._said = said
    yield mod, off, on
    for name in ("TAOS_LEVERS_FILE", "TAOS_LEVERS_ON_FILE"):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(mod)


def test_reward_in_wait_is_opt_in_and_off_by_default(levers):
    mod, off, on = levers
    assert "reward_in_wait" in mod.OPT_IN and "reward_in_wait" not in mod.LEVERS
    assert mod.LEVERS == ROUND_TIME, "the four round-time levers stay as they were"
    assert not mod.enabled("reward_in_wait")
    assert mod.states() == {n: True for n in ROUND_TIME}
    assert mod.opt_in_states() == {"offmain_build": False, "reward_in_wait": False}
    assert set(mod.states() | mod.opt_in_states()) == set(ROUND_TIME) | {"offmain_build", "reward_in_wait"}, \
        "the report publishes every lever, lever_reward_in_wait included"


def test_naming_it_in_the_on_file_turns_it_on_and_naming_it_off_as_well_turns_it_off(levers):
    mod, off, on = levers
    on.write_text("reward_in_wait\n")
    assert mod.enabled("reward_in_wait") and mod.opt_in_states()["reward_in_wait"]
    assert mod.enabled("prereq_in_wait"), "the lever it needs stays on"
    off.write_text("reward_in_wait\n")
    assert not mod.enabled("reward_in_wait"), "named off wins over named on"
    off.write_text("")
    on.write_text("")
    assert not mod.enabled("reward_in_wait"), "the line removed, it is off again"


def test_the_environment_turns_it_on(tmp_path, monkeypatch):
    monkeypatch.setenv("TAOS_LEVERS_FILE", str(tmp_path / "off"))
    monkeypatch.setenv("TAOS_LEVERS_ON_FILE", str(tmp_path / "on"))
    monkeypatch.setenv("TAOS_LEVERS_ON", "reward_in_wait")
    monkeypatch.delenv("TAOS_LEVERS_OFF", raising=False)
    import taos.im.validator.levers as mod
    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "_say", lambda msg: None)
    try:
        assert mod.enabled("reward_in_wait")
    finally:
        monkeypatch.delenv("TAOS_LEVERS_ON")
        importlib.reload(mod)


def test_the_off_files_state_line_names_an_opt_in_lever_named_off_as_off_not_as_unknown(levers):
    mod, off, on = levers
    on.write_text("reward_in_wait\n")
    off.write_text("reward_in_wait\nno_such_lever\n")
    assert not mod.enabled("reward_in_wait")
    line = next(m for m in reversed(mod._said) if m.startswith("levers ("))
    assert "reward_in_wait off" in line
    assert "not a lever, ignored: no_such_lever" in line
    assert "ignored: reward_in_wait" not in line and "ignored: no_such_lever, reward_in_wait" not in line
