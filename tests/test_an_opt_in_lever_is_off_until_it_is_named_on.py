# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A lever that changes the round's path and has not yet been read on a live network is off unless it is named on.

The round-time levers are on unless turned off (levers.py). An opt-in lever (OPT_IN) is the other way round: off
until it is named in TAOS_LEVERS_ON or in the levers-on file (TAOS_LEVERS_ON_FILE, default ~/.taos/levers_on), read
the same way as the off file, so it goes on with `echo offmain_build >> ~/.taos/levers_on` and off when the line is
removed, with no restart. Naming it off as well turns it off. The report publishes its state beside the others, as
validator_gauges lever_<name>, and the round-time levers keep their own default."""
import importlib
import inspect
import os

import pytest

from taos.im.validator import levers as lv


@pytest.fixture
def levers(tmp_path, monkeypatch):
    off, on = tmp_path / "levers_off", tmp_path / "levers_on"
    monkeypatch.setenv("TAOS_LEVERS_FILE", str(off))
    monkeypatch.setenv("TAOS_LEVERS_ON_FILE", str(on))
    monkeypatch.delenv("TAOS_LEVERS_OFF", raising=False)
    monkeypatch.delenv("TAOS_LEVERS_ON", raising=False)
    mod = importlib.reload(lv)
    monkeypatch.setattr(mod, "CHECK_S", 0.0)
    yield mod, off, on
    for name in ("TAOS_LEVERS_FILE", "TAOS_LEVERS_ON_FILE"):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(lv)


def _write(path, text, ns):
    path.write_text(text)
    os.utime(path, ns=(ns, ns))


def test_offmain_build_is_an_opt_in_lever_off_by_default(levers):
    mod, _, _ = levers
    assert "offmain_build" in mod.OPT_IN and "offmain_build" not in mod.LEVERS
    assert not mod.enabled("offmain_build")
    assert mod.opt_in_states() == {"offmain_build": False}
    assert mod.states() == {n: True for n in mod.LEVERS}, "the round-time levers keep their default, on"


def test_naming_it_in_the_on_file_turns_it_on_and_removing_the_line_turns_it_off(levers):
    mod, _, on = levers
    _write(on, "offmain_build  # testnet hour on\n", 1_000_000_000)
    assert mod.enabled("offmain_build") and mod.opt_in_states() == {"offmain_build": True}
    _write(on, "# off again\n", 2_000_000_000)
    assert not mod.enabled("offmain_build")


def test_naming_it_off_wins_over_naming_it_on(levers):
    mod, off, on = levers
    _write(on, "offmain_build\n", 1_000_000_000)
    _write(off, "offmain_build\n", 1_000_000_000)
    assert not mod.enabled("offmain_build")


def test_the_environment_turns_it_on_and_a_round_time_lever_named_on_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TAOS_LEVERS_FILE", str(tmp_path / "none_off"))
    monkeypatch.setenv("TAOS_LEVERS_ON_FILE", str(tmp_path / "none_on"))
    monkeypatch.setenv("TAOS_LEVERS_ON", "offmain_build,push_in_wait")
    try:
        mod = importlib.reload(lv)
        said = []
        monkeypatch.setattr(mod, "_say", lambda msg: said.append(msg))
        assert mod.enabled("offmain_build") and mod.enabled("push_in_wait")
        mod.opt_in_states()
        assert len(said) == 1 and "TAOS_LEVERS_ON" in said[0] and "offmain_build on" in said[0], said
    finally:
        monkeypatch.delenv("TAOS_LEVERS_ON")
        importlib.reload(lv)


def test_the_report_publishes_the_opt_in_levers_beside_the_round_time_ones():
    from taos.im.neurons import validator
    src = inspect.getsource(validator)
    assert "'levers': levers.states() | levers.opt_in_states()" in src
