# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Every round-time lever can be turned off on a live validator in seconds, with no restart.

Each lever that moves work into the miners' wait, or changes what the validator does on the round's path, can be
turned off at once. A lever is on unless it is named in TAOS_LEVERS_OFF (read at start) or in the levers file
(TAOS_LEVERS_FILE, default ~/.taos/levers_off, one name a line), which the validator re-reads when it changes, looking
at most every CHECK_S seconds. A lever turned off takes its other path: the data-service push after the forward, the
report's copy of the whole payload. Each change is logged, and the report publishes each lever's state as
validator_gauges lever_<name>."""
import importlib
import inspect
import os
import types

import pytest

from taos.im.neurons import validator
from taos.im.validator import levers as lv


@pytest.fixture
def levers(tmp_path, monkeypatch):
    path = tmp_path / "levers_off"
    monkeypatch.setenv("TAOS_LEVERS_FILE", str(path))
    monkeypatch.delenv("TAOS_LEVERS_OFF", raising=False)
    mod = importlib.reload(lv)
    monkeypatch.setattr(mod, "CHECK_S", 0.0)
    yield mod, path
    monkeypatch.delenv("TAOS_LEVERS_FILE", raising=False)
    importlib.reload(lv)


def _write(path, text, ns):
    path.write_text(text)
    os.utime(path, ns=(ns, ns))


def test_every_lever_is_on_without_a_file(levers):
    mod, _ = levers
    assert {"push_in_wait", "pack_shared"} <= set(mod.LEVERS), "push_in_wait and pack_shared are levers"
    assert all(mod.enabled(n) for n in mod.LEVERS) and mod.states() == {n: True for n in mod.LEVERS}


def test_the_file_turns_a_lever_off_and_back_on_without_a_restart(levers):
    mod, path = levers
    _write(path, "push_in_wait  # the query looked slow\n\n", 1_000_000_000)
    assert not mod.enabled("push_in_wait") and mod.enabled("pack_shared")
    _write(path, "# nothing off\n", 2_000_000_000)
    assert mod.enabled("push_in_wait")
    path.unlink()
    assert mod.enabled("push_in_wait")


def test_the_file_is_looked_at_no_more_often_than_its_period(levers, monkeypatch):
    mod, path = levers
    monkeypatch.setattr(mod, "CHECK_S", 3600.0)
    assert mod.enabled("push_in_wait")
    _write(path, "push_in_wait\n", 3_000_000_000)
    assert mod.enabled("push_in_wait"), "inside the period the last reading stands"


def test_the_environment_turns_a_lever_off_from_the_start(tmp_path, monkeypatch):
    monkeypatch.setenv("TAOS_LEVERS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("TAOS_LEVERS_OFF", "pack_shared, push_in_wait")
    try:
        mod = importlib.reload(lv)
        assert not mod.enabled("pack_shared") and not mod.enabled("push_in_wait")
    finally:
        monkeypatch.delenv("TAOS_LEVERS_OFF")
        monkeypatch.delenv("TAOS_LEVERS_FILE")
        importlib.reload(lv)


def test_a_change_is_logged_once_and_an_unknown_name_is_named(levers, monkeypatch):
    mod, path = levers
    said = []
    monkeypatch.setattr(mod, "_say", lambda msg: said.append(msg))
    _write(path, "push_in_wait\npush_in_wiat\n", 4_000_000_000)
    mod.enabled("push_in_wait")
    mod.enabled("push_in_wait")
    assert len(said) == 1 and "push_in_wait off" in said[0] and "push_in_wiat" in said[0], said


def test_the_push_starts_in_the_wait_only_while_its_lever_is_on():
    hook = inspect.getsource(validator.Validator.on_query_sent)
    guard = hook.index("levers.enabled('push_in_wait')")
    assert guard < hook.index("self._sim_push_ts = state.timestamp"), "off: the post-forward spawn pushes it"


def test_a_push_lever_turned_off_leaves_the_push_to_the_forwards_end(levers):
    mod, path = levers
    _write(path, "push_in_wait\n", 5_000_000_000)
    spawned = []
    stub = types.SimpleNamespace(engine=types.SimpleNamespace(mode="simulation"), _sim_push_url=lambda: "http://x",
                                 _push_sim_state_bg=lambda state, url: spawned.append(state.timestamp))
    state = types.SimpleNamespace(timestamp=7)
    validator.Validator.on_query_sent(stub, state)
    assert not spawned and getattr(stub, "_sim_push_ts", None) != state.timestamp


def test_the_report_copies_the_whole_payload_while_its_lever_is_off():
    src = inspect.getsource(validator)
    at = src.index("def _pack_report_chunked():")
    body = src[at:at + 900]
    assert "if levers.enabled('pack_shared') else frozenset()" in body


def test_the_report_publishes_each_levers_state():
    from taos.im.validator import report
    src = inspect.getsource(report.publish_validator_gauges)
    assert "lever_" in src and "'levers'" in src
    assert "'levers': levers.states()" in inspect.getsource(validator)
