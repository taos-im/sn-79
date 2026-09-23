# SPDX-License-Identifier: MIT
"""The validator's self-update at a simulation run end, as the 0.6.2 deploy relies on it.

The intended deploy: push to the branch the host tracks well before the run boundary and touch nothing on
the host; at the run's end the validator pulls, rebuilds the engine if its sources moved, starts the new
simulation from the new config, and restarts itself if its own code moved. Three properties that path did
not have on 23 September 2026:

1. The hourly check restarted the validator mid-run for a push that changed validator code and the
   simulator CONFIG (it deferred only for engine cpp/py changes), so a config-plus-scoring push would have
   applied half at the hour and half at the boundary.
2. A tree pulled by hand before the boundary (local == remote at the end) counted as nothing to update, so
   the running process kept its old code while the new engine started.
3. The restart fell back to `pm2 start --name=validator python validator.py <five flags>` whenever the pm2
   entry was not literally named "validator", dropping --port, --prometheus.port, --neuron.timeout, the
   engine and gentrx arguments and the process name. The mainnet entry is validator-sim-<network>.
"""
import os
import sys
import types
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import taos.im.neurons.validator as vmod  # noqa: E402
import taos.im.validator.update as upd  # noqa: E402


class _Change:
    def __init__(self, path):
        self.b_path = path
        self.a_path = path


class _Diff:
    change_type = ("M",)

    def __init__(self, paths):
        self._paths = paths

    def iter_change_type(self, _cht):
        return [_Change(p) for p in self._paths]


class _Commit:
    def __init__(self, sha, paths_to_next=()):
        self.hexsha = sha
        self._paths = paths_to_next

    def diff(self, _other):
        return _Diff(self._paths)


def _duck(start_sha, head_sha, head_paths=(), remote_endpoint="wss://entrypoint-finney.opentensor.ai:443", mode="simulation"):
    calls = {"pull": 0, "alerts": []}
    remote = SimpleNamespace(pull=lambda: calls.__setitem__("pull", calls["pull"] + 1))
    start = _Commit(start_sha, head_paths)
    head = _Commit(head_sha)
    repo = SimpleNamespace(remotes={"origin": remote}, head=SimpleNamespace(commit=head), commit=lambda sha: start if sha == start_sha else head)
    d = SimpleNamespace(
        config=SimpleNamespace(subtensor=SimpleNamespace(chain_endpoint=remote_endpoint), repo=SimpleNamespace(remote="origin")),
        repo=repo, repo_path=None, engine=SimpleNamespace(mode=mode), _start_commit=start_sha,
        pagerduty_alert=lambda msg, details=None: calls["alerts"].append(msg),
    )
    return d, calls


def _spy(monkeypatch):
    seen = {"update_validator": 0, "rebuild_simulator": 0, "restart_simulator": []}
    monkeypatch.setattr(vmod, "update_validator", lambda self: seen.__setitem__("update_validator", seen["update_validator"] + 1), raising=False)
    monkeypatch.setattr(vmod, "rebuild_simulator", lambda self: seen.__setitem__("rebuild_simulator", seen["rebuild_simulator"] + 1), raising=False)
    monkeypatch.setattr(vmod, "restart_simulator", lambda self, end=False: seen["restart_simulator"].append(end), raising=False)
    return seen


def test_classify_changes_sorts_paths_into_the_four_classes():
    old = _Commit("a", ["taos/im/validator/reward.py", "simulate/trading/src/cpp/agent/X.cpp",
                        "simulate/trading/run/config/simulation_0.xml", "simulate/trading/python/foo.py"])
    d = SimpleNamespace(repo_path=__import__("pathlib").Path("/r"), simulator_config_file="/r/simulate/trading/run/config/simulation_0.xml")
    v, cfg, spy, cpp = upd.classify_changes(d, old, _Commit("b"))
    assert (v, cfg, spy, cpp) == (True, True, True, True)
    v, cfg, spy, cpp = upd.classify_changes(d, _Commit("a", ["docs/x.md"]), _Commit("b"))
    assert (v, cfg, spy, cpp) == (False, False, False, False)


def test_hourly_check_defers_the_validator_restart_when_the_config_moved_too(monkeypatch):
    d, calls = _duck("s", "s")
    seen = _spy(monkeypatch)
    monkeypatch.setattr(vmod, "check_repo", lambda self: (True, True, False, False), raising=False)
    assert vmod.Validator.update_repo(d) is True
    assert seen["update_validator"] == 0 and calls["pull"] == 0, "a config change must wait for the run end"


def test_hourly_check_restarts_the_validator_when_only_its_code_moved(monkeypatch):
    d, calls = _duck("s", "s")
    seen = _spy(monkeypatch)
    monkeypatch.setattr(vmod, "check_repo", lambda self: (True, False, False, False), raising=False)
    vmod.Validator.update_repo(d)
    assert seen["update_validator"] == 1 and calls["pull"] == 1


def test_run_end_applies_a_tree_that_moved_under_the_process(monkeypatch):
    """Pulled by hand before the boundary: local == remote, so check_repo reports nothing, but HEAD is not
    the commit this process started on. The end must rebuild for the engine changes in that span and
    restart the validator for its own."""
    d, calls = _duck("start", "head", head_paths=["taos/im/validator/debeta.py", "simulate/trading/src/cpp/agent/X.cpp"])
    seen = _spy(monkeypatch)
    monkeypatch.setattr(vmod, "check_repo", lambda self: (False, False, False, False), raising=False)
    monkeypatch.setattr(vmod, "classify_changes", upd.classify_changes, raising=False)
    d.repo_path = __import__("pathlib").Path("/r"); d.simulator_config_file = "/r/simulate/trading/run/config/simulation_0.xml"
    vmod.Validator.update_repo(d, end=True)
    assert seen["rebuild_simulator"] == 1
    assert seen["restart_simulator"] == [True]
    assert seen["update_validator"] == 1


def test_run_end_with_nothing_moved_only_restarts_the_simulator(monkeypatch):
    d, calls = _duck("s", "s")
    seen = _spy(monkeypatch)
    monkeypatch.setattr(vmod, "check_repo", lambda self: (False, False, False, False), raising=False)
    vmod.Validator.update_repo(d, end=True)
    assert seen["rebuild_simulator"] == 0 and seen["update_validator"] == 0 and seen["restart_simulator"] == [True]


def _fake_jlist(entries):
    import json
    class R:
        stdout = json.dumps(entries)
        returncode = 0
        stderr = ""
    return R()


def test_pm2_self_entry_resolves_the_supervising_entry_from_the_process_tree(monkeypatch):
    entries = [{"name": "validator-sim-mainnet", "pid": 4242, "pm_id": 7}, {"name": "simulator", "pid": 999, "pm_id": 2}]
    monkeypatch.setattr(upd.subprocess, "run", lambda *a, **k: _fake_jlist(entries))
    chain = [SimpleNamespace(pid=31337), SimpleNamespace(pid=4242), SimpleNamespace(pid=1)]
    monkeypatch.setattr(upd, "_process_chain", lambda: chain)
    e = upd.pm2_self_entry()
    assert e and e["name"] == "validator-sim-mainnet" and e["pm_id"] == 7


def test_update_validator_restarts_its_own_pm2_entry_and_never_a_truncated_command_line(monkeypatch):
    ran = []
    monkeypatch.setattr(upd, "run_process", lambda cmd, cwd: SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(upd, "pm2_self_entry", lambda: {"name": "validator-sim-mainnet", "pid": 4242, "pm_id": 7})
    monkeypatch.setattr(upd.subprocess, "run", lambda cmd, **k: (ran.append(cmd), SimpleNamespace(returncode=0, stdout="[]", stderr=""))[1])
    d = SimpleNamespace(repo_path=__import__("pathlib").Path(REPO_ROOT), config=None, pagerduty_alert=lambda *a, **k: None)
    upd.update_validator(d)
    assert ["pm2", "restart", "7"] in ran
    assert not any("python validator.py" in " ".join(c) for c in ran if isinstance(c, list))


def test_update_validator_refuses_a_truncated_restart_when_no_entry_resolves(monkeypatch):
    ran, alerts = [], []
    monkeypatch.setattr(upd, "run_process", lambda cmd, cwd: SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(upd, "pm2_self_entry", lambda: None)
    monkeypatch.setattr(upd.subprocess, "run", lambda cmd, **k: (ran.append(cmd), _fake_jlist([{"name": "something-else", "pid": 1, "pm_id": 0}]))[1])
    d = SimpleNamespace(repo_path=__import__("pathlib").Path(REPO_ROOT), config=None, pagerduty_alert=lambda msg, details=None: alerts.append(msg))
    upd.update_validator(d)
    assert not any("python validator.py" in " ".join(c) for c in ran if isinstance(c, list)), "the bare start drops every flag but five"
    assert alerts, "the operator must be told the process was left on the old code"


def test_the_localnet_guard_can_be_lifted_for_the_acceptance_suite(monkeypatch):
    d, calls = _duck("s", "s", remote_endpoint="ws://localhost:9944")
    seen = _spy(monkeypatch)
    monkeypatch.setattr(vmod, "check_repo", lambda self: (True, False, False, False), raising=False)
    monkeypatch.delenv("TAOS_AUTO_UPDATE_ON_LOCALNET", raising=False)
    vmod.Validator.update_repo(d)
    assert seen["update_validator"] == 0, "localnet stays hands-off by default"
    monkeypatch.setenv("TAOS_AUTO_UPDATE_ON_LOCALNET", "1")
    vmod.Validator.update_repo(d)
    assert seen["update_validator"] == 1, "with the guard lifted the same path runs on localnet"
