# SPDX-License-Identifier: MIT
"""The validator's self-restart at a run end used to follow the new engine's start within seconds, on the same
thread that had to handle the engine's start event: the event arrived while the process was being replaced, the
replacement loaded the state saved before it, and the old run's last window rode into the new run unshifted.

At the run end the update installs the code and leaves the restart to the run that is about to open: the engine
handler restarts the validator once the start event has been handled and the state saved on the new clock. The
hourly path, which runs mid-run with no boundary to wait for, restarts at once as before.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taos.im.neurons.validator as vmod  # noqa: E402
import taos.im.validator.update as upd  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _validator():
    return SimpleNamespace(repo_path=Path(REPO_ROOT), config=None, pagerduty_alert=lambda *a, **k: None)


def test_at_the_run_end_the_update_installs_and_leaves_the_restart_to_the_new_run(monkeypatch):
    ran = []
    monkeypatch.setattr(upd, "run_process", lambda cmd, cwd: (ran.append(cmd), SimpleNamespace(returncode=0, stderr=""))[1])
    monkeypatch.setattr(upd, "restart_validator_process", lambda self: ran.append("restart"))
    d = _validator()
    upd.update_validator(d, restart=False)
    assert ran == [["pip", "install", "-e", "."]], "installed, not restarted"
    assert isinstance(d._restart_when_run_opens, float)


def test_the_hourly_update_still_restarts_at_once(monkeypatch):
    ran = []
    monkeypatch.setattr(upd, "run_process", lambda cmd, cwd: SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(upd, "restart_validator_process", lambda self: ran.append("restart"))
    d = _validator()
    upd.update_validator(d)
    assert ran == ["restart"] and getattr(d, "_restart_when_run_opens", None) is None


def test_a_failed_install_owes_no_restart(monkeypatch):
    monkeypatch.setattr(upd, "run_process", lambda cmd, cwd: SimpleNamespace(returncode=1, stderr="boom"))
    monkeypatch.setattr(upd, "restart_validator_process", lambda self: (_ for _ in ()).throw(AssertionError("must not restart")))
    d = _validator()
    try:
        upd.update_validator(d, restart=False)
    except Exception:
        pass
    assert getattr(d, "_restart_when_run_opens", None) is None


def test_restart_validator_process_restarts_the_supervising_entry(monkeypatch):
    ran = []
    monkeypatch.setattr(upd, "pm2_self_entry", lambda: {"name": "validator-sim-mainnet", "pid": 4242, "pm_id": 7})
    monkeypatch.setattr(upd.subprocess, "run", lambda cmd, **k: (ran.append(cmd), SimpleNamespace(returncode=0, stdout="[]", stderr=""))[1])
    upd.restart_validator_process(_validator())
    assert ["pm2", "restart", "7"] in ran


class _Commit:
    def __init__(self, sha):
        self.hexsha = sha

    def diff(self, _other):
        return SimpleNamespace(iter_change_type=lambda _c: [])


def _duck(end_ok=True):
    remote = SimpleNamespace(pull=lambda: None)
    head = _Commit("s")
    repo = SimpleNamespace(remotes={"origin": remote}, head=SimpleNamespace(commit=head), commit=lambda sha: head)
    return SimpleNamespace(
        config=SimpleNamespace(subtensor=SimpleNamespace(chain_endpoint="wss://entrypoint-finney.opentensor.ai:443"),
                               repo=SimpleNamespace(remote="origin")),
        repo=repo, repo_path=None, engine=SimpleNamespace(mode="simulation"), _start_commit="s",
        pagerduty_alert=lambda msg, details=None: None,
    )


def test_the_run_end_asks_for_the_deferred_restart_and_the_hour_for_an_immediate_one(monkeypatch):
    seen = []
    monkeypatch.setattr(vmod, "update_validator", lambda self, **kw: seen.append(kw), raising=False)
    monkeypatch.setattr(vmod, "rebuild_simulator", lambda self: None, raising=False)
    monkeypatch.setattr(vmod, "restart_simulator", lambda self, end=False: None, raising=False)
    monkeypatch.setattr(vmod, "check_repo", lambda self: (True, False, False, False), raising=False)
    vmod.Validator.update_repo(_duck(), end=True)
    assert seen == [{"restart": False}], "the run end waits for the new run to open"
    seen.clear()
    vmod.Validator.update_repo(_duck())
    assert seen == [{}], "the hour restarts at once"
