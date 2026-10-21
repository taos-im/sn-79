# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""An engine whose process is up but which has sent no state for longer than the check allows pages the operator.

The health check read pm2's `online` (or a matching process) as healthy whatever the state clock said, so an engine
that hung with its process still running paged nobody. It pages now, once per silence: the dedup key is the time of
the last state, so every later check of the same silence updates one incident. The verdict stays healthy, so the
monitor does not restart it: a slow validator leaves the engine waiting in the same silence, and a resume replays a
hang. Silence under the allowance, and a stopped process, behave as before.
"""

import json
import time

import pytest

import taos.im.validator.update as U


class _Engine:
    def __init__(self, mode):
        self.mode = mode


class _V:
    def __init__(self, mode, state_age):
        self.engine = _Engine(mode)
        self.last_state_time = time.time() - state_age
        self.start_time = time.time() - 3600
        self.pages = []

    def pagerduty_alert(self, incident_text, method="unknown", dedup_key=None, details=None, **kw):
        self.pages.append({"text": incident_text, "method": method, "dedup_key": dedup_key, "details": details})


class _Run:
    def __init__(self, stdout):
        self.stdout = stdout


def _pm2(monkeypatch, processes):
    jlist = json.dumps([{"name": n, "pm2_env": {"status": s}} for n, s in processes.items()])
    monkeypatch.setattr(U.subprocess, "run", lambda *a, **kw: _Run(jlist))


class _Proc:
    def __init__(self, cmdline):
        self.info = {"cmdline": cmdline, "name": "taosim"}


def _procs(monkeypatch, cmdlines):
    monkeypatch.setattr(U.psutil, "process_iter", lambda attrs=None: iter([_Proc(c) for c in cmdlines]))


@pytest.mark.parametrize("mode,process", [("simulation", "simulator"), ("exchange", "exchange")])
def test_an_online_engine_with_no_state_past_the_allowance_pages_once_per_silence(monkeypatch, mode, process):
    _pm2(monkeypatch, {process: "online"})
    v = _V(mode, 400)
    assert U.check_simulator(v) is True, "a silent engine is paged, not restarted"
    assert len(v.pages) == 1
    page = v.pages[0]
    assert "no state" in page["text"] and "online" in page["text"]
    assert page["dedup_key"] and str(int(v.last_state_time)) in page["dedup_key"]
    assert page["details"]["last_state_age_s"] >= 400

    U.check_simulator(v)
    assert len(v.pages) == 2 and v.pages[1]["dedup_key"] == page["dedup_key"], "one silence is one incident"

    v.last_state_time = time.time() - 350
    U.check_simulator(v)
    assert v.pages[2]["dedup_key"] != page["dedup_key"], "a later silence opens its own incident"


def test_a_simulator_found_without_pm2_pages_the_same_way(monkeypatch):
    _pm2(monkeypatch, {})
    _procs(monkeypatch, [["../build/src/cpp/taosim", "-c", "latest"]])
    v = _V("simulation", 400)
    assert U.check_simulator(v) is True
    assert len(v.pages) == 1 and "no state" in v.pages[0]["text"]


@pytest.mark.parametrize("mode,process", [("simulation", "simulator"), ("exchange", "exchange")])
def test_a_recent_state_pages_nothing(monkeypatch, mode, process):
    _pm2(monkeypatch, {process: "online"})
    v = _V(mode, 10)
    assert U.check_simulator(v) is True
    assert v.pages == []


@pytest.mark.parametrize("mode,process", [("simulation", "simulator"), ("exchange", "exchange")])
def test_a_stopped_engine_still_pages_as_stopped_and_reads_unhealthy(monkeypatch, mode, process):
    _pm2(monkeypatch, {process: "stopped"})
    v = _V(mode, 400)
    assert U.check_simulator(v) is False
    assert len(v.pages) == 1 and "stopped" in v.pages[0]["text"]
