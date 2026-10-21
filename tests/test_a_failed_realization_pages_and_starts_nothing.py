# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A MULTI-ASSET RUN THAT STOPS ON A FAILED REALIZATION PAGES, AND THE VALIDATOR STARTS NOTHING.

The engine stops the run at the next barrier when a realization throws and, before it exits, posts one
EVENT_SIMULATION_FAILED notice to the lifecycle route, carrying the realization, its background, the
simulation time and the error. The validator:

  * raises a PagerDuty incident at the highest severity the helper supports, with those four facts;
  * does not take it for a run end: no on_end, no update_repo, no new simulation;
  * keeps it to itself: the notice never reaches the miner-forwarding path;
  * holds its simulator monitor, which would otherwise resume the stopped engine from its latest
    checkpoint within minutes, until a state arrives again (the operator's relaunch).
"""

import asyncio
import inspect
import types

import pytest

validator = pytest.importorskip("taos.im.neurons.validator")

FAILED = "EVENT_SIMULATION_FAILED"


def _failure_notice(**payload):
    body = {"realization": 3, "background": "bg", "time": 7_200_000_000_000, "error": "boom"}
    body.update(payload)
    return {
        "timestamp": body["time"],
        "delay": 0,
        "source": "SIMULATION",
        "target": "*",
        "type": FAILED,
        "payload": body,
    }


class _Request:
    def __init__(self, body: bytes):
        self._body = body

    async def stream(self):
        yield self._body


class _Engine:
    def __init__(self):
        self.calls = []

    def on_start(self, *args):
        self.calls.append("on_start")

    def on_end(self, *args):
        self.calls.append("on_end")


def _stub_validator():
    pages = []

    def pagerduty_alert(incident_text, method="unknown", dedup_key=None, details=None,
                        event_class="ERROR", severity="error"):
        pages.append({"text": incident_text, "details": details, "severity": severity, "dedup_key": dedup_key})

    def update_repo(*args, **kwargs):
        raise AssertionError("a failure notice ran update_repo")

    stub = types.SimpleNamespace(
        engine=_Engine(),
        pagerduty_alert=pagerduty_alert,
        update_repo=update_repo,
        simulation_failed_at=None,
        last_state_time=None,
    )
    stub.on_simulation_failed = types.MethodType(validator.Validator.on_simulation_failed, stub)
    return stub, pages


def _post(stub, messages, monkeypatch):
    import msgspec

    forwarded = []

    async def notify(self, notices):
        forwarded.extend(notices)

    monkeypatch.setattr(validator, "notify", notify, raising=False)
    body = msgspec.json.encode({"messages": messages})
    asyncio.run(validator.Validator.account(stub, _Request(body)))
    return forwarded


def test_the_failure_notice_pages_at_the_highest_severity_with_the_four_facts(monkeypatch):
    stub, pages = _stub_validator()

    _post(stub, [_failure_notice()], monkeypatch)

    assert len(pages) == 1, "one notice must raise exactly one page"
    page = pages[0]
    assert page["severity"] == "critical"
    for key, value in {"realization": 3, "background": "bg", "time": 7_200_000_000_000, "error": "boom"}.items():
        assert page["details"][key] == value
    for fact in ("3", "bg", "boom"):
        assert fact in page["text"]


def test_the_failure_notice_starts_no_end_of_run_path(monkeypatch):
    stub, _ = _stub_validator()

    _post(stub, [_failure_notice()], monkeypatch)

    assert "on_end" not in stub.engine.calls, "a failed run was taken for a run end"
    assert stub.simulation_failed_at is not None, "the simulator monitor was not held"


def test_the_failure_notice_never_reaches_the_miner_forwarding_path(monkeypatch):
    stub, _ = _stub_validator()

    forwarded = _post(stub, [_failure_notice()], monkeypatch)

    assert forwarded == [], f"the failure notice reached notify(): {forwarded}"
    assert FAILED in validator._NOTICES_CONSUMED_NOT_FORWARDED
    assert not validator._notice_is_forwarded_to_miners(FAILED)


def test_a_notice_missing_fields_still_pages(monkeypatch):
    stub, pages = _stub_validator()

    _post(stub, [{"timestamp": 5, "delay": 0, "source": "SIMULATION", "target": "*", "type": FAILED}], monkeypatch)

    assert len(pages) == 1 and pages[0]["severity"] == "critical"


def test_the_monitor_is_held_until_a_state_arrives_after_the_failure():
    held = validator._simulator_restart_is_held
    assert not held(None, None)
    assert not held(None, 100.0)
    assert held(100.0, None), "no state since the failure: the stopped engine must be left stopped"
    assert held(100.0, 99.0)
    assert not held(100.0, 101.0), "a state after the failure is the relaunch; the monitor resumes"


@pytest.mark.parametrize("method", ["monitor", "_sync_and_check"])
def test_both_simulator_restart_paths_ask_the_hold_first(method):
    src = inspect.getsource(getattr(validator.Validator, method))
    assert "_simulator_restart_is_held(" in src, f"{method} restarts the simulator without asking the hold"
    assert src.index("_simulator_restart_is_held(") < src.index("restart_simulator"), (
        f"{method} asks the hold only after it has restarted the simulator"
    )
