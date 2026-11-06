# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A validator stopped before its engine exists has nothing to save, so its stop runs no save and logs no failure.

The engine loads the saved state when it starts, and every fill is accounted through it, so until it exists the
validator holds nothing a restart would lose. A stop in that window runs no stop save and logs no failure.
"""
import asyncio
import signal
from types import SimpleNamespace

import bittensor as bt
import pytest

import taos.im.validator.cleanup as C
from taos.im.validator import persistence as P


def _starting_validator():
    return SimpleNamespace(
        _cleanup_done=False, main_loop=asyncio.new_event_loop(), shared_state_rewarding=False,
        shared_state_saving=False, shared_state_reporting=False, maintaining=False, compressing=False, querying=False,
        simulation_timestamp=0, step=0,
    )


def test_only_a_simulation_engine_saves_on_stop():
    assert not P.saves_on_stop(SimpleNamespace())
    assert P.saves_on_stop(SimpleNamespace(engine=SimpleNamespace(mode="simulation")))
    assert not P.saves_on_stop(SimpleNamespace(engine=SimpleNamespace(mode="exchange")))


def test_a_stop_before_the_engine_exists_saves_nothing_and_logs_no_failure(monkeypatch):
    errors, saves = [], []
    monkeypatch.setattr(bt.logging, "error", lambda msg, *a, **k: errors.append(str(msg)))
    monkeypatch.setattr(C, "save_on_stop", lambda v: saves.append(v))
    for name in ("cleanup_executors", "cleanup_ipc", "cleanup_event_loop"):
        monkeypatch.setattr(C, name, lambda v: None)
    v = _starting_validator()
    try:
        C.cleanup(v)
    finally:
        v.main_loop.close()
    assert saves == [] and errors == [] and v._cleanup_finished


def test_a_stop_before_the_engine_exists_logs_no_stop_save_failure(monkeypatch):
    errors = []
    monkeypatch.setattr(bt.logging, "error", lambda msg, *a, **k: errors.append(str(msg)))
    for name in ("cleanup_executors", "cleanup_ipc", "cleanup_event_loop"):
        monkeypatch.setattr(C, name, lambda v: None)
    v = _starting_validator()
    try:
        C.cleanup(v)
    finally:
        v.main_loop.close()
    assert not any("Stop save failed" in e for e in errors), errors


@pytest.fixture
def _signal_handlers():
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    yield
    for s, h in saved.items():
        signal.signal(s, h)


def test_a_second_signal_before_the_engine_exists_ends_the_stop(_signal_handlers):
    import taos.im.neurons.validator as V

    calls = []
    stopping = SimpleNamespace(_cleanup_done=True, cleanup=lambda: calls.append("cleanup"))
    V.Validator._setup_signal_handlers(stopping)
    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert calls == ["cleanup"], "the stop is not held for a save: cleanup, already done, returns and the process exits"
