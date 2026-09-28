# SPDX-License-Identifier: MIT
"""At a run boundary the validator that ends the old run starts the new engine and then restarts itself onto
the updated code. The new engine's start event lands while the old process is going down, so the process that
comes up never receives it: it loads the state saved at the old run's end, on the old clock, and the first state
update it handles belongs to a run that opened without it. Nothing shifted the histories, so the old run's last
window rides into the new run with stamps ahead of the clock, inside every window, for as long as the new clock
takes to pass them.

Two things close that gap. A state update whose clock has gone backwards onto a run that has only just opened is
that missed start, and the engine applies the run change there, exactly as the start event would have. And a
validator restart owed at the run end waits until the new run has opened and its state has been saved on the new
clock, so the start event is handled by a process that stays up long enough to save it.
"""
import asyncio
import os
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taos.im.validator.engines.simulation as sim  # noqa: E402
import taos.im.validator.trade as trade  # noqa: E402
import taos.im.validator.update as upd  # noqa: E402

H = 3600_000_000_000
S = 1_000_000_000
OLD_END = 24 * H
FRESH = 30 * 60 * S


def test_a_clock_that_goes_back_onto_a_young_run_is_a_missed_run_change():
    assert trade.missed_run_change(OLD_END, 600 * S, fresh_threshold=FRESH)


def test_a_resume_that_steps_back_on_a_mature_run_is_not():
    # a checkpoint resume can land minutes behind the last state the validator saw, and it fires the start
    # event itself; a run that is hours old did not just open
    assert not trade.missed_run_change(10 * H, 10 * H - 5 * 60 * S, fresh_threshold=FRESH)


def test_a_clock_that_moves_forward_is_never_a_run_change():
    assert not trade.missed_run_change(600 * S, 601 * S, fresh_threshold=FRESH)
    assert not trade.missed_run_change(0, 600 * S, fresh_threshold=FRESH)
    assert not trade.missed_run_change(None, 600 * S, fresh_threshold=FRESH)


def test_history_clock_health_reports_how_far_the_stamps_sit_ahead_of_the_clock():
    v = SimpleNamespace(simulation_timestamp=600 * S, debeta_invn_hist={0: {OLD_END: 1.0, 500 * S: 1.0}, 1: {}},
                        _run_changes_recovered=1, _histories_rebased_at_load=7)
    h = trade.history_clock_health(v)
    assert h == {"history_stamp_ahead_ns": OLD_END - 600 * S, "run_changes_recovered": 1, "histories_rebased_at_load": 7}
    v.debeta_invn_hist = {0: {500 * S: 1.0}}
    assert trade.history_clock_health(v)["history_stamp_ahead_ns"] == 0
    assert trade.history_clock_health(SimpleNamespace())["history_stamp_ahead_ns"] == 0


class _Loop:
    """The validator's main loop runs in its own thread; the run change saves through it."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


class _Now:
    """A timer that fires at once, so the test sees the restart the engine schedules."""

    def __init__(self, delay, fn):
        self.fn = fn

    def start(self):
        self.fn()


def _engine(tmp_path, monkeypatch, *, clock=OLD_END):
    calls = {"shift": [], "saves": 0, "restarts": 0, "seed": [], "config_loads": 0}
    v = SimpleNamespace()
    v.simulation_timestamp = clock
    v.simulation = SimpleNamespace(volumeDecimals=4, book_count=2, miner_wealth=1.0, logDir=None, simulation_id=None,
                                   grace_period=0, log_window=H, publish_interval=1)
    v.config = SimpleNamespace(
        scoring=SimpleNamespace(kappa=SimpleNamespace(lookback=3 * H),
                                activity=SimpleNamespace(trade_volume_assessment_period=24 * H)),
        subtensor=SimpleNamespace(network="finney"), neuron=SimpleNamespace(full_path=str(tmp_path)))
    v.effective_max_uids = 3
    v._scoring_shadow = None
    v.start_time = 1.0
    v.start_timestamp = 0
    v.last_state_time = 2.0
    v.step_rates = [1.0]
    v._shadow_applied_ts = clock
    v.update_repo = lambda end=False: None
    loop = _Loop()
    v.main_loop = loop.loop

    async def _save():
        calls["saves"] += 1

    v._save_state_sync = _save
    e = sim.SimulationEngine.__new__(sim.SimulationEngine)
    e.validator = v
    monkeypatch.setattr(e, "_load_config", lambda: calls.__setitem__("config_loads", calls["config_loads"] + 1))
    monkeypatch.setattr(e, "_notify_seed_log_dir_change", lambda d: calls["seed"].append(d))
    monkeypatch.setattr(e, "compress_outputs", lambda start=False: None)
    monkeypatch.setattr(e, "load_fundamental", lambda: None)
    monkeypatch.setattr(trade, "shift_simulation_histories", lambda self, old, new, **kw: calls["shift"].append((old, new)))
    monkeypatch.setattr(upd, "restart_validator_process", lambda self: calls.__setitem__("restarts", calls["restarts"] + 1))
    monkeypatch.setattr(sim.threading, "Timer", _Now)
    return e, v, calls, loop


def _state(tmp_path, ts):
    return SimpleNamespace(timestamp=ts, logDir=str(tmp_path / "logs" / "20260927_170001"))


def test_the_first_update_of_a_run_that_opened_without_the_process_applies_the_run_change(tmp_path, monkeypatch):
    e, v, calls, loop = _engine(tmp_path, monkeypatch)
    try:
        e.on_tick(_state(tmp_path, 600 * S))
        assert calls["shift"] == [(OLD_END, 0)], "from the old run's end onto the new run's start, as the start event does"
        assert calls["saves"] == 1, "the shifted state is on disk before anything else runs"
        assert v.simulation_timestamp == 600 * S
        assert v.simulation.simulation_id == "20260927_1700"
        assert calls["seed"] == [str(tmp_path / "logs" / "20260927_170001")]
        assert v._run_changes_recovered == 1
        e.on_tick(_state(tmp_path, 601 * S))
        assert calls["shift"] == [(OLD_END, 0)] and calls["saves"] == 1, "a clock that moves on is left alone"
    finally:
        loop.stop()


def test_a_process_that_saw_the_start_event_does_not_shift_again_at_the_first_update(tmp_path, monkeypatch):
    e, v, calls, loop = _engine(tmp_path, monkeypatch)
    try:
        e.on_start(0, SimpleNamespace(logDir=str(tmp_path / "logs" / "20260927_170001")))
        assert calls["shift"] == [(OLD_END, 0)] and v.simulation_timestamp == 0
        e.on_tick(_state(tmp_path, 600 * S))
        assert calls["shift"] == [(OLD_END, 0)]
        assert not hasattr(v, "_run_changes_recovered") or v._run_changes_recovered == 0
    finally:
        loop.stop()


def test_a_mature_run_whose_clock_steps_back_is_not_reopened(tmp_path, monkeypatch):
    e, v, calls, loop = _engine(tmp_path, monkeypatch, clock=10 * H)
    try:
        e.on_tick(_state(tmp_path, 10 * H - 5 * 60 * S))
        assert calls["shift"] == [] and calls["saves"] == 0
    finally:
        loop.stop()


def test_a_restart_owed_at_the_run_end_follows_the_start_event_once_the_state_is_saved(tmp_path, monkeypatch):
    e, v, calls, loop = _engine(tmp_path, monkeypatch)
    v._restart_when_run_opens = time.time()
    try:
        e.on_start(0, SimpleNamespace(logDir=str(tmp_path / "logs" / "20260927_170001")))
        assert calls["saves"] == 1 and calls["restarts"] == 1
        assert v._restart_when_run_opens is None
    finally:
        loop.stop()


def test_the_owed_restart_also_follows_a_recovered_run_change(tmp_path, monkeypatch):
    e, v, calls, loop = _engine(tmp_path, monkeypatch)
    v._restart_when_run_opens = time.time()
    try:
        e.on_tick(_state(tmp_path, 600 * S))
        assert calls["saves"] == 1 and calls["restarts"] == 1 and v._restart_when_run_opens is None
    finally:
        loop.stop()


def test_an_owed_restart_left_waiting_too_long_happens_at_the_next_update_after_a_save(tmp_path, monkeypatch):
    e, v, calls, loop = _engine(tmp_path, monkeypatch, clock=600 * S)
    v._restart_when_run_opens = time.time() - 2 * 3600
    try:
        e.on_tick(_state(tmp_path, 601 * S))
        assert calls["shift"] == [] and calls["saves"] == 1 and calls["restarts"] == 1
        assert v._restart_when_run_opens is None
    finally:
        loop.stop()


def test_nothing_restarts_when_no_restart_is_owed(tmp_path, monkeypatch):
    e, v, calls, loop = _engine(tmp_path, monkeypatch)
    try:
        e.on_start(0, SimpleNamespace(logDir=str(tmp_path / "logs" / "20260927_170001")))
        e.on_tick(_state(tmp_path, 600 * S))
        assert calls["restarts"] == 0
    finally:
        loop.stop()
