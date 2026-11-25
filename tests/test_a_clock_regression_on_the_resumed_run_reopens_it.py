"""A state update whose clock is behind the one the validator holds, on the SAME run, re-opens the run.

When a full relaunch resumes the engine from a checkpoint, the engine's start notice can go out while the validator
is still coming up, and the first state update then arrives with the checkpoint's clock, behind the restored
histories. A fallback for a missed start notice that only covers a clock near zero (a fresh run) would leave the
histories on the old clock, stamped ahead of it, with the skill leg, the floor and the inventory gauges carrying
the previous run. A clock that steps back on the same run is a resume, and the run is re-opened exactly as the
start notice would have done it; a start notice that then arrives late for the same episode is absorbed, while one
for a new run still applies.
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import taos.im.validator.engines.simulation as sim  # noqa: E402

S = 1_000_000_000
H = 3600 * S
RUN = "20261006_1200"
LOGDIR = "/r/logs/20261006_120000"


def _engine(locked=RUN, held=20 * H):
    v = SimpleNamespace()
    v.config = SimpleNamespace(subtensor=SimpleNamespace(network="local"), neuron=SimpleNamespace(full_path="/tmp"))
    v.simulation_timestamp = held
    v.simulation = SimpleNamespace(logDir=LOGDIR, simulation_id=locked, log_window=H, publish_interval=1)
    v.update_repo = lambda end=False: None
    e = sim.SimulationEngine.__new__(sim.SimulationEngine)
    e.validator = v
    e._fresh_threshold = lambda: H
    e._notify_seed_log_dir_change = lambda d: None
    e.compress_outputs = lambda start=False: None
    opened = []

    def _open_run(old_ts, new_ts, log_dir, *, reason):
        opened.append((old_ts, new_ts, log_dir, reason))
        v.simulation_timestamp = new_ts
    e._open_run = _open_run
    return e, v, opened


def test_a_clock_behind_the_held_one_on_the_same_run_reopens_the_run():
    e, v, opened = _engine()
    e.on_tick(SimpleNamespace(timestamp=9 * H + S, logDir=LOGDIR))
    assert opened == [(20 * H, 9 * H + S, LOGDIR, opened[0][3])] and "resum" in opened[0][3]
    assert v.simulation_timestamp == 9 * H + S and v.simulation.simulation_id == RUN
    assert v._run_changes_recovered == 1, "the dashboard's repair counter sees it, as it sees the near-zero repair"


def test_a_late_start_notice_for_the_reopened_run_is_absorbed():
    e, v, opened = _engine()
    e.on_tick(SimpleNamespace(timestamp=9 * H + S, logDir=LOGDIR))
    e.on_start(9 * H, SimpleNamespace(logDir=LOGDIR))
    assert len(opened) == 1, "the notice is the same episode the update already opened"


def test_a_start_notice_for_a_new_run_still_applies_after_a_reopen():
    e, v, opened = _engine()
    e.on_tick(SimpleNamespace(timestamp=9 * H + S, logDir=LOGDIR))
    e.on_start(0, SimpleNamespace(logDir="/r/logs/20261009_120000"))
    assert len(opened) == 2 and opened[1][1] == 0


def test_a_monotone_clock_does_not_reopen():
    e, v, opened = _engine()
    e.on_tick(SimpleNamespace(timestamp=20 * H + S, logDir=LOGDIR))
    e.on_tick(SimpleNamespace(timestamp=20 * H + 2 * S, logDir=LOGDIR))
    assert opened == [] and v.simulation_timestamp == 20 * H + 2 * S


def test_a_near_zero_clock_still_takes_the_missed_run_change_path():
    e, v, opened = _engine()
    e.on_tick(SimpleNamespace(timestamp=S, logDir="/r/logs/20261009_120000"))
    assert len(opened) == 1 and opened[0][1] == 0 and "first state update" in opened[0][3]


def test_an_older_run_is_still_refused_before_any_reopen():
    e, v, opened = _engine()
    with pytest.raises(sim.RunWentBackwardError):
        e.on_tick(SimpleNamespace(timestamp=9 * H, logDir="/r/logs/20261002_120000"))
    assert opened == [] and v.simulation_timestamp == 20 * H
