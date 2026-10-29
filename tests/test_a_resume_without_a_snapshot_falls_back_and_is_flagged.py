# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A resume the validator holds no snapshot for falls back to re-opening the run, and says so where it is seen.

The snapshot can be missing: the validator did not see the engine's checkpoint directory, was down when the engine
wrote it, or the engine resumed from an older checkpoint than the validator kept. The validator then re-opens the run
on the resumed clock as before, so the replayed stretch counts twice in every window and the positions held at the
checkpoint are reset. That is logged and counted on the history-clock gauges. A new run is handled as before and
never reloads a snapshot.
"""
import logging
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import taos.im.validator.checkpoint_snapshot as cs  # noqa: E402
import taos.im.validator.engines.simulation as sim  # noqa: E402
from taos.im.validator.trade import history_clock_health  # noqa: E402

S = 1_000_000_000
H = 3600 * S
RUN = "20261006_1603"
DEV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _engine(tmp_path, locked=RUN, held=20 * H):
    v = SimpleNamespace()
    v.config = SimpleNamespace(subtensor=SimpleNamespace(network="local"),
                               neuron=SimpleNamespace(full_path=str(tmp_path / "validator")))
    os.makedirs(v.config.neuron.full_path, exist_ok=True)
    v.simulation_timestamp = held
    logdir = str(tmp_path / "logs" / "20261006_160323")
    v.simulation = SimpleNamespace(logDir=logdir, simulation_id=locked, log_window=H, publish_interval=S)
    v.effective_max_uids = 4
    v.update_repo = lambda end=False: None
    e = sim.SimulationEngine.__new__(sim.SimulationEngine)
    e.validator = v
    e._fresh_threshold = lambda: H
    e._notify_seed_log_dir_change = lambda d: None
    e.compress_outputs = lambda start=False: None
    e._carry_fundamental = lambda state: None
    opened, restored = [], []

    def _open_run(old_ts, new_ts, log_dir, *, reason):
        opened.append((old_ts, new_ts, reason))
        v.simulation_timestamp = new_ts
    e._open_run = _open_run
    e._resume_from_snapshot = lambda old_ts, state, snap: restored.append(snap["meta"]["ts"])
    return e, v, logdir, opened, restored


def _snapshot(v, logdir, run, ts):
    """A snapshot written as the validator writes it, bound to the engine's checkpoint directory for ts."""
    os.makedirs(os.path.join(logdir, "ckpt", f"{ts}.ckptd"), exist_ok=True)
    meta = cs.snapshot_meta(v, run, ts, cs.engine_checkpoint(logdir, ts))
    cs.write_snapshot(v, cs.snapshot_path(v, run, ts), meta, {})


def test_a_resume_with_no_snapshot_re_opens_the_run_as_before_logs_it_and_counts_it(tmp_path, caplog):
    e, v, logdir, opened, restored = _engine(tmp_path)
    with caplog.at_level(logging.WARNING):
        e.on_tick(SimpleNamespace(timestamp=9 * H + S, logDir=logdir))
    assert restored == [] and opened == [(20 * H, 9 * H + S, opened[0][2])] and "resum" in opened[0][2]
    assert v._resumes_unrestored == 1 and v._run_changes_recovered == 1
    line = next(r.getMessage() for r in caplog.records if "RESUME NOT SEEN" in r.getMessage())
    assert "no scoring snapshot" in line and str(9 * H) in line, line
    health = history_clock_health(v)
    assert health["resumes_unrestored"] == 1 and health["resumes_restored"] == 0


def test_a_resume_with_a_snapshot_at_the_checkpoint_reloads_it(tmp_path):
    e, v, logdir, opened, restored = _engine(tmp_path)
    v._last_accounted_ts = 9 * H
    _snapshot(v, logdir, RUN, 9 * H)
    e.on_tick(SimpleNamespace(timestamp=9 * H + S, logDir=logdir))
    assert opened == [] and restored == [9 * H]


def test_a_snapshot_older_than_the_checkpoint_resumed_is_not_used(tmp_path):
    """Only the snapshot at the checkpoint the engine resumed from is exact: an older one would drop the stretch
    between them, so the validator falls back and flags it."""
    e, v, logdir, opened, restored = _engine(tmp_path)
    v._last_accounted_ts = 8 * H
    _snapshot(v, logdir, RUN, 8 * H)
    e.on_tick(SimpleNamespace(timestamp=9 * H + S, logDir=logdir))
    assert restored == [] and len(opened) == 1 and v._resumes_unrestored == 1


def test_a_new_run_is_unchanged_and_never_reloads_a_snapshot(tmp_path):
    e, v, logdir, opened, restored = _engine(tmp_path)
    v._last_accounted_ts = 0
    _snapshot(v, logdir, RUN, 0)
    new_dir = str(tmp_path / "logs" / "20261009_120000")
    e.on_tick(SimpleNamespace(timestamp=S, logDir=new_dir))
    assert restored == [] and len(opened) == 1 and opened[0][1] == 0 and "first state update" in opened[0][2]
    assert getattr(v, "_resumes_unrestored", 0) == 0, "a new run is not a resume"
    e2, v2, logdir2, opened2, restored2 = _engine(tmp_path / "b")
    e2.on_start(0, SimpleNamespace(logDir=str(tmp_path / "b" / "logs" / "20261009_120000")))
    assert opened2 and opened2[0][1] == 0 and restored2 == []


def test_a_forward_tick_is_unchanged(tmp_path):
    e, v, logdir, opened, restored = _engine(tmp_path)
    st = SimpleNamespace(timestamp=20 * H + S, logDir=logdir)
    e.on_tick(st)
    assert opened == [] and restored == [] and v.simulation_timestamp == 20 * H + S
    assert getattr(st, "_checkpoint_snapshot", None) is None, "no checkpoint at the previous state, nothing due"
    os.makedirs(os.path.join(logdir, "ckpt", f"{20 * H + S}.ckptd"))
    v.miner_presence, v.miner_stats = {}, {}
    st2 = SimpleNamespace(timestamp=20 * H + 2 * S, logDir=logdir)
    e.on_tick(st2)
    assert st2._checkpoint_snapshot[0] == 20 * H + S, "the engine checkpointed the previous state"


def test_the_counters_reach_the_gauges():
    report = open(os.path.join(DEV, "taos/im/validator/report.py")).read()
    clock_block = report.split("_clock = getattr")[1]
    loop = re.search(r'for _gauge in \(([^)]*)\):\n\s*self\.prometheus_validator_gauges', clock_block)
    assert loop and "resumes_unrestored" in loop.group(1) and "resumes_restored" in loop.group(1)
