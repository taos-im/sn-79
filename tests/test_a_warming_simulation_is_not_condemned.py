# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A simulation that is warming up must not be killed for being silent.

The engine publishes no state during its grace period: the publish is gated on warmingUp(), and so
is the checkpoint writer. That period is configured in SIM seconds -- 600 in the acceptance config,
which is ten real minutes at 1x and far longer on a loaded box.

The health check allowed a flat 300 wall-clock seconds and then declared the engine unhealthy. So a
cold-started simulation was killed several minutes before it could ever speak. restart_simulator
read that verdict as a failed resume, rewrote the pm2 entry to a bare `taosim -c latest`, and
started another -- which died the same way. And because no checkpoint exists until grace ends, none
of them was ever resumable, so every restart cold-started again.

Left alone it repeats: each restart cold-starts, dies at the same point and never advances, while
every log line reports the engine restarted successfully. It also means a deliberate cold start can
never survive, so nothing that needs a fresh simulation can have one.

Silence during grace is expected; stillness is not. So the question became whether the engine is
still writing its episode log.
"""

import os

import pytest

import taos.im.validator.update as U


class _Sim:
    def __init__(self, log_dir):
        self.logDir = str(log_dir)


class _V:
    def __init__(self, log_dir, started_at):
        self.simulation = _Sim(log_dir)
        self.start_time = started_at
        self.last_state_time = None


def _write(d, name, size):
    (d / name).write_bytes(b"x" * size)


class _AliveProc:
    info = {"name": "taosim"}


@pytest.fixture(autouse=True)
def _engine_present(monkeypatch):
    """Report the engine as running unless a test says otherwise.

    Without this these tests pass only on a box that happens to have taosim up, which makes them
    assert the box rather than the code.
    """
    monkeypatch.setattr(U.psutil, "process_iter", lambda attrs=None: iter([_AliveProc()]))


def _fresh_state():
    U._ENGINE_PROGRESS.clear()


def test_a_growing_episode_log_reads_as_healthy(tmp_path):
    _fresh_state()
    v = _V(tmp_path, 0)
    _write(tmp_path, "L2-0.log", 100)
    assert U._engine_log_is_growing(v) is True, "the first observation has nothing to compare and must not condemn"
    _write(tmp_path, "L2-1.log", 500)
    assert U._engine_log_is_growing(v) is True, "an engine still writing its episode is warming up, not wedged"


def test_a_static_episode_log_reads_as_unhealthy(tmp_path):
    _fresh_state()
    v = _V(tmp_path, 0)
    _write(tmp_path, "L2-0.log", 100)
    U._engine_log_is_growing(v)
    assert U._engine_log_is_growing(v) is False, "an engine that has stopped writing is the case this must still catch"


def test_a_new_episode_starts_its_own_measurement(tmp_path):
    """Otherwise a smaller new episode reads as shrinkage and is condemned at its first check."""
    _fresh_state()
    big = tmp_path / "old"
    big.mkdir()
    _write(big, "L2-0.log", 10_000)
    U._engine_log_is_growing(_V(big, 0))
    small = tmp_path / "new"
    small.mkdir()
    _write(small, "L2-0.log", 10)
    assert U._engine_log_is_growing(_V(small, 0)) is True, "a fresh episode must not inherit the previous one's size"


def test_an_unreadable_directory_never_condemns(tmp_path):
    _fresh_state()
    assert U._engine_log_is_growing(_V(tmp_path / "missing", 0)) is True
    v = _V(tmp_path, 0)
    v.simulation.logDir = None
    assert U._engine_log_is_growing(v) is True


def test_the_health_check_consults_progress_rather_than_the_clock_alone():
    """The 300s allowance must no longer be the final word when no state has arrived."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "taos", "im", "validator", "update.py")).read()
    body = src[src.index("def check_simulator"):]
    assert "_engine_log_is_growing(self)" in body, (
        "check_simulator still condemns on the wall clock alone, so a warming simulation dies at 300s"
    )


def test_the_restart_keeps_the_wrapper_and_the_kill_timeout():
    """pm2 re-runs whatever is registered here verbatim for the rest of the box's life.

    A bare `taosim -c latest` replaces the entry run_mvtrx.sh installed and takes two things with
    it: start_simulator.sh reads the one-shot cold-start marker, so nothing can ask for a new
    simulation once this has run, and without --kill-timeout pm2 SIGKILLs 1600ms after the signal,
    well before the engine can write its shutdown checkpoint -- turning every intended stop into a
    crash with nothing clean to resume from.
    """
    src = open(os.path.join(os.path.dirname(__file__), "..", "taos", "im", "validator", "update.py")).read()
    i = src.index("_run_dir = (self.repo_path")
    block = src[i:i + 1200]
    assert "start_simulator.sh" in block, "the restart bypasses the wrapper, so a cold start can never be honoured"
    assert block.count("--kill-timeout") == 2, "both branches must keep the kill timeout, or a stop is a crash"


def test_the_public_release_falls_back_to_the_binary():
    """start_simulator.sh is NOT in the public carve.

    Registering it unconditionally would make every restart on a deployed validator run a script
    that is not there, and the engine would never come back -- breaking the exact path this
    function exists to serve. The fallback is what those hosts had all along.
    """
    src = open(os.path.join(os.path.dirname(__file__), "..", "taos", "im", "validator", "update.py")).read()
    i = src.index("_run_dir = (self.repo_path")
    block = src[i:i + 1200]
    assert "start_simulator.sh').exists()" in block, "the wrapper is used without checking it is present"
    assert "taosim -c latest" in block, "there is no fallback for a host that ships no wrapper"


def test_a_dead_engine_is_unhealthy_immediately(tmp_path, monkeypatch):
    """The production recovery path must not be slowed by the warm-up allowance.

    Growth is the right question only for a process that is still there. A crashed or stopped
    engine writes nothing, and if absence were judged by growth it would cost a whole check
    interval to notice -- delaying the very restart this monitor exists to perform.
    """
    _fresh_state()
    monkeypatch.setattr(U.psutil, "process_iter", lambda attrs=None: iter([]))
    v = _V(tmp_path, 0)
    _write(tmp_path, "L2-0.log", 100)
    assert U._engine_log_is_growing(v) is False, "a stopped engine must be caught on the first check, not the second"


def test_a_live_engine_mid_grace_is_still_healthy(tmp_path, monkeypatch):
    _fresh_state()

    class _P:
        info = {"name": "taosim"}

    monkeypatch.setattr(U.psutil, "process_iter", lambda attrs=None: iter([_P()]))
    v = _V(tmp_path, 0)
    _write(tmp_path, "L2-0.log", 100)
    assert U._engine_log_is_growing(v) is True
