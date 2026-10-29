# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The scoring child's lag is a validator gauge: the number of states main has teed beyond the newest one the child
has reported applying (its parity frames and its scores both name the state they were computed on). A child that
falls behind the stream is then seen on the board before a verify happens to land on it. The lag is counted on
timestamps, not on the child's applied counter, which runs on across re-INITs while the states a snapshot covers are
never applied."""
import socket
import threading
import time
from pathlib import Path

import bittensor as bt

from taos.im.validator.scoring_shadow import ScoringShadow, _send_frame

DEV = Path(__file__).resolve().parents[1]
REPORT = (DEV / "taos/im/validator/report.py").read_text()


class _Rec:
    def __call__(self, msg, *a, **k):
        pass


def _idle(monkeypatch):
    for level in ("info", "warning", "error"):
        monkeypatch.setattr(bt.logging, level, _Rec())
    sh = ScoringShadow(child_fn=lambda *a: None)
    monkeypatch.setattr(sh, "is_alive", lambda: True)
    return sh


def test_the_lag_counts_the_states_teed_beyond_the_childs_report(monkeypatch):
    sh = _idle(monkeypatch)
    assert sh.health()["child_lag"] == 0
    for ts in range(1, 6):
        sh.tee(b"x", ts)
    sh._child_applied(3)
    assert sh.health()["child_lag"] == 2
    sh._child_applied(5)
    assert sh.health()["child_lag"] == 0


def test_a_run_change_does_not_count_the_old_runs_states(monkeypatch):
    sh = _idle(monkeypatch)
    for ts in (900, 901):
        sh.tee(b"x", ts)
    sh.emit_sim_start(901, 0)
    sh.tee(b"x", 1)
    sh._child_applied(1)
    assert sh.health()["child_lag"] == 0


def _wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end and not pred():
        time.sleep(0.02)
    return pred()


def test_the_reader_takes_the_lag_from_parity_and_scores_frames(monkeypatch):
    sh = _idle(monkeypatch)
    sh._sock, peer = socket.socketpair()
    threading.Thread(target=sh._read_loop, daemon=True).start()
    try:
        for ts in range(1, 6):
            sh.tee(b"x", ts)
        _send_frame(peer, ("parity", (2, {}, {}, 2, 0.0)))
        assert _wait_for(lambda: sh.health()["child_lag"] == 3), sh.health()
        _send_frame(peer, ("scores", (5, {"trading": []}, 0.1)))
        assert _wait_for(lambda: sh.health()["child_lag"] == 0), sh.health()
    finally:
        peer.close()
        sh._sock.close()


def test_the_lag_reaches_the_validator_gauges():
    assert '("scoring_shadow_child_lag", "child_lag")' in REPORT
