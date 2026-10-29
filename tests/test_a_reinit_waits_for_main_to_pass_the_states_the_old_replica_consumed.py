# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A re-INIT must not leave the fresh replica short of a state.

The listener tees each state to the child the moment it arrives, ahead of main, whose rounds queue behind the reward
lock. When main requests a re-INIT, the states already teed reach the child before the reinit frame and are applied
to the replica it then drops. The INIT snapshot carries main's state as of the round it is taken in, so a state the
old replica consumed that main had not yet applied is in neither the snapshot nor the child's pre-INIT buffer: the
fresh replica misses it, the next parity points read a MISMATCH on every structure (the per-uid P&L counts short by
exactly the missing states), and a second re-INIT follows. The INIT now waits until main has applied the last state
teed before the reinit frame, bounded so that a clock that never reaches it (a run change) still re-initialises."""
import time
from types import SimpleNamespace

import bittensor as bt

import taos.im.validator.scoring_shadow as ss
from taos.im.validator.scoring_shadow import ScoringShadow, compute_parity_components, parity_vectors
from taos.im.validator.trade import update_trade_volumes

from test_scoring_shadow import _S, _fresh_main, _raw_state, _state

# no scoring boundary (5 s) among them, so the child never pauses for a score
_TRADES = {
    6: [(1, 2, 0, 3.0, 100.0, 0)],
    7: [(2, 1, 0, 1.5, 101.0, 1)],
    8: [(3, 1, 1, 2.0, 55.0, 0)],
    9: [(1, 3, 1, 2.0, 56.0, 1)],
    11: [(2, 3, 0, 1.0, 99.0, 0)],
    12: [(3, 2, 1, 1.0, 57.0, 1)],
}


def _main_round(shadow, main, ts):
    """Main's round: apply the state, then, while the child is not initialized, the INIT branch of _reward."""
    update_trade_volumes(main, _state(ts * _S, _TRADES[ts]))
    main._shadow_applied_ts = ts * _S
    if not shadow.initialized:
        shadow.send_init(main).result(timeout=60)


def test_a_reinit_with_the_listener_ahead_restores_parity():
    main = _fresh_main()
    shadow = ScoringShadow(parity_ns=12 * _S)
    shadow.start()
    try:
        _main_round(shadow, main, 6)
        # the listener has teed 7, 8 and 9 while main is still on 7
        for ts in (7, 8, 9):
            shadow.tee(_raw_state(ts * _S, _TRADES[ts]), ts * _S)
        _main_round(shadow, main, 7)
        shadow.request_reinit()  # a VERIFY mismatch in main's round at 7
        for ts in (8, 9):
            _main_round(shadow, main, ts)
        for ts in (11, 12):
            shadow.tee(_raw_state(ts * _S, _TRADES[ts]), ts * _S)
            _main_round(shadow, main, ts)
        shadow.record_main_digest(12 * _S, (compute_parity_components(main), parity_vectors(main)))
        end = time.time() + 60
        while time.time() < end and shadow.matches + shadow.mismatches == 0:
            time.sleep(0.1)
        assert (shadow.matches, shadow.mismatches) == (1, 0), (
            f"parity after the re-INIT: {shadow.matches} match, {shadow.mismatches} mismatch")
    finally:
        shadow.stop()


class _Rec:
    def __init__(self):
        self.msgs = []

    def __call__(self, msg, *a, **k):
        self.msgs.append(str(msg))


def _idle(monkeypatch):
    for level in ("info", "warning", "error"):
        monkeypatch.setattr(bt.logging, level, _Rec())
    sh = ScoringShadow(child_fn=lambda *a: None)
    monkeypatch.setattr(sh, "is_alive", lambda: True)
    return sh


def test_the_floor_is_the_last_state_teed_before_the_reinit_frame(monkeypatch):
    sh = _idle(monkeypatch)
    for ts in (7, 8, 9):
        sh.tee(b"x", ts)
    sh.request_reinit()
    sh.tee(b"x", 10)
    assert sh.init_waits(8) and sh.init_waits(8), "main has not applied 9 yet"
    assert not sh.init_waits(9), "main has applied every state the dropped replica consumed"
    assert not sh.init_waits(8), "the floor is spent once the INIT may run"


def test_an_init_that_waits_sends_nothing_and_stays_uninitialized(monkeypatch):
    sh = _idle(monkeypatch)
    sh.tee(b"x", 9)
    sh.request_reinit()
    fut = sh.send_init(SimpleNamespace(_shadow_applied_ts=8))
    assert fut.result(timeout=1) is None and not sh.initialized


def test_the_wait_is_bounded(monkeypatch):
    monkeypatch.setattr(ss, "SHADOW_REINIT_WAIT_MAX", 3)
    sh = _idle(monkeypatch)
    sh.tee(b"x", 900)
    sh.request_reinit()
    assert [sh.init_waits(5) for _ in range(4)] == [True, True, True, False], (
        "a clock that never reaches the floor re-initialises after the bound")


def test_a_run_change_or_a_fresh_child_clears_the_floor(monkeypatch):
    sh = _idle(monkeypatch)
    sh.tee(b"x", 900)
    sh.request_reinit()
    sh.emit_sim_start(900, 0)
    assert not sh.init_waits(1), "the new run's clock starts below the old run's floor"
    sh.tee(b"x", 900)
    sh.request_reinit()
    monkeypatch.setattr(sh._ctx, "Process", lambda **k: SimpleNamespace(
        start=lambda: None, join=lambda timeout=None: None, is_alive=lambda: False, kill=lambda: None))
    sh.start()
    assert not sh.init_waits(1), "a fresh child has consumed no state"
    sh.stop()


def test_the_first_init_does_not_wait(monkeypatch):
    sh = _idle(monkeypatch)
    sh.tee(b"x", 9)
    assert not sh.init_waits(1)
