# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator's state is saved once a simulated minute by default (every 12th scoring interval at 5 s).

The save is written by the scoring service while the main process waits under the reward lock, so it is taken once a
simulated minute. A crash loses at most the minute since the last save, out
of a three-hour lookback; an engine resume reads the scoring snapshot kept at its checkpoint, not this file.
SAVE_EVERY_INTERVALS still overrides it."""
import importlib
import os
from types import SimpleNamespace

from taos.im.validator import persistence


def _reloaded(env):
    old = os.environ.pop("SAVE_EVERY_INTERVALS", None)
    try:
        if env is not None:
            os.environ["SAVE_EVERY_INTERVALS"] = env
        return importlib.reload(persistence)._SAVE_EVERY_INTERVALS
    finally:
        os.environ.pop("SAVE_EVERY_INTERVALS", None)
        if old is not None:
            os.environ["SAVE_EVERY_INTERVALS"] = old
        importlib.reload(persistence)


def test_the_default_is_every_twelfth_interval_and_the_variable_overrides_it():
    assert _reloaded(None) == 12
    assert _reloaded("3") == 3


def test_a_save_is_scheduled_once_a_simulated_minute_at_the_five_second_interval(monkeypatch):
    monkeypatch.setattr(persistence, "_SAVE_EVERY_INTERVALS", 12)
    # schedule_save's debug line asks for the current loop, which an earlier test in the suite may have unset
    monkeypatch.setattr(persistence.asyncio, "get_event_loop", lambda: None)
    calls = []
    loop = SimpleNamespace(call_soon_threadsafe=lambda f: calls.append(1), create_task=lambda c: None)
    S = 10 ** 9
    fired = []
    for t in range(0, 600 * S, S):
        v = SimpleNamespace(last_state=SimpleNamespace(timestamp=t), config=SimpleNamespace(
            scoring=SimpleNamespace(interval=5 * S)), shared_state_saving=False, querying=False, main_loop=loop,
            _cleanup_done=False, step=t // S)
        before = len(calls)
        persistence.schedule_save(v)
        if len(calls) > before:
            fired.append(t // S)
    assert fired == [4 + 60 * k for k in range(10)]
