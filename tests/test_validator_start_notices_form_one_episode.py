# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The multi-asset engine posts one SIMULATION_START per realization; the validator treats them as
one episode, and a later episode as a new one.

WHY THIS EXISTS. The orchestrator sends N start notices at the shared grid start, each naming its
realization's own log directory, in one HTTP batch. on_start absorbs the duplicates by timestamp
and normalises the realization directory to the run root. Both halves were in place before any
engine path relied on them; now one does. And the dedupe had a hole: the remembered timestamp was
never forgotten, so the NEXT episode, which also starts at 0, was absorbed as "a further
realization" and its restart shift, config reload and fundamental load never ran. on_end now
forgets it.

The engine is driven with its collaborators stubbed at the seams on_start itself crosses (config
reload, history shift, output compression, fundamental load, state save), so what is exercised is
exactly the episode logic and nothing behind it.
"""

import types

import pytest

pytest.importorskip("taos.im.validator.engines.simulation")

from taos.im.validator.engines import simulation as engine_mod  # noqa: E402
from taos.im.validator import trade as trade_mod  # noqa: E402

RUN_ROOT = "/runs/20260917_170000"
REALIZATIONS = [(0, "simulation_0", 0, None), (1, "highVol", 0, None), (2, "highVol", 1, None)]


def _engine(monkeypatch):
    """A SimulationEngine with a stub validator, its heavy collaborators replaced by recorders."""
    engine = engine_mod.SimulationEngine.__new__(engine_mod.SimulationEngine)
    calls = {"load_config": 0, "shift": 0, "compress": 0, "fundamental": 0, "update_repo": 0}

    sim = types.SimpleNamespace(
        backgrounds=[object()],
        realizations=lambda: REALIZATIONS,
        logDir=None,
        simulation_id=None,
        volumeDecimals=4,
        book_count=28,
        book_ids=[0, 1, 16, 17, 32, 33, 48, 49],
        miner_wealth=1_000_000.0,
        grace_period=600_000_000_000,
    )
    v = types.SimpleNamespace(
        simulation=sim,
        simulation_timestamp=0,
        start_time=None,
        start_timestamp=None,
        last_state_time=None,
        step_rates=[],
        effective_max_uids=4,
        config=types.SimpleNamespace(scoring=types.SimpleNamespace(
            kappa=types.SimpleNamespace(lookback=10),
            activity=types.SimpleNamespace(trade_volume_assessment_period=10),
        )),
        _scoring_shadow=None,
        _shadow_applied_ts=None,
        main_loop=None,
        fundamental_price={},
        pending_notices={},
        _save_state_sync=lambda: None,
        update_repo=lambda **_kw: calls.__setitem__("update_repo", calls["update_repo"] + 1),
    )
    engine.validator = v
    engine._book_ids = None

    def bump(key):
        def _f(*_a, **_kw):
            calls[key] += 1
        return _f

    engine._load_config = bump("load_config")
    engine.compress_outputs = bump("compress")
    engine.load_fundamental = bump("fundamental")
    engine._notify_seed_log_dir_change = lambda *_a, **_kw: None
    monkeypatch.setattr(trade_mod, "shift_simulation_histories", bump("shift"))
    monkeypatch.setattr(
        engine_mod.asyncio, "run_coroutine_threadsafe",
        lambda *_a, **_kw: types.SimpleNamespace(result=lambda *_x, **_y: None),
    )
    return engine, v, calls


def _start(log_dir):
    return types.SimpleNamespace(logDir=log_dir)


def test_three_realization_starts_at_one_timestamp_are_one_episode(monkeypatch):
    engine, v, calls = _engine(monkeypatch)
    for name, instance in (("simulation_0", 0), ("highVol", 0), ("highVol", 1)):
        engine.on_start(0, _start(f"{RUN_ROOT}/{name}-{instance}"))

    assert calls["load_config"] == 1 and calls["shift"] == 1, "the restart runs once per episode"
    assert calls["compress"] == 1 and calls["fundamental"] == 1
    assert v.simulation.logDir == RUN_ROOT, "a realization's directory normalises to the run root"
    assert v.start_timestamp == 0


def test_a_later_episode_at_the_same_timestamp_is_a_new_episode_after_the_end_notice(monkeypatch):
    """Every shipped config starts at 0. Without on_end forgetting the last start, the second
    episode a long-lived validator sees is absorbed as a duplicate and never reset."""
    engine, v, calls = _engine(monkeypatch)
    engine.on_start(0, _start(f"{RUN_ROOT}/simulation_0-0"))
    engine.on_start(0, _start(f"{RUN_ROOT}/highVol-0"))
    assert calls["shift"] == 1

    engine.on_end()
    assert calls["update_repo"] == 1
    assert v.simulation.logDir is None and v.simulation.simulation_id is None

    engine.on_start(0, _start("/runs/20260918_090000/simulation_0-0"))
    assert calls["shift"] == 2, "the next episode's start must run the restart shift again"
    assert calls["load_config"] == 2
    assert v.simulation.logDir == "/runs/20260918_090000"


def test_a_start_naming_the_run_root_itself_is_left_as_is(monkeypatch):
    """A single-config start names the run root directly; normalisation must not strip a level."""
    engine, v, _calls = _engine(monkeypatch)
    v.simulation.backgrounds = None                # single-market config: no realizations
    engine.on_start(0, _start(RUN_ROOT))
    assert v.simulation.logDir == RUN_ROOT
