# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A run change's history shift waits for a checkpoint snapshot that is being written.

The snapshot of the validator's state at an engine checkpoint is pickled under the reward lock, structure by
structure, at the start of the next state's reward. A resume the validator holds no snapshot for re-opens the run on
the thread handling the engine's events, and the shift there replaces the structures. Run while the pickle is half
way through, it would leave a snapshot whose first structures are the checkpoint's and whose last are the shifted
ones, and a later resume to that checkpoint would reload the mix. The shift takes the reward lock too, so it runs only
once the snapshot is written.
"""
import asyncio
import os
import threading
from types import SimpleNamespace

import test_scoring_shadow as tss
import test_shadow_snapshot_completeness as tsc
import test_a_resume_to_a_checkpoint_reloads_the_scoring_snapshot as trs
import taos.im.validator.checkpoint_snapshot as cs
import taos.im.validator.engines.simulation as sim
import taos.im.validator.trade as trade
from taos.im.validator.scoring_shadow import _STRUCT_NAMES, _rebuild_structs
from taos.im.validator.trade import update_trade_volumes

_S = tss._S
T = 6 * _S


def _engine(m):
    e = sim.SimulationEngine.__new__(sim.SimulationEngine)
    e.validator = m
    m.simulation.log_window = 3600 * _S
    m.update_repo = lambda end=False: None
    e._fresh_threshold = lambda: 0
    e._notify_seed_log_dir_change = lambda d: None
    e.compress_outputs = lambda start=False: None
    e._carry_fundamental = lambda state: None
    e._load_config = lambda: None
    e.load_fundamental = lambda: None

    async def _save():
        return None
    m._save_state_sync = _save
    return e


def test_the_shift_of_a_resume_without_a_snapshot_waits_for_the_snapshot_being_written(tmp_path, monkeypatch):
    m = trs._main(tmp_path)
    for st in tsc._debeta_rounds(1, 6):
        st.logDir = m.simulation.logDir
        update_trade_volumes(m, st)
    m.simulation_timestamp = T
    trs._engine_checkpoint(m, T)
    at_t = {name: tsc._plain(getattr(m, name, {})) for name in _STRUCT_NAMES}

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    m.main_loop = loop
    m._reward_lock = asyncio.Lock()
    m.config.scoring.debeta = SimpleNamespace(making_fee_weight=0.0)   # the shift reads the fee weight

    entered, gate = threading.Event(), threading.Event()

    class _Held(cs._PlainPickler):
        dumps = 0

        def dump(self, obj):
            _Held.dumps += 1
            if _Held.dumps == 4:            # past the first structures, before the rest
                entered.set()
                gate.wait(30)
            return super().dump(obj)
    monkeypatch.setattr(cs, "_PlainPickler", _Held)

    shifted = threading.Event()
    real_shift = trade.shift_simulation_histories

    def _shift(*a, **k):
        shifted.set()
        return real_shift(*a, **k)
    monkeypatch.setattr(trade, "shift_simulation_histories", _shift)

    nxt = SimpleNamespace(timestamp=T + _S, logDir=m.simulation.logDir)
    assert cs.mark_if_checkpointed(m, nxt, T, trs.RUN)
    m.simulation_timestamp = T + _S

    async def _reward():
        async with m._reward_lock:
            return await cs.take_due_snapshot(m, nxt)
    writing = asyncio.run_coroutine_threadsafe(_reward(), loop)
    assert entered.wait(30), "the snapshot write reached the held record"

    e = _engine(m)
    back = SimpleNamespace(timestamp=3 * _S, logDir=m.simulation.logDir)
    tick = threading.Thread(target=e.on_tick, args=(back,))
    tick.start()
    started_while_held = shifted.wait(1.0)
    gate.set()
    writing.result(timeout=60).result(timeout=60)
    tick.join(60)
    assert not tick.is_alive()
    assert not started_while_held, "the shift ran while the snapshot was half written"
    assert shifted.is_set() and getattr(m, "_resumes_unrestored", 0) == 1

    snap = cs.load_snapshot(cs.snapshot_path(m, trs.RUN, T))
    rebuilt = _rebuild_structs(snap["parts"])
    for name in _STRUCT_NAMES:
        assert tsc._plain(rebuilt[name]) == at_t[name], f"{name} in the snapshot is not the checkpoint's"
    loop.call_soon_threadsafe(loop.stop)
    assert os.path.exists(cs.snapshot_path(m, trs.RUN, T))
