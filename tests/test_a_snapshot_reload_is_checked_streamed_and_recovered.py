# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The reload of a checkpoint snapshot is checked first, streamed one structure at a time, and recovered whole.

Read whole, a mainnet snapshot sat in memory beside the state it was about to replace, so a resume peaked at about
twice the state. Now the snapshot is checked from its meta record alone (run, checkpoint clock, the clock it was taken
at, uid count, checkpoint identity) before anything of the validator is touched, and then reloaded structure by
structure, each old one dropped before the next is read. A record that cannot be read partway would leave some
structures at the checkpoint and the rest after it; the validator then reloads its saved state files through its
normal loader and re-opens the run on the resumed clock, as it does without a snapshot, counted and logged.
"""
import asyncio
import logging
import os
import pickle
import re
import shutil
import threading
import weakref
from types import SimpleNamespace

import test_scoring_shadow as tss
import test_shadow_snapshot_completeness as tsc
import test_a_resume_to_a_checkpoint_reloads_the_scoring_snapshot as trs
import taos.im.validator.checkpoint_snapshot as cs
import taos.im.validator.engines.simulation as sim
from taos.im.validator import persistence
from taos.im.validator.scoring_shadow import _STRUCT_NAMES, _rebuild_structs
from taos.im.validator.trade import shift_simulation_histories

_S = tss._S
T = trs.T


def _validator(tmp_path, files=None):
    """The test validator with what the engine's initialiser and the state loader read, and its engine."""
    m = trs._main(tmp_path)
    e = sim.SimulationEngine.__new__(sim.SimulationEngine)
    e.validator = m
    e._book_ids = list(range(tss._BOOKS))
    e._sim_config = m.simulation
    m.engine = e
    m.subnet_info = SimpleNamespace(max_uids=tss._UIDS)
    m.benchmark_agents = []
    m.config.neuron.reset = False
    m.simulation_state_file, m.validator_state_file = files or (str(tmp_path / "simulation.mp"),
                                                               str(tmp_path / "validator.mp"))
    m.start_time, m.start_timestamp, m.step_rates = 0.0, 0, []
    m.pending_notices = {u: [] for u in range(tss._UIDS)}
    m.simulation.log_window = 3600 * _S
    m.update_repo = lambda end=False: None
    e._fresh_threshold = lambda: 0
    e._notify_seed_log_dir_change = lambda d: None
    e.compress_outputs = lambda start=False: None
    e._carry_fundamental = lambda state: None
    e._load_config = lambda: None
    e.load_fundamental = lambda: None
    return m, e


def _abandoned_run(m):
    """To T, the snapshot at T taken on the next state, and the stretch the engine abandons; then the validator's
    periodic save at the end of that stretch."""
    trs._run_to_t(m)
    for st in trs.ABANDONED:
        trs._round(m, st, trs._answered)
    asyncio.run(persistence.save_state_sync(m))


def _with_loop(m):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    m.main_loop = loop
    m._reward_lock = asyncio.Lock()

    async def _save():
        return None
    m._save_state_sync = _save
    m.config.scoring.debeta = SimpleNamespace(making_fee_weight=0.0)   # the shift reads the fee weight
    return loop


def _record_offsets(path):
    with open(path, "rb") as f:
        out = []
        while True:
            at = f.tell()
            try:
                rec = pickle.load(f)
            except EOFError:
                return out
            out.append((at, rec[0], rec[1] if rec[0] in ("struct", "whole") else None))


def _state_view(m):
    out = {name: tsc._plain(getattr(m, name, {})) for name in _STRUCT_NAMES}
    out["clocks"] = (m._last_accounted_ts, m._last_prune_timestamp, m.simulation_timestamp)
    out["scores"] = [round(float(x), 9) for x in m.scores.tolist()]
    for name in ("_trading_score_ema", "_trading_score_ema_n", "_trading_score_ema_ts", "hotkeys", "miner_stats"):
        out[name] = tsc._plain(getattr(m, name))
    return out


def test_the_check_reads_the_meta_record_alone(tmp_path):
    m, _ = _validator(tmp_path)
    _abandoned_run(m)
    path = cs.snapshot_path(m, trs.RUN, T)
    first = _record_offsets(path)[1][0]
    with open(path, "rb+") as f:
        f.truncate(first)
    checked = cs.check_snapshot(m, trs.RUN, T, m.simulation.logDir)
    assert checked is not None and checked["meta"]["ts"] == T and checked["path"] == path


def test_a_snapshot_that_fails_its_check_leaves_every_structure_untouched(tmp_path, monkeypatch):
    m, e = _validator(tmp_path)
    _abandoned_run(m)
    os.utime(os.path.join(m.simulation.logDir, "ckpt", f"{T}.ckptd"), ns=(3, 3))
    before = {name: getattr(m, name, None) for name in _STRUCT_NAMES}
    monkeypatch.setattr(cs, "restore_snapshot_from", lambda v, p: (_ for _ in ()).throw(AssertionError("read")))
    reopened = []
    e._open_run = lambda old, new, log_dir, *, reason: reopened.append((old, new))
    st = trs.REPLAY[0]
    st.logDir = m.simulation.logDir
    e.on_tick(st)
    assert reopened and m._resumes_unrestored == 1
    replaced = [name for name, obj in before.items() if getattr(m, name, None) is not obj]
    assert not replaced, f"replaced before the check: {replaced}"


def test_each_old_structure_is_dropped_before_the_next_is_read(tmp_path, monkeypatch):
    """The peak of a reload is the state plus the largest structure: when a structure is rebuilt, every one swapped
    before it has already let go of the structure it replaced."""
    m, _ = _validator(tmp_path)
    _abandoned_run(m)

    class _Held(dict):
        pass
    old = {}
    for name in _STRUCT_NAMES:
        held = _Held(getattr(m, name, None) or {})
        setattr(m, name, held)
        old[name] = weakref.ref(held)
        del held
    order, alive_at = [], {}
    real = cs._rebuild_structs

    def _watch(parts):
        (name,) = parts
        alive_at[name] = [n for n in order if old[n]() is not None]
        order.append(name)
        return real(parts)
    monkeypatch.setattr(cs, "_rebuild_structs", _watch)
    assert cs.restore_snapshot_from(m, cs.snapshot_path(m, trs.RUN, T)) == len(_STRUCT_NAMES)
    assert order == list(_STRUCT_NAMES)
    assert all(not alive for alive in alive_at.values()), {k: v for k, v in alive_at.items() if v}
    snap = cs.load_snapshot(cs.snapshot_path(m, trs.RUN, T))
    whole = _rebuild_structs(snap["parts"])
    for name in _STRUCT_NAMES:
        assert tsc._plain(getattr(m, name)) == tsc._plain(whole[name]), f"{name} differs from the whole-file reload"


def test_a_reload_that_fails_partway_is_replaced_by_the_saved_state(tmp_path, caplog):
    m, e = _validator(tmp_path / "v")
    _abandoned_run(m)
    saved_at = m.simulation_timestamp
    path = cs.snapshot_path(m, trs.RUN, T)
    structs = [(at, name) for at, kind, name in _record_offsets(path) if kind == "struct"]
    at, broken = structs[len(structs) // 2]
    with open(path, "rb+") as f:
        f.seek(at)
        f.write(b"\xff")
    loop = _with_loop(m)
    st = trs.REPLAY[0]
    st.logDir = m.simulation.logDir
    with caplog.at_level(logging.WARNING):
        e.on_tick(st)
    loop.call_soon_threadsafe(loop.stop)
    assert m._resumes_unrestored == 1 and getattr(m, "_resumes_restored", 0) == 0 and m._run_changes_recovered == 1
    line = next(r.getMessage() for r in caplog.records if "RESUME NOT RESTORED" in r.getMessage())
    swapped = int(re.search(r"after (\d+) of", line).group(1))
    # the structure before the broken record was read whole but not yet swapped
    assert 0 < swapped < len(_STRUCT_NAMES) and _STRUCT_NAMES.index(broken) == swapped + 1, line

    # the reference: a validator that never saw the snapshot, loaded from the same files and shifted the same way
    ref, eref = _validator(tmp_path / "ref", files=(m.simulation_state_file, m.validator_state_file))
    fresh = _rebuild_structs({})
    for name in _STRUCT_NAMES:
        setattr(ref, name, fresh.get(name, {}))
    ref._last_accounted_ts = ref._last_prune_timestamp = None
    eref.initialize_structures()
    persistence.load_state(ref)
    assert ref.simulation_timestamp == saved_at
    ref.config.scoring.debeta = SimpleNamespace(making_fee_weight=0.0)
    shift_simulation_histories(
        ref, saved_at, st.timestamp, book_count=tss._BOOKS, volume_decimals=4,
        lookback=ref.config.scoring.kappa.lookback,
        volume_assessment_period=ref.config.scoring.activity.trade_volume_assessment_period,
        miner_wealth=ref.simulation.miner_wealth, effective_max_uids=tss._UIDS)
    ref.simulation_timestamp = st.timestamp
    mine, theirs = _state_view(m), _state_view(ref)
    for key in theirs:
        assert mine[key] == theirs[key], f"{key}: the recovered state is not the saved state's"
    shutil.rmtree(os.path.dirname(path))
