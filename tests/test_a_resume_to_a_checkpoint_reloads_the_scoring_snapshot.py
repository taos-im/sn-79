# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A resume to an engine checkpoint restarts the validator's scoring from the same instant.

An engine that is killed or crashes resumes from its newest periodic checkpoint T, behind the clock the validator
holds. Re-opening the run on the resumed clock shifts every history back by the regression and keeps the entries of
the stretch the engine abandoned, so the replay counts that stretch twice in every window, and it resets the
inventory, marks and open positions as for a new run, forgetting every position held at T.

The engine writes the checkpoint for T after publishing the state at T, before it publishes the next one, so the
validator reads the boundary on disk at the next state: on its tick it keeps the query-side windows, and at the start
of its reward, under the reward lock and before that state is accounted, every scoring structure. On a resume to T
the validator reloads that snapshot. These tests run the validator's own functions over a synthetic run: to T, past
T on a stretch the engine abandons, back to T, and the replay; the result equals a run that went straight from T.
"""
import asyncio
import os
import shutil
import threading
from collections import deque
from types import SimpleNamespace

import torch

import test_scoring_shadow as tss
import test_shadow_snapshot_completeness as tsc
import taos.im.validator.checkpoint_snapshot as cs
import taos.im.validator.engines.simulation as sim
from taos.im.validator.scoring_shadow import _STRUCT_NAMES, compute_parity_components
from taos.im.validator.trade import shift_simulation_histories, update_trade_volumes

_S = tss._S
RUN = "20261023_0900"
T = 6 * _S


def _main(tmp_path, run=RUN):
    m = tss._fresh_main()
    m.config.neuron = SimpleNamespace(full_path=str(tmp_path / "validator"))
    m.config.subtensor = SimpleNamespace(network="local")
    os.makedirs(m.config.neuron.full_path, exist_ok=True)
    m.simulation.logDir = str(tmp_path / "logs" / run)
    m.simulation.simulation_id = run
    m.simulation_timestamp = 0
    m.scores = torch.zeros(tss._UIDS, dtype=torch.float32)
    m.gentrx_scores = torch.zeros(tss._UIDS, dtype=torch.float32)
    m._gentrx_ema = {}
    m.device = "cpu"
    m.unnormalized_scores = {u: 0.0 for u in range(tss._UIDS)}
    m.hotkeys = [f"hk{u}" for u in range(tss._UIDS)]
    m.deregistered_uids = []
    m._trading_score_ema, m._trading_score_ema_n, m._trading_score_ema_ts = {}, {}, None
    m._debeta_pool_ema = {}
    m.miner_presence = {u: deque(maxlen=5) for u in range(tss._UIDS)}
    m.miner_stats = {u: {"requests": 0, "timeouts": 0, "failures": 0, "rejections": 0, "call_time": []}
                     for u in range(tss._UIDS)}
    return m


def _engine_checkpoint(m, ts):
    """The engine's checkpoint directory for `ts`, as CheckpointManager names it under the run directory."""
    path = os.path.join(m.simulation.logDir, "ckpt", f"{ts}.ckptd")
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, "common.bin"), "wb").close()
    return path


def _round(m, st, answered):
    """One state, in the validator's order: the tick (the boundary read, then this state's query), then the
    reward (the snapshot, then the accounting, then the scoring at an interval boundary)."""
    st.logDir = m.simulation.logDir
    cs.mark_if_checkpointed(m, st, m.simulation_timestamp, m.simulation.simulation_id)
    for uid in range(tss._UIDS):
        m.miner_presence[uid].append(answered(uid, st.timestamp))
        m.miner_stats[uid]["requests"] += 1
    m.simulation_timestamp = st.timestamp
    pending = asyncio.run(cs.take_due_snapshot(m, st))
    if pending is not None:
        pending.result(timeout=60)
    update_trade_volumes(m, st)
    if st.timestamp % m.config.scoring.interval == 0:
        for uid in range(tss._UIDS):
            vol = sum(m.volume_sums[uid].values())
            m._trading_score_ema[uid] = 0.5 * m._trading_score_ema.get(uid, 0.0) + 0.5 * vol
            m._trading_score_ema_n[uid] = m._trading_score_ema_n.get(uid, 0) + 1
            m.scores[uid] = 0.9 * m.scores[uid] + 0.1 * vol
            # the GenTRX EMA and its moving average step at every boundary on the service's current scores
            m._gentrx_ema[uid] = 0.1 * (uid + 1) + 0.9 * m._gentrx_ema.get(uid, 0.0)
            m.gentrx_scores[uid] = 0.9 * m.gentrx_scores[uid] + 0.1 * m._gentrx_ema[uid]
            m.unnormalized_scores[uid] = vol
        m._trading_score_ema_ts = st.timestamp


def _answered(uid, ts):
    return (uid + ts // _S) % 3 != 0


def _answered_on_replay(uid, ts):
    return (uid * 2 + ts // _S) % 4 != 0


def _replay_round(k):
    """The resumed engine's own state at k s: other fills than the abandoned stretch carried at the same clock."""
    fills = []
    for j in range(6):
        taker, maker = 1 + (j + k) % 3, 1 + (j + k + 1) % 3
        price = 100.0 + ((k * 5 + j) % 7) * 0.25
        fills.append((taker, maker, j % 2, 0.75 + j * 0.25, price if j % 2 == 0 else price * 0.55, (j + k) % 2))
    return tsc._fill_state(k * _S, fills)


BEFORE = tsc._debeta_rounds(1, 6)                     # 1..6 s, the checkpoint at 6 s
ABANDONED = tsc._debeta_rounds(7, 4)                  # 7..10 s, run by the engine and then lost with it
REPLAY = [_replay_round(k) for k in range(7, 13)]     # 7..12 s, the resumed engine's run


def _run_to_t(m):
    for st in BEFORE:
        _round(m, st, _answered)
        if st.timestamp == T:
            _engine_checkpoint(m, T)


def _view(m):
    """Everything the snapshot is for, in a comparable form."""
    out = {name: tsc._plain(getattr(m, name, {})) for name in _STRUCT_NAMES}
    out["_last_prune_timestamp"] = m._last_prune_timestamp
    out["_last_accounted_ts"] = m._last_accounted_ts
    out["scores"] = [round(float(x), 9) for x in m.scores.tolist()]
    out["gentrx_scores"] = [round(float(x), 9) for x in m.gentrx_scores.tolist()]
    for name in ("unnormalized_scores", "hotkeys", "deregistered_uids", "_trading_score_ema", "_trading_score_ema_n",
                 "_trading_score_ema_ts", "_debeta_pool_ema", "_gentrx_ema", "miner_stats"):
        out[name] = tsc._plain(getattr(m, name))
    out["miner_presence"] = {u: (list(d), d.maxlen) for u, d in m.miner_presence.items()}
    return out


def _straight(tmp_path):
    m = _main(tmp_path / "straight")
    _run_to_t(m)
    for st in REPLAY:
        _round(m, st, _answered_on_replay)
    return m


def _resume(m):
    """The first state after the resume steps back to T plus one step: the validator finds and reloads T."""
    first = REPLAY[0]
    snap = cs.find_snapshot(m, RUN, first.timestamp - m.simulation.publish_interval, m.simulation.logDir)
    assert snap is not None, "the snapshot the validator kept at the checkpoint is found and verified"
    cs.restore_snapshot(m, snap)
    m.simulation_timestamp = snap["meta"]["ts"]


def test_a_resume_reloads_the_snapshot_and_equals_a_run_straight_from_the_checkpoint(tmp_path):
    straight = _straight(tmp_path)

    broken = _main(tmp_path / "broken")
    _run_to_t(broken)
    assert any(p["longs"] or p["shorts"] for b in broken.open_positions.values() for p in b.values()), (
        "the fixture must hold positions at the checkpoint")
    for st in ABANDONED:
        _round(broken, st, _answered)
    assert _view(broken) != _view(straight)
    _resume(broken)
    for st in REPLAY:
        _round(broken, st, _answered_on_replay)

    mine, ref = _view(broken), _view(straight)
    for key in ref:
        assert mine[key] == ref[key], f"{key} differs from a run that went straight from the checkpoint"
    assert compute_parity_components(broken) == compute_parity_components(straight)
    assert broken.debeta_capbuy_hist and broken.debeta_mtm_hist, "the fixture must exercise the de-beta histories"


def test_re_opening_the_run_instead_counts_the_replayed_stretch_twice_and_forgets_the_positions(tmp_path):
    """What the reload replaces: the shift on the resumed clock keeps the abandoned entries and resets the
    positions held at the checkpoint."""
    straight = _straight(tmp_path)
    reopened = _main(tmp_path / "reopened")
    _run_to_t(reopened)
    for st in ABANDONED:
        _round(reopened, st, _answered)
    first = REPLAY[0].timestamp
    reopened.config.scoring.debeta = SimpleNamespace(making_fee_weight=0.0)   # the shift reads the fee weight
    shift_simulation_histories(
        reopened, reopened.simulation_timestamp, first, book_count=tss._BOOKS, volume_decimals=4,
        lookback=reopened.config.scoring.kappa.lookback,
        volume_assessment_period=reopened.config.scoring.activity.trade_volume_assessment_period,
        miner_wealth=50000.0, effective_max_uids=tss._UIDS)
    del reopened.config.scoring.debeta
    for st in REPLAY:
        _round(reopened, st, _answered_on_replay)
    a, b = _view(reopened), _view(straight)
    assert a["volume_sums"] != b["volume_sums"] and a["open_positions"] != b["open_positions"]
    assert sum(sum(v.values()) for v in reopened.volume_sums.values()) > \
        sum(sum(v.values()) for v in straight.volume_sums.values()), "the replayed stretch is counted twice"


def test_the_snapshot_is_taken_only_where_the_engine_checkpointed_and_before_the_next_state_is_accounted(tmp_path):
    m = _main(tmp_path)
    _run_to_t(m)
    _round(m, ABANDONED[0], _answered)
    names = sorted(os.listdir(cs.snapshot_dir(m)))
    assert names == [os.path.basename(cs.snapshot_path(m, RUN, T))], names
    snap = cs.load_snapshot(cs.snapshot_path(m, RUN, T))
    assert snap["meta"]["ts"] == T and snap["meta"]["last_accounted_ts"] == T
    assert [len(d) for d in snap["fields"]["miner_presence"].values()] == [5] * tss._UIDS
    assert snap["fields"]["miner_stats"][0]["requests"] == 6, "the query windows as they stood after T's query"


def test_a_snapshot_whose_engine_checkpoint_was_rewritten_is_not_used(tmp_path):
    m = _main(tmp_path)
    _run_to_t(m)
    _round(m, ABANDONED[0], _answered)
    ckpt = os.path.join(m.simulation.logDir, "ckpt", f"{T}.ckptd")
    shutil.rmtree(ckpt)
    _engine_checkpoint(m, T)
    os.utime(ckpt, ns=(1, 1))
    assert cs.find_snapshot(m, RUN, T, m.simulation.logDir) is None


def test_the_snapshots_follow_the_engines_retention(tmp_path):
    """The engine keeps its newest checkpoints (ckptNumLastFilesToKeep) and any it marked to keep; the validator
    keeps a snapshot exactly while the engine keeps its checkpoint, and never more than its own cap."""
    m = _main(tmp_path)
    kept = []
    for st in tsc._debeta_rounds(1, 13):
        _round(m, st, _answered)
        ts = st.timestamp
        if ts % (2 * _S) == 0 and ts <= 12 * _S:
            _engine_checkpoint(m, ts)
            kept.append(ts)
            if ts == 4 * _S:
                open(os.path.join(m.simulation.logDir, "ckpt", f"{ts}.ckptd", "keep"), "w").close()
            for old in [k for k in kept if k != 4 * _S][:-3]:
                shutil.rmtree(os.path.join(m.simulation.logDir, "ckpt", f"{old}.ckptd"))
                kept.remove(old)
    on_disk = sorted(int(n.split(".")[-2]) for n in os.listdir(cs.snapshot_dir(m)))
    engine = sorted(int(n.split(".")[0]) for n in os.listdir(os.path.join(m.simulation.logDir, "ckpt")))
    assert on_disk == engine == [4 * _S, 8 * _S, 10 * _S, 12 * _S], (on_disk, engine)
    assert len(on_disk) <= cs.SNAPSHOT_CAP


def test_the_snapshots_never_exceed_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SNAPSHOT_CAP", 2)
    m = _main(tmp_path)
    for st in tsc._debeta_rounds(1, 7):
        _round(m, st, _answered)
        if st.timestamp <= 6 * _S:
            _engine_checkpoint(m, st.timestamp)
    on_disk = sorted(int(n.split(".")[-2]) for n in os.listdir(cs.snapshot_dir(m)))
    assert on_disk == [5 * _S, 6 * _S], on_disk


def test_a_snapshot_of_another_run_is_removed_at_the_next_write(tmp_path):
    m = _main(tmp_path)
    _run_to_t(m)
    _round(m, ABANDONED[0], _answered)
    other = cs.snapshot_path(m, "20261001_0000", T)
    open(other, "wb").close()
    _engine_checkpoint(m, 8 * _S)
    for st in ABANDONED[1:3]:
        _round(m, st, _answered)
    assert not os.path.exists(other)


# --- the engine's tick routes a resume to the reload ---------------------------------------------------------------


def _loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def _engine(m):
    e = sim.SimulationEngine.__new__(sim.SimulationEngine)
    e.validator = m
    m.simulation.log_window = 3600 * _S
    m.update_repo = lambda end=False: None
    m.main_loop = _loop()
    m._reward_lock = asyncio.Lock()
    saved = []

    async def _save():
        saved.append(m.simulation_timestamp)
    m._save_state_sync = _save
    e._fresh_threshold = lambda: 30 * 60 * _S
    e._notify_seed_log_dir_change = lambda d: None
    e.compress_outputs = lambda start=False: None
    e._carry_fundamental = lambda state: None
    opened = []

    def _open_run(old_ts, new_ts, log_dir, *, reason):
        opened.append((old_ts, new_ts, reason))
        m.simulation_timestamp = new_ts
    e._open_run = _open_run
    return e, opened, saved


def _tick_round(e, m, st, answered):
    st.logDir = m.simulation.logDir
    e.on_tick(st)
    for uid in range(tss._UIDS):
        m.miner_presence[uid].append(answered(uid, st.timestamp))
        m.miner_stats[uid]["requests"] += 1
    pending = asyncio.run_coroutine_threadsafe(cs.take_due_snapshot(m, st), m.main_loop).result(timeout=60)
    if pending is not None:
        pending.result(timeout=60)
    update_trade_volumes(m, st)


def test_the_tick_reloads_the_snapshot_on_a_resume_instead_of_re_opening_the_run(tmp_path):
    straight = _main(tmp_path / "straight")
    es, _, _ = _engine(straight)
    for st in BEFORE:
        _tick_round(es, straight, st, _answered)
        if st.timestamp == T:
            _engine_checkpoint(straight, T)
    for st in REPLAY:
        _tick_round(es, straight, st, _answered_on_replay)

    m = _main(tmp_path / "broken")
    e, opened, saved = _engine(m)
    for st in BEFORE:
        _tick_round(e, m, st, _answered)
        if st.timestamp == T:
            _engine_checkpoint(m, T)
    for st in ABANDONED:
        _tick_round(e, m, st, _answered)
    for st in REPLAY:
        _tick_round(e, m, st, _answered_on_replay)

    assert opened == [], "a resume with a snapshot does not re-open the run"
    assert saved == [T], "the restored state is saved at the checkpoint's clock"
    assert m._resumes_restored == 1 and getattr(m, "_resumes_unrestored", 0) == 0
    assert m.simulation.simulation_id == RUN and m.simulation_timestamp == REPLAY[-1].timestamp
    mine, ref = _view(m), _view(straight)
    for key in ref:
        assert mine[key] == ref[key], f"{key} differs from a run that went straight from the checkpoint"


def test_a_checkpoint_inside_the_first_half_hour_is_resumed_not_read_as_a_new_run(tmp_path):
    """The near-zero rule reads any step back under thirty simulated minutes as a new run; on the same run with
    a snapshot at the checkpoint it is the resume it is."""
    m = _main(tmp_path)
    e, opened, _ = _engine(m)
    e._fresh_threshold = lambda: 3600 * _S
    for st in BEFORE:
        _tick_round(e, m, st, _answered)
        if st.timestamp == T:
            _engine_checkpoint(m, T)
    for st in ABANDONED:
        _tick_round(e, m, st, _answered)
    _tick_round(e, m, REPLAY[0], _answered_on_replay)
    assert opened == [] and m._resumes_restored == 1 and m.simulation.simulation_id == RUN


def test_the_scoring_service_is_re_initialised_from_the_restored_state_without_tripping_the_storm_breaker(tmp_path):
    m = _main(tmp_path)
    e, _, _ = _engine(m)
    calls = []
    m._scoring_shadow = SimpleNamespace(request_reinit=lambda **kw: calls.append(kw))
    m._shadow_applied_ts = 0
    for st in BEFORE:
        _tick_round(e, m, st, _answered)
        if st.timestamp == T:
            _engine_checkpoint(m, T)
    for st in ABANDONED:
        _tick_round(e, m, st, _answered)
    REPLAY[0].logDir = m.simulation.logDir
    e.on_tick(REPLAY[0])
    assert len(calls) == 1 and calls[0].get("storm") is False, calls
    assert m._shadow_applied_ts == T


def test_a_resume_reinit_counts_as_a_reinit_but_not_toward_the_storm():
    from taos.im.validator.scoring_shadow import ScoringShadow

    shadow = ScoringShadow.__new__(ScoringShadow)
    shadow.cutover_suspended = False
    shadow.reinits = 0
    shadow.initialized = True
    hits = []
    shadow._storm = SimpleNamespace(hit=lambda now: hits.append(now) or False, count=0, window=900.0)
    shadow._ring, shadow._score_ring, shadow._pending_child_parity = {}, {}, {}
    shadow._pending_child_scores, shadow._eager_inputs = {}, {7 * _S: {"x": 1}}
    shadow._submit_lock, shadow._last_teed_ts = threading.Lock(), 9 * _S
    sent = []
    shadow._executor = SimpleNamespace(submit=lambda fn: sent.append(fn))
    shadow.request_reinit(storm=False)
    assert shadow._init_floor_ts == 9 * _S, "the INIT after a resume waits for main like any other"
    assert shadow.reinits == 1 and not shadow.initialized and hits == [] and len(sent) == 1
    assert shadow._eager_inputs == {}, "a boundary shipped to the dropped replica is never answered by the new one"
    # a validator that restarts into a resume has no replica yet: the child still drops what it buffered, but
    # nothing was re-initialised, so the board reads the restart as a restart
    shadow.request_reinit(storm=False)
    assert shadow.reinits == 1 and len(sent) == 2


def test_a_checkpoint_the_resumed_engine_writes_again_is_snapshotted_again(tmp_path):
    """The replay reaches the instants of the abandoned stretch's checkpoints and the engine writes them again; the
    snapshot taken in the abandoned stretch does not stand in for the replay's."""
    m = _main(tmp_path)
    _run_to_t(m)
    _round(m, ABANDONED[0], _answered)                  # 7 s: the snapshot at 6 s
    _engine_checkpoint(m, 7 * _S)
    _round(m, ABANDONED[1], _answered)                  # 8 s: the abandoned snapshot at 7 s
    abandoned = cs.load_snapshot(cs.snapshot_path(m, RUN, 7 * _S))["meta"]["checkpoint"]
    _resume(m)
    _round(m, REPLAY[0], _answered_on_replay)           # 7 s, replayed; the engine writes 7 s again
    ckpt = os.path.join(m.simulation.logDir, "ckpt", f"{7 * _S}.ckptd")
    open(os.path.join(ckpt, "0.bin"), "wb").close()
    os.utime(ckpt, ns=(2, 2))
    _round(m, REPLAY[1], _answered_on_replay)           # 8 s: the replay's snapshot at 7 s
    snap = cs.find_snapshot(m, RUN, 7 * _S, m.simulation.logDir)
    assert snap is not None and snap["meta"]["checkpoint"] != abandoned
