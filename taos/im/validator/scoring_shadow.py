# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Shadow scoring service — Q3 Stage A (parity-gated groundwork for the split).

Goal: move all history-consuming background work (volume/PnL accounting, reward,
save, report prep) out of the validator main process. Before any cutover, this
shadow proves the replication property: a subprocess fed (a) one snapshot of the
history structures and (b) the same raw state bytes per round maintains
BIT-IDENTICAL structures by running the very same trade.update_trade_volumes.

Mechanics:
- Main tees each round's raw msgpack state bytes over a socketpair (sent on a
  single-thread executor: FIFO framing, ~10ms GIL-released syscalls off-loop).
- One-time INIT under _reward_lock: each history structure is plainified
  (lambda-factory defaultdicts are unpicklable), pickled and streamed; the child
  rebuilds the exact defaultdict/deque shells and applies buffered rounds newer
  than the snapshot.
- Both sides compute a cheap parity digest at deterministic sim-timestamps
  (counts + running totals; ~10ms); the child ships its digest back and main
  logs [SHADOW-PARITY] MATCH/MISMATCH against its own ring.

Default OFF (SCORING_SHADOW=1 or a repo-root .scoring_shadow sentinel enables);
zero effect on scoring — the child's structures are observed, never consumed.
"""
import hashlib
import multiprocessing
import os
import pickle
import struct
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHADOW_PARITY_NS = int(os.environ.get("SHADOW_PARITY_NS", "10000000000"))

_STRUCT_NAMES = [
    "trade_volumes", "volume_sums", "maker_volume_sums", "taker_volume_sums",
    "self_volume_sums", "fee_sums", "roundtrip_volumes", "roundtrip_volume_sums",
    "realized_pnl_history", "agent_pnl_by_book", "agent_pnl_total",
    "open_positions", "inventory_history", "initial_balances",
    "recent_trades", "recent_miner_trades",
    # Shadow-reward inputs: mutated in place by score_uids each interval, so
    # once initialized the child's own scoring keeps them in lockstep with main.
    "kappa_values", "activity_factors", "pnl_factors",
]


def shadow_enabled() -> bool:
    if os.environ.get("SCORING_SHADOW", "0") == "1":
        return True
    try:
        repo_root = Path(__file__).resolve().parents[3]
        return (repo_root / ".scoring_shadow").exists()
    except Exception:
        return False


def cutover_enabled() -> bool:
    """Cutover mode: the scoring service computes rewards authoritatively and
    main adopts them (with per-interval fallback and periodic verification).

    DEFAULT ON (validated live: full score-vector VERIFY, history parity through
    dereg resets, automatic in-process fallback). Disable with SCORING_PROC=0 or
    a repo-root .no_scoring_proc sentinel — the quick rollback switches. The
    legacy .scoring_proc opt-in sentinel is now redundant (harmless if present)."""
    env = os.environ.get("SCORING_PROC")
    if env == "0":
        return False
    if env == "1":
        return True
    try:
        repo_root = Path(__file__).resolve().parents[3]
        return not (repo_root / ".no_scoring_proc").exists()
    except Exception:
        return True


def _plainify(obj):
    """Recursively convert dict-likes to plain dicts so lambda-factory
    defaultdicts pickle; deques, tuples and model objects pass through
    (they pickle natively). Values are never copied deeper than needed."""
    if isinstance(obj, dict):
        return {k: _plainify(v) for k, v in obj.items()}
    if isinstance(obj, deque):
        return deque(_plainify(v) for v in obj)
    if isinstance(obj, list):
        return [_plainify(v) for v in obj]
    return obj


def _rebuild_structs(parts: dict) -> dict:
    """Rebuild the exact container shells main uses, so missing-key behaviour
    (load-bearing defaultdict auto-creation) is identical in the shadow."""
    out = {}

    def dd_float_2(plain):
        d = defaultdict(lambda: defaultdict(float))
        for uid, books in plain.items():
            inner = defaultdict(float)
            inner.update(books)
            d[uid] = inner
        return d

    for name in ("volume_sums", "maker_volume_sums", "taker_volume_sums",
                 "self_volume_sums", "fee_sums", "roundtrip_volume_sums",
                 "agent_pnl_by_book"):
        out[name] = dd_float_2(parts.get(name, {}))

    apt = defaultdict(float)
    apt.update(parts.get("agent_pnl_total", {}))
    out["agent_pnl_total"] = apt

    rtv = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for uid, books in parts.get("roundtrip_volumes", {}).items():
        for book_id, vols in books.items():
            inner = defaultdict(float)
            inner.update(vols)
            rtv[uid][book_id] = inner
    out["roundtrip_volumes"] = rtv

    rpn = defaultdict(lambda: defaultdict(dict))
    for uid, hist in parts.get("realized_pnl_history", {}).items():
        inner = defaultdict(dict)
        inner.update(hist)
        rpn[uid] = inner
    out["realized_pnl_history"] = rpn

    pos = defaultdict(lambda: defaultdict(lambda: {'longs': deque(), 'shorts': deque()}))
    for uid, books in parts.get("open_positions", {}).items():
        for book_id, p in books.items():
            pos[uid][book_id] = {
                'longs': deque(p.get('longs', ())),
                'shorts': deque(p.get('shorts', ())),
            }
    out["open_positions"] = pos

    for name in ("trade_volumes", "inventory_history", "initial_balances",
                 "recent_trades", "recent_miner_trades",
                 "kappa_values", "activity_factors", "pnl_factors"):
        out[name] = parts.get(name, {})

    return out


def compute_parity_components(s) -> dict:
    """Per-structure digests of the history state (duck-typed: works on the
    Validator and the shadow container alike). Sorted keys + sorted-value sums
    make each insertion-order independent; exact float repr keeps them
    bit-sensitive to real divergence. Kept per-field so a mismatch names the
    diverging structure instead of just failing a combined hash."""
    def sum_books(d):
        return tuple(
            (int(u), float(sum(sorted(float(x) for x in b.values()))))
            for u, b in sorted(d.items())
        )

    def h(v):
        return hashlib.md5(repr(v).encode()).hexdigest()[:12]

    return {
        "n_pnl": h(sum(len(hh) for hh in s.realized_pnl_history.values())),
        "n_tv": h(sum(len(r.get('total', {})) for b in s.trade_volumes.values() for r in b.values())),
        "n_rt": h(sum(len(v) for b in s.roundtrip_volumes.values() for v in b.values())),
        "n_pos": h(sum(len(p['longs']) + len(p['shorts'])
                       for b in s.open_positions.values() for p in b.values())),
        "pnl_total": h(tuple(sorted((int(u), float(v)) for u, v in s.agent_pnl_total.items()))),
        "pnl_book": h(sum_books(s.agent_pnl_by_book)),
        "vol": h(sum_books(s.volume_sums)),
        "rt": h(sum_books(s.roundtrip_volume_sums)),
        "prune": h(getattr(s, "_last_prune_timestamp", None)),
    }


def pnl_len_vector(s) -> dict:
    """{uid: len(realized_pnl_history[uid])} for non-empty hists — the uid-level
    breakdown behind the n_pnl component, shipped with parity frames so an
    n_pnl mismatch names WHICH uids diverge (and by how much)."""
    return {int(u): len(h) for u, h in s.realized_pnl_history.items() if h}


def compute_parity_digest(s) -> str:
    """Combined digest over the per-structure components."""
    comps = compute_parity_components(s)
    return hashlib.md5(repr(sorted(comps.items())).encode()).hexdigest()


class ShadowState:
    """Attribute container exposing exactly the surface trade.update_trade_volumes
    (and, for shadow-reward, score_uids) touches, so the shadow runs the SAME
    functions on the SAME shapes."""

    def __init__(self, structs: dict, knobs: dict):
        for name, value in structs.items():
            setattr(self, name, value)
        self._last_prune_timestamp = knobs.get("_last_prune_timestamp")
        self.step = knobs.get("step", 0)
        self.effective_max_uids = knobs["effective_max_uids"]
        self.config = knobs["config"]
        self.simulation = knobs["simulation"]
        # Shadow-reward inputs (see shadow_score). kappa_cache stays empty —
        # the fingerprint cache is off by default and score_uids gates on it.
        self.scoring_config = knobs.get("scoring_config")
        self.simulation_config = knobs.get("simulation_config")
        self.scoring_interval = knobs.get("scoring_interval", 0)
        self.validator_uid = knobs.get("validator_uid")
        self.kappa_cache = {}
        self.deregistered_uids = []

    def pagerduty_alert(self, message, details=None):
        print(f"[SHADOW] pagerduty (suppressed): {message}", flush=True)


def shadow_score(shadow: ShadowState, sim_ts: int, deregs: list,
                 gentrx_scores=None, gentrx_ema=None,
                 trading_ema=None, trading_ema_n=None, trading_ema_ts=None) -> dict:
    """Run the SAME scoring main runs (score_uids + Pareto distribute) on the
    shadow's structures. Mutates kappa/activity/pnl factors in place exactly as
    main's reward does, keeping the shadow in lockstep.

    GenTRX inputs are main-side state: shipped in via score_at (cutover mode)
    or empty (shadow mode — the trading comparison is unaffected either way;
    the EMA state round-trips through main so the child stays stateless).

    Returns {'trading', 'gentrx', 'factors', 'gentrx_ema'} — everything main
    adopts in cutover mode.
    """
    from taos.im.validator.reward import (apply_reward_floor, apply_track_record_ema,
                                          distribute_rewards, score_uids)

    shadow.deregistered_uids = list(deregs)
    all_uids = list(range(shadow.effective_max_uids))
    _ema = dict(gentrx_ema or {})
    validator_data = {
        'kappa_values': shadow.kappa_values,
        'kappa_cache': shadow.kappa_cache,
        'activity_factors': shadow.activity_factors,
        'pnl_factors': shadow.pnl_factors,
        'roundtrip_volumes': shadow.roundtrip_volumes,
        'realized_pnl_history': shadow.realized_pnl_history,
        'config': shadow.scoring_config,
        'simulation_config': shadow.simulation_config,
        'simulation_timestamp': sim_ts,
        'uids': all_uids,
        'deregistered_uids': shadow.deregistered_uids,
        'device': 'cpu',
        'gentrx_scores': dict(gentrx_scores or {}),
        'gentrx_ema': _ema,
    }
    trading_scores, gentrx_scores_out = score_uids(validator_data)
    # Track-record EMA: same pre-round state main ships (parity contract with
    # reward.get_rewards); the child stays stateless — the state round-trips.
    _tema = dict(trading_ema or {})
    _tema_n = dict(trading_ema_n or {})
    _tema_ts = trading_ema_ts
    _hl = (shadow.scoring_config.get('scoring', {}) or {}).get('score_ema_halflife', 0)
    if _hl and _hl > 0:
        trading_scores, _tema_ts = apply_track_record_ema(
            trading_scores, all_uids, shadow.deregistered_uids,
            sim_ts, _hl, _tema, _tema_n, _tema_ts)
    # Identical allocation pipeline to main's get_rewards: soft floor THEN Pareto.
    # (The floor was missing here — a pre-existing divergence whenever
    # rewarding.floor.enabled is on; a no-op when it is off.)
    floored = apply_reward_floor(
        [trading_scores[uid] for uid in all_uids], shadow.scoring_config
    )
    distributed = distribute_rewards(floored, shadow.scoring_config)
    return {
        'trading': [float(x) for x in distributed.tolist()],
        'gentrx': [float(gentrx_scores_out[uid]) for uid in all_uids],
        'factors': {
            'kappa_values': shadow.kappa_values,
            'activity_factors': shadow.activity_factors,
            'pnl_factors': shadow.pnl_factors,
        },
        'gentrx_ema': validator_data.get('gentrx_ema', {}),
        'trading_ema': _tema,
        'trading_ema_n': _tema_n,
        'trading_ema_ts': _tema_ts,
    }


def child_save_validator_state(shadow, light: dict, path: str) -> int:
    """Write the validator-state file from the child's replica structures plus
    main's shipped light fields — the exact dict persistence.build_validator_state
    produces, stream-packed at the same depth and atomically replaced, so the
    file is indistinguishable from a main-side save. The ongoing parity digests
    are the proof the heavy subtrees match main's. Returns bytes written."""
    import msgpack
    from taos.im.validator.persistence import (
        _SAVE_STREAM_DEPTH, _stream_pack,
        snapshot_inventory_history, snapshot_realized_pnl_history,
        snapshot_volume_sums, snapshot_trade_volumes,
        snapshot_roundtrip_volumes, snapshot_open_positions,
    )
    vols = snapshot_volume_sums(shadow)
    data = {
        "step": light["step"],
        "simulation_timestamp": light["simulation_timestamp"],
        "hotkeys": light["hotkeys"],
        "scores": light["scores"],
        "gentrx_scores": light["gentrx_scores"],
        "activity_factors": shadow.activity_factors,
        "pnl_factors": shadow.pnl_factors,
        "inventory_history": snapshot_inventory_history(shadow),
        "kappa_values": shadow.kappa_values,
        "realized_pnl_history": snapshot_realized_pnl_history(shadow),
        "open_positions": snapshot_open_positions(shadow),
        "unnormalized_scores": light["unnormalized_scores"],
        "deregistered_uids": light["deregistered_uids"],
        "trading_score_ema": light.get("trading_score_ema", {}),
        "trading_score_ema_ts": light.get("trading_score_ema_ts"),
        "trading_score_ema_n": light.get("trading_score_ema_n", {}),
        "trade_volumes": snapshot_trade_volumes(shadow),
        "roundtrip_volumes": snapshot_roundtrip_volumes(shadow),
        "volume_sums": vols["volume_sums"],
        "maker_volume_sums": vols["maker_volume_sums"],
        "taker_volume_sums": vols["taker_volume_sums"],
        "self_volume_sums": vols["self_volume_sums"],
        "roundtrip_volume_sums": vols["roundtrip_volume_sums"],
        "miner_stats": light["miner_stats"],
    }
    packer = msgpack.Packer(use_bin_type=True)
    tmp = f"{path}.tmp.shadow"
    try:
        with open(tmp, "wb", buffering=1024 * 1024) as f:
            total = _stream_pack(packer, f.write, data, _SAVE_STREAM_DEPTH)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return total


def _send_frame(sock, obj) -> None:
    data = pickle.dumps(obj, protocol=5)
    sock.sendall(struct.pack("Q", len(data)) + data)


def _recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("shadow socket closed")
        buf += chunk
    return buf


def _recv_frame(sock):
    (size,) = struct.unpack("Q", _recv_exact(sock, 8))
    return pickle.loads(_recv_exact(sock, size))


def apply_state_bytes(shadow: ShadowState, raw: bytes) -> int:
    """Parse the raw msgpack state exactly as the engine does and apply the same
    accounting. Returns the state timestamp."""
    import msgpack
    from taos.im.protocol import MarketSimulationStateUpdate
    from taos.im.validator.trade import update_trade_volumes

    d = msgpack.unpackb(raw, raw=False, use_list=True, strict_map_key=False)
    state = MarketSimulationStateUpdate.parse_dict(d)
    shadow.step += 1
    update_trade_volumes(shadow, state)
    # Derive resets from the state itself (same data main scans) and apply at
    # the same position — immediately after this round's update.
    validator_uid = getattr(shadow, 'validator_uid', None)
    if validator_uid is not None:
        from taos.im.validator.trade import collect_reset_uids, reset_agent_histories
        pending, _failed = collect_reset_uids(state, validator_uid)
        if pending:
            book_ids = list(range(shadow.simulation_config['book_count']))
            for uid in sorted(pending):
                reset_agent_histories(shadow, uid, book_ids)
            print(f"[SHADOW] resets applied (derived): {sorted(pending)}", flush=True)
    return state.timestamp


def _shadow_child_main(sock, cores, parity_ns):
    """Child process: INIT -> apply teed rounds -> parity digests back.

    Shadow-reward protocol: after applying an interval-boundary round X (where
    main also scores), the child PAUSES applies — side-buffering later state
    frames — until main's ("score_at", (X, sim_ts_used, deregs)) frame arrives
    (sent after main's reward completes, carrying the exact live inputs main
    used). It then scores on histories that are exactly post-X, ships the
    trading scores back, and drains the side buffer. A missing score_at (main
    skipped/errored its reward) times out and applies continue.
    """
    import queue as _queue

    try:
        if cores:
            try:
                os.sched_setaffinity(0, set(cores))
            except OSError:
                pass
        print(f"[SHADOW] child up (pid {os.getpid()}, cores {cores or 'unpinned'})", flush=True)

        frames = _queue.Queue()

        def _drain_socket():
            try:
                while True:
                    frames.put(_recv_frame(sock))
            except (EOFError, OSError):
                frames.put(("stop", None))

        threading.Thread(target=_drain_socket, daemon=True, name="shadow-sock-drain").start()

        shadow = None
        base_ts = None
        init_parts = {}
        buffered = []       # pre-INIT state frames
        side_buffer = []    # states arriving while awaiting a score_at
        awaiting_score_at = None
        pending_score_inputs = {}   # eager inputs by boundary ts
        applied = 0
        last_applied_ts = 0
        pending_save = None         # (expect_ts, path, light, mono_deadline)

        def _maybe_save():
            # Fire the pending save once the replica has caught up to its
            # snapshot point (states arrive FIFO, so >= means "at or one past").
            nonlocal pending_save
            if pending_save is None or shadow is None:
                return
            expect_ts, path, light, deadline = pending_save
            if last_applied_ts >= expect_ts:
                pending_save = None
                if last_applied_ts > expect_ts:
                    print(f"[SHADOW] save snapshot at ts={last_applied_ts} > requested {expect_ts}", flush=True)
                t0 = time.time()
                try:
                    nbytes = child_save_validator_state(shadow, light, path)
                    _send_frame(sock, ("save_done", (expect_ts, nbytes, time.time() - t0)))
                    print(f"[SHADOW] validator state saved ({nbytes/1e6:.1f}MB, {time.time()-t0:.2f}s)", flush=True)
                except Exception as e:
                    try:
                        _send_frame(sock, ("save_err", (expect_ts, f"{type(e).__name__}: {e}")))
                    except Exception:
                        pass
            elif time.monotonic() > deadline:
                pending_save = None
                try:
                    _send_frame(sock, ("save_err", (expect_ts, f"snapshot point never reached (applied={last_applied_ts})")))
                except Exception:
                    pass

        def _score_and_send(s_ts, sim_ts, deregs, gtx_scores, gtx_ema,
                            t_ema=None, t_ema_n=None, t_ema_ts=None):
            t0 = time.time()
            try:
                result = shadow_score(shadow, sim_ts, deregs, gtx_scores, gtx_ema,
                                      t_ema, t_ema_n, t_ema_ts)
                _send_frame(sock, ("scores", (s_ts, result, time.time() - t0)))
            except Exception as e:
                import traceback
                print(f"[SHADOW] scoring failed at ts={s_ts}: {e}\n{traceback.format_exc()}", flush=True)
                try:
                    _send_frame(sock, ("scores_err", (s_ts, f"{type(e).__name__}: {e}")))
                except Exception:
                    pass

        def _apply_one(ts, raw):
            nonlocal applied, awaiting_score_at, last_applied_ts
            t0 = time.time()
            apply_state_bytes(shadow, raw)
            applied += 1
            last_applied_ts = ts
            if ts % parity_ns == 0:
                _send_frame(sock, ("parity", (ts, compute_parity_components(shadow), pnl_len_vector(shadow), applied, time.time() - t0)))
            if shadow.scoring_interval and ts % shadow.scoring_interval == 0:
                eager = pending_score_inputs.pop(ts, None)
                if eager is not None:
                    # eager path: inputs arrived at tee time — compute NOW
                    # (during main's lock-queue delay) with no boundary pause.
                    _score_and_send(ts, *eager)
                else:
                    awaiting_score_at = (ts, time.monotonic())

        while True:
            try:
                kind, payload = frames.get(timeout=5.0)
            except _queue.Empty:
                _maybe_save()
                if awaiting_score_at and time.monotonic() - awaiting_score_at[1] > 120.0:
                    print(f"[SHADOW] score_at for ts={awaiting_score_at[0]} never arrived — skipping score", flush=True)
                    awaiting_score_at = None
                    pending, side_buffer = side_buffer, []
                    for ts, raw in pending:
                        if awaiting_score_at is None:
                            _apply_one(ts, raw)
                        else:
                            side_buffer.append((ts, raw))
                continue
            if kind == "stop":
                break
            if kind == "init_part":
                name, blob = payload
                init_parts[name] = pickle.loads(blob)
            elif kind == "init_done":
                base_ts, knobs = payload
                shadow = ShadowState(_rebuild_structs(init_parts), knobs)
                # Child-local loky sizing: score on the child's own cores (or a
                # small unpinned pool when it has no dedicated slice).
                kcfg = shadow.scoring_config['scoring']['kappa']
                kcfg['reward_cores'] = list(cores) if cores else []
                if not cores and kcfg.get('parallel_workers') == -1:
                    # -1 means one loky worker per core; with no dedicated slice
                    # fall back to a small unpinned pool. Explicit 0 (serial)
                    # and positive counts pass through unchanged.
                    kcfg['parallel_workers'] = 2
                init_parts = None
                replayed = 0
                for ts, raw in buffered:
                    if ts > base_ts and awaiting_score_at is None:
                        _apply_one(ts, raw)
                        replayed += 1
                    elif ts > base_ts:
                        side_buffer.append((ts, raw))
                buffered = []
                print(f"[SHADOW] init applied base_ts={base_ts} (replayed {replayed} buffered rounds)", flush=True)
            elif kind == "state":
                ts, raw = payload
                if shadow is None:
                    buffered.append((ts, raw))
                    if len(buffered) > 64:
                        print("[SHADOW] init overdue — dropping oldest buffered round", flush=True)
                        buffered.pop(0)
                    continue
                if ts <= base_ts:
                    continue
                if awaiting_score_at is not None:
                    side_buffer.append((ts, raw))
                    continue
                _apply_one(ts, raw)
            elif kind == "sim_start":
                old_ts, new_ts = payload
                if shadow is not None:
                    from taos.im.validator.trade import shift_simulation_histories
                    shift_simulation_histories(
                        shadow, old_ts, new_ts,
                        book_count=shadow.simulation_config['book_count'],
                        volume_decimals=shadow.simulation.volumeDecimals,
                        lookback=shadow.config.scoring.kappa.lookback,
                        volume_assessment_period=shadow.config.scoring.activity.trade_volume_assessment_period,
                        miner_wealth=shadow.simulation.miner_wealth,
                        effective_max_uids=shadow.effective_max_uids,
                    )
                    # New sim restarts near ts=0 — the old-run guard must not
                    # drop its frames, and any held boundary is void.
                    base_ts = -1
                    awaiting_score_at = None
                    pending, side_buffer = side_buffer, []
                    for ts, raw in pending:
                        if awaiting_score_at is None:
                            _apply_one(ts, raw)
                        else:
                            side_buffer.append((ts, raw))
                    print(f"[SHADOW] sim_start applied ({old_ts} -> {new_ts})", flush=True)
            elif kind == "resets":
                if shadow is not None:
                    from taos.im.validator.trade import reset_agent_histories
                    book_ids = list(range(shadow.simulation_config['book_count']))
                    for uid in payload:
                        reset_agent_histories(shadow, uid, book_ids)
                    print(f"[SHADOW] resets applied: {payload}", flush=True)
            elif kind == "reinit":
                shadow = None
                base_ts = None
                init_parts = {}
                buffered = []
                side_buffer = []
                awaiting_score_at = None
                pending_save = None
                print("[SHADOW] state dropped — awaiting fresh INIT", flush=True)
            elif kind == "save":
                expect_ts, path, light = payload
                if shadow is None:
                    try:
                        _send_frame(sock, ("save_err", (expect_ts, "not initialized")))
                    except Exception:
                        pass
                else:
                    pending_save = (expect_ts, path, light, time.monotonic() + 60.0)
            elif kind == "score_inputs":
                s_ts, sim_ts, deregs, gtx_scores, gtx_ema = payload[:5]
                t_ema = payload[5] if len(payload) > 5 else None
                t_ema_n = payload[6] if len(payload) > 6 else None
                t_ema_ts = payload[7] if len(payload) > 7 else None
                if shadow is not None and awaiting_score_at is not None and awaiting_score_at[0] == s_ts:
                    # inputs arrived after the boundary was applied — score now
                    awaiting_score_at = None
                    _score_and_send(s_ts, sim_ts, deregs, gtx_scores, gtx_ema,
                                    t_ema, t_ema_n, t_ema_ts)
                    pending, side_buffer = side_buffer, []
                    for ts, raw in pending:
                        if awaiting_score_at is None:
                            _apply_one(ts, raw)
                        else:
                            side_buffer.append((ts, raw))
                else:
                    pending_score_inputs[s_ts] = (sim_ts, deregs, gtx_scores, gtx_ema,
                                                  t_ema, t_ema_n, t_ema_ts)
                    while len(pending_score_inputs) > 4:
                        pending_score_inputs.pop(next(iter(pending_score_inputs)))
            elif kind == "score_at":
                s_ts, sim_ts, deregs = payload[0], payload[1], payload[2]
                gentrx_scores = payload[3] if len(payload) > 3 else None
                gentrx_ema = payload[4] if len(payload) > 4 else None
                t_ema = payload[5] if len(payload) > 5 else None
                t_ema_n = payload[6] if len(payload) > 6 else None
                t_ema_ts = payload[7] if len(payload) > 7 else None
                if shadow is None or awaiting_score_at is None or awaiting_score_at[0] != s_ts:
                    print(f"[SHADOW] unexpected score_at ts={s_ts} (awaiting={awaiting_score_at}) — ignored", flush=True)
                    if shadow is not None:
                        _send_frame(sock, ("scores_err", (s_ts, "not at boundary")))
                else:
                    awaiting_score_at = None
                    _score_and_send(s_ts, sim_ts, deregs, gentrx_scores, gentrx_ema,
                                    t_ema, t_ema_n, t_ema_ts)
                    pending, side_buffer = side_buffer, []
                    for ts, raw in pending:
                        if awaiting_score_at is None:
                            _apply_one(ts, raw)
                        else:
                            side_buffer.append((ts, raw))
            _maybe_save()
    except (EOFError, OSError):
        pass
    except Exception as e:
        import traceback
        print(f"[SHADOW] child fatal: {e}\n{traceback.format_exc()}", flush=True)
    finally:
        try:
            sock.close()
        except Exception:
            pass


class ScoringShadow:
    """Owner-side handle: tee stream, one-time INIT, parity ring + comparison."""

    def __init__(self, cores=None, parity_ns=SHADOW_PARITY_NS, child_fn=_shadow_child_main):
        self.parity_ns = parity_ns
        self._cores = list(cores) if cores else []
        self._child_fn = child_fn
        self._ctx = multiprocessing.get_context("spawn")
        self._sock = None
        self._proc = None
        self._reader = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scoring_shadow")
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._dropped = 0
        self.initialized = False
        # Two-sided rings: the child usually computes AHEAD of main (it applies
        # a round the moment it is teed, while main's _reward runs seconds later
        # behind _reward_lock) — so whichever side arrives first stashes its
        # value and the second side completes the comparison.
        self._ring = {}
        self._score_ring = {}
        self._pending_child_parity = {}
        self._pending_child_scores = {}
        self._score_futures = {}
        # Eager scoring (cutover): inputs shipped at boundary-tee time so the
        # child computes DURING main's lock-queue delay; results arriving before
        # main asks are stashed here and returned instantly by request_scores.
        self._eager_inputs = {}
        self._score_results = {}
        # Save offload (Stage B v2): the child writes the validator-state file
        # from its replica; main only ships the light main-only fields.
        self._save_futures = {}
        self._last_teed_ts = 0
        self.matches = 0
        self.mismatches = 0
        self.score_matches = 0
        self.score_mismatches = 0
        self._consecutive_mismatches = 0

    def start(self) -> None:
        import socket as _socket
        self._sock, child_sock = _socket.socketpair()
        # daemon=False: the child spawns loky workers for shadow scoring, which
        # daemonic processes cannot. Orphan safety comes from the socket: when
        # the parent dies the child's recv EOFs and its loop exits cleanly.
        self._proc = self._ctx.Process(
            target=self._child_fn,
            args=(child_sock, self._cores, self.parity_ns),
            daemon=False,
            name="scoring-shadow",
        )
        self._proc.start()
        child_sock.close()
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="shadow-parity-reader")
        self._reader.start()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    # ── main -> child ────────────────────────────────────────────────────────

    def tee(self, raw: bytes, timestamp: int) -> None:
        """Fire-and-forget send of one round's raw state bytes (FIFO executor).
        Bounded: if the child stalls, frames are dropped (shadow-only — never
        block or bloat the main process for the shadow's sake)."""
        if not self.is_alive():
            return
        with self._pending_lock:
            if self._pending >= 8:
                self._dropped += 1
                return
            self._pending += 1

        def _send():
            try:
                _send_frame(self._sock, ("state", (timestamp, raw)))
            except Exception:
                pass
            finally:
                with self._pending_lock:
                    self._pending -= 1

        self._executor.submit(_send)
        self._last_teed_ts = timestamp

    def emit_sim_start(self, old_ts: int, new_ts: int) -> None:
        """Simulation restart: the child must run the same history shift.
        FIFO-ordered with the state tee (old-sim frames before, new-sim after)."""
        if not self.is_alive():
            return

        def _send():
            try:
                _send_frame(self._sock, ("sim_start", (old_ts, new_ts)))
            except Exception:
                pass

        self._executor.submit(_send)

    def emit_resets(self, uids: list) -> None:
        """Deregistration resets: the child zeroes the same UIDs' histories."""
        if not self.is_alive():
            return

        def _send():
            try:
                _send_frame(self._sock, ("resets", list(uids)))
            except Exception:
                pass

        self._executor.submit(_send)

    def tee_score_inputs(self, timestamp: int, sim_ts: int, deregs: list,
                         gentrx_scores=None, gentrx_ema=None,
                         trading_ema=None, trading_ema_n=None, trading_ema_ts=None) -> None:
        """Eager scoring: ship the boundary round's scoring inputs at TEE time,
        so the child computes during the seconds main's _reward spends queued
        behind _reward_lock — request_scores then collects a (usually) finished
        result instead of holding the lock through the full compute. The same
        inputs are recorded for the verify pin (eager_inputs_for)."""
        if not self.is_alive() or not self.initialized:
            return
        pin = {
            'simulation_timestamp': sim_ts,
            'deregistered_uids': list(deregs),
            'gentrx_scores': dict(gentrx_scores or {}),
            'gentrx_ema': dict(gentrx_ema or {}),
            'trading_ema': dict(trading_ema or {}),
            'trading_ema_n': dict(trading_ema_n or {}),
            'trading_ema_ts': trading_ema_ts,
        }
        self._stash(self._eager_inputs, timestamp, pin, 4)

        def _send():
            try:
                _send_frame(self._sock, (
                    "score_inputs",
                    (timestamp, sim_ts, pin['deregistered_uids'],
                     pin['gentrx_scores'], pin['gentrx_ema'],
                     pin['trading_ema'], pin['trading_ema_n'], pin['trading_ema_ts']),
                ))
            except Exception:
                pass

        self._executor.submit(_send)

    def eager_inputs_for(self, timestamp: int):
        """The exact inputs shipped for this boundary (verify must pin to these),
        or None if the boundary wasn't eagerly dispatched."""
        return self._eager_inputs.get(timestamp)

    def request_scores(self, timestamp: int, sim_ts: int, deregs: list,
                       gentrx_scores=None, gentrx_ema=None, timeout: float = 45.0,
                       eager: bool = False,
                       trading_ema=None, trading_ema_n=None, trading_ema_ts=None):
        """Cutover mode: collect this boundary's full scoring result, blocking
        until it arrives (call via run_in_executor — the wait is GIL-free).
        eager=True means the inputs were already shipped at tee time (the child
        is computing or done — do not send a score_at); otherwise the score_at
        pull path runs as before. Returns the result dict {'trading','gentrx',
        'factors','gentrx_ema'} or None on ANY failure — the caller falls back
        to the in-process get_rewards and a dead child is respawned.
        """
        import concurrent.futures
        ready = self._score_results.pop(timestamp, None)
        if ready is not None:
            return ready
        if not self.is_alive():
            try:
                self.start()
                self.initialized = False  # fresh child needs a new INIT
            except Exception:
                pass
            return None
        if not self.initialized:
            return None
        fut = concurrent.futures.Future()
        self._score_futures[timestamp] = fut

        if not eager:
            def _send():
                try:
                    _send_frame(self._sock, (
                        "score_at",
                        (timestamp, sim_ts, list(deregs),
                         dict(gentrx_scores or {}), dict(gentrx_ema or {}),
                         dict(trading_ema or {}), dict(trading_ema_n or {}), trading_ema_ts),
                    ))
                except Exception:
                    pass

            self._executor.submit(_send)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            return None
        finally:
            self._score_futures.pop(timestamp, None)

    def request_save(self, path: str, light: dict):
        """Save offload: ask the child to write the validator-state file from
        its replica once it has applied the last teed round. `light` carries
        the main-only fields (step/scores/hotkeys/miner_stats/...). Returns
        (expect_ts, Future) — the future resolves to (bytes, child_io_secs) on
        success or None on child failure — or None when the offload is
        unavailable (caller falls back to the in-process save path)."""
        if not self.is_alive() or not self.initialized or not self._last_teed_ts:
            return None
        import concurrent.futures
        expect_ts = self._last_teed_ts
        fut = concurrent.futures.Future()
        self._save_futures[expect_ts] = fut

        def _send():
            try:
                _send_frame(self._sock, ("save", (expect_ts, path, light)))
            except Exception:
                pass

        self._executor.submit(_send)
        return expect_ts, fut

    def request_reinit(self) -> None:
        """Self-heal: drop the child's state and re-run INIT on the next round
        (triggered automatically after consecutive parity mismatches)."""
        import bittensor as bt
        bt.logging.warning("[SHADOW] requesting re-INIT (parity self-heal)")
        self.initialized = False
        self._ring.clear()
        self._score_ring.clear()
        self._pending_child_parity.clear()
        self._pending_child_scores.clear()

        def _send():
            try:
                _send_frame(self._sock, ("reinit", None))
            except Exception:
                pass

        self._executor.submit(_send)

    def send_init(self, validator) -> None:
        """Build + stream the INIT snapshot. MUST be called while the caller
        holds _reward_lock (structures frozen); runs synchronously on the
        shadow executor so it is FIFO-ordered with the tee frames."""
        self.initialized = True
        base_ts = validator._shadow_applied_ts

        def _do_init():
            t0 = time.time()
            try:
                for name in _STRUCT_NAMES:
                    plain = _plainify(getattr(validator, name, {}))
                    _send_frame(self._sock, ("init_part", (name, pickle.dumps(plain, protocol=5))))
                    del plain
                from taos.im.validator.reward import build_scoring_config, build_simulation_config_dict
                knobs = {
                    "_last_prune_timestamp": getattr(validator, "_last_prune_timestamp", None),
                    "step": validator.step,
                    "effective_max_uids": validator.effective_max_uids,
                    "config": _config_duck(validator.config),
                    "simulation": _simulation_duck(validator.simulation),
                    # Shadow-reward: the exact dicts main's get_rewards consumes
                    # (single source of truth — see reward.build_scoring_config).
                    "scoring_config": build_scoring_config(validator),
                    "simulation_config": build_simulation_config_dict(validator),
                    "scoring_interval": validator.config.scoring.interval,
                    "validator_uid": getattr(validator, 'uid', None),
                }
                _send_frame(self._sock, ("init_done", (base_ts, knobs)))
                import bittensor as bt
                bt.logging.info(f"[SHADOW] init snapshot sent (base_ts={base_ts}, {time.time()-t0:.2f}s)")
            except Exception as e:
                import bittensor as bt
                bt.logging.warning(f"[SHADOW] init failed: {e}")

        return self._executor.submit(_do_init)

    def _compare_parity(self, ts: int, mine, theirs, applied=None, apply_s=None) -> None:
        import bittensor as bt
        mine_comps, mine_vec = mine if isinstance(mine, tuple) else (mine, None)
        theirs_comps, theirs_vec = theirs if isinstance(theirs, tuple) else (theirs, None)
        mine, theirs = mine_comps, theirs_comps
        if mine == theirs:
            self.matches += 1
            bt.logging.info(
                f"[SHADOW-PARITY] ts={ts} MATCH ({self.matches} ok / {self.mismatches} bad, "
                f"dropped={self._dropped})"
            )
        else:
            self.mismatches += 1
            self._consecutive_mismatches += 1
            if isinstance(mine, dict) and isinstance(theirs, dict):
                # Name the diverging structures — turns a rare mismatch into a
                # self-diagnosing event instead of an opaque hash difference.
                _diff = sorted(
                    k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)
                )
                bt.logging.error(
                    f"[SHADOW-PARITY] ts={ts} MISMATCH diverged_structures={_diff} "
                    f"(main={ {k: mine.get(k) for k in _diff} } shadow={ {k: theirs.get(k) for k in _diff} })"
                )
                if 'n_pnl' in _diff and mine_vec is not None and theirs_vec is not None:
                    _uid_diff = {
                        u: (mine_vec.get(u, 0), theirs_vec.get(u, 0))
                        for u in set(mine_vec) | set(theirs_vec)
                        if mine_vec.get(u, 0) != theirs_vec.get(u, 0)
                    }
                    bt.logging.error(
                        f"[SHADOW-PARITY] ts={ts} n_pnl uid-level diff (main,shadow): {_uid_diff}"
                    )
            else:
                bt.logging.error(f"[SHADOW-PARITY] ts={ts} MISMATCH main={mine} shadow={theirs}")
            if self._consecutive_mismatches >= 2:
                self._consecutive_mismatches = 0
                self.request_reinit()
            return
        self._consecutive_mismatches = 0

    def _compare_scores(self, ts: int, mine: list, theirs: list) -> None:
        import bittensor as bt
        if len(mine) == len(theirs) and all(a == b for a, b in zip(mine, theirs)):
            self.score_matches += 1
            bt.logging.info(
                f"[SHADOW-SCORES] ts={ts} MATCH ({self.score_matches} ok / "
                f"{self.score_mismatches} bad, n={len(theirs)})"
            )
        else:
            self.score_mismatches += 1
            _diffs = [
                (i, a, b) for i, (a, b) in enumerate(zip(mine, theirs)) if a != b
            ] if len(mine) == len(theirs) else []
            _max = max((abs(a - b) for _, a, b in _diffs), default=float('nan'))
            bt.logging.error(
                f"[SHADOW-SCORES] ts={ts} MISMATCH n_main={len(mine)} n_shadow={len(theirs)} "
                f"diff_uids={len(_diffs)} max_abs_diff={_max:.3e} first={_diffs[:3]}"
            )

    @staticmethod
    def _stash(ring: dict, key, value, cap: int) -> None:
        ring[key] = value
        while len(ring) > cap:
            ring.pop(next(iter(ring)))

    def record_main_digest(self, timestamp: int, digest: str) -> None:
        theirs = self._pending_child_parity.pop(timestamp, None)
        if theirs is not None:
            self._compare_parity(timestamp, digest, theirs)
        else:
            self._stash(self._ring, timestamp, digest, 16)

    def on_main_scored(self, timestamp: int, sim_ts_used: int, deregs_used: list, trading_scores: list,
                       trading_ema=None, trading_ema_n=None, trading_ema_ts=None) -> None:
        """Called after main's get_rewards completes on a scoring round: record
        main's trading scores for comparison and release the child (which is
        holding applies at this boundary) with the exact live inputs main used."""
        theirs = self._pending_child_scores.pop(timestamp, None)
        if theirs is not None:
            self._compare_scores(timestamp, list(trading_scores), theirs)
        else:
            self._stash(self._score_ring, timestamp, list(trading_scores), 8)
        if not self.is_alive():
            return

        def _send():
            try:
                _send_frame(self._sock, ("score_at", (timestamp, sim_ts_used, list(deregs_used),
                                                      None, None,
                                                      dict(trading_ema or {}), dict(trading_ema_n or {}),
                                                      trading_ema_ts)))
            except Exception:
                pass

        self._executor.submit(_send)

    # ── child -> main ────────────────────────────────────────────────────────

    def _read_loop(self):
        import bittensor as bt
        try:
            while True:
                kind, payload = _recv_frame(self._sock)
                if kind == "parity":
                    ts, comps, vec, applied, apply_s = payload
                    mine = self._ring.pop(ts, None)
                    if mine is None:
                        # Child computed ahead of main (normal: it applies on tee,
                        # main applies seconds later behind _reward_lock) — stash
                        # and let record_main_digest complete the comparison.
                        self._stash(self._pending_child_parity, ts, (comps, vec), 32)
                    else:
                        self._compare_parity(ts, mine, (comps, vec))
                elif kind == "scores":
                    ts, result, score_s = payload
                    fut = self._score_futures.pop(ts, None)
                    if fut is not None:
                        # cutover mode: resolve the awaiting _reward directly
                        fut.set_result(result)
                        continue
                    if ts in self._eager_inputs:
                        # eager result landed before main asked — stash for
                        # request_scores to return instantly
                        self._stash(self._score_results, ts, result, 4)
                        continue
                    trading = result['trading'] if isinstance(result, dict) else result
                    mine = self._score_ring.pop(ts, None)
                    if mine is None:
                        self._stash(self._pending_child_scores, ts, trading, 8)
                    else:
                        self._compare_scores(ts, mine, trading)
                elif kind == "scores_err":
                    ts, msg = payload
                    bt.logging.warning(f"[SHADOW] child scoring error at ts={ts}: {msg}")
                    fut = self._score_futures.pop(ts, None)
                    if fut is not None:
                        fut.set_result(None)
                elif kind == "save_done":
                    ts, nbytes, secs = payload
                    fut = self._save_futures.pop(ts, None)
                    if fut is not None:
                        fut.set_result((nbytes, secs))
                elif kind == "save_err":
                    ts, msg = payload
                    bt.logging.warning(f"[SHADOW] child save failed at ts={ts}: {msg}")
                    fut = self._save_futures.pop(ts, None)
                    if fut is not None:
                        fut.set_result(None)
        except (EOFError, OSError):
            pass
        except Exception as e:
            bt.logging.warning(f"[SHADOW] parity reader: {e}")

    def stop(self) -> None:
        try:
            self._executor.submit(lambda: _send_frame(self._sock, ("stop", None)))
            self._executor.shutdown(wait=True)
        except Exception:
            pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.kill()
            self._proc = None


def _config_duck(config):
    """Minimal picklable stand-in exposing exactly the config paths trade.py reads."""
    from types import SimpleNamespace
    return SimpleNamespace(
        scoring=SimpleNamespace(
            activity=SimpleNamespace(
                trade_volume_assessment_period=config.scoring.activity.trade_volume_assessment_period,
                trade_volume_sampling_interval=config.scoring.activity.trade_volume_sampling_interval,
            ),
            kappa=SimpleNamespace(lookback=config.scoring.kappa.lookback),
        )
    )


def _simulation_duck(simulation):
    from types import SimpleNamespace
    return SimpleNamespace(
        volumeDecimals=simulation.volumeDecimals,
        miner_wealth=simulation.miner_wealth,
    )
