# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""GenTRX gradient exchange — event-driven aggregation server.

HTTP endpoints:
  POST /gentrx/state                  — receive sim state ticks from validator
  GET  /gentrx/assignment?miner_uid=N — fetch assignment for a miner
  GET  /gentrx/version                — check current model version
  GET  /gentrx/scores?since_round=N   — poll latest scores
  GET  /gentrx/metrics                — Prometheus metrics (scrape target for Grafana)

Two bucket types, each committed on-chain (state arrives over HTTP — no state bucket):
  - Validator bucket (GENTRX_VALIDATOR_S3_*): checkpoints/ + data/ + proposals/
  - Per-miner buckets (GENTRX_AGENT_S3_*): gradients/ — one per miner

Run standalone:
    python -m GenTRX.src.gradient_server \\
        --checkpoint $REPO/checkpoints/GenTRX/best.pt \\
        --val-data $REPO/data/server \\
        --subtensor-network local --netuid 1

All filesystem paths are resolved to absolute at construction. Pass
absolute paths or `$VAR`-expanded paths to keep the invocation
independent of the launching shell's CWD.
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import math
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from GenTRX.src.gradient_store import GradientStore
from GenTRX.src.version import TRAIN_REGIME_VERSION, checkpoint_stamp

logger = logging.getLogger("GenTRX.src.gradient_server")


def _tag_to_ns(tag: str) -> int:
    """Convert ddHHMMSS tag back to nanoseconds.

    Also handles plain integer strings (sim format fallback) and an optional
    `_NNNN` disambiguating suffix that `_flush_book_parquet` appends when
    multiple flushes share the same sim-second tag (cap-triggered flushes
    within one second). The suffix is stripped here so downstream filtering
    is unaffected.
    """
    if "_" in tag:
        tag = tag.split("_", 1)[0]
    if len(tag) == 8:
        dd = int(tag[0:2])
        hh = int(tag[2:4])
        mm = int(tag[4:6])
        ss = int(tag[6:8])
        if hh < 24 and mm < 60 and ss < 60:
            total_secs = dd * 86400 + hh * 3600 + mm * 60 + ss
            return total_secs * 1_000_000_000
    return int(tag)


def _ts_to_tag(ts_ns: int) -> str:
    """Convert nanosecond timestamp to ddHHMMSS tag for filenames."""
    total_secs = ts_ns // 1_000_000_000
    days = total_secs // 86400
    remainder = total_secs % 86400
    hours = remainder // 3600
    minutes = (remainder % 3600) // 60
    seconds = remainder % 60
    return f"{days:02d}{hours:02d}{minutes:02d}{seconds:02d}"


# Heavy fields (_gradient_data, _comp, _score*) are intentionally dropped;
# they get re-fetched and recomputed on the next collection tick.
_ASSIGNMENT_PERSIST_FIELDS = (
    "round",
    "books",
    "data",
    "ts_start",
    "ts_end",
    "model_version",
)


def _json_safe(obj):
    # Diverged gradients yield inf/nan losses that starlette's encoder (allow_nan=False) 500s on.
    if isinstance(obj, float):
        if obj != obj:
            return 0.0
        if obj == float("inf"):
            return 1e30
        if obj == float("-inf"):
            return -1e30
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _serialize_assignments(d: dict) -> dict:
    out: dict[str, dict] = {}
    for uid, a in d.items():
        state = a.get("_state")
        if state not in ("DELIVERED", "GRADIENT_IN", "SCORED"):
            continue
        skel: dict = {}
        for k in _ASSIGNMENT_PERSIST_FIELDS:
            if k in a:
                skel[k] = a[k]
        out[str(uid)] = skel
    return out


def _restore_assignment_dict(src: dict, target: dict) -> None:
    now = time.time()
    for uid_str, skel in src.items():
        try:
            uid = int(uid_str)
        except (TypeError, ValueError):
            continue
        if not isinstance(skel, dict):
            continue
        target[uid] = {
            "round": skel.get("round"),
            "books": skel.get("books", []) or [],
            "data": skel.get("data", []) or [],
            "ts_start": int(skel.get("ts_start", 0) or 0),
            "ts_end": int(skel.get("ts_end", 0) or 0),
            "model_version": skel.get("model_version"),
            "_state": "DELIVERED",
            "_created_at": now,
            "_delivered_at": now,
            "_gradient_data": None,
            "_score": None,
        }


def _filter_by_timestamp(
    files: list[Path],
    ts_start: int,
    ts_end: int,
) -> list[Path]:
    """Filter interval parquet files by timestamp range (nanoseconds).

    Supports two filename formats:
      - ddHHMMSS-ddHHMMSS.parquet (live format, e.g. "01013000-01020000.parquet")
      - NNNNNNNN-NNNNNNNN.parquet (sim format, raw integers)

    A file is included if its range overlaps with [ts_start, ts_end].
    ts_start=0 means from beginning, ts_end=0 means to end.
    """
    result = []
    for f in files:
        parts = f.stem.split("-")
        if len(parts) != 2:
            result.append(f)  # can't parse, include by default
            continue
        try:
            file_start = _tag_to_ns(parts[0])
            file_end = _tag_to_ns(parts[1])
        except ValueError:
            result.append(f)
            continue

        if ts_end and file_start >= ts_end:
            continue
        if ts_start and file_end <= ts_start:
            continue
        result.append(f)
    return result


def normalize_seed_checkpoint(
    path: Path, current_n_types: int, current_regime: int, label_smooth_sigma: float
) -> str:
    """Make a locally-deployed seed checkpoint usable under the current code,
    automatically, on any box — so no manual re-stamp step is needed at deploy.

    Returns one of "absent" | "current" | "restamped" | "quarantined":

      - "absent"      — no file; caller fresh-inits a current-format model.
      - "quarantined" — the seed's order-type head has a different ``n_types``
        than this build (e.g. an old n_types=3 seed under n_types=5 code). It
        cannot be loaded or scored, so it is renamed aside; the caller then
        fresh-inits a compatible model instead of crash-trapping on the first
        gradient. A trained baseline requires deploying the current-format seed.
      - "restamped"   — same architecture, older ``train_regime_version`` (e.g.
        the current-format release seed shipped under an earlier regime stamp).
        Re-stamped in place to the current regime so it is adopted as the
        baseline with no manual step. Only the LOCAL seed file is touched; the
        aggregator bucket's own regime gate (resume/sync) is unaffected.

    Idempotent: a current, correctly-stamped seed returns "current" untouched.
    """
    import torch

    if not path.exists():
        return "absent"
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("seed normalize: could not read %s: %s", path, exc)
        return "current"

    seed_n_types = (ckpt.get("model_config") or {}).get("n_types")
    if seed_n_types is not None and int(seed_n_types) != int(current_n_types):
        quarantine = path.with_name(path.name + f".incompatible-n_types{int(seed_n_types)}")
        try:
            path.rename(quarantine)
        except OSError as exc:
            logger.error(
                "seed normalize: seed n_types=%s incompatible with build n_types=%s "
                "but could not move it aside (%s) — leaving in place",
                seed_n_types, current_n_types, exc,
            )
            return "current"
        logger.error(
            "SEED INCOMPATIBLE: %s has n_types=%s, this build uses n_types=%s "
            "(scoring would crash on it). Moved to %s; starting from a fresh "
            "n_types=%s model. Deploy the current-format release seed to resume "
            "from a trained baseline.",
            path.name, seed_n_types, current_n_types, quarantine.name, current_n_types,
        )
        return "quarantined"

    seed_regime = int(ckpt.get("train_regime_version", 0) or 0)
    if seed_regime < int(current_regime):
        ckpt.update(checkpoint_stamp(label_smooth_sigma))
        try:
            torch.save(ckpt, str(path))
        except Exception as exc:
            logger.error("seed normalize: re-stamp save failed for %s: %s", path, exc)
            return "current"
        logger.info(
            "seed re-stamped: %s train_regime_version %d -> %d (architecture "
            "compatible; adopted as current baseline)",
            path.name, seed_regime, current_regime,
        )
        return "restamped"
    return "current"


class GradientAggregator:
    """Thread-safe gradient accumulator + periodic aggregation.

    Assignment protocol:
      - Per-round held-out split, rotating and disjoint from that round's
        training books, pushed by the validator.
      - Each round, assigns each miner a random overlapping sample of training
        books over a shared held-out window, plus a timestamp window biased
        toward recent data.
      - Round advances after each aggregation cycle.
    """

    # Default training window: 5 minutes of sim time
    DEFAULT_WINDOW_NS = 300_000_000_000

    def __init__(
        self,
        checkpoint_path: str,
        val_data_path: str,
        output_path: str | None = None,
        log_path: str | None = None,
        books_per_miner: int = 3,
        val_fraction: float = 0.10,
        min_score: float = -0.1,
        warmup_rounds: int = 5,
        rollback: bool = True,
        label_smooth_sigma: float = 1.0,
        interval: float = 30.0,
        max_val_batches: int = 10,
        seq_len: int = 512,
        max_loader_files: int = 0,
        seed: int = 42,
        window_ns: int | None = None,
        chain: Any | None = None,
        validator_store: GradientStore | None = None,
        is_aggregator: bool = True,
        parquet_interval_ns: int = 300_000_000_000,  # 5 min, matches training window
        max_pending_rows_per_book: int = 30_000,
        loop_sleep_s: float = 5.0,
        round_grace_s: float = 30.0,
        max_gradient_bytes: int = 20 * 1024 * 1024,
        overfit_ratio: float = 3.0,
        reject_untagged: bool = True,
        proposal_norm_ratio: float = 10.0,
        max_grad_norm: float | None = None,
        max_agg_norm: float | None = None,
        max_agg_element: float | None = None,
        no_startup_cleanup: bool = True,
        keep_checkpoints: int = 10,
        keep_proposals: int = 10,
        checkpoint_interval: int = 6,
        keep_version_deltas: int = 24,
        publish_state_hash: bool = False,
        s3_cache_retention_hours: float = 24.0,
        blocks_per_round: int = 25,
        block_time_s: float = 12.0,
        forward_scoring: bool = True,
        agg_top_k_frac: float = 1.0,
        validator_uid: str = "",
        bucket_prefix: str = "",
    ):
        # Default to "0" so unscoped invocations (proxy/localnet) write
        # proposals/scores under a stable path; multi-validator setups
        # must pass --validator-uid to avoid collisions in shared buckets.
        self._validator_uid: str = (
            str(validator_uid) if validator_uid not in (None, "") else "0"
        )
        self._bucket_prefix: str = bucket_prefix
        # Kept beside the prefix so an assignment can name the shard in words as well as in the key.
        self._mode: str = (bucket_prefix.rstrip("/").rsplit("/", 1)[-1] if bucket_prefix else "")
        # Resolve all filesystem paths to absolute at construction time.
        # If the operator passes a relative --checkpoint, the audit log
        # and any derived paths would otherwise depend on the launching
        # process's CWD.
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.val_data_path = (
            str(Path(val_data_path).expanduser().resolve())
            if val_data_path
            else val_data_path
        )
        self.output_path = (
            Path(output_path).expanduser().resolve()
            if output_path
            else self.checkpoint_path
        )
        self.log_path = (
            Path(log_path).expanduser().resolve()
            if log_path
            else self.output_path.parent / "aggregation.jsonl"
        )
        self.books_per_miner = books_per_miner
        self.val_fraction = val_fraction
        self.min_score = min_score
        self.warmup_rounds = warmup_rounds
        self.rollback = rollback
        self.label_smooth_sigma = label_smooth_sigma
        self.interval = interval
        self.max_val_batches = max_val_batches
        # Sequence length for scoring loaders — must match the miners' training
        # seq_len so the model is evaluated at the context it was trained on.
        self.seq_len = seq_len
        # Own-data check is a cheap proof-of-work gate (did the miner actually
        # train on its assigned pages), not a quality signal — so it runs on
        # far fewer batches than the held-out reward eval.
        self._proof_batches = max(4, max_val_batches // 4)
        # Cap files fetched per loader build (0 = unlimited); bounds the S3
        # list+download time that dominates t_loader when scoring only needs
        # max_val_batches batches.
        self.max_loader_files = max_loader_files
        self.window_ns = window_ns or self.DEFAULT_WINDOW_NS
        # Retention on the validator bucket. 0 disables pruning (objects
        # accumulate indefinitely; operator handles cleanup themselves).
        # Defaults: 10 checkpoints (~470 MB at 47 MB each), 10 proposals.
        self.keep_checkpoints = keep_checkpoints
        self.keep_proposals = keep_proposals
        # Full checkpoint uploaded every N versions; in between, only the small
        # per-version delta is published and miners advance by applying it.
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.keep_version_deltas = keep_version_deltas
        self.publish_state_hash = publish_state_hash
        self.s3_cache_retention_hours = s3_cache_retention_hours
        self._last_s3_cache_prune: float = 0.0
        # Validator-advice adaptive top-k: cumulative emitted-event + execution
        # counts -> trade density -> advised gradient top_k_frac in the assignment.
        self._adv_events = 0
        self._adv_trades = 0
        # Forward/temporal scoring: score gradients on data that arrived
        # AFTER the miner's training window (contamination-proof). Toggle for rollback.
        self._forward_scoring = bool(forward_scoring)
        # Top-k fraction kept when re-compressing the aggregated delta that is
        # applied to the canonical model (and published as the version delta).
        # 1.0 = dense: apply the full mean direction, not a sparse subset whose
        # largest-magnitude coords skew toward per-miner page overfit.
        self._agg_top_k_frac = max(0.0, min(1.0, float(agg_top_k_frac)))
        # Round-completion fallback timing. The validator drives round
        # closure via POST /gentrx/round (block-sync); these only feed the
        # fallback that fires when the validator stops pushing.
        self.blocks_per_round = blocks_per_round
        self.block_time_s = block_time_s

        # Single validator bucket: checkpoints/ + data/ + proposals/ all in one place.
        # Committed to chain so miners and the aggregator discover it without
        # pre-configuration.
        self.validator_store = validator_store
        # When True (uid-0 aggregator): reads cross-validator scores from chain,
        # aggregates gradients, publishes checkpoint.
        # When False (sibling validator): scores miners, publishes scores only.
        self.is_aggregator: bool = is_aggregator

        # On-chain bucket discovery (optional). When set, gradients are
        # collected from per-miner buckets using committed read credentials
        # instead of from the validator's single bucket.
        self._chain = chain
        self._miner_buckets: dict[int, Any] = {}
        # Static buckets registered via /register_bucket (benchmark miners that
        # can't commit to chain). Merged into _miner_buckets after each refresh.
        self._static_buckets: dict[int, Any] = {}
        # Cooldown so we don't re-query chain every loop iteration when the
        # result is empty (no miners committed yet). Both call sites go through
        # _refresh_miner_buckets().
        self._miner_buckets_queried_at: float = 0.0
        self._miner_buckets_refresh_s: float = 30.0
        self._version = 0
        self._agg_round = 0
        self._last_round_log_ts: float = 0.0
        self._last_push_block: int | None = (
            None  # block number from most recent POST /round
        )
        self._lock = threading.Lock()
        # Guards `_assignments` + `_prev_round_assignments` state-transitions.
        # Separate from `_lock` (which holds the wide critical sections around
        # checkpoint load/save) so the API path doesn't stall behind I/O.
        # RLock: nested re-entry from inside `_create_assignment_for` ->
        # `_create_round_assignments` is safe.
        self._assignments_lock = threading.RLock()
        # Guards `_scoring_cache` get/set/clear so eager-score callers don't
        # see a half-cleared cache. The model object itself is mutated in place
        # by aggregation; we evict on the boundary rather than locking the
        # forward pass.
        self._scoring_cache_lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Monitoring identity. sim_id is the simulator-issued identifier for
        # the current sim run, sourced from the first state packet's config
        # block (state.config.simulation_id). sim_epoch bumps when we observe
        # sim time going backwards inside one gradient server lifetime —
        # catches sim restarts the simulator forgot to re-id. Both are stamped
        # on every aggregation.jsonl event so the dashboard can tell sessions
        # apart instead of blending them.
        self._sim_id: str | None = None
        self._sim_epoch: int = 0
        self._last_seen_sim_ts: int = 0
        self._dedup_drops_total: int = 0
        self._pre_bind_drops: int = 0
        # Reorder buffer for out-of-order state ticks. Ticks are held when
        # they arrive with a sim-timestamp jump larger than _reorder_jump_ns
        # (suggesting earlier ticks are still in flight). Held ticks are
        # flushed in sim-timestamp order once either (a) a filling tick
        # arrives that closes the gap, or (b) _reorder_timeout_s elapses.
        # Fast path: if the buffer is empty and the tick is in order, it is
        # passed directly to _process_tick without buffering overhead.
        self._reorder_buf: list = []  # min-heap: (sim_ts, seq, wall_time, tick)
        self._reorder_seq: int = 0
        self._reorder_timeout_s: float = 30.0  # states arrive every 5-10 s wall-clock
        # EMA of observed sim-timestamp increments between consecutive in-order
        # ticks. Used as the "expected publish interval"; a new tick is held in
        # the reorder buffer when its increment exceeds this value, giving
        # earlier in-flight ticks time to arrive. Self-calibrates from the
        # first few ticks; falls back to parquet_interval // 60 until then.
        self._ts_increment_ema: float = 0.0
        self._ts_ema_alpha: float = 0.2
        self._ts_ema_samples: int = 0
        self._ts_ema_fallback: int = parquet_interval_ns // 60  # 5 s sim time default
        # Sim id from the pending-rows staging file.
        self._restored_sim_id: str | None = None
        # Sim id from the bucket marker (data/<uid>/.sim_id).
        self._bucket_sim_id: str | None = None
        # Last sim_id durably written to the bucket marker. Re-asserted on each
        # flush so the marker always matches the data on disk (a bind-time write
        # can fail silently and strand a stale marker → false wipe on restart).
        self._sim_marker_written: str | None = None
        # Sim transition handling. Set to True when an 'ESE' marker is seen
        # (or when the heuristic detects a restart without an explicit
        # marker). Picked up by the aggregation loop on its next tick,
        # which wipes stale parquets under the validator bucket's data/
        # prefix. Checkpoints and proposals are intentionally not touched.
        self._data_cleanup_pending: bool = False
        # Last model version stamped onto an assignment. When _version advances
        # past this, the round-completion fallback timer is extended so miners
        # have time to download the new checkpoint before uploading.
        self._last_assigned_version: int = 0
        # Multiplier applied to the round-completion fallback estimate when
        # the checkpoint just rolled.
        self._round_rollover_mult: float = 1.5

        # Latest scores (in-memory, served via GET /gentrx/scores)
        self._latest_scores: dict | None = None

        # Aggregation snapshot — updated on each aggregation; embedded in
        # the scores payload so the validator-side Prometheus collector can
        # track training progress without needing a second HTTP call.
        self._rounds_aggregated_total: int = 0
        self._rollbacks_total: int = 0
        # Stable-shape defaults so every gentrx_training{stat=X} series exists
        # from the first scrape, even before any round has aggregated. Dashboard
        # panels see 0 instead of "no data" until real values overwrite.
        self._last_aggregation: dict = {
            "round": 0,
            "version": 0,
            "n_assigned": 0,
            "n_delivered": 0,
            "n_collected": 0,
            "n_version_mismatched": 0,
            "n_scored": 0,
            "n_accepted": 0,
            "loss_before": 0.0,
            "loss_after": 0.0,
            "loss_improvement_pct": 0.0,
            "t_score_s": 0.0,
            "t_aggregate_s": 0.0,
            "t_total_s": 0.0,
            "t_load_s": 0.0,
            "t_proposal_eval_s": 0.0,
            "t_save_ckpt_s": 0.0,
            "t_loader_build_s": 0.0,
            "rolled_back": 0,
            "rollback_rate_10w": 0.0,
            "rollback_rate_50w": 0.0,
            "grad_norm_mean": 0.0,
            "grad_norm_min": 0.0,
            "grad_norm_max": 0.0,
            "grad_norm_median": 0.0,
            "grad_norm_std": 0.0,
            "overlap_pairs_checked": 0,
            "overlap_pairs_high": 0,
            "overlap_mean": 0.0,
            "overlap_max": 0.0,
            "loader_cache_hits": 0,
            "loader_cache_misses": 0,
            "loader_cache_hit_rate": 0.0,
            "proposals_evaluated": 0,
            "proposals_skipped": 0,
        }
        for field in ("order_type", "price", "vol_int", "vol_dec", "interval"):
            self._last_aggregation[f"per_field_loss_before_{field}"] = 0.0
            self._last_aggregation[f"per_field_loss_after_{field}"] = 0.0
        self._rollback_history: deque[bool] = deque(maxlen=50)

        # ---- Assignment lifecycle tracking ----
        # States: PENDING → DATA_READY → DELIVERED → GRADIENT_IN → SCORED
        # miner_uid → MinerAssignment dict — HOLDS ONLY THE CURRENT ROUND.
        self._assignments: dict[int, dict] = {}
        # Closing-round assignments preserved across a POST /round that
        # advances _agg_round. Without this the previous round's _state /
        # _gradient_data would be overwritten by the new round's installation
        # before the aggregation loop had a chance to collect + score.
        # uid → MinerAssignment dict (round field on each assignment tells us
        # which round it belongs to). Cleared after aggregation completes.
        self._prev_round_assignments: dict[int, dict] = {}
        # Rounds queued for aggregation by POST /round. The aggregation loop
        # drains this set on each iteration so heavy aggregation never runs
        # on the FastAPI event loop.
        self._pending_aggregation_rounds: set[int] = set()
        # Wall-clock time when each round was queued by POST /round. Used
        # to bound the post-close grace window before stragglers are
        # scored 0.
        self._pending_aggregation_at: dict[int, float] = {}

        self._all_books: list[str] = []
        self._round_val_seeds: dict[int, int] = {}
        self._round_val_books: dict[int, frozenset[str]] = {}
        self._seed = seed

        # Track simulation progress (updated as data arrives)
        self._max_timestamp_ns: int = 0

        # S3: track processed gradient keys to avoid double-scoring
        self._processed_grad_keys: set[str] = set()
        # Negative cache: keys that returned 404/error — skip until next round
        self._failed_grad_keys: set[str] = set()
        # Per-miner boto3 S3 client cache. Constructing a boto3.client costs
        # ~50-100 ms (TLS handshake, signer load, endpoint resolution); under
        # the per-tick miner-gradient poll loop we'd otherwise build one per
        # miner per tick. Keyed by (endpoint, bucket, access_key) so a miner
        # rotating credentials yields a fresh client. Cleared on
        # `_refresh_miner_buckets` when the bucket map changes.
        self._miner_s3_clients: dict[tuple, Any] = {}

        # Price/volume scaling from simulator config (set from first state packet)
        self._price_scale: int | None = None
        self._vol_scale: int | None = None

        # Sim-time span used only to tail-flush a stalled book's partial page;
        # the row cap below is the primary flush trigger.
        self._parquet_interval_ns: int = parquet_interval_ns
        self._loop_sleep_s: float = loop_sleep_s
        self._round_grace_s: float = round_grace_s
        self._max_gradient_bytes: int = max_gradient_bytes
        # Scoring-model cache for eager scoring. Loaded on the first eager
        # score per round, reused for siblings, evicted at aggregation
        # boundary (because aggregation mutates the model in place). None
        # means "not loaded yet for this round".
        # Shape: {"round": int, "model": ..., "model_cfg": ..., "tokenizer_cfg": ...,
        #         "tokenizer": ..., "device": str}
        self._scoring_cache: dict | None = None
        # Per-round shared held-out baseline for single-pass scoring; evicted with
        # the scoring cache at aggregation boundaries.
        self._held_baseline_cache: dict[tuple, float] = {}
        self._loader_cache: dict[tuple, Any] = {}
        self._loader_cache_hits: int = 0
        self._loader_cache_misses: int = 0
        # Per-round data-fetch instrumentation (reset in _clear_scoring_cache).
        # Splits t_loader into its real drivers: S3 listing vs downloading.
        self._fetch_list_local: int = 0
        self._fetch_list_s3: int = 0
        self._fetch_list_s: float = 0.0
        self._fetch_file_hits: int = 0
        self._fetch_downloads: int = 0
        self._fetch_download_bytes: int = 0
        self._fetch_errors: int = 0
        self.overfit_ratio: float = overfit_ratio
        self._reject_untagged: bool = reject_untagged
        self.proposal_norm_ratio: float = proposal_norm_ratio
        # Reject a miner gradient whose L2 norm exceeds this. None disables;
        # set once the per-miner grad_norm log reveals the legitimate range.
        self.max_grad_norm: float | None = max_grad_norm
        # Scale the AGGREGATED delta (after compress) so its L2 norm is at
        # most this. None disables. Use when one valid-but-large gradient
        # dominates the proposal and pushes the apply over fp32-stability on
        # a fresh-init model — capping the magnitude preserves direction
        # while preventing forward-pass overflow. Distinct from
        # `max_grad_norm`, which rejects per-miner *inputs* outright.
        self.max_agg_norm: float | None = max_agg_norm
        # Element-wise clamp on the aggregated delta: every sparse value is
        # clamped to ±max_agg_element after compression. Catches single-
        # weight overshoots that the norm cap misses (a delta with low total
        # norm can still have one outlier element that breaks softmax).
        # Slightly direction-distorting at the saturating entries; preserves
        # the rest. None disables.
        self.max_agg_element: float | None = max_agg_element
        # Operator override: never wipe data/ on the startup sim-id bind (a pm2
        # restart resumes the existing data). A genuine rollover detected at
        # runtime still cleans up.
        self._no_startup_cleanup: bool = no_startup_cleanup
        self._pending_rows: dict[int, list[dict]] = {}  # book_id → rows
        self._pending_interval_start: dict[int, int] = {}  # book_id → interval start ts
        # Page size: a book flushes a parquet once its staging buffer reaches
        # this many rows, so active books emit uniform fixed-row pages. The
        # sim-time interval is demoted to a stalled-book tail flush. 0 disables
        # the row trigger (sim-time interval flush only). Parquet filenames are
        # content-range-based, so flushes are transparent to the dataloader.
        self._max_pending_rows_per_book: int = max(0, int(max_pending_rows_per_book))
        # Registry of written parquets (avoids S3 LIST for data-readiness checks)
        # book_id → [(filename, ts_start_ns, ts_end_ns), ...]
        self._written_parquets: dict[int, list[tuple[str, int, int]]] = {}
        # Page selection: recency-biased with a stale-revisit tail so untouched
        # or long-untouched pages still get covered. _page_last_round maps a
        # "book/filename" key to the round it was last assigned in.
        self._page_last_round: dict[str, int] = {}
        self._stale_revisit_frac: float = 0.2
        self._recency_window: int = 4
        # Books to keep out of the val pool (e.g. one identified as bad data).
        self._val_book_blocklist: set[str] = set()
        # Validator-pushed held-out split, keyed by the gradient's round (not
        # _agg_round, which advances before scoring). Falls back to local derivation.
        self._pushed_val_books: dict[int, frozenset] = {}
        # Per-book flush lock: prevents concurrent _flush_book_parquet calls for
        # the same book (cap-triggered flushes can race via two _process_tick
        # threads, causing duplicate S3 PUTs and over-deletion from the staging
        # buffer). Lazy-initialized via setdefault on first use.
        self._book_flush_locks: dict[int, threading.Lock] = {}
        # Per-book engine state persists across _process_tick calls
        self._engines: dict[int, Any] = {}
        self._order_sides: dict[int, dict[int, bool]] = {}
        self._last_ts: dict[int, int] = {}
        self._session_open_mid: dict[int, int | None] = {}
        # Local staging file for in-progress parquet buffer: persisted
        # periodically so a restart can continue filling the current window
        # rather than starting a fresh 5-min accumulation.
        # SCOPED BY MODE, not just by directory. checkpoint_path is .resolve()d above, so two
        # aggregators whose --checkpoint paths differ only by a symlink collapse to the SAME parent.
        # That happened: the exchange server restored the simulation's buffer ("Restored pending
        # buffer: 128 books, 165419 rows"), exchange rows were written into simulation books, and the
        # exchange mechanism never accumulated a page of its own. A directory split alone could not
        # prevent it; the filename must carry the mode. Empty mode keeps the legacy name so an older
        # invocation does not orphan an existing buffer.
        self._pending_staging_path: Path = self.checkpoint_path.parent / (
            f"pending_rows.{self._mode}.msgpack" if self._mode else "pending_rows.msgpack"
        )
        self._last_pending_save_ts: float = 0.0
        self._ingest_ticks: int = 0
        self._ingest_books: int = 0
        self._ingest_events: int = 0
        self._last_ingest_log: float = 0.0

        # S3 data cache: downloaded parquets are cached locally to avoid
        # re-downloading every aggregation cycle. Key: S3 key → local path.
        self._s3_cache_dir: Path | None = None
        self._s3_cached_files: dict[str, Path] = {}

        # Fresh-model flag: set when we create a seed checkpoint (no prior
        # training).  Enables warmup — first N rounds accept all gradients
        # and skip val rollback so the model can bootstrap from random weights.
        self._fresh_start = False
        # Anchors the warmup window to whatever _agg_round value we're
        # starting from. The gate must not key on `_agg_round < warmup_rounds`
        # alone, because a regime-incompatible restart restores _agg_round
        # from disk (often >> warmup_rounds) and the warmup would never fire.
        # Set in `start()` after state restore, when `_fresh_start` is True.
        self._warmup_anchor_round: int = -1

        # Lazy-loaded
        self._model = None
        self._global_val_loader = None
        # Fixed held-out eval: a frozen file list + loader, built once, evaluated
        # on the committed model every round to give a comparable CE curve (the
        # per-round forward-val loss is on rotating ranges, so it can't show a
        # trend). _fixed_val_files is the immutable file list captured at build.
        self._fixed_val_loader = None
        self._fixed_val_files: list | None = None

    @property
    def _in_warmup(self) -> bool:
        """True while the warmup window after a fresh start is still active."""
        return (
            self._fresh_start
            and self._warmup_anchor_round >= 0
            and (self._agg_round - self._warmup_anchor_round < self.warmup_rounds)
        )

    @property
    def _effective_min_score(self) -> float:
        """Min score threshold, disabled during warmup rounds after fresh start."""
        if self._in_warmup:
            return float("-inf")
        return self.min_score

    @property
    def _effective_rollback(self) -> bool:
        """Rollback check, disabled during warmup rounds after fresh start."""
        if self._in_warmup:
            return False
        return self.rollback

    def _discover_books(self) -> list[str]:
        """Discover available books from incoming state, S3, or filesystem.

        Sources in priority order — first non-empty wins:
          1. _written_parquets — definitive, set after first parquet flush.
          2. _pending_rows / _engines — books seen in state ticks but not yet
             flushed (lets us discover from tick #1, before the 5-min flush).
          3. data_store.list_books() — restart resilience: gradient server
             came back up against a populated S3 bucket.
          4. val_data_path filesystem scan — legacy local-only mode.

        Returns [] when nothing is known yet. Callers (_create_round_assignments,
        _create_assignment_for) early-return on empty, so the aggregator simply
        waits for state to arrive instead of inventing fake book IDs.

        Note: the live network is expected to run with ~128 books per validator
        with room to grow; nothing in this module hard-codes that count.
        """
        # 1. Flushed parquets (in-memory registry, no I/O)
        if self._written_parquets:
            return [str(bid) for bid in sorted(self._written_parquets.keys())]

        # 2. State ticks observed but not yet flushed
        if self._pending_rows:
            return [str(bid) for bid in sorted(self._pending_rows.keys())]

        # 3. S3 — restart against persistent bucket
        if self.validator_store is not None:
            try:
                book_ids = self.validator_store.list_books(self._validator_uid)
                if book_ids:
                    return [str(bid) for bid in book_ids]
            except Exception as exc:
                logger.debug("S3 book discovery failed: %s", exc)

        # 4. Local filesystem (legacy local-only mode)
        data_path = Path(self.val_data_path)
        if data_path.exists():
            books = [
                d.name
                for d in sorted(data_path.iterdir())
                if d.is_dir() and (d / "intervals").is_dir()
            ]
            if books:
                return books

        return []

    def _get_val_books(self, round_id: int | None = None) -> set[str]:
        if round_id is None:
            round_id = getattr(self, "_agg_round", 0)

        # Authoritative: the validator pushed the split for this round.
        pushed = self._pushed_val_books.get(round_id)
        if pushed is not None:
            return set(pushed)

        cached = self._round_val_books.get(round_id)
        if cached is not None:
            return set(cached)

        if not self._all_books:
            self._all_books = self._discover_books()
        if not self._all_books:
            return set()

        # Fallback derivation (validator didn't push a split for this round —
        # e.g. server-driven heartbeat-loss mode). Rotates per round like the
        # validator's, disjoint from that round's train (which the server
        # derives as the complement), so it stays self-consistent. Sorted so the
        # sample is order-stable. A book identified as bad data can be kept out
        # via _val_book_blocklist.
        # Exclude books actually assigned for training this round (when the
        # assignments are known) so a derived split can't hold out a book a
        # miner trained on. The authoritative pushed split guarantees this by
        # construction; a lost/late push must not silently reintroduce the
        # train/score overlap the held-out design removes.
        trained: set[str] = set()
        for _src in (self._assignments, self._prev_round_assignments):
            for _a in list(_src.values()):
                if _a.get("round") == round_id:
                    trained.update(str(b) for b in (_a.get("books") or []))
        _blocked = self._val_book_blocklist
        eligible = sorted(
            (b for b in self._all_books if b not in _blocked and str(b) not in trained),
            key=lambda b: int(b) if str(b).isdigit() else b,
        )
        if not eligible:
            # Every book was trained (or blocklisted) — can't hold out disjoint;
            # fall back to blocklist-only, then all books.
            eligible = sorted(
                (b for b in self._all_books if b not in _blocked),
                key=lambda b: int(b) if str(b).isdigit() else b,
            )
        if not eligible:
            eligible = sorted(self._all_books, key=lambda b: int(b) if str(b).isdigit() else b)
        n_val = max(1, int(len(eligible) * self.val_fraction))
        rng = random.Random(
            hashlib.sha256(f"{self._seed}:{round_id}:val".encode()).hexdigest()
        )
        val_set = set(rng.sample(eligible, min(n_val, len(eligible))))
        self._round_val_books[round_id] = frozenset(val_set)
        logger.info(
            "Val pool (round %d): %s (%d/%d, val_fraction=%.2f, rotating-fallback)",
            round_id,
            sorted(val_set),
            len(val_set),
            len(eligible),
            self.val_fraction,
        )
        return val_set

    def _get_train_books(self, round_id: int | None = None) -> list[str]:
        if not self._all_books:
            self._all_books = self._discover_books()
        val = self._get_val_books(round_id)
        return [b for b in self._all_books if b not in val]

    def update_max_timestamp(self, ts_ns: int) -> None:
        """Update the maximum observed timestamp (called as data arrives)."""
        if ts_ns > self._max_timestamp_ns:
            self._max_timestamp_ns = ts_ns

    def get_assignment(self, miner_uid: int) -> dict | None:
        """Return an assignment for a miner, only if data is ready.

        Creates an assignment on demand if one doesn't exist for this UID.
        The gradient server doesn't need to know miner UIDs upfront —
        any UID can request an assignment and get one created.

        Returns None if data is not ready yet (PENDING state).
        Returns the assignment dict when state is DATA_READY.
        Also transitions to DELIVERED and stamps `_delivered_at`.
        """
        a = self._assignments.get(miner_uid)

        # Create on demand if no assignment exists for this UID
        if a is None or a.get("round") != self._agg_round:
            self._create_assignment_for(miner_uid)

        # Retry page resolution for an assignment still waiting on data.
        a = self._assignments.get(miner_uid)
        if a is not None and a.get("_state") == "PENDING":
            self._apply_pages(a, miner_uid)

        a = self._assignments.get(miner_uid)
        if a is None:
            return None
        if a.get("_state") != "DATA_READY":
            return None

        # Transition to DELIVERED, stamp the delivery time. Used by the
        # heartbeat-loss fallback in _round_complete.
        a["_state"] = "DELIVERED"
        a["_delivered_at"] = time.time()
        logger.info(
            "[GTX] Assignment delivered: miner=%d round=%d books=%s ts=[%d,%d] data=%d files",
            miner_uid,
            a.get("round", 0),
            a.get("books", []),
            int(a.get("ts_start", 0)),
            int(a.get("ts_end", 0)),
            len(a.get("data", [])),
        )
        a["advice"] = {"top_k_frac": self._advise_top_k()}
        return {k: v for k, v in a.items() if not k.startswith("_")}

    def _advise_top_k(self) -> float | None:
        """Advise gradient top_k_frac from observed execution density: few trades =>
        sparse/weak gradient signal => keep MORE of the delta (bigger top_k). Returns
        None until enough events are seen (miner then falls back to its own default)."""
        ev = self._adv_events
        if ev < 5000:
            return None
        frac = self._adv_trades / ev
        lo, hi, ref = 0.10, 0.40, 0.01
        return round(max(lo, min(hi, lo + (1.0 - min(frac, ref) / ref) * (hi - lo))), 3)

    @staticmethod
    def _page_key(book_id, fname: str) -> str:
        return f"{book_id}/{fname}"

    def _select_books_for_miner(self, miner_uid: int) -> list[str] | None:
        """IID sample of train books for a miner (overlap allowed, not disjoint).

        Matches the validator's flavor-B policy so a miner served by this
        fallback path trains on the same kind of data draw.
        """
        train_books = self._get_train_books(self._agg_round)
        if not train_books:
            train_books = list(self._all_books)
        if not train_books:
            return None
        book_rng = random.Random(
            hashlib.sha256(f"{self._agg_round}:{miner_uid}:books".encode()).hexdigest()
        )
        k = min(self.books_per_miner, len(train_books))
        return book_rng.sample(train_books, k)

    def _resolve_pages(self, books: list[str], miner_uid: int):
        """Pick one fixed-row page per book and return (data_keys, ts_start, ts_end).

        Page choice is recency-biased (recent data) with a stale-revisit tail so
        untouched / long-untouched pages still get covered. Returns None if no
        page exists for any book yet (assignment stays PENDING). ts_start/ts_end
        span the chosen pages so the scorer can match held-out val data. Records
        each chosen page's assignment round for the stale-revisit logic.
        """
        page_rng = random.Random(
            hashlib.sha256(f"{self._agg_round}:{miner_uid}:page".encode()).hexdigest()
        )
        keys: list[str] = []
        spans: list[tuple[int, int]] = []
        for book_id in books:
            pages = self._written_parquets.get(int(book_id), [])
            if not pages:
                continue
            if page_rng.random() < self._stale_revisit_frac:
                # Least-recently-assigned page; never-assigned pages (-1) win.
                idx = min(
                    range(len(pages)),
                    key=lambda i: self._page_last_round.get(
                        self._page_key(book_id, pages[i][0]), -1
                    ),
                )
            else:
                # Recency: uniform over the most recent pages.
                lo = max(0, len(pages) - self._recency_window)
                idx = page_rng.randrange(lo, len(pages))
            fname, f_start, f_end = pages[idx]
            keys.append(f"data/{self._validator_uid}/{book_id}/intervals/{fname}")
            spans.append((f_start, f_end))
            self._page_last_round[self._page_key(book_id, fname)] = self._agg_round
        if not keys:
            return None
        return keys, min(s for s, _ in spans), max(e for _, e in spans)

    def _apply_pages(self, a: dict, miner_uid: int) -> bool:
        """Resolve pages onto a PENDING assignment. Returns True if DATA_READY."""
        sel = self._resolve_pages(a.get("books", []), miner_uid)
        if sel is None:
            return False
        a["data"], a["ts_start"], a["ts_end"] = sel
        a["_state"] = "DATA_READY"
        return True

    def _make_assignment_dict(self, books: list[str]) -> dict:
        """PENDING assignment skeleton; pages are resolved by _apply_pages."""
        return {
            "round": self._agg_round,
            "model_version": self._version,
            "books": books,
            "ts_start": 0,
            "ts_end": 0,
            "data": [],
            # THE SHARD THIS AGGREGATOR READS. A miner answers exactly one aggregator, and this is the
            # prefix its gradient must land under -- the gather path looks for
            # f"{self._bucket_prefix}gradients/{uid}/{round:08d}.grad" and nowhere else. Sending it
            # removes the miner's need to guess. A shard derived from the miner's own per-tick
            # trading mechanism is unrelated to which aggregator issued the work, so on a
            # dual-mechanism stack the gradient lands on the other shard and the round closes empty
            # while the miner logs a successful upload.
            "bucket_prefix": self._bucket_prefix,
            "mode": self._mode,
            "data_source": "s3" if self.validator_store else "local",
            "data_endpoint": self.validator_store.endpoint_url
            if self.validator_store
            else "",
            "data_bucket": self.validator_store.bucket if self.validator_store else "",
            "data_access_key": self.validator_store.access_key
            if self.validator_store
            else "",
            "data_secret_key": self.validator_store.secret_key
            if self.validator_store
            else "",
            "_state": "PENDING",
            "_created_at": time.time(),
            "_delivered_at": None,
            "_gradient_data": None,
            "_score": None,
        }

    def _create_round_assignments(self) -> None:
        """Create assignments for all miners for the current round.

        Called after aggregation completes (or at startup). Each miner gets
        one fixed-row page per assigned book, resolved at creation time.
        """
        self._all_books = self._discover_books()
        if not self._all_books:
            return

        # Number of miners — chain commitments first, then previous
        # assignments, then a default. _refresh_miner_buckets() handles the
        # empty-cache cooldown so this is cheap even when none have committed.
        self._refresh_miner_buckets()
        n_miners = max(len(self._miner_buckets), len(self._assignments), 2)

        created = ready = 0
        for miner_uid in range(n_miners):
            books = self._select_books_for_miner(miner_uid)
            if not books:
                continue
            a = self._make_assignment_dict(books)
            if self._apply_pages(a, miner_uid):
                ready += 1
            self._assignments[miner_uid] = a
            created += 1

        rolled = self._version > self._last_assigned_version
        self._last_assigned_version = self._version
        logger.info(
            "[GTX] round=%d: created %d assignments (%d data-ready, books_per_miner=%d%s)",
            self._agg_round,
            created,
            ready,
            self.books_per_miner,
            ", model rollover" if rolled else "",
        )

    def _create_assignment_for(self, miner_uid: int) -> None:
        """Create an assignment for a specific miner UID on demand."""
        self._all_books = self._discover_books()
        if not self._all_books:
            return
        books = self._select_books_for_miner(miner_uid)
        if not books:
            return
        a = self._make_assignment_dict(books)
        self._apply_pages(a, miner_uid)
        self._assignments[miner_uid] = a
        # Mark this version as assigned (idempotent: first assignment of
        # the round drives the rollover-bump check; on-demand creates
        # inherit the same value).
        self._last_assigned_version = max(self._last_assigned_version, self._version)
        logger.debug(
            "Assignment created on demand: miner=%d round=%d books=%s",
            miner_uid,
            self._agg_round,
            books,
        )

    def _round_estimate_s(self) -> float:
        """Server-side estimate of how long a round should take.

        Used only by the round-completion fallback in `_round_complete`,
        which fires when the validator stops pushing `POST /gentrx/round`.
        Derived from `blocks_per_round * block_time_s`. On a fresh
        checkpoint roll the estimate is bumped so miners have time to
        download the new model before uploading; the bump fires on the
        first round whose model_version advanced past the previous one.
        """
        base = self.blocks_per_round * self.block_time_s
        if self._version > self._last_assigned_version:
            return base * self._round_rollover_mult
        return base

    def _check_data_readiness(self) -> None:
        """Move PENDING assignments to DATA_READY once their pages exist."""
        # Snapshot: this runs on the aggregation thread while the event-loop
        # thread (POST /round rollover, get_assignment) can pop/insert keys —
        # iterating the live dict would raise "changed size during iteration".
        for uid, a in list(self._assignments.items()):
            if a.get("_state") != "PENDING":
                continue
            if self._apply_pages(a, uid):
                logger.debug(
                    "Assignment data ready: miner=%d round=%d files=%d",
                    uid,
                    a["round"],
                    len(a.get("data", [])),
                )

    def _get_max_data_timestamp(self) -> int:
        """Maximum sim timestamp (ns) we've observed.

        In-memory `_max_timestamp_ns` is the source of truth — it's updated by
        every tick processed in `_process_tick`. Only when we have nothing
        in-memory (fresh server, no state yet) do we consult S3 / filesystem
        for restart resilience. Stale data in S3 must not poison the live
        timestamp once state starts flowing.
        """
        if self._max_timestamp_ns > 0:
            return self._max_timestamp_ns

        if not self._all_books:
            self._all_books = self._discover_books()
        if not self._all_books:
            return 0

        max_ns = 0

        if self.validator_store is not None:
            for book_id in self._all_books[:3]:
                try:
                    filenames = self.validator_store.list_data(
                        self._validator_uid, book_id=int(book_id)
                    )
                    for fname in filenames:
                        parts = Path(fname).stem.split("-")
                        if len(parts) == 2:
                            try:
                                end_ns = _tag_to_ns(parts[1])
                                max_ns = max(max_ns, end_ns)
                            except ValueError:
                                pass
                except Exception:
                    continue
        else:
            data_path = Path(self.val_data_path)
            for book_id in self._all_books[:3]:
                book_dir = data_path / book_id / "intervals"
                if not book_dir.is_dir():
                    continue
                for f in book_dir.glob("*.parquet"):
                    parts = f.stem.split("-")
                    if len(parts) == 2:
                        try:
                            end_ns = _tag_to_ns(parts[1])
                            max_ns = max(max_ns, end_ns)
                        except ValueError:
                            pass

        return max_ns

    def _collect_val_files(self) -> list[Path]:
        """Collect all parquet files from val books.

        Uses the global val book split. Reads all available data for val books
        from S3 or local filesystem.
        """
        val_books = self._get_val_books(self._agg_round)
        if not val_books:
            return []

        files = []
        for book_id in val_books:
            book_files = self._get_book_files(book_id)
            files.extend(book_files)

        logger.info(
            "Val files: %d files from %d val books",
            len(files),
            len(val_books),
        )
        return files

    # ------------------------------------------------------------------
    # S3 data loading
    # ------------------------------------------------------------------

    def _get_s3_cache_dir(self) -> Path:
        """Local mirror of validator parquets.

        Deterministic path under output_path.parent so restarts reuse the
        same directory instead of leaking a fresh tempdir each launch.
        """
        if self._s3_cache_dir is None:
            self._s3_cache_dir = self.output_path.parent / "s3_cache"
            logger.info("S3 data cache: %s", self._s3_cache_dir)
        self._s3_cache_dir.mkdir(parents=True, exist_ok=True)
        return self._s3_cache_dir

    def _fetch_s3_book_files(
        self,
        book_id: str,
        ts_start: int = 0,
        ts_end: int = 0,
        limit: int = 0,
    ) -> list[Path]:
        """List and download parquets for a book from S3, with timestamp filtering.

        Files are cached locally — only downloaded once per session.
        Returns list of local Path objects ready for OrderDataset.
        """
        if self.validator_store is None:
            return []

        # Scoring only ever reads this validator's own data shard, and every
        # file this process flushes is recorded in _written_parquets (cold
        # starts rebuild it from S3 once). So the local record is a complete
        # index — use it and skip the per-build S3 LIST entirely. Fall back to
        # list_data only for books with no local record.
        known = self._written_parquets.get(int(book_id))
        if known:
            filenames = [fname for fname, _, _ in known]
            self._fetch_list_local += 1
        else:
            _t0 = time.time()
            try:
                filenames = self.validator_store.list_data(
                    self._validator_uid, book_id=int(book_id)
                )
            except Exception as exc:
                self._fetch_errors += 1
                logger.debug("S3 list_data failed for book %s: %s", book_id, exc)
                return []
            self._fetch_list_s3 += 1
            self._fetch_list_s += time.time() - _t0

        if not filenames:
            return []

        # Build Path-like objects for timestamp filtering
        # _filter_by_timestamp works on Path objects (uses .stem)
        pseudo_paths = [Path(f) for f in filenames]
        if ts_start or ts_end:
            pseudo_paths = _filter_by_timestamp(pseudo_paths, ts_start, ts_end)

        # Keep the most recent files when capped (freshest data).
        if limit and len(pseudo_paths) > limit:
            pseudo_paths = sorted(pseudo_paths, key=lambda p: p.name)[-limit:]

        # Download each file to local cache
        cache_dir = self._get_s3_cache_dir()
        local_files = []
        for pp in pseudo_paths:
            fname = pp.name
            cache_key = f"{book_id}/{fname}"

            if cache_key in self._s3_cached_files:
                self._fetch_file_hits += 1
                local_files.append(self._s3_cached_files[cache_key])
                continue

            local_path = cache_dir / str(book_id) / "intervals" / fname
            if local_path.is_file() and local_path.stat().st_size > 0:
                # Warm cache from a prior process: file already on disk
                # under the deterministic cache dir. Register and skip download.
                self._fetch_file_hits += 1
                self._s3_cached_files[cache_key] = local_path
                local_files.append(local_path)
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = self.validator_store.get_data(
                    self._validator_uid, book_id=int(book_id), filename=fname
                )
                local_path.write_bytes(data)
                self._fetch_downloads += 1
                self._fetch_download_bytes += len(data)
                self._s3_cached_files[cache_key] = local_path
                local_files.append(local_path)
            except Exception as exc:
                self._fetch_errors += 1
                logger.warning(
                    "S3 download failed: book=%s file=%s: %s", book_id, fname, exc
                )

        return local_files

    def _get_book_files(
        self,
        book_id: str,
        ts_start: int = 0,
        ts_end: int = 0,
        limit: int = 0,
    ) -> list[Path]:
        """Get parquet files for a book — from S3 if data_store is set, else filesystem."""
        if self.validator_store is not None:
            return self._fetch_s3_book_files(book_id, ts_start, ts_end, limit=limit)

        # Filesystem fallback
        data_path = Path(self.val_data_path)
        book_dir = data_path / str(book_id) / "intervals"
        if not book_dir.is_dir():
            return []
        book_files = sorted(book_dir.glob("*.parquet"))
        if ts_start or ts_end:
            book_files = _filter_by_timestamp(book_files, ts_start, ts_end)
        if limit and len(book_files) > limit:
            book_files = book_files[-limit:]
        return book_files

    def _fetch_one_data_file(self, book_id: str, fname: str) -> Path | None:
        """Resolve one page to a local file, downloading + caching if needed."""
        cache_key = f"{book_id}/{fname}"
        if cache_key in self._s3_cached_files:
            return self._s3_cached_files[cache_key]

        if self.validator_store is not None:
            local_path = (
                self._get_s3_cache_dir() / str(book_id) / "intervals" / fname
            )
            if local_path.is_file() and local_path.stat().st_size > 0:
                self._s3_cached_files[cache_key] = local_path
                return local_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = self.validator_store.get_data(
                    self._validator_uid, book_id=int(book_id), filename=fname
                )
                local_path.write_bytes(data)
                self._s3_cached_files[cache_key] = local_path
                return local_path
            except Exception as exc:
                logger.warning(
                    "S3 download failed: book=%s file=%s: %s", book_id, fname, exc
                )
                return None

        p = Path(self.val_data_path) / str(book_id) / "intervals" / fname
        return p if p.is_file() else None

    def _get_files_from_keys(self, data_keys: list[str]) -> list[Path]:
        """Resolve assignment data keys to local files.

        Keys are `data/<uid>/<book_id>/intervals/<fname>`.
        """
        files: list[Path] = []
        for key in data_keys:
            parts = key.split("/")
            if len(parts) < 5:
                continue
            local = self._fetch_one_data_file(parts[-3], parts[-1])
            if local is not None:
                files.append(local)
        return files

    # ------------------------------------------------------------------
    # DataLoader builders (use S3 or filesystem transparently)
    # ------------------------------------------------------------------

    def _forward_val_ranges(self, ranges):
        """Forward/temporal scoring: map each trained window [s,e] to the
        FUTURE window (e, latest] — score a gradient on data that arrived AFTER the
        miner's deadline. Contamination-proof and method-agnostic. Falls back to the
        trained window when no future data has arrived yet, or when disabled."""
        if not self._forward_scoring:
            return ranges
        fwd_end = int(self._max_timestamp_ns or 0)
        out = []
        for s, e in ranges:
            e = int(e or 0)
            if e > 0 and fwd_end > e:
                out.append((e, fwd_end))
            else:
                out.append((int(s or 0), e))
        return out

    @staticmethod
    def _held_baseline_key(grad_round: int, version: int, fwd_ranges):
        """Cache key for a held-out baseline loss.

        MUST be keyed on the FORWARD ranges the loader was actually built from
        (see _build_val_loader_for_ranges, which keys its loader cache the same
        way), NOT the raw training ranges. _forward_val_ranges maps [s,e] to
        (e, _max_timestamp_ns], and _max_timestamp_ns advances as data streams
        in, so keying on raw ranges would collide two distinct loaders onto one
        cached baseline and score a miner across mismatched held-out data.
        """
        return (grad_round, version, tuple(sorted((int(s), int(e)) for s, e in fwd_ranges)))

    def _build_val_loader_for_ranges(
        self,
        ts_ranges: list[tuple[int, int]],
        tokenizer,
        device: str,
        round_id: int | None = None,
    ):
        """Build a DataLoader from val books for specific timestamp ranges.

        For each time range that miners trained on, load the held-out val books
        that overlap. Tests whether the aggregated gradient generalizes to
        unseen books in the same time periods. `round_id` selects that round's
        held-out split (the split rotates), defaulting to the aggregating round.
        """
        from torch.utils.data import DataLoader

        from GenTRX.src.dataloader import OrderDataset, ChunkSampler

        if not ts_ranges:
            return None

        if round_id is None:
            round_id = self._agg_round

        cache_key = (
            "val", round_id, tuple(sorted((int(s), int(e)) for s, e in ts_ranges))
        )
        cached = self._loader_cache.get(cache_key)
        if cached is not None:
            self._loader_cache_hits += 1
            return cached
        self._loader_cache_misses += 1

        val_books = self._get_val_books(round_id)
        if not val_books:
            return None

        files = []
        remaining = self.max_loader_files
        for ts_start, ts_end in ts_ranges:
            for book_id in val_books:
                lim = max(0, remaining) if self.max_loader_files else 0
                book_files = self._get_book_files(book_id, ts_start, ts_end, limit=lim)
                files.extend(book_files)
                if self.max_loader_files:
                    remaining -= len(book_files)
                    if remaining <= 0:
                        break
            if self.max_loader_files and remaining <= 0:
                break

        # Deduplicate (multiple ranges may overlap the same files)
        seen = set()
        unique_files = []
        for f in files:
            key = str(f)
            if key not in seen:
                seen.add(key)
                unique_files.append(f)

        if not unique_files:
            # Fallback: the forward window may lead all flushed val pages early
            # in a run (data still catching up). Rather than drop held scoring,
            # score on each val book's most recent pages — still a clean holdout
            # (val books are never assigned for training), just not forward.
            seen = set()
            for book_id in val_books:
                for f in self._get_book_files(book_id, 0, 0, limit=8):
                    key = str(f)
                    if key not in seen:
                        seen.add(key)
                        unique_files.append(f)
            if unique_files:
                logger.warning(
                    "  Val loader: no files in ranges; fell back to %d recent "
                    "val pages from %d books — a data-readiness issue if this "
                    "recurs (val books should have flushed pages before scoring)",
                    len(unique_files), len(val_books),
                )

        if not unique_files:
            logger.info("  No val files found for %d time ranges", len(ts_ranges))
            return None

        logger.info(
            "  Val loader: %d files from %d ranges, %d val books",
            len(unique_files),
            len(ts_ranges),
            len(val_books),
        )
        ds = OrderDataset(unique_files, seq_len=self.seq_len, tokenizer=tokenizer, max_cached=2)
        # num_workers=0: in-process iteration. With num_workers>0 + persistent_workers=True,
        # each cached loader forked 2 ~1GB worker processes that _loader_cache.clear()
        # didn't terminate, accumulating hundreds of zombies and 10+ GB RAM over a few
        # hours. Single-process iteration is slower per-batch but bounds RAM.
        loader = DataLoader(
            ds,
            batch_size=64,
            sampler=ChunkSampler(ds, shuffle=False),
            num_workers=0,
            pin_memory=False,
        )
        self._loader_cache[cache_key] = loader
        return loader

    def _log_fixed_val(self, model, tokenizer, device: str) -> None:
        """Evaluate the committed model on a frozen held-out set and log it.

        The file list is captured once (held-out val books, oldest pages first
        so they persist) and reused every round, so the logged CE is comparable
        across versions. Best-effort: a build/eval failure logs and returns.
        """
        from torch.utils.data import DataLoader

        from GenTRX.src.dataloader import OrderDataset, ChunkSampler
        from GenTRX.src.distributed import _eval_loss_per_field

        try:
            if self._fixed_val_loader is None:
                val_books = self._get_val_books(self._agg_round)
                files: list = []
                for book_id in sorted(val_books):
                    files.extend(self._get_book_files(book_id, 0, 0))
                # Oldest pages first: stable across the run (new pages append at
                # the tail, pruning hits the tail's local cache not these).
                files = sorted({str(f): f for f in files}.values(), key=str)[:8]
                if not files:
                    return
                self._fixed_val_files = files
                ds = OrderDataset(files, seq_len=self.seq_len, tokenizer=tokenizer, max_cached=2)
                self._fixed_val_loader = DataLoader(
                    ds,
                    batch_size=64,
                    sampler=ChunkSampler(ds, shuffle=False),
                    num_workers=0,
                    pin_memory=False,
                )
                logger.info(
                    "[GTX] fixed_val set: %d frozen files from %d held-out books",
                    len(files), len(val_books),
                )

            loss, per_field = _eval_loss_per_field(
                model, self._fixed_val_loader, device,
                self.max_val_batches, self.label_smooth_sigma,
            )
            logger.info(
                "[GTX] fixed_val v=%d loss=%.4f per_field=%s",
                self._version, loss, {k: round(v, 4) for k, v in per_field.items()},
            )
            self._log_event({
                "type": "fixed_val",
                "round": self._agg_round,
                "version": self._version,
                "loss": loss,
                **{f"fixed_val_{k}": v for k, v in per_field.items()},
            })
        except Exception as exc:
            logger.warning("[GTX] fixed_val eval failed: %s", exc)

    @property
    def version(self) -> int:
        return self._version

    def _log_event(self, event: dict) -> None:
        """Append a JSON event to the aggregation log.

        Every event is stamped with `sim_id` (from the simulator's config) and
        `sim_epoch` (incremented when sim time resets). Lets the dashboard
        separate runs that would otherwise blend together because version +
        round both reset on each gradient server restart.

        sim_id is None until the first state packet arrives — events emitted
        before that (e.g. the run_start marker) carry sim_id=null and are
        bucketed together by the dashboard.

        Also mirrors to wandb when `--wandb-project` is set (or
        `WANDB_PROJECT` env var). Soft no-op otherwise.
        """
        event["timestamp"] = time.time()
        event.setdefault("sim_id", self._sim_id)
        event.setdefault("sim_epoch", self._sim_epoch)
        self._rotate_log_if_large()
        with open(self.log_path, "a") as f:
            f.write(_json.dumps(event) + "\n")
        # Mirror aggregation events into the snapshot dict + counters so
        # the validator-side Prometheus collector can read them off the
        # next /gentrx/scores payload.
        if event.get("type") == "aggregation":
            self._last_aggregation.update(
                {
                    k: v
                    for k, v in event.items()
                    if k
                    in (
                        "round",
                        "n_assigned",
                        "n_delivered",
                        "n_collected",
                        "n_version_mismatched",
                        "n_scored",
                        "n_accepted",
                        "loss_before",
                        "loss_after",
                        "loss_improvement_pct",
                        "t_proposal_eval_s",
                        "t_save_ckpt_s",
                        "t_loader_build_s",
                        "grad_norm_mean",
                        "grad_norm_min",
                        "grad_norm_max",
                        "grad_norm_median",
                        "grad_norm_std",
                        "overlap_pairs_checked",
                        "overlap_pairs_high",
                        "overlap_mean",
                        "overlap_max",
                        "loader_cache_hits",
                        "loader_cache_misses",
                        "loader_cache_hit_rate",
                        "fetch_list_local",
                        "fetch_list_s3",
                        "fetch_list_s",
                        "fetch_file_hits",
                        "fetch_downloads",
                        "fetch_download_mb",
                        "fetch_errors",
                        "proposals_evaluated",
                        "proposals_skipped",
                        "rolled_back",
                        "sibling_only",
                        "version",
                    )
                    or k.startswith("per_field_loss_")
                }
            )
            if event.get("n_accepted", 0) > 0 and not event.get("rolled_back"):
                self._rounds_aggregated_total += 1
            if event.get("rolled_back"):
                self._rollbacks_total += 1
            # Only rounds that actually made a rollback decision count toward
            # the rate; no-accepted and sibling-only paths skip the gate.
            if "rolled_back" in event and not event.get("sibling_only"):
                self._rollback_history.append(bool(event["rolled_back"]))
                if self._rollback_history:
                    last10 = list(self._rollback_history)[-10:]
                    self._last_aggregation["rollback_rate_10w"] = sum(last10) / len(last10)
                    self._last_aggregation["rollback_rate_50w"] = (
                        sum(self._rollback_history) / len(self._rollback_history)
                    )
        try:
            from GenTRX.src import wandb_ops

            wandb_ops.log_event(event)
        except Exception:
            pass  # logging must never break the caller

    def _rotate_log_if_large(
        self, max_bytes: int = 50 * 1024 * 1024, keep: int = 5
    ) -> None:
        """Size-based rotation for the JSONL audit log.

        Written via raw `open(..., "a")`, so RotatingFileHandler doesn't
        apply. Check size before each append. After rotation the file
        does not exist; the next `open("a")` creates it fresh.
        """
        try:
            if not self.log_path.exists():
                return
            if self.log_path.stat().st_size < max_bytes:
                return

            def _backup(i: int) -> Path:
                return self.log_path.with_name(f"{self.log_path.name}.{i}")

            oldest = _backup(keep)
            if oldest.exists():
                oldest.unlink()
            for i in range(keep - 1, 0, -1):
                src = _backup(i)
                if src.exists():
                    src.rename(_backup(i + 1))
            self.log_path.rename(_backup(1))
        except Exception:
            pass  # rotation must never break the caller

    def _sync_from_uid0(self) -> bool:
        """Download the latest checkpoint from uid 0's chain-committed bucket.

        Called at startup (bootstrap) and at the start of each aggregation round
        (model sync).  Keeps sibling validators and their scoring aligned with
        the canonical model published by the aggregator.

        Returns True if a new checkpoint was downloaded, False otherwise.
        No-op if this server IS the aggregator, chain is not set, uid 0 has
        no bucket commitment, or the latest version is not newer than local.
        """
        if self.is_aggregator:
            return False  # we ARE uid 0 — we publish, not pull
        try:
            from GenTRX.src.chain import BucketInfo
            from GenTRX.src.gradient_store import GradientStore

            bucket_info = None
            if self._chain is not None:
                bucket_info = self._chain.get_bucket(0)

            if bucket_info is None:
                bucket_info = BucketInfo.from_aggregator_env()
                if bucket_info is not None:
                    logger.info(
                        "uid-0 chain bucket unavailable; using "
                        "GENTRX_AGGREGATOR_S3_* env override"
                    )

            if bucket_info is None:
                logger.debug("No uid-0 bucket available; skipping sync")
                return False

            store = GradientStore(
                endpoint_url=bucket_info.endpoint_url,
                bucket=bucket_info.bucket_name,
                access_key=bucket_info.access_key_id,
                secret_key=bucket_info.secret_access_key,
                region=bucket_info.region,
                prefix=self._bucket_prefix,
            )
            version = store.get_latest_version(0)
            if version <= 0:
                logger.debug("uid-0 bucket has no checkpoint yet")
                return False
            if version <= self._version:
                return False  # already current
            try:
                uid0_meta = store.get_latest_meta(0)
                if int(uid0_meta.get("train_regime_version", 0)) < TRAIN_REGIME_VERSION:
                    logger.warning(
                        "uid-0 checkpoint v%d predates train regime %d — "
                        "not syncing onto a newer-regime sibling",
                        version,
                        TRAIN_REGIME_VERSION,
                    )
                    return False
            except Exception as exc:
                logger.warning("Could not read uid-0 regime stamp: %s", exc)

            logger.info(
                "Syncing from uid-0 checkpoint v%d → v%d at %s/%s",
                self._version,
                version,
                bucket_info.endpoint_url,
                bucket_info.bucket_name,
            )
            ckpt_bytes = store.get_checkpoint(0, version)
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_bytes(ckpt_bytes)
            self._version = version
            logger.info(
                "Model sync complete: checkpoint v%d written to %s",
                version,
                self.checkpoint_path,
            )
            return True
        except Exception as exc:
            logger.warning("uid-0 sync failed: %s", exc)
            return False

    def _bootstrap_from_own_bucket(self) -> bool:
        """Aggregator-only: pull our own bucket head into checkpoint_path at boot.

        `_sync_from_uid0` is a publish-side no-op for the aggregator, so when the
        local seed is missing or was quarantined (wrong n_types) the aggregator
        would fresh-init and then only *version-track* the bucket head — the
        "Resumed vN" branch never reloads weights — so scoring would run on a
        fresh (random) model and the next aggregation would publish fresh-derived
        weights over a trained head. This self-heals that: download the head and
        adopt it as the local scoring checkpoint, but ONLY when it is
        architecture-compatible (n_types matches the build). An incompatible head
        is left untouched so the fresh-init + rebaseline path handles it exactly
        as before.

        Bootstrap-only (called when checkpoint_path is absent); the per-round
        sibling sync stays disabled for the aggregator.
        """
        if not self.is_aggregator or self.validator_store is None:
            return False
        try:
            import io

            import torch

            from GenTRX.src.model import ModelConfig as _MC

            version = self.validator_store.get_head_version(self._validator_uid)
            if version <= 0:
                version = self.validator_store.get_latest_existing_version(
                    self._validator_uid
                )
            if version <= 0:
                return False
            data = self.validator_store.get_checkpoint(self._validator_uid, version)
            ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            head_n_types = int((ckpt.get("model_config") or {}).get("n_types", -1))
            if head_n_types != int(_MC().n_types):
                logger.warning(
                    "Aggregator bootstrap: bucket head v%d has n_types=%d, build "
                    "uses n_types=%d — not pulling; fresh-init will handle it.",
                    version,
                    head_n_types,
                    _MC().n_types,
                )
                return False
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_bytes(data)
            self._version = version
            self._fresh_start = False
            logger.info(
                "Aggregator bootstrap: pulled bucket head v%d into %s "
                "(n_types=%d) — scoring resumes from the trained baseline "
                "instead of a fresh model.",
                version,
                self.checkpoint_path,
                head_n_types,
            )
            return True
        except Exception as exc:
            logger.warning("Aggregator bootstrap pull failed: %s", exc)
            return False

    def start(self) -> None:
        """Start the background aggregation thread.

        On first start, uploads the initial checkpoint to S3 so agents can
        bootstrap without needing a local checkpoint file.
        """
        if self._running:
            return

        # Auto-normalize the local seed so a deploy to any box needs no manual
        # re-stamp: a current-architecture seed with an old/absent regime stamp
        # is re-stamped to the current regime; an architecture-incompatible seed
        # (wrong n_types) is quarantined so the fresh-init below produces a
        # usable model instead of crash-trapping on the first gradient.
        from GenTRX.src.model import ModelConfig as _MC

        _seed_action = normalize_seed_checkpoint(
            self.checkpoint_path,
            current_n_types=_MC().n_types,
            current_regime=TRAIN_REGIME_VERSION,
            label_smooth_sigma=self.label_smooth_sigma,
        )
        if _seed_action == "quarantined":
            self._fresh_start = True

        # Ensure a checkpoint exists.
        # Priority: (1) local file, (2) bucket head, (3) fresh model.
        if not self.checkpoint_path.exists():
            # Siblings pull uid-0's chain-committed head; the aggregator pulls
            # its own bucket head (its _sync_from_uid0 is a publish-side no-op),
            # so a missing/quarantined seed self-heals to the trained baseline
            # instead of scoring on a fresh-init model.
            if not self._sync_from_uid0():
                self._bootstrap_from_own_bucket()

        if not self.checkpoint_path.exists():
            logger.info(
                "No checkpoint at %s — initializing fresh model", self.checkpoint_path
            )
            try:
                from dataclasses import asdict

                import torch

                from GenTRX.src.model import ModelConfig, OrderModel
                from GenTRX.src.tokenizer import TokenizerConfig

                model_cfg = ModelConfig()
                tokenizer_cfg = TokenizerConfig()
                model = OrderModel(model_cfg)
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": asdict(model_cfg),
                        "tokenizer_config": asdict(tokenizer_cfg),
                        "epoch": 0,
                        "step": 0,
                        "loss": float("inf"),
                        **checkpoint_stamp(self.label_smooth_sigma),
                    },
                    self.checkpoint_path,
                )
                self._fresh_start = True
                logger.info(
                    "Fresh model checkpoint saved: %s (warmup=%d rounds)",
                    self.checkpoint_path,
                    self.warmup_rounds,
                )
            except Exception as exc:
                logger.error("Failed to initialize fresh model: %s", exc)

        # Upload initial checkpoint to aggregator bucket so agents can bootstrap.
        # On restart, scan for the highest existing version rather than always
        # resetting to v1 (which prune would immediately delete, leaving
        # latest.json pointing at a non-existent file).
        if self.validator_store is not None and self.checkpoint_path.exists():
            try:
                existing_version = self.validator_store.get_latest_existing_version(
                    self._validator_uid
                )
                if existing_version > 0:
                    # Bucket already has checkpoints — resume from the highest,
                    # unless its training regime predates ours. A regime bump
                    # (e.g. strict→soft CE) makes the old model's scores
                    # incomparable, so we keep its weights as init but force
                    # warmup and drop the data/ collected under the old regime.
                    # Reconcile against the head pointer: checkpoints/ only holds
                    # baselines (every checkpoint_interval versions), so the head
                    # can be ahead of the latest baseline file. The local
                    # output_path checkpoint holds those head weights on a
                    # same-host restart.
                    head_version = self.validator_store.get_head_version(
                        self._validator_uid
                    )
                    self._version = max(existing_version, head_version)
                    if self._regime_incompatible():
                        # A train-regime bump (e.g. strict→soft CE) makes the old
                        # model's scores incomparable, so re-warmup from the
                        # existing weights. The order data is regime-independent,
                        # so KEEP it — do not wipe data/ (that also stops a
                        # missing/old stamp from nuking data on every restart).
                        self._fresh_start = True
                        logger.warning(
                            "Existing checkpoint v%d predates train regime %d — "
                            "keeping weights as init and forcing %d warmup round(s); "
                            "data/ retained (regime-independent)",
                            self._version,
                            TRAIN_REGIME_VERSION,
                            self.warmup_rounds,
                        )
                        # Publish the local current-format checkpoint as a new
                        # version NOW so miners/siblings bootstrap from a
                        # compatible model. Without this the published pointer
                        # keeps serving the stale incompatible checkpoint: an
                        # n_types=5 miner loads an n_types=3 model, crashes in
                        # compute_loss, never submits a gradient, so aggregation
                        # (which would otherwise re-publish) never runs — the
                        # bucket can't self-heal and the whole round deadlocks.
                        self._rebaseline_incompatible_bucket()
                    else:
                        logger.info(
                            "Resumed from existing checkpoint v%d in aggregator bucket",
                            self._version,
                        )
                else:
                    # Fresh bucket — upload local checkpoint as v1.
                    data = self.checkpoint_path.read_bytes()
                    self._version = 1
                    self.validator_store.put_checkpoint(
                        self._validator_uid, self._version, data,
                        meta=checkpoint_stamp(self.label_smooth_sigma),
                    )
                    logger.info(
                        "Initial checkpoint uploaded to aggregator (v%d, %.1f MB)",
                        self._version,
                        len(data) / 1e6,
                    )
                    self._prune_checkpoints()
            except Exception as exc:
                logger.error("Failed to upload initial checkpoint: %s", exc)

        # Restore in-progress parquet buffer from local staging file so a
        # restart continues filling the current window, not a fresh one.
        # Skipped only when a runtime sim-rollover queued a cleanup; regime
        # changes alone retain prior staging data.
        if not self._data_cleanup_pending:
            self._restore_pending_rows()
            # Restore written-parquet registry from S3 so a restart doesn't
            # force a full 5-min re-accumulation before assignments can be created.
            if self.validator_store is not None:
                self._restore_written_parquets()

        # A full crash can leave _agg_round stale — either at 0 (no local round
        # to restore) or at a partial value if the crash hit after an increment
        # but before the next persistence. The round is block-derived
        # (round = block // blocks_per_round, sim-independent), so cross-check
        # against the chain and snap to it if the gap exceeds warmup_rounds.
        # Without the drift catch, a stale round silently rejects every miner
        # gradient as version-mismatched once live miners advance past us.
        # Chain failure is non-fatal.
        if self._chain is not None and self.blocks_per_round > 0:
            try:
                block = int(self._chain.subtensor.get_current_block())
                seeded = block // self.blocks_per_round
                drift = seeded - self._agg_round
                if self._agg_round == 0 and seeded > 0:
                    self._agg_round = seeded
                    logger.info(
                        "Seeded agg_round=%d from chain block %d (no local round to restore)",
                        seeded,
                        block,
                    )
                elif seeded > 0 and abs(drift) > self.warmup_rounds:
                    logger.warning(
                        "Reseeding agg_round %d → %d from chain (drift=%d > warmup_rounds=%d; "
                        "stale local round would reject live miner gradients)",
                        self._agg_round,
                        seeded,
                        drift,
                        self.warmup_rounds,
                    )
                    self._agg_round = seeded
            except Exception as exc:
                logger.warning("Could not seed agg_round from chain: %s", exc)

        # Anchor the warmup window. After state restore + chain reseed,
        # `_agg_round` reflects the round we'll actually start aggregating
        # from — that's the moment to peg the warmup gate against, so
        # `warmup_rounds` rounds of accept-all/no-rollback fire from here
        # regardless of how high the restored counter is.
        if self._fresh_start:
            self._warmup_anchor_round = self._agg_round
            logger.info(
                "Warmup window anchored at round %d (warmup_rounds=%d)",
                self._warmup_anchor_round,
                self.warmup_rounds,
            )

        self._running = True
        self._thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self._thread.start()
        logger.info("Aggregation thread started (interval=%ds)", self.interval)
        # Drop a synthetic event so dashboards see the gradient-server-start
        # boundary even before any aggregation has happened. sim_id is null
        # at this point — it gets attached as soon as the first state packet
        # arrives, and a separate `sim_bind` event will mark the moment.
        self._log_event({"type": "server_start", "interval": self.interval})

    def _save_pending_rows(self) -> None:
        """Persist staging state. Skipped when _sim_id is None so files stay identity-paired."""
        if self._sim_id is None:
            return
        try:
            import msgpack as _mp

            payload = _mp.packb(
                {
                    "pending_rows": {
                        str(bid): rows for bid, rows in self._pending_rows.items()
                    },
                    "interval_start": {
                        str(bid): ts for bid, ts in self._pending_interval_start.items()
                    },
                    "last_ts": {str(bid): ts for bid, ts in self._last_ts.items()},
                    "max_ts": self._max_timestamp_ns,
                    "sim_id": self._sim_id,
                    "last_seen_sim_ts": self._last_seen_sim_ts,
                    "agg_round": self._agg_round,
                    "assignments": _serialize_assignments(dict(self._assignments)),
                    "prev_round_assignments": _serialize_assignments(
                        dict(self._prev_round_assignments)
                    ),
                },
                use_bin_type=True,
            )
            tmp = self._pending_staging_path.with_suffix(".tmp")
            tmp.write_bytes(payload)
            tmp.rename(self._pending_staging_path)
        except Exception as exc:
            logger.debug("pending rows persist failed: %s", exc)

    def _restore_pending_rows(self) -> None:
        """Restore in-progress parquet buffer from local disk after restart.

        Skipped if the staging file doesn't exist or is unreadable. The
        sim-id check in _process_tick (against `_restored_sim_id`) plus
        the backwards-time check (against `_max_timestamp_ns`) wipe the
        restored buffer if the sim was swapped between save and restore.
        """
        if not self._pending_staging_path.exists():
            return
        try:
            import msgpack as _mp

            from GenTRX.src.orderbook import MatchingEngine

            raw = _mp.unpackb(
                self._pending_staging_path.read_bytes(), raw=False, strict_map_key=False
            )
            pending = raw.get("pending_rows", {})
            interval_start = raw.get("interval_start", {})
            last_ts = raw.get("last_ts", {})
            max_ts = int(raw.get("max_ts", 0))
            self._restored_sim_id = raw.get("sim_id")
            self._last_seen_sim_ts = int(raw.get("last_seen_sim_ts", 0) or 0)

            for bid_str, rows in pending.items():
                bid = int(bid_str)
                self._pending_rows[bid] = rows
                self._pending_interval_start[bid] = int(interval_start.get(bid_str, 0))
                self._last_ts[bid] = int(last_ts.get(bid_str, 0))
                # Initialise engine placeholder so _process_tick's
                # "if book_id not in self._engines" branch does NOT fire
                # and overwrite our restored pending_rows with [].
                if bid not in self._engines:
                    self._engines[bid] = MatchingEngine()
                    self._order_sides[bid] = {}
                    self._session_open_mid[bid] = None

            if max_ts > self._max_timestamp_ns:
                self._max_timestamp_ns = max_ts

            restored_round = raw.get("agg_round")
            if isinstance(restored_round, int) and restored_round > 0:
                self._agg_round = restored_round

            _restore_assignment_dict(raw.get("assignments") or {}, self._assignments)
            _restore_assignment_dict(
                raw.get("prev_round_assignments") or {},
                self._prev_round_assignments,
            )
            # Any prior-round assignments restored from disk are owed a
            # scoring pass: the validator already closed that round and
            # moved on, so we re-queue it for the drain path.
            for a in self._prev_round_assignments.values():
                rnd = a.get("round")
                if isinstance(rnd, int):
                    self._pending_aggregation_rounds.add(rnd)
                    self._pending_aggregation_at.setdefault(rnd, time.time())

            total_rows = sum(len(r) for r in self._pending_rows.values())
            logger.info(
                "Restored pending buffer: %d books, %d rows, max_ts=%d, "
                "agg_round=%d, assignments=%d, prev_round_assignments=%d, "
                "restored_sim_id=%s",
                len(self._pending_rows),
                total_rows,
                max_ts,
                self._agg_round,
                len(self._assignments),
                len(self._prev_round_assignments),
                self._restored_sim_id,
            )
        except Exception as exc:
            logger.warning("pending rows restore failed: %s", exc)

    def _restore_written_parquets(self) -> None:
        """Scan S3 and rebuild _written_parquets from existing data/ parquets.

        Called once at startup so a restart doesn't force a fresh 5-min
        accumulation before the first assignment can be created.

        Bounded by a wall-clock budget (GENTRX_RESTORE_BUDGET_S, default 60s):
        list_books / list_data paginate every object under the shard, so a
        bucket that has grown large across a long run would otherwise hang
        startup here — before uvicorn even binds. Each S3 list runs in an
        abandonable daemon thread (it only reads and returns, so dropping it on
        timeout can't corrupt shared state); once the budget is spent we stop
        and let the un-scanned pages re-accumulate over the next window.
        """
        budget_s = float(os.environ.get("GENTRX_RESTORE_BUDGET_S", "60") or 60)
        t0 = time.monotonic()

        def _bounded(fn, timeout_s: float):
            """Return (value, timed_out). A timed-out worker is abandoned (daemon;
            reaped at process exit) and its result discarded — never mutates state."""
            box: dict = {}

            def _run():
                try:
                    box["v"] = fn()
                except Exception as exc:  # noqa: BLE001 - surfaced to caller below
                    box["e"] = exc

            th = threading.Thread(target=_run, daemon=True)
            th.start()
            th.join(max(0.0, timeout_s))
            if th.is_alive():
                return None, True
            if "e" in box:
                raise box["e"]
            return box.get("v"), False

        try:
            book_ids, timed_out = _bounded(
                lambda: self.validator_store.list_books(self._validator_uid), budget_s
            )
            if timed_out:
                logger.warning(
                    "parquet restore: list_books exceeded %.0fs budget — skipping "
                    "restore; pages re-accumulate over the next window",
                    budget_s,
                )
                return
        except Exception as exc:
            logger.warning("Could not list S3 books for parquet restore: %s", exc)
            return

        book_ids = book_ids or []
        restored = 0
        scanned = 0
        for bid in book_ids:
            remaining = budget_s - (time.monotonic() - t0)
            if remaining <= 0:
                logger.warning(
                    "parquet restore: %.0fs budget spent after %d/%d books "
                    "(%d parquet(s)) — remainder re-accumulates",
                    budget_s, scanned, len(book_ids), restored,
                )
                break
            try:
                fnames, timed_out = _bounded(
                    lambda b=bid: self.validator_store.list_data(self._validator_uid, b),
                    remaining,
                )
            except Exception as exc:
                logger.debug("list_data failed for book %s: %s", bid, exc)
                continue
            if timed_out:
                logger.warning(
                    "parquet restore: list_data(book=%s) exceeded remaining budget "
                    "— stopping; remainder re-accumulates",
                    bid,
                )
                break
            scanned += 1
            for fname in fnames or []:
                if not fname.endswith(".parquet"):
                    continue
                stem = fname[: -len(".parquet")]
                parts = stem.split("-")
                if len(parts) == 2:
                    try:
                        ts_min = _tag_to_ns(parts[0])
                        ts_max = _tag_to_ns(parts[1])
                    except Exception:
                        continue
                    if bid not in self._written_parquets:
                        self._written_parquets[bid] = []
                    self._written_parquets[bid].append((fname, ts_min, ts_max))
                    if ts_max > self._max_timestamp_ns:
                        self._max_timestamp_ns = ts_max
                    restored += 1

        marker = self._read_bucket_sim_marker()
        if marker is not None:
            self._bucket_sim_id = marker
        if restored:
            logger.info(
                "Restored %d parquet(s) across %d book(s) from S3 (bucket_sim_id=%s)",
                restored,
                len(self._written_parquets),
                self._bucket_sim_id,
            )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _aggregation_loop(self) -> None:
        """Event-driven aggregation loop.

        Round advancement is driven by the validator via POST /gentrx/round.
        This loop only handles the per-round work:
          1. Drain rounds queued for aggregation by POST /round
          2. Create assignments if none exist yet (needs data first)
          3. Check data readiness for PENDING assignments
          4. Collect gradients from miner buckets for DELIVERED assignments
          5. Aggregate when the current round is complete (all in or timer expired)
          6. Create next round's assignments after aggregation

        Polls every 5s — cheap when nothing is happening.
        """
        while self._running:
            try:
                # Sim transition cleanup: fires on the first loop tick after
                # an ESE marker (or a heuristic sim-restart) set the flag.
                # Wipes stale parquets from the validator bucket's data/
                # prefix; checkpoints and proposals are intentionally left
                # alone. Runs before the rest of the loop so a half-done
                # cleanup never overlaps a fresh assignment creation.
                if self._data_cleanup_pending:
                    self._run_data_cleanup()

                # Rolling eviction of the local parquet mirror. Throttled to
                # once per 5 min — the cost is a recursive directory scan,
                # cheap but not free.
                _now = time.time()
                if _now - self._last_s3_cache_prune > 300:
                    self._prune_s3_cache()
                    self._last_s3_cache_prune = _now

                # Tail flush for stalled / sparse books. Active books flush at
                # the row cap in _process_tick; this lands a partial page once a
                # book stops receiving rows (global sim clock advanced past its
                # last event by a full interval) so it does not linger. With the
                # cap disabled the sim-time interval is the primary trigger, so
                # fall back to the interval-start comparison.
                if self._max_timestamp_ns > 0:
                    _cap = self._max_pending_rows_per_book
                    # A ROBUST CLOCK, NOT THE MAXIMUM.
                    #
                    # "Has the sim advanced past this book" used self._max_timestamp_ns, which is a
                    # MAX over every book -- so one book with a bad timestamp defines the clock for
                    # all of them. Book 70 held 1530540000000000 (425 sim-hours,
                    # exchange-scale) while the other 128 sat at ~2.45e12 (0.68h). Every one of them
                    # was then 1,528,085s "behind" a 300s interval, so every book tail-flushed on
                    # every loop -- pages of 1-3 sim-seconds and ~217 rows against train_seq_len=512,
                    # which reads downstream as "no trainable pages" and no gradient, on a stack where
                    # nothing was actually wrong with training.
                    #
                    # The median cannot be moved by a single outlier, so one poisoned book can no
                    # longer stall the other 128. The outlier is reported rather than silently
                    # tolerated: it is a real symptom (here, exchange-scale time in a simulation
                    # aggregator) and hiding it would trade one silent failure for another.
                    _lasts = sorted(v for v in self._last_ts.values() if v > 0)
                    if _lasts:
                        _clock = _lasts[len(_lasts) // 2]
                        _hi = _lasts[-1]
                        if (_hi - _clock) >= self._parquet_interval_ns and not getattr(
                            self, "_clock_outlier_warned", False
                        ):
                            self._clock_outlier_warned = True
                            _who = [b for b, v in self._last_ts.items() if v == _hi]
                            logger.warning(
                                "book clock outlier: book(s) %s at ts=%d are %.0fs ahead of the "
                                "median book (%d). Using the median as the sim clock so one book "
                                "cannot stall the rest; investigate the source of that timestamp.",
                                _who, _hi, (_hi - _clock) / 1e9, _clock,
                            )
                    else:
                        _clock = self._max_timestamp_ns
                    for _bid in list(self._pending_rows):
                        if not self._pending_rows[_bid]:
                            continue
                        if _cap:
                            _last = self._last_ts.get(_bid, 0)
                            if _last > 0 and (
                                _clock - _last
                            ) >= self._parquet_interval_ns:
                                self._flush_book_parquet(_bid)
                        else:
                            _start = self._pending_interval_start.get(_bid, 0)
                            if _start > 0 and (
                                _clock - _start
                            ) >= self._parquet_interval_ns:
                                self._flush_book_parquet(_bid)

                # Drain any rounds queued by POST /gentrx/round. The HTTP
                # handler installs new assignments and advances _agg_round
                # synchronously, but leaves aggregation of the closing round
                # to us so the event loop never blocks on torch work.
                if self._pending_aggregation_rounds:
                    # Give late-arriving gradients a final collection pass
                    # BEFORE aggregating. Without this, a miner whose
                    # upload lands after POST /round for the next round
                    # gets scored 0 even though the gradient is already
                    # in S3. Collection iterates both _assignments and
                    # _prev_round_assignments, so it catches the entries
                    # POST /round moved into _prev. Eager scoring runs
                    # inline on each newly-collected gradient.
                    self._collect_round_gradients()

                    # Drain a round when either:
                    #   1. every miner the validator delivered an
                    #      assignment to has been scored already (eager
                    #      path completed during the round) — score now,
                    #      no waiting; or
                    #   2. the post-close grace window has elapsed —
                    #      score whoever is in, leave stragglers at 0.
                    # The validator's 25-block round IS the training-time
                    # budget; the grace is just slack for uploads landing
                    # right after POST /round.
                    _now = time.time()
                    ready = {
                        rnd
                        for rnd in self._pending_aggregation_rounds
                        if self._is_round_ready_for_aggregation(rnd, _now)
                    }

                    # Snapshot which rounds we're aggregating NOW. A
                    # concurrent POST /round can add new rounds to
                    # _pending_aggregation_rounds and new entries to
                    # _prev_round_assignments while we're working — we
                    # must not touch anything outside this snapshot.
                    draining = ready
                    for rnd in sorted(draining):
                        saved_round = self._agg_round
                        self._agg_round = rnd
                        try:
                            self._aggregate_round()
                        finally:
                            self._agg_round = saved_round
                        self._pending_aggregation_rounds.discard(rnd)
                        self._pending_aggregation_at.pop(rnd, None)
                    # Drop _prev_round_assignments entries for rounds we
                    # just aggregated only — leave any entries belonging to
                    # a later round that POST /round added concurrently.
                    for uid, a in list(self._prev_round_assignments.items()):
                        if a.get("round", -1) in draining:
                            self._prev_round_assignments.pop(uid, None)

                # State arrives over HTTP (POST /state) and is processed in
                # the FastAPI handler — this loop only manages the assignment
                # state machine + gradient collection.

                # Step 1: create assignments if none exist for current round
                has_current = any(
                    a.get("round") == self._agg_round
                    for a in list(self._assignments.values())
                )
                if not has_current:
                    self._create_round_assignments()

                # Step 2: check if PENDING assignments now have data
                self._check_data_readiness()

                # Step 3: collect gradients for DELIVERED assignments
                self._collect_round_gradients()

                # Periodic status log: every ~block (block_time_s seconds).
                _now = time.monotonic()
                if _now - self._last_round_log_ts >= self.block_time_s:
                    cur_assignments = [
                        a
                        for a in list(self._assignments.values())
                        if a.get("round") == self._agg_round
                    ]
                    pending = sum(
                        1 for a in cur_assignments if a.get("_state") == "PENDING"
                    )
                    data_ready = sum(
                        1 for a in cur_assignments if a.get("_state") == "DATA_READY"
                    )
                    delivered = sum(
                        1
                        for a in cur_assignments
                        if a.get("_state") in ("DELIVERED", "GRADIENT_IN")
                    )
                    gradient_in = sum(
                        1 for a in cur_assignments if a.get("_state") == "GRADIENT_IN"
                    )
                    total = len(cur_assignments)
                    if delivered > 0:
                        # Active round: show countdown to next round
                        round_s = self._round_estimate_s()
                        delivered_times = [
                            a["_delivered_at"]
                            for a in cur_assignments
                            if a.get("_delivered_at")
                            and a.get("_state") in ("DELIVERED", "GRADIENT_IN")
                        ]
                        timing_suffix = ""
                        if (
                            self._last_push_block is not None
                            and self.blocks_per_round > 0
                            and delivered_times
                        ):
                            elapsed_blocks = round(
                                (time.time() - min(delivered_times)) / self.block_time_s
                            )
                            cur_block = self._last_push_block + elapsed_blocks
                            next_block = (self._agg_round + 1) * self.blocks_per_round
                            secs = max(
                                0.0, (next_block - cur_block) * self.block_time_s
                            )
                            timing_suffix = (
                                f"; next round at block ~{next_block} (~{secs:.0f}s)"
                            )
                        elif delivered_times and round_s > 0:
                            elapsed_s = time.time() - min(delivered_times)
                            remaining_s = max(0.0, round_s - elapsed_s)
                            timing_suffix = f"; ~{remaining_s:.0f}s until round closes"
                        # RETAINED PAYLOAD, once per round. Cheap: len() on bytes already in memory,
                        # over at most a few dozen assignments. Reported as MB because the design
                        # budget (two rounds, ~290MB) and the failure (11.5GB) are both MB-scale.
                        try:
                            _held_n = 0
                            _held_b = 0
                            for _src in (self._assignments, self._prev_round_assignments):
                                for _a in list(_src.values()):
                                    _gd = _a.get("_gradient_data")
                                    if _gd is not None:
                                        _held_n += 1
                                        _held_b += len(_gd)
                            _held = (f"; retained {_held_n} gradient(s) "
                                     f"{_held_b / 1048576:.0f}MB across "
                                     f"{len(self._assignments)}+{len(self._prev_round_assignments)} assignments")
                        except Exception as _e:
                            # SAY WHY IT FAILED. A bare swallow here hid the reason for two rounds and
                            # made a working instrument look like a no-op.
                            _held = f"; retained=UNAVAILABLE ({type(_e).__name__}: {_e})"
                        logger.info(
                            f"[GTX] round={self._agg_round}: {gradient_in}/{delivered} gradients in "
                            f"({delivered}/{total} delivered){timing_suffix}{_held}"
                        )
                    else:
                        # Waiting for validator to push round (data may still be accumulating)
                        data_age_s = ""
                        if self._written_parquets:
                            data_age_s = ", data ready"
                            if self._pending_rows and self._max_timestamp_ns > 0:
                                # min: pick the most advanced book (smallest interval
                                # start = largest elapsed) so the display shows how
                                # close we are to the NEXT parquet flush.
                                min_start = min(
                                    (
                                        self._pending_interval_start.get(bid, 0)
                                        for bid in self._pending_rows
                                    ),
                                    default=0,
                                )
                                if min_start > 0:
                                    elapsed_ns = self._max_timestamp_ns - min_start
                                    pct = min(
                                        100,
                                        int(
                                            100 * elapsed_ns / self._parquet_interval_ns
                                        ),
                                    )
                                    data_age_s += f", next window {pct}%"
                        elif self._max_timestamp_ns > 0:
                            min_start = min(
                                (
                                    self._pending_interval_start.get(bid, 0)
                                    for bid in self._pending_rows
                                ),
                                default=0,
                            )
                            if min_start > 0:
                                elapsed_ns = self._max_timestamp_ns - min_start
                                pct = min(
                                    100,
                                    int(100 * elapsed_ns / self._parquet_interval_ns),
                                )
                                data_age_s = (
                                    f", accumulating data ({pct}% to first window)"
                                )
                        states = f"pending={pending} data_ready={data_ready}"
                        logger.info(
                            f"[GTX] round={self._agg_round}: waiting for validator push "
                            f"({states}{data_age_s})"
                        )
                    self._last_round_log_ts = _now

                # Persist pending rows every loop iteration (~5s) so restarts
                # lose at most one loop tick of accumulation.
                self._save_pending_rows()
                self._last_pending_save_ts = _now

                # Step 4: check if current round is complete → aggregate
                # Fallback path when the validator stops pushing POST /round
                # (e.g. disconnect): the loop closes the round on its
                # heartbeat-loss timer so scoring doesn't stall.
                if self._round_complete():
                    self._aggregate_round()
                    self._agg_round += 1
                    # Siblings: sync canonical model from uid-0 before the new
                    # round so scoring uses the latest checkpoint. Once per round.
                    if not self.is_aggregator:
                        self._sync_from_uid0()
                    self._create_round_assignments()

            except Exception as exc:
                logger.error("Aggregation loop error: %s", exc)

            time.sleep(self._loop_sleep_s)  # short poll — most iterations are cheap

    def _refresh_miner_buckets(self) -> None:
        """Pull miner bucket commitments from chain, cooldown-rate-limited.

        Re-queries when the cache is non-empty (catches new registrations) at
        most once per `_miner_buckets_refresh_s`. Importantly also rate-limits
        the empty-cache case — a chain with no commitments would otherwise
        cause a query every loop iteration.
        """
        if self._chain is None:
            return
        now = time.time()
        if now - self._miner_buckets_queried_at < self._miner_buckets_refresh_s:
            return
        try:
            import asyncio

            t_start = time.time()
            loop = asyncio.new_event_loop()
            try:
                fresh = loop.run_until_complete(self._chain.get_miner_buckets())
            finally:
                loop.close()
            t_query = time.time() - t_start
            if fresh:
                self._miner_buckets = fresh
            elif not self._miner_buckets:
                self._miner_buckets = {}
            # Static buckets (benchmark miners not on-chain) always win.
            if self._static_buckets:
                self._miner_buckets.update(self._static_buckets)
            self._miner_buckets_queried_at = now
            logger.info(
                "[GTX] miner_buckets_refresh n=%d (static=%d) t=%.2fs",
                len(self._miner_buckets),
                len(self._static_buckets),
                t_query,
            )
            self._log_event(
                {
                    "type": "miner_buckets_refresh",
                    "n_buckets": len(self._miner_buckets),
                    "t_query_s": t_query,
                }
            )
        except Exception as exc:
            logger.debug("Failed to refresh miner buckets: %s", exc)

    def _collect_round_gradients(self) -> None:
        """Check miner S3 buckets for gradients from DELIVERED assignments.

        Iterates both the live current-round (_assignments) and the preserved
        closing-round (_prev_round_assignments). A gradient uploaded for the
        closing round is still collectable until that round's aggregation
        finishes and clears _prev_round_assignments.
        """
        # Snapshot before iterating — push_round (HTTP handler) mutates these
        # dicts concurrently from the uvicorn event loop.
        for source in (dict(self._assignments), dict(self._prev_round_assignments)):
            for uid, a in source.items():
                if a.get("_state") != "DELIVERED":
                    continue
                if a.get("_gradient_data") is not None:
                    continue  # already collected

                # Try to read gradient from miner's bucket
                grad_data = self._try_read_miner_gradient(uid, a["round"])
                if grad_data is not None:
                    a["_gradient_data"] = grad_data
                    a["_state"] = "GRADIENT_IN"
                    logger.info(
                        "[GTX] round=%d gradient received: uid=%d (%.1f KB)",
                        a["round"],
                        uid,
                        len(grad_data) / 1024,
                    )
                    self._log_event(
                        {
                            "type": "gradient_received",
                            "round": a["round"],
                            "miner": uid,
                            "bytes": len(grad_data),
                        }
                    )
                    # Score eagerly so the round can drain the moment all
                    # expected gradients are in. Failure here is fine —
                    # the drain pass re-attempts via _score_round.
                    self._score_eagerly(uid, a, grad_data)

    def _try_read_miner_gradient(self, uid: int, round_id: int) -> bytes | None:
        """Try to read a gradient from a miner's S3 bucket. Returns None if not found."""
        dedup_key = f"miner_{uid}/round_{round_id}"
        if dedup_key in self._processed_grad_keys:
            return None
        if dedup_key in self._failed_grad_keys:
            return None

        # Per-miner buckets (production + localnet)
        if self._chain is not None:
            self._refresh_miner_buckets()
            bucket_info = self._miner_buckets.get(uid)
            if bucket_info is None:
                return None

            try:
                import boto3
                from botocore.config import Config as BotoConfig
                from botocore.exceptions import ClientError

                client = boto3.client(
                    "s3",
                    endpoint_url=bucket_info.endpoint_url,
                    aws_access_key_id=bucket_info.access_key_id,
                    aws_secret_access_key=bucket_info.secret_access_key,
                    region_name=getattr(bucket_info, "region", "auto"),
                    config=BotoConfig(
                        signature_version="s3v4",
                        s3={"addressing_style": "path"},
                        connect_timeout=5,
                        read_timeout=10,
                        request_checksum_calculation="when_required",
                        response_checksum_validation="when_required",
                    ),
                )
                key = f"{self._bucket_prefix}gradients/{uid}/{round_id:08d}.grad"
                logger.debug(
                    "[GTX] gradient_get uid=%d round=%d bucket=%s key=%s",
                    uid,
                    round_id,
                    bucket_info.bucket_name,
                    key,
                )
                try:
                    t_start = time.time()
                    resp = client.get_object(Bucket=bucket_info.bucket_name, Key=key)
                    data = resp["Body"].read()
                    t_get = time.time() - t_start
                    self._processed_grad_keys.add(dedup_key)
                    logger.info(
                        "[GTX] gradient_get uid=%d round=%d bytes=%d t=%.2fs",
                        uid,
                        round_id,
                        len(data),
                        t_get,
                    )
                    return data
                except ClientError as exc:
                    error_code = exc.response.get("Error", {}).get("Code", "")
                    if error_code == "NoSuchKey":
                        logger.debug(
                            "[GTX] gradient_get uid=%d round=%d not yet uploaded (NoSuchKey)",
                            uid,
                            round_id,
                        )
                    else:
                        logger.warning(
                            "[GTX] gradient_get uid=%d round=%d S3 error: %s",
                            uid,
                            round_id,
                            exc,
                        )
                        self._failed_grad_keys.add(dedup_key)
                    return None
            except Exception as exc:
                logger.warning(
                    "[GTX] gradient_get uid=%d round=%d unexpected error: %s",
                    uid,
                    round_id,
                    exc,
                )
                self._failed_grad_keys.add(dedup_key)

        return None

    def _is_round_ready_for_aggregation(self, rnd: int, now: float) -> bool:
        """Decide if a queued (POST /round-closed) round can drain now.

        Two triggers:
          1. Early complete: no DELIVERED or GRADIENT_IN remain for this
             round. Either every delivered miner uploaded and was already
             eagerly scored, or no miner was ever delivered to (an empty
             round). Nothing to wait for.
          2. Grace expired: at least `round_grace_s` seconds have elapsed
             since POST /round arrived for this round. Score whoever is
             in; leave stragglers at 0.
        """
        queued_at = self._pending_aggregation_at.get(rnd)
        if queued_at is not None and (now - queued_at) >= self._round_grace_s:
            return True

        for source in (self._assignments, self._prev_round_assignments):
            for a in list(source.values()):
                if a.get("round") != rnd:
                    continue
                if a.get("_state") in ("DELIVERED", "GRADIENT_IN"):
                    return False
        return True

    def _round_complete(self) -> bool:
        """Check if the current round is ready for aggregation.

        A round is complete when all DELIVERED miners have submitted
        their gradients (GRADIENT_IN), or the heartbeat-loss fallback
        fires.

        The fallback exists only because the validator normally closes
        rounds via `POST /gentrx/round`. If the validator stops pushing
        (disconnect, restart, partition), the server force-closes the
        round at `oldest_delivery + _round_estimate_s() + _round_grace_s`
        so scoring does not stall indefinitely.
        """
        delivered = [
            a
            for a in list(self._assignments.values())
            if a.get("_state") in ("DELIVERED", "GRADIENT_IN")
            and a.get("round") == self._agg_round
        ]
        if not delivered:
            return False

        all_in = all(a["_state"] == "GRADIENT_IN" for a in delivered)
        if all_in:
            logger.info(
                "[GTX] round=%d complete: all %d gradients received",
                self._agg_round,
                len(delivered),
            )
            return True

        # Heartbeat-loss fallback: oldest delivery + estimate + grace
        delivered_times = [
            a["_delivered_at"] for a in delivered if a.get("_delivered_at")
        ]
        if not delivered_times:
            return False

        oldest_delivery = min(delivered_times)
        expected_round_s = self._round_estimate_s()
        elapsed = time.time() - oldest_delivery

        if elapsed > expected_round_s + self._round_grace_s:
            n_in = sum(1 for a in delivered if a["_state"] == "GRADIENT_IN")
            n_missing = len(delivered) - n_in
            logger.info(
                "[GTX] round=%d heartbeat-loss force-close (%.0fs > %.0fs): %d/%d gradients received, %d missing",
                self._agg_round,
                elapsed,
                expected_round_s + self._round_grace_s,
                n_in,
                len(delivered),
                n_missing,
            )
            return True

        return False

    def _aggregate_round(self) -> None:
        """Score and aggregate gradients for the current round.

        Collects all GRADIENT_IN assignments, scores them, aggregates
        accepted gradients into the model, and publishes a new checkpoint.
        Miners that didn't submit get score=0 for this round.

        Looks in BOTH _assignments (live current round) and
        _prev_round_assignments (closing round preserved across the last
        POST /round). Filters by `round == self._agg_round` so we only pick
        up entries that belong to the round we're aggregating.
        """
        round_assignments = []
        for source in (dict(self._assignments), dict(self._prev_round_assignments)):
            for uid, a in source.items():
                if a.get("round") == self._agg_round and a.get("_state") in (
                    "GRADIENT_IN",
                    "DELIVERED",
                    "SCORED",
                ):
                    round_assignments.append((uid, a))

        # Build pending list for the existing _drain_and_aggregate scoring logic.
        # Eager-scored entries (state=SCORED, _comp set) are passed through
        # so _score_round's idempotent path returns the cached score+comp
        # without re-running GPU work.
        pending = []
        for uid, a in round_assignments:
            grad_data = a.get("_gradient_data")
            state = a.get("_state")
            if grad_data is not None and state in ("GRADIENT_IN", "SCORED"):
                pending.append((uid, self._agg_round, a, grad_data))
            elif state == "DELIVERED":
                a["_state"] = "SCORED"
                a["_score"] = 0.0
                logger.info(
                    "Miner %d: no gradient for round %d (score=0)", uid, self._agg_round
                )

        if not pending:
            logger.info("[GTX] round=%d: no gradients to aggregate", self._agg_round)
            # Empty publish; without this /scores keeps the prior round's payload.
            self._deliver_scores(
                [], [], [], self._effective_min_score, round_assignments
            )
            self._log_event(
                {
                    "type": "aggregation",
                    "round": self._agg_round,
                    "n_scored": 0,
                    "n_accepted": 0,
                    "version": self._version,
                }
            )
            return

        logger.info(
            "[GTX] round=%d: aggregating %d gradients", self._agg_round, len(pending)
        )

        self._score_and_aggregate(pending, round_assignments)

        for uid, a in round_assignments:
            a["_state"] = "SCORED"

    def _score_and_aggregate(self, pending: list, round_assignments: list) -> None:
        """Orchestrator: reuse the eager-scoring cache when present, else load.

        Most rounds will arrive here with the model already loaded (the
        first eager score during the round triggered _get_scoring_cache).
        Stragglers go through _score_round, which is a no-op for already-
        scored gradients. Aggregation then runs once on the assembled
        scored list, and the cache is dropped so the next round loads
        the freshly-published checkpoint.

        round_assignments is the full (uid, assignment) set for the round
        including non-submitters; it flows to _deliver_scores so the
        payload always covers every assigned miner.
        """
        t_total_start = time.time()

        t_load_start = time.time()
        cache = self._get_scoring_cache(self._agg_round)
        if cache is None:
            logger.warning(
                "[GTX] aggregate_round=%d: scoring model unavailable, skipping",
                self._agg_round,
            )
            return
        model = cache["model"]
        model_cfg = cache["model_cfg"]
        tokenizer_cfg = cache["tokenizer_cfg"]
        tokenizer = cache["tokenizer"]
        device = cache["device"]
        t_load = time.time() - t_load_start

        t_score_start = time.time()
        scored = self._score_round(pending, model, tokenizer, device)
        t_score = time.time() - t_score_start

        t_agg_start = time.time()
        self._aggregate_accepted(
            scored,
            model,
            model_cfg,
            tokenizer_cfg,
            tokenizer,
            device,
            round_assignments,
        )
        t_agg = time.time() - t_agg_start

        # Drop the cache so the next round picks up the freshly-published
        # checkpoint (aggregation mutated `model` in place, and a new
        # checkpoint was written by _aggregate_accepted).
        self._clear_scoring_cache()

        t_total = time.time() - t_total_start
        logger.info(
            "[GTX] aggregate_round=%d: n_pending=%d t_load=%.2fs t_score=%.2fs t_aggregate=%.2fs t_total=%.2fs",
            self._agg_round,
            len(pending),
            t_load,
            t_score,
            t_agg,
            t_total,
        )
        # Stash timings on the aggregator — _aggregate_accepted below will
        # also set loss_before/loss_after/rolled_back on this same dict.
        self._last_aggregation.update(
            {
                "round": self._agg_round,
                "n_pending": len(pending),
                "t_load_s": t_load,
                "t_score_s": t_score,
                "t_aggregate_s": t_agg,
                "t_total_s": t_total,
                "timestamp": time.time(),
            }
        )

    def _get_scoring_cache(self, rnd: int) -> dict | None:
        """Load (or reuse) the scoring model+tokenizer for round `rnd`.

        Cached for the lifetime of the round so eagerly-scored gradients
        don't each pay the checkpoint-load cost. Evicted at aggregation
        boundaries (`_clear_scoring_cache`) because aggregation mutates
        the model in place.
        """
        if self._scoring_cache is not None and self._scoring_cache.get("round") == rnd:
            return self._scoring_cache

        try:
            import torch

            from GenTRX.src.model import ModelConfig, OrderModel
            from GenTRX.src.tokenizer import OrderTokenizer, TokenizerConfig
            from GenTRX.src.train import load_checkpoint

            device = "cuda" if torch.cuda.is_available() else "cpu"
            with self._lock:
                ckpt = load_checkpoint(str(self.checkpoint_path), device="cpu")
                raw_cfg = ckpt.get("model_config", {})
                model_cfg = ModelConfig(
                    **{
                        k: v
                        for k, v in raw_cfg.items()
                        if k in ModelConfig.__dataclass_fields__
                    }
                )
                tok_dict = ckpt.get("tokenizer_config")
                tokenizer_cfg = (
                    TokenizerConfig.from_dict(tok_dict)
                    if tok_dict
                    else TokenizerConfig()
                )
                model = OrderModel(model_cfg).to(device)
                model.load_state_dict(ckpt["model_state_dict"])
            tokenizer = OrderTokenizer(tokenizer_cfg)
            self._scoring_cache = {
                "round": rnd,
                "model": model,
                "model_cfg": model_cfg,
                "tokenizer_cfg": tokenizer_cfg,
                "tokenizer": tokenizer,
                "device": device,
            }
            return self._scoring_cache
        except Exception as exc:
            logger.warning("Scoring model load failed for round %d: %s", rnd, exc)
            return None

    def _clear_scoring_cache(self) -> None:
        """Drop the cached scoring model. Called after aggregation."""
        self._scoring_cache = None
        self._held_baseline_cache.clear()
        # Loader cache lives for one round; cross-round reuse risks stale
        # _written_parquets snapshots when new data arrives.
        if self._loader_cache:
            self._loader_cache.clear()
        self._loader_cache_hits = 0
        self._loader_cache_misses = 0
        self._fetch_list_local = 0
        self._fetch_list_s3 = 0
        self._fetch_list_s = 0.0
        self._fetch_file_hits = 0
        self._fetch_downloads = 0
        self._fetch_download_bytes = 0
        self._fetch_errors = 0

    def _score_eagerly(
        self, miner_uid: int, assignment: dict, grad_bytes: bytes
    ) -> bool:
        """Score a freshly-collected gradient on the same loop tick it arrived.

        Returns True if scoring succeeded (assignment now stamped + state →
        SCORED), False otherwise (assignment stays GRADIENT_IN; the drain
        path will retry). Best-effort: failure here is not fatal because
        `_score_round` covers stragglers and any retries at drain time.
        """
        rnd = assignment.get("round")
        if rnd is None:
            return False
        cache = self._get_scoring_cache(rnd)
        if cache is None:
            return False
        result = self._score_one_gradient(
            miner_uid,
            assignment,
            grad_bytes,
            cache["model"],
            cache["tokenizer"],
            cache["device"],
        )
        if result is None:
            return False
        assignment["_state"] = "SCORED"
        return True

    def _score_one_gradient(
        self,
        miner_uid: int,
        assignment: dict,
        grad_bytes: bytes,
        model,
        tokenizer,
        device: str,
    ):
        """Evaluate one miner's gradient. Returns (score, comp) or None on failure.

        Stamps `_score_own`, `_score_held`, `_overfitting`, `_score`, and
        `_comp` on the assignment dict. Idempotent: callers that re-invoke
        with the same assignment get the cached result without re-running
        the GPU work, so we cleanly tolerate the inevitable race where a
        gradient is read on one loop tick and aggregation runs on the next.
        """
        from GenTRX.src.distributed import evaluate_gradient
        from GenTRX.src.gradient import deserialize

        cached_comp = assignment.get("_comp")
        cached_score = assignment.get("_score")
        if cached_comp is not None and cached_score is not None:
            return cached_score, cached_comp

        t_start = time.time()
        try:
            if len(grad_bytes) > self._max_gradient_bytes:
                logger.warning(
                    "  miner %d: gradient too large (%.1f MB)",
                    miner_uid,
                    len(grad_bytes) / 1e6,
                )
                return None
            expected_shapes = {name: p.shape for name, p in model.named_parameters()}
            try:
                comp = deserialize(grad_bytes, expected_shapes=expected_shapes)
            except ValueError as exc:
                logger.warning("  miner %d: gradient rejected: %s", miner_uid, exc)
                return None

            expected_v = int(assignment.get("model_version", 0) or 0)
            trained_v = int(getattr(comp.metadata, "model_v_trained", 0) or 0)
            logger.info(
                "  miner %d: trained=v%d expected=v%d",
                miner_uid, trained_v, expected_v,
            )
            # Off by default so a miner that has not yet shipped version-tagging
            # is accepted; enable once all miners tag their gradients.
            if self._reject_untagged and not trained_v:
                logger.warning(
                    "  miner %d: gradient untagged (model_v_trained=0), rejected",
                    miner_uid,
                )
                assignment["_version_mismatched"] = True
            elif not trained_v:
                # Soft path: log so we can still see who's submitting without
                # version metadata, but accept the gradient into aggregation.
                logger.info(
                    "  miner %d: gradient untagged (pre-patch miner, accepted)",
                    miner_uid,
                )
                assignment["_version_mismatched"] = False
            elif expected_v and trained_v != expected_v:
                logger.warning(
                    "  miner %d: model_v mismatch (trained=%d, assignment=%d)",
                    miner_uid,
                    trained_v,
                    expected_v,
                )
                assignment["_version_mismatched"] = True
            else:
                assignment["_version_mismatched"] = False

            grad_norm_sq = 0.0
            for _, (_, vals, _) in comp.sparse.items():
                grad_norm_sq += float(vals.double().pow(2).sum().item())
            grad_norm = grad_norm_sq ** 0.5
            assignment["_grad_norm"] = grad_norm
            logger.info("  miner %d: grad_norm=%.4g", miner_uid, grad_norm)
            if self.max_grad_norm is not None and grad_norm > self.max_grad_norm:
                logger.warning(
                    "  miner %d: gradient rejected, norm %.4g > max %.4g",
                    miner_uid, grad_norm, self.max_grad_norm,
                )
                return None

            t_loader_start = time.time()
            miner_loader = self._build_miner_loader(assignment, tokenizer, device)
            if miner_loader is None:
                logger.warning("  miner %d: no data for assigned books", miner_uid)
                return None

            t_own_start = time.time()
            score_own = evaluate_gradient(
                model, comp, miner_loader, device, self._proof_batches,
                self.label_smooth_sigma,
            )
            t_own = time.time() - t_own_start

            miner_ranges = [
                (assignment.get("ts_start", 0), assignment.get("ts_end", 0))
            ]
            t_held_loader_start = time.time()
            grad_round = int(assignment.get("round", self._agg_round))
            # Resolve the forward ranges ONCE and key both the loader and the
            # baseline on them. _forward_val_ranges maps [s,e] to (e, _max_ts],
            # and _max_ts advances as data streams in, so two miners with the
            # same raw [ts_start, ts_end] scored moments apart get DIFFERENT
            # forward windows (and different loaders). Keying the baseline on the
            # raw ranges instead would collide those distinct loaders onto one
            # cached baseline → a miner scored as baseline_A − loss_after_B
            # across mismatched held-out data. Key on fwd_ranges so the cache
            # entry always matches the loader it was computed against.
            fwd_ranges = self._forward_val_ranges(miner_ranges)
            held_loader = self._build_val_loader_for_ranges(
                fwd_ranges, tokenizer, device,
                round_id=grad_round,
            )
            t_loader_build = (
                (t_own_start - t_loader_start)
                + (time.time() - t_held_loader_start)
            )
            t_held = 0.0
            if held_loader is not None:
                t_held_start = time.time()
                # Cache the baseline per (round, model version, forward ranges).
                # Miners landing on the identical forward window share one
                # baseline (single-pass); distinct windows get their own.
                held_base_key = self._held_baseline_key(
                    grad_round, self._version, fwd_ranges
                )
                held_base = self._held_baseline_cache.get(held_base_key)
                if held_base is None:
                    from GenTRX.src.distributed import _eval_loss
                    held_base = _eval_loss(
                        model, held_loader, device, self.max_val_batches,
                        self.label_smooth_sigma,
                    )
                    self._held_baseline_cache[held_base_key] = held_base
                score_held = evaluate_gradient(
                    model, comp, held_loader, device, self.max_val_batches,
                    self.label_smooth_sigma, loss_before=held_base,
                )
                t_held = time.time() - t_held_start
                # Held-out is the reward; overfitting stays a logged diagnostic, not a penalty.
                overfitting = score_own > score_held * self.overfit_ratio
                score = score_held
            else:
                # A held loader should always exist (permanent val pool + recent-page
                # fallback); empty is a data problem to flag, not a normal path.
                # Do NOT fall back to rewarding score_own — that pays memorization
                # of the miner's own training pages, the exact contamination the
                # held-out design removes. Flag it so it's excluded from reward
                # and aggregation outside warmup (warmup still bootstraps on it).
                logger.error(
                    "  miner %d: NO held-out val loader (round %d, ranges %s) — "
                    "not rewardable this round; check val pool / data flush",
                    miner_uid, self._agg_round, miner_ranges,
                )
                score_held = None
                overfitting = False
                score = score_own
                assignment["_no_held"] = True

            assignment["_score_own"] = score_own
            assignment["_score_held"] = score_held
            assignment["_overfitting"] = overfitting
            assignment["_score"] = score
            assignment["_comp"] = comp
            assignment["_t_loader_build"] = t_loader_build

            t_total = time.time() - t_start
            logger.info(
                "[GTX] score_miner=%d: score_own=%+.4f score_held=%s%s combined=%+.4f t_own=%.2fs t_held=%.2fs t_loader=%.2fs t_total=%.2fs",
                miner_uid,
                score_own,
                f"{score_held:+.4f}" if score_held is not None else "n/a",
                " [OVERFIT]" if overfitting else "",
                score,
                t_own,
                t_held,
                t_loader_build,
                t_total,
            )
            return score, comp
        except Exception as exc:
            logger.warning(
                "  miner %d: scoring failed: %s", miner_uid, exc, exc_info=True
            )
            return None

    def _score_round(self, pending: list, model, tokenizer, device: str) -> list:
        """Score gradients that haven't been scored yet; collect prior scores.

        Eager scoring (called from _collect_round_gradients) usually fills
        in `_score` and `_comp` on the assignment well before the round
        drains. This pass picks up any stragglers — gradients that landed
        inside the round-grace window — and assembles the full (uid, win,
        score, comp, assignment) tuple list the aggregator expects.
        """
        scored = []
        for miner_uid, window_id, assignment, grad_bytes in pending:
            result = self._score_one_gradient(
                miner_uid,
                assignment,
                grad_bytes,
                model,
                tokenizer,
                device,
            )
            if result is None:
                continue
            score, comp = result
            scored.append((miner_uid, window_id, score, comp, assignment))
        return scored

    def _compute_index_overlap(
        self,
        accepted: list,
        *,
        min_param_size: int = 100,
        high_threshold: float = 0.9,
    ) -> dict:
        """Per-pair Jaccard overlap on top-k indices, size-weighted across params.

        Two miners producing identical index sets are either copying each
        other or training on identical-enough data — either way worth
        flagging. The values may differ honestly so we ignore them; only
        the chosen index positions are compared. Parameters under
        `min_param_size` elements are skipped to avoid noise on small
        biases.

        Returns dict with keys overlap_pairs_checked, overlap_pairs_high,
        overlap_mean, overlap_max.
        """
        if len(accepted) < 2:
            return {}

        comps = [(uid, c) for uid, _, _, c, _ in accepted]
        param_names: set[str] = set()
        for _, c in comps:
            if not param_names:
                param_names = {
                    n for n, (_, _, shape) in c.sparse.items()
                    if shape.numel() >= min_param_size
                }
            else:
                param_names &= set(c.sparse.keys())

        if not param_names:
            return {}

        pair_overlaps: list[float] = []
        pair_high = 0
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                _, ci = comps[i]
                _, cj = comps[j]
                weighted_sum = 0.0
                total_weight = 0.0
                for name in param_names:
                    idx_i = ci.sparse[name][0]
                    idx_j = cj.sparse[name][0]
                    shape = ci.sparse[name][2]
                    size = float(shape.numel())
                    si = set(idx_i.tolist())
                    sj = set(idx_j.tolist())
                    union = si | sj
                    if not union:
                        continue
                    jacc = len(si & sj) / len(union)
                    weighted_sum += jacc * size
                    total_weight += size
                if total_weight == 0:
                    continue
                pair_overlap = weighted_sum / total_weight
                pair_overlaps.append(pair_overlap)
                if pair_overlap >= high_threshold:
                    pair_high += 1

        if not pair_overlaps:
            return {}

        return {
            "overlap_pairs_checked": len(pair_overlaps),
            "overlap_pairs_high": pair_high,
            "overlap_mean": sum(pair_overlaps) / len(pair_overlaps),
            "overlap_max": max(pair_overlaps),
        }

    def _is_gradient_acceptable(self, score: float, assignment: dict) -> bool:
        """Gate deciding which scored gradients enter the aggregation that produces
        the published proposal (training-integrity critical; unit-tested in
        tests/test_gradient_accept.py).

        - Below/at the effective min score: rejected (not a positive signal).
        - Version-mismatched (outside warmup): trained against an older model than
          the one we'd apply to — its direction-of-improvement is for a checkpoint
          that no longer exists; letting it in pollutes the proposal (and, on a
          fresh-regime random init, can push the apply over fp32 stability).
          During warmup anything is accepted (model bootstrapping from scratch).
        - No held-out loader (outside warmup): the score is own-data loss
          (memorization), not a generalization signal."""
        if score <= self._effective_min_score:
            return False
        if not self._in_warmup and assignment.get("_version_mismatched"):
            return False
        if not self._in_warmup and assignment.get("_no_held"):
            return False
        return True

    def _aggregate_accepted(
        self,
        scored: list,
        model,
        model_cfg,
        tokenizer_cfg,
        tokenizer,
        device: str,
        round_assignments: list,
    ) -> None:
        """Aggregate accepted gradients and publish result.

        Aggregation-of-aggregations pattern:
          - ALL validators (aggregator + siblings): score miners, aggregate
            accepted gradients into a compressed delta, publish the delta as
            proposals/{block:08d}.grad to their own bucket.  This is proof of
            work — you can't produce a valid delta without actually scoring.
          - Aggregator (uid 0): also fetches proposals from sibling buckets,
            evaluates each against held-out val data, picks the best (or
            averages top proposals), and publishes the canonical checkpoint.

        Sibling validators never publish checkpoints — only proposals + scores.
        """
        import torch

        from GenTRX.src.distributed import _eval_loss, _eval_loss_per_field, model_state_hash
        from GenTRX.src.gradient import (
            aggregate, apply_gradient, compress, decompress, serialize,
        )

        threshold = self._effective_min_score

        def _is_acceptable(s: float, a: dict) -> bool:
            return self._is_gradient_acceptable(s, a)

        accepted = [(m, w, s, c, a) for m, w, s, c, a in scored if _is_acceptable(s, a)]
        rejected = [(m, w, s, a) for m, w, s, _, a in scored if not _is_acceptable(s, a)]


        n_assigned = len(round_assignments)
        n_delivered = sum(1 for _, a in round_assignments if a.get("_delivered_at"))
        n_collected = sum(1 for _, a in round_assignments if a.get("_comp") is not None)
        n_version_mismatched = sum(
            1 for _, a in round_assignments if a.get("_version_mismatched")
        )
        t_loader_build_miners = sum(
            a.get("_t_loader_build", 0.0) for _, a in round_assignments
        )

        grad_norms = sorted(
            a["_grad_norm"]
            for _, a in round_assignments
            if a.get("_grad_norm") is not None
        )
        if grad_norms:
            n_g = len(grad_norms)
            mean_g = sum(grad_norms) / n_g
            var_g = sum((x - mean_g) ** 2 for x in grad_norms) / n_g
            grad_norm_stats = {
                "grad_norm_mean": mean_g,
                "grad_norm_min": grad_norms[0],
                "grad_norm_max": grad_norms[-1],
                "grad_norm_median": grad_norms[n_g // 2],
                "grad_norm_std": var_g ** 0.5,
            }
        else:
            grad_norm_stats = {}

        overlap_stats = self._compute_index_overlap(accepted)
        grad_norm_stats.update(overlap_stats)
        cache_total = self._loader_cache_hits + self._loader_cache_misses
        if cache_total > 0:
            grad_norm_stats["loader_cache_hits"] = self._loader_cache_hits
            grad_norm_stats["loader_cache_misses"] = self._loader_cache_misses
            grad_norm_stats["loader_cache_hit_rate"] = (
                self._loader_cache_hits / cache_total
            )
        grad_norm_stats["fetch_list_local"] = self._fetch_list_local
        grad_norm_stats["fetch_list_s3"] = self._fetch_list_s3
        grad_norm_stats["fetch_list_s"] = self._fetch_list_s
        grad_norm_stats["fetch_file_hits"] = self._fetch_file_hits
        grad_norm_stats["fetch_downloads"] = self._fetch_downloads
        grad_norm_stats["fetch_download_mb"] = self._fetch_download_bytes / 1e6
        grad_norm_stats["fetch_errors"] = self._fetch_errors

        # Initialised here so all four log paths see them; populated only
        # when the val eval runs (otherwise empty dicts spread to nothing).
        per_field_before: dict[str, float] = {}
        per_field_after: dict[str, float] = {}

        if self._in_warmup:
            logger.info(
                "  Warmup round %d/%d — min_score disabled, rollback disabled",
                (self._agg_round - self._warmup_anchor_round) + 1,
                self.warmup_rounds,
            )

        # Log + deliver scores (all validators)
        for m, w, s, _, a in scored:
            self._log_event(
                {
                    "type": "gradient_score",
                    "miner": m,
                    "window": w,
                    "score": s,
                    "score_own": a.get("_score_own"),
                    "score_held": a.get("_score_held"),
                    "overfitting": a.get("_overfitting", False),
                    "accepted": s > threshold,
                    "books": a.get("books", []),
                    "version": self._version,
                }
            )

        self._deliver_scores(scored, accepted, rejected, threshold, round_assignments)

        if not accepted:
            logger.info("  No accepted gradients — model unchanged")
            self._log_event(
                {
                    "type": "aggregation",
                    "round": self._agg_round,
                    "n_assigned": n_assigned,
                    "n_delivered": n_delivered,
                    "n_collected": n_collected,
                    "n_version_mismatched": n_version_mismatched,
                    "n_scored": len(scored),
                    "n_accepted": 0,
                    "t_loader_build_s": t_loader_build_miners,
                    "version": self._version,
                    **grad_norm_stats,
                }
            )
            return

        # --- Phase 1: Local aggregation (all validators) ---
        # Every validator aggregates its own accepted gradients into a delta.
        # `aggregate()` returns a fully-dense CompressedGradient (every index
        # kept). Re-compress at agg_top_k_frac: the same object is both applied
        # to the model and published as the version delta, so the two stay
        # consistent for delta-follow reconstruction. Default 1.0 (dense).
        local_agg_dense = aggregate([c for _, _, _, c, _ in accepted])
        local_agg = compress(decompress(local_agg_dense), top_k_frac=self._agg_top_k_frac)

        # Aggregated-delta safety: always log the L2 norm + max |element|
        # so the operator can see the distribution and tune the caps. The
        # norm cap scales the whole delta down proportionally; the element
        # cap clamps individual sparse entries that exceed the bound.
        agg_norm_sq = 0.0
        agg_max_elem = 0.0
        for _, (_, vals, _) in local_agg.sparse.items():
            if vals.numel() == 0:
                continue
            agg_norm_sq += float(vals.double().pow(2).sum().item())
            m = float(vals.double().abs().max().item())
            if m > agg_max_elem:
                agg_max_elem = m
        agg_norm = agg_norm_sq ** 0.5
        logger.info(
            "  local_agg norm=%.4g max_elem=%.4g (cap_norm=%s cap_elem=%s)",
            agg_norm, agg_max_elem,
            f"{self.max_agg_norm:.4g}" if self.max_agg_norm is not None else "off",
            f"{self.max_agg_element:.4g}" if self.max_agg_element is not None else "off",
        )

        if self.max_agg_norm is not None and self.max_agg_norm > 0 and agg_norm > self.max_agg_norm:
            scale = self.max_agg_norm / agg_norm
            for name, (idx, vals, shape) in list(local_agg.sparse.items()):
                local_agg.sparse[name] = (idx, vals * scale, shape)
            logger.warning(
                "  local_agg norm %.4g > cap %.4g → scaled by %.4g",
                agg_norm, self.max_agg_norm, scale,
            )

        if (
            self.max_agg_element is not None and self.max_agg_element > 0
            and agg_max_elem > self.max_agg_element
        ):
            cap = self.max_agg_element
            n_clamped = 0
            for name, (idx, vals, shape) in list(local_agg.sparse.items()):
                clipped = vals.clamp(-cap, cap)
                n_clamped += int((clipped != vals).sum().item())
                local_agg.sparse[name] = (idx, clipped, shape)
            logger.warning(
                "  local_agg max_elem %.4g > cap %.4g → clamped %d entries",
                agg_max_elem, cap, n_clamped,
            )

        local_delta = decompress(local_agg)

        # Publish the local delta as a proposal — proof of work.
        # Siblings publish proposals only; aggregator publishes proposals + checkpoint.
        if self.validator_store is not None:
            try:
                proposal_bytes = serialize(local_agg)
                self.validator_store.put_proposal(
                    self._validator_uid, self._agg_round, proposal_bytes
                )
                logger.info(
                    "  Published proposal for round %d (%.1f KB)",
                    self._agg_round,
                    len(proposal_bytes) / 1024,
                )
                self._prune_proposals()
            except Exception as exc:
                logger.warning("  Failed to publish proposal: %s", exc)

        # --- Phase 2: Aggregator evaluates proposals (uid 0 only) ---
        # Fetch proposals from sibling validators, evaluate each against held-out
        # val data, pick the best delta (lowest val loss).  Falls back to local
        # aggregation if no sibling proposals are available.
        trained_ranges = [
            (a.get("ts_start", 0), a.get("ts_end", 0))
            for _, _, _, _, a in accepted
            if a.get("ts_start")
        ]
        t_val_loader_start = time.time()
        agg_round_id = next(
            (int(a.get("round", self._agg_round)) for _, _, _, _, a in accepted),
            self._agg_round,
        )
        val_loader = self._build_val_loader_for_ranges(
            self._forward_val_ranges(trained_ranges), tokenizer, device,
            round_id=agg_round_id,
        )
        t_val_loader_build = time.time() - t_val_loader_start
        if val_loader is None:
            val_loader = self._global_val_loader
        t_loader_build_total = t_loader_build_miners + t_val_loader_build

        best_delta = local_delta
        best_comp = local_agg  # compressed form of the applied delta (for publishing)
        best_label = "local"
        best_loss = float("inf")
        baseline_loss = 0.0
        t_proposal_eval = 0.0

        if val_loader is not None:
            original_state = {k: v.clone() for k, v in model.state_dict().items()}
            baseline_loss, per_field_before = _eval_loss_per_field(
                model, val_loader, device, self.max_val_batches,
                self.label_smooth_sigma,
            )

            if self.is_aggregator:
                # Evaluate all proposals (local + sibling) and pick the best
                expected_shapes = {n: p.shape for n, p in model.named_parameters()}
                proposals = self._fetch_validator_proposals(
                    self._agg_round, expected_shapes
                )
                # Always include our own local delta in the candidate set
                candidates = [("local", local_agg)]
                candidates.extend(proposals)

                local_norm_sq = sum(
                    float(vals.double().pow(2).sum().item())
                    for _, (_, vals, _) in local_agg.sparse.items()
                )
                local_norm = local_norm_sq ** 0.5
                norm_threshold = local_norm * self.proposal_norm_ratio
                proposals_skipped = 0

                t_proposal_eval_start = time.time()
                for label, comp in candidates:
                    if label != "local" and local_norm > 0:
                        cand_norm = sum(
                            float(vals.double().pow(2).sum().item())
                            for _, (_, vals, _) in comp.sparse.items()
                        ) ** 0.5
                        if cand_norm > norm_threshold:
                            logger.warning(
                                "  Proposal %s skipped: norm=%.2f exceeds %.2f×local (%.2f)",
                                label,
                                cand_norm,
                                self.proposal_norm_ratio,
                                local_norm,
                            )
                            proposals_skipped += 1
                            continue
                    model.load_state_dict(original_state)
                    delta = decompress(comp)
                    # Pre-apply diagnostic for this candidate's delta.
                    try:
                        _d_norm_sq = 0.0
                        _d_max = 0.0
                        for _t in delta.delta.values():
                            if _t.numel() == 0:
                                continue
                            _d_norm_sq += float(_t.double().pow(2).sum().item())
                            _m = float(_t.double().abs().max().item())
                            if _m > _d_max:
                                _d_max = _m
                        logger.info(
                            "  Proposal %s pre-apply: delta norm=%.4g max_abs=%.4g",
                            label, _d_norm_sq ** 0.5, _d_max,
                        )
                    except Exception as exc:
                        logger.warning("  Proposal %s delta stats failed: %s", label, exc)
                    apply_gradient(model, delta)
                    loss = _eval_loss(
                        model, val_loader, device, self.max_val_batches,
                        self.label_smooth_sigma,
                    )
                    logger.info(
                        "  Proposal %s: val loss %r (baseline %.4f, finite=%s)",
                        label, loss, baseline_loss, math.isfinite(loss),
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best_delta = delta
                        best_comp = comp
                        best_label = label
                t_proposal_eval = time.time() - t_proposal_eval_start
                grad_norm_stats["proposals_evaluated"] = len(candidates) - proposals_skipped
                grad_norm_stats["proposals_skipped"] = proposals_skipped

                # Restore and apply the winner
                model.load_state_dict(original_state)

            # Apply the chosen delta and check rollback. Detailed pre/post
            # state logging so the actual failure point is in the log even
            # when the inputs to apply_gradient look benign.
            try:
                bd_norm_sq = 0.0
                bd_max_abs = 0.0
                bd_n = 0
                for _name, _t in best_delta.delta.items():
                    if _t.numel() == 0:
                        continue
                    bd_norm_sq += float(_t.double().pow(2).sum().item())
                    m = float(_t.double().abs().max().item())
                    if m > bd_max_abs:
                        bd_max_abs = m
                    bd_n += int(_t.numel())
                logger.info(
                    "  pre-apply best_delta(label=%s): n_elems=%d norm=%.4g max_abs=%.4g",
                    best_label, bd_n, bd_norm_sq ** 0.5, bd_max_abs,
                )
            except Exception as exc:
                logger.warning("  pre-apply best_delta stats failed: %s", exc)

            try:
                w_max_abs = 0.0
                for _p in model.parameters():
                    if _p.numel() == 0:
                        continue
                    m = float(_p.data.double().abs().max().item())
                    if m > w_max_abs:
                        w_max_abs = m
                logger.info(
                    "  pre-apply model param max_abs=%.4g (baseline_loss=%r)",
                    w_max_abs, baseline_loss,
                )
            except Exception as exc:
                logger.warning("  pre-apply model param stats failed: %s", exc)

            apply_gradient(model, best_delta)

            try:
                w_max_abs_after = 0.0
                w_nonfinite = 0
                for _p in model.parameters():
                    if _p.numel() == 0:
                        continue
                    m = float(_p.data.double().abs().max().item())
                    if m > w_max_abs_after:
                        w_max_abs_after = m
                    w_nonfinite += int((~torch.isfinite(_p.data)).sum().item())
                logger.info(
                    "  post-apply model param max_abs=%.4g n_nonfinite=%d",
                    w_max_abs_after, w_nonfinite,
                )
            except Exception as exc:
                logger.warning("  post-apply model param stats failed: %s", exc)

            # Both aggregator and siblings evaluate the applied delta against
            # the val set. Gating this on `is_aggregator` and falling
            # back to `new_loss = best_loss` does not work for siblings: for
            # them the candidates loop above is also gated on
            # `is_aggregator` — so `best_loss` stays at its initial
            # `float("inf")` and the rollback check always fires. Run the
            # eval unconditionally so siblings get a real `new_loss` for the
            # rollback decision.
            new_loss, per_field_after = _eval_loss_per_field(
                model, val_loader, device, self.max_val_batches,
                self.label_smooth_sigma,
            )
            logger.info(
                "  post-apply new_loss=%r baseline=%r per_field_after=%s",
                new_loss, baseline_loss,
                {k: round(v, 4) for k, v in per_field_after.items()},
            )

            for name, val in per_field_before.items():
                grad_norm_stats[f"per_field_loss_before_{name}"] = val
            for name, val in per_field_after.items():
                grad_norm_stats[f"per_field_loss_after_{name}"] = val

            # A non-finite update must never be committed, even during warmup
            # when the worse-than-baseline rollback is otherwise disabled.
            if not math.isfinite(new_loss) or (
                self._effective_rollback and new_loss > baseline_loss
            ):
                model.load_state_dict(original_state)
                logger.warning(
                    "  Rollback: val %.4f → %.4f (worse)", baseline_loss, new_loss
                )
                if best_label == "local":
                    for _, _, _, _, a in accepted:
                        a["_was_rollback_winner"] = True
                # The model was reverted, so the effective post-round loss
                # equals the baseline. Reporting `loss_after=new_loss` (the
                # rejected proposal's eval) would surface as a spurious
                # regression on downstream dashboards even though nothing
                # actually changed. Overwrite per-field after-stats too —
                # individual fields can be inf even when the total isn't.
                # The attempted value stays only in the warning log above,
                # not in the structured event, so dashboards don't render a
                # second series alongside the corrected one.
                for name, val in per_field_before.items():
                    grad_norm_stats[f"per_field_loss_after_{name}"] = val
                self._log_event(
                    {
                        "type": "aggregation",
                        "round": self._agg_round,
                        "n_assigned": n_assigned,
                        "n_delivered": n_delivered,
                        "n_collected": n_collected,
                        "n_version_mismatched": n_version_mismatched,
                        "n_scored": len(scored),
                        "n_accepted": len(accepted),
                        "loss_before": baseline_loss,
                        "loss_after": baseline_loss,
                        "loss_improvement_pct": 0.0,
                        "t_proposal_eval_s": t_proposal_eval,
                        "t_loader_build_s": t_loader_build_total,
                        "rolled_back": True,
                        "version": self._version,
                        **grad_norm_stats,
                    }
                )
                return
        else:
            new_loss = 0.0
            if not self.is_aggregator:
                # Sibling without val data: we published the proposal, nothing more to do
                logger.info("  Sibling: proposal published, no val check")
                self._log_event(
                    {
                        "type": "aggregation",
                        "round": self._agg_round,
                        "n_assigned": n_assigned,
                        "n_delivered": n_delivered,
                        "n_collected": n_collected,
                        "n_version_mismatched": n_version_mismatched,
                        "n_scored": len(scored),
                        "n_accepted": len(accepted),
                        "version": self._version,
                        "sibling_only": True,
                        **grad_norm_stats,
                    }
                )
                return
            logger.warning("  No val data — applying local delta without check")
            apply_gradient(model, best_delta)

        # --- Phase 3: Save checkpoint (aggregator publishes canonical) ---
        t_save_start = time.time()
        with self._lock:
            import io as _io
            from dataclasses import asdict

            ckpt_dict = {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_cfg),
                "tokenizer_config": asdict(tokenizer_cfg),
                "step": self._version + 1,
                "loss": new_loss,
                "epoch": 0,
                **checkpoint_stamp(self.label_smooth_sigma),
            }
            tmp_path = self.output_path.with_suffix(".tmp")
            torch.save(ckpt_dict, tmp_path)
            os.rename(tmp_path, self.output_path)
            self._version += 1
            t_local_save = time.time() - t_save_start

            # Only the aggregator (uid 0) publishes the canonical model. Every
            # version it publishes the small applied delta + head pointer;
            # the full checkpoint is uploaded only every `checkpoint_interval`
            # versions (and v1), so miners normally advance by applying deltas.
            t_s3_put = 0.0
            ckpt_bytes = 0
            is_baseline = (
                self._version == 1 or self._version % self.checkpoint_interval == 0
            )
            if self.validator_store is not None and self.is_aggregator:
                try:
                    self.validator_store.put_version_delta(
                        self._validator_uid, self._version, serialize(best_comp)
                    )
                    meta = {}
                    if self.publish_state_hash:
                        meta["state_hash"] = model_state_hash(model)
                    self.validator_store.put_head_version(
                        self._validator_uid, self._version, meta=meta
                    )
                    self.validator_store.prune_version_deltas(
                        self._validator_uid, self.keep_version_deltas
                    )
                except Exception as exc:
                    logger.error("Failed to publish version delta/head: %s", exc)

                if is_baseline:
                    try:
                        t_s3_start = time.time()
                        buf = _io.BytesIO()
                        torch.save(ckpt_dict, buf)
                        ckpt_bytes = len(buf.getvalue())
                        self.validator_store.put_checkpoint(
                            self._validator_uid, self._version, buf.getvalue(),
                            meta=checkpoint_stamp(self.label_smooth_sigma),
                        )
                        t_s3_put = time.time() - t_s3_start
                        self._prune_checkpoints()
                    except Exception as exc:
                        logger.error("Failed to upload checkpoint to S3: %s", exc)
            logger.info(
                "[GTX] checkpoint_save v=%d baseline=%s t_local=%.2fs t_s3=%.2fs bytes=%d",
                self._version,
                is_baseline,
                t_local_save,
                t_s3_put,
                ckpt_bytes,
            )

        # Comparable CE curve on a frozen held-out set (the loss_before/after
        # above is on rotating forward ranges, so it can't reveal a trend).
        self._log_fixed_val(model, tokenizer, device)

        self._log_event(
            {
                "type": "aggregation",
                "round": self._agg_round,
                "n_assigned": n_assigned,
                "n_delivered": n_delivered,
                "n_collected": n_collected,
                "n_version_mismatched": n_version_mismatched,
                "n_scored": len(scored),
                "n_accepted": len(accepted),
                "loss_before": baseline_loss,
                "loss_after": new_loss,
                "loss_improvement_pct": (
                    (baseline_loss - new_loss) / max(abs(baseline_loss), 1e-9)
                ),
                "t_proposal_eval_s": t_proposal_eval,
                "t_save_ckpt_s": t_local_save + t_s3_put,
                "t_loader_build_s": t_loader_build_total,
                "rolled_back": False,
                "version": self._version,
                **grad_norm_stats,
            }
        )
        logger.info(
            "[GTX] round=%d aggregated %d/%d: loss %.4f → %.4f, new_version=%d",
            self._agg_round,
            len(accepted),
            len(scored),
            baseline_loss,
            new_loss,
            self._version,
        )

    def _deliver_scores(
        self,
        scored,
        accepted,
        rejected,
        threshold,
        round_assignments,
    ) -> None:
        """Stash scores for GET /gentrx/scores; non-submitters in round_assignments appear at 0."""
        if (
            self._latest_scores is not None
            and self._agg_round < self._latest_scores.get("round", -1)
        ):
            logger.debug(
                "[GTX] _deliver_scores: dropping out-of-order round %d (latest=%d)",
                self._agg_round,
                self._latest_scores["round"],
            )
            return
        scored_uids = {m for m, _, _, _, _ in scored}
        # `accepted` must reflect what actually entered aggregation, not just
        # score > threshold: dedup-dropped duplicates and version-mismatched
        # submitters can carry a positive score but must NOT be rewarded (the
        # reward path keys purely on this flag).
        accepted_uids = {m for m, _, _, _, _ in accepted}
        scores_dict: dict[str, dict] = {}
        for uid, a in round_assignments:
            score = a.get("_score")
            if score is None:
                score = 0.0
            is_submitter = uid in scored_uids
            score_own = a.get("_score_own")
            score_held = a.get("_score_held")
            grad_norm = a.get("_grad_norm")
            scores_dict[str(uid)] = {
                "score": score,
                "score_own": float(score_own) if score_own is not None else 0.0,
                "score_held": float(score_held) if score_held is not None else 0.0,
                "overfitting": bool(a.get("_overfitting", False)),
                "accepted": bool(uid in accepted_uids),
                "was_rollback_winner": bool(a.get("_was_rollback_winner", False)),
                "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                "books": a.get("books", []),
            }
        self._latest_scores = _json_safe({
            "round": self._agg_round,
            "model_version": self._version,
            "scores": scores_dict,
            "n_scored": len(scored),
            "n_accepted": len(accepted),
            "n_rejected": len(rejected),
            "aggregation": dict(self._last_aggregation),
            "counters": {
                "rounds_aggregated_total": self._rounds_aggregated_total,
                "rollbacks_total": self._rollbacks_total,
            },
            "config": {
                "min_score": self.min_score,
                "max_grad_norm": self.max_grad_norm,
                "overfit_ratio": self.overfit_ratio,
                "books_per_miner": self.books_per_miner,
                "val_fraction": self.val_fraction,
            },
        })

    def _fetch_validator_proposals(
        self,
        round_id: int,
        expected_shapes: dict,
    ) -> list[tuple[str, Any]]:
        """Fetch proposals/{round:08d}.grad from sibling validator buckets.

        Returns [(label, CompressedGradient), ...] — one per validator that
        published a proposal for this round.  Filtered by validator_permit
        to prevent miners from injecting fake proposals.

        Only called by the aggregator.
        """
        if self._chain is None:
            return []

        try:
            import asyncio

            from GenTRX.src.gradient import deserialize
            from GenTRX.src.gradient_store import GradientStore

            loop = asyncio.new_event_loop()
            try:
                all_buckets: dict = loop.run_until_complete(
                    self._chain.get_miner_buckets()
                )
            finally:
                loop.close()

            if not all_buckets:
                return []

            metagraph = self._chain.metagraph
            validator_permits = getattr(metagraph, "validator_permit", None)

            proposals = []
            for uid, bucket_info in all_buckets.items():
                if validator_permits is not None:
                    if uid >= len(validator_permits) or not validator_permits[uid]:
                        continue
                try:
                    store = GradientStore(
                        endpoint_url=bucket_info.endpoint_url,
                        bucket=bucket_info.bucket_name,
                        access_key=bucket_info.access_key_id,
                        secret_key=bucket_info.secret_access_key,
                        region=getattr(bucket_info, "region", "auto"),
                        prefix=self._bucket_prefix,
                    )
                    raw = store.get_proposal(uid, round_id)
                    if raw is None:
                        continue
                    comp = deserialize(raw, expected_shapes=expected_shapes)
                    proposals.append((f"validator-{uid}", comp))
                    logger.debug(
                        "  Fetched proposal from validator uid-%d (%.1f KB)",
                        uid,
                        len(raw) / 1024,
                    )
                except ValueError as exc:
                    logger.warning("Proposal from uid-%d rejected: %s", uid, exc)
                except Exception as exc:
                    logger.debug("Failed to read proposal from uid-%d: %s", uid, exc)

            if proposals:
                logger.info(
                    "  Fetched %d sibling proposals for round %d",
                    len(proposals),
                    round_id,
                )
            return proposals

        except Exception as exc:
            logger.warning("Validator proposal fetch failed: %s", exc)
            return []

    def _build_miner_loader(
        self,
        assignment: dict,
        tokenizer,
        device: str,
    ):
        """Build a DataLoader for a miner's assigned books + timestamp range.

        Reads from S3 when store is set, otherwise from local filesystem.
        """
        from torch.utils.data import DataLoader

        from GenTRX.src.dataloader import OrderDataset, ChunkSampler

        books = assignment.get("books", [])
        data_keys = assignment.get("data") or []
        if not books and not data_keys:
            return None

        # Score the exact pages the miner trained on (the assignment names
        # them). Fall back to book + ts-range resolution for any legacy
        # assignment that carries no data keys.
        if data_keys:
            cache_key = ("miner", tuple(sorted(data_keys)))
        else:
            cache_key = (
                "miner",
                tuple(sorted(str(b) for b in books)),
                int(assignment.get("ts_start", 0)),
                int(assignment.get("ts_end", 0)),
            )
        cached = self._loader_cache.get(cache_key)
        if cached is not None:
            self._loader_cache_hits += 1
            return cached
        self._loader_cache_misses += 1

        if data_keys:
            files = self._get_files_from_keys(data_keys)
        else:
            ts_start = assignment.get("ts_start", 0)
            ts_end = assignment.get("ts_end", 0)
            files = []
            for book_id in books:
                files.extend(self._get_book_files(book_id, ts_start, ts_end))

        if not files:
            _books_with_files = sum(
                1 for b in self._written_parquets if self._written_parquets[b]
            )
            logger.warning(
                "  no parquets for books=%s data=%d (books_with_any_flushed_data=%d/%d)",
                list(books),
                len(data_keys),
                _books_with_files,
                len(self._written_parquets) or 128,
            )
            return None

        ds = OrderDataset(files, seq_len=self.seq_len, tokenizer=tokenizer, max_cached=2)
        # See _build_val_loader_for_ranges for the rationale on num_workers=0;
        # same forked-worker accumulation problem applies here.
        loader = DataLoader(
            ds,
            batch_size=64,
            sampler=ChunkSampler(ds, shuffle=False),
            num_workers=0,
            pin_memory=False,
        )
        self._loader_cache[cache_key] = loader
        return loader

    def receive_state(self, data: bytes) -> None:
        """Receive a msgpack-encoded state tick from the validator.

        Called from the POST /gentrx/state endpoint. Processes the tick
        in-memory and accumulates rows for parquet flushing.
        """
        import msgpack as _msgpack

        try:
            tick = _msgpack.unpackb(
                data, raw=False, use_list=True, strict_map_key=False
            )
        except Exception as exc:
            logger.warning("Failed to unpack state: %s", exc)
            return
        self._enqueue_tick(tick)

    def _enqueue_tick(self, tick: dict) -> None:
        """Route a state tick through the reorder buffer.

        Fast path: if the buffer is empty and the tick is in order (ts >=
        last_seen or no ts), process immediately with no overhead.

        Slow path: if the tick arrives with a sim-time jump larger than
        _reorder_jump_ns, or while the buffer already holds earlier ticks,
        push onto the min-heap and drain whatever is ready.
        """
        import heapq as _heapq

        tick_ts = int(tick.get("ts", 0) or 0)
        wall_now = time.monotonic()

        in_order = not tick_ts or tick_ts >= self._last_seen_sim_ts

        # Update EMA from in-order positive increments only
        if tick_ts and self._last_seen_sim_ts and tick_ts > self._last_seen_sim_ts:
            increment = tick_ts - self._last_seen_sim_ts
            if self._ts_ema_samples == 0:
                self._ts_increment_ema = float(increment)
            else:
                self._ts_increment_ema = (
                    self._ts_ema_alpha * increment
                    + (1 - self._ts_ema_alpha) * self._ts_increment_ema
                )
            self._ts_ema_samples += 1

        expected_ns = (
            self._ts_increment_ema
            if self._ts_ema_samples >= 5
            else self._ts_ema_fallback
        )
        large_jump = (
            tick_ts
            and self._last_seen_sim_ts
            and (tick_ts - self._last_seen_sim_ts) > expected_ns
        )

        if not large_jump and not self._reorder_buf and in_order:
            self._process_tick(tick)
            return

        self._reorder_seq += 1
        _heapq.heappush(
            self._reorder_buf, (tick_ts or 0, self._reorder_seq, wall_now, tick)
        )
        self._drain_reorder_buf(wall_now)

    def _drain_reorder_buf(self, wall_now: float | None = None) -> None:
        """Process buffered ticks that are ready.

        A tick is ready when:
          - its sim-ts gap to last_seen_sim_ts is within _reorder_jump_ns
            (a filling tick has arrived, closing the gap), OR
          - it has waited at least _reorder_timeout_s wall-clock seconds.
        """
        import heapq as _heapq

        if wall_now is None:
            wall_now = time.monotonic()
        expected_ns = (
            self._ts_increment_ema
            if self._ts_ema_samples >= 5
            else self._ts_ema_fallback
        )
        while self._reorder_buf:
            sim_ts, _seq, enqueued_at, tick = self._reorder_buf[0]
            gap = sim_ts - self._last_seen_sim_ts if self._last_seen_sim_ts else 0
            timed_out = (wall_now - enqueued_at) >= self._reorder_timeout_s
            if timed_out or gap <= expected_ns:
                _heapq.heappop(self._reorder_buf)
                self._process_tick(tick)
            else:
                break

    def _reset_sim_buffers(self) -> None:
        """Drop every in-memory tick buffer so the next sim starts clean.

        Does not touch the on-S3 bucket — that's scheduled separately via
        `_data_cleanup_pending` and executed by `_run_data_cleanup` on the
        next aggregation-loop tick. Checkpoints, proposals, round counters,
        and model version are all preserved.
        """
        self._pending_rows = {}
        self._pending_interval_start = {}
        self._written_parquets = {}
        self._book_flush_locks = {}
        self._last_seen_sim_ts = 0
        self._reorder_buf = []
        self._reorder_seq = 0
        self._ts_increment_ema = 0.0
        self._ts_ema_samples = 0
        # Without this, the backwards-time check in _process_tick would
        # re-fire for every new-sim tick whose ts < old_max.
        self._max_timestamp_ns = 0
        self._restored_sim_id = None
        self._bucket_sim_id = None
        # Engines are per-book; drop them so the next sim replays events
        # against a fresh matching engine instead of stale depth.
        self._engines = {}
        # Invalidate the local pending-rows staging file so a subsequent
        # restart doesn't restore data from the old sim.
        try:
            if self._pending_staging_path.exists():
                self._pending_staging_path.unlink()
        except Exception:
            pass

    def _prune_checkpoints(self) -> None:
        """Trim this validator's checkpoints/<uid>/ to keep_checkpoints newest .pt files.

        Skips `latest.json` via the suffix filter — that pointer must
        survive every prune. No-op when keep_checkpoints<=0 or the
        validator store isn't wired (proxy-only / bootstrap states).
        """
        if self.validator_store is None or self.keep_checkpoints <= 0:
            return
        try:
            n = self.validator_store.prune_keep_latest(
                f"checkpoints/{self._validator_uid}/",
                keep=self.keep_checkpoints,
                suffix=".pt",
            )
            if n:
                logger.info(
                    "[GTX] pruned %d old checkpoint(s), keeping latest %d",
                    n,
                    self.keep_checkpoints,
                )
        except Exception as exc:
            logger.debug("checkpoint prune failed: %s", exc)

    def _regime_incompatible(self) -> bool:
        """True when the bucket's latest checkpoint predates our train regime.

        Reads only latest.json (no full checkpoint download). Missing stamp
        reads as regime 0, below any current regime. Defaults to compatible
        when the store or pointer is unreadable, so a transient read error
        never forces a spurious warmup.
        """
        if self.validator_store is None:
            return False
        try:
            meta = self.validator_store.get_latest_meta(self._validator_uid)
        except Exception as exc:
            logger.warning("Could not read checkpoint regime stamp: %s", exc)
            return False
        return int(meta.get("train_regime_version", 0)) < TRAIN_REGIME_VERSION

    def _rebaseline_incompatible_bucket(self) -> bool:
        """Publish the local current-format checkpoint as a new bucket version.

        Called at aggregator startup when the published checkpoint predates the
        current regime. The local checkpoint at this point is guaranteed
        current-format (an incompatible seed was quarantined + fresh-inited, or
        the deployed seed was re-stamped — see start()), so publishing it makes
        miners/siblings bootstrap from a compatible model instead of the stale
        one. Additive: writes a new (higher) version + updates latest.json/head;
        old versions are left for the normal prune. Idempotent: after this the
        bucket reads current-regime, so a later start won't re-fire. Best-effort
        — a publish failure is logged, not fatal (the first aggregation would
        still eventually publish, and the next start retries).

        Aggregator-only; siblings pull, they don't publish.
        """
        if not self.is_aggregator or self.validator_store is None:
            return False
        if not self.checkpoint_path.exists():
            return False
        try:
            new_v = int(self._version) + 1
            data = self.checkpoint_path.read_bytes()
            meta = checkpoint_stamp(self.label_smooth_sigma)
            self.validator_store.put_checkpoint(self._validator_uid, new_v, data, meta=meta)
            self.validator_store.put_head_version(self._validator_uid, new_v, meta=meta)
            self._version = new_v
            logger.warning(
                "Re-baselined aggregator bucket to v%d (regime %d): superseded the "
                "stale pre-regime checkpoint so miners/siblings bootstrap a "
                "compatible model (avoids the load-crash -> no-gradient -> "
                "no-aggregation deadlock)",
                new_v,
                TRAIN_REGIME_VERSION,
            )
            return True
        except Exception as exc:
            logger.error(
                "bucket re-baseline failed (%s) — first aggregation or next "
                "start will retry",
                exc,
            )
            return False

    def _read_bucket_sim_marker(self) -> str | None:
        if self.validator_store is None:
            return None
        try:
            return self.validator_store.get_sim_marker(self._validator_uid)
        except Exception:
            return None

    def _write_bucket_sim_marker(self, sim_id: str) -> None:
        if self.validator_store is None or not sim_id:
            return
        try:
            self.validator_store.put_sim_marker(self._validator_uid, sim_id)
        except Exception as exc:
            logger.warning("bucket sim_id marker write failed: %s", exc)

    def _ensure_sim_marker(self) -> None:
        """Keep data/<uid>/.sim_id consistent with the data we flush. Retries
        each flush until the write succeeds, so a transient failure can't leave
        a stale marker that wrongly trips the sim-transition wipe on restart."""
        if self.validator_store is None or not self._sim_id:
            return
        if self._sim_marker_written == self._sim_id:
            return
        try:
            self.validator_store.put_sim_marker(self._validator_uid, self._sim_id)
            self._sim_marker_written = self._sim_id
        except Exception as exc:
            logger.warning("sim_id marker write failed (retry next flush): %s", exc)

    def _prune_proposals(self) -> None:
        """Trim this validator's proposals/<uid>/ to keep_proposals newest .grad files."""
        if self.validator_store is None or self.keep_proposals <= 0:
            return
        try:
            n = self.validator_store.prune_keep_latest(
                f"proposals/{self._validator_uid}/",
                keep=self.keep_proposals,
                suffix=".grad",
            )
            if n:
                logger.info(
                    "[GTX] pruned %d old proposal(s), keeping latest %d",
                    n,
                    self.keep_proposals,
                )
        except Exception as exc:
            logger.debug("proposal prune failed: %s", exc)

    def _prune_s3_cache(self) -> int:
        """Evict local mirror parquets older than s3_cache_retention_hours.

        No-op when retention is disabled (<=0) or the cache dir has not
        been created yet. Also drops stale `_s3_cached_files` entries that
        point at unlinked files.
        """
        if self.s3_cache_retention_hours <= 0 or self._s3_cache_dir is None:
            return 0
        if not self._s3_cache_dir.exists():
            return 0
        cutoff = time.time() - self.s3_cache_retention_hours * 3600.0
        removed = 0
        for path in self._s3_cache_dir.rglob("*.parquet"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        if removed:
            stale = [k for k, p in self._s3_cached_files.items() if not p.is_file()]
            for k in stale:
                del self._s3_cached_files[k]
            logger.info(
                "[GTX] s3_cache eviction: removed %d file(s) older than %.0fh",
                removed,
                self.s3_cache_retention_hours,
            )
        return removed

    def _clear_local_data_cache(self) -> None:
        """Wipe the local s3_cache mirror on sim transition.

        Pairs with the S3-side `delete_prefix(data/<uid>/)` so the next sim
        starts from a clean local mirror as well as a clean bucket.
        """
        import shutil

        self._s3_cached_files.clear()
        if self._loader_cache:
            self._loader_cache.clear()
        if self._s3_cache_dir is not None and self._s3_cache_dir.exists():
            try:
                shutil.rmtree(self._s3_cache_dir)
                logger.info("[GTX] Local data cache wiped: %s", self._s3_cache_dir)
            except OSError as exc:
                logger.warning("Local data cache wipe failed: %s", exc)

    def _run_data_cleanup(self) -> int:
        """Delete stale parquets under data/ in the validator bucket and
        wipe the local mirror.

        Returns the number of S3 objects removed. No-op on the S3 side when
        no validator store is configured (proxy-only setups, or bootstrap
        before S3 is wired); the local mirror is wiped regardless. Clears
        `_data_cleanup_pending` even on failure so one transient S3 error
        does not loop the cleanup attempt forever.
        """
        self._data_cleanup_pending = False
        n = 0
        if self.validator_store is not None:
            try:
                n = self.validator_store.delete_prefix(
                    f"data/{self._validator_uid}/"
                )
                if n:
                    logger.info(
                        "[GTX] Sim transition cleanup: removed %d parquets from data/",
                        n,
                    )
            except Exception as exc:
                logger.warning("Sim transition cleanup failed: %s", exc)
        self._clear_local_data_cache()
        # delete_prefix also removed data/<uid>/.sim_id — re-push the marker for
        # the current binding so the bucket reflects the live sim (otherwise the
        # _sim_marker_written guard would think it's still written and skip it).
        self._sim_marker_written = None
        self._ensure_sim_marker()
        return n

    def _process_tick(self, tick: dict) -> None:
        """Process a single sim state tick into training rows.

        Replays events through MatchingEngine, accumulates rows per book,
        flushes to parquet when sim time crosses an interval boundary.
        Sim restarts are detected via sim_id change or ESE/ESS markers only —
        timestamp decreases are ignored as states may arrive out of order.
        """
        from GenTRX.src.orderbook import MatchingEngine
        from GenTRX.src.util.schema import (
            ASK,
            BID,
            CANCEL,
            DEFAULT_PRICE_DECIMALS,
            DEFAULT_VOLUME_DECIMALS,
            EXEC_BUY,
            EXEC_SELL,
            LOB_DEPTH,
        )

        # Sim lifecycle markers from state.notices. 'ESE' flags a sim end:
        # clear in-memory buffers so no sim-A tail contaminates sim-B ticks,
        # queue the S3 data/ cleanup for the next aggregation-loop tick, and
        # return without touching books on this tick (ESE ticks carry no
        # trading data worth retaining). 'ESS' on its own is the new-sim
        # marker; if cleanup was still pending we run it as a safety net.
        sim_events = tick.get("sim_events") or []
        if "ESE" in sim_events:
            self._reset_sim_buffers()
            self._data_cleanup_pending = True
            logger.info(
                "[GTX] Sim end received; in-memory state cleared, "
                "S3 data/ cleanup queued for next aggregation-loop tick"
            )
            return
        if "ESS" in sim_events:
            cfg = tick.get("config") or {}
            incoming_sim_id = cfg.get("simulation_id") if cfg else None
            # Idempotent for spool-replay/duplicate ESS: only reset when the
            # incoming sim_id actually differs from the current binding.
            if self._sim_id is not None and (
                incoming_sim_id is None or incoming_sim_id != self._sim_id
            ):
                self._sim_epoch += 1
                self._reset_sim_buffers()
                self._data_cleanup_pending = True
                self._sim_id = None
            if self._data_cleanup_pending:
                logger.info(
                    "[GTX] Sim start received with cleanup still pending; "
                    "running cleanup now before new sim data flushes"
                )
                self._run_data_cleanup()

        # Identity guard: drop ticks that can't be tied to a sim_id.
        incoming_sim_id = tick.get("sim_id") or (tick.get("config") or {}).get(
            "simulation_id"
        )
        if self._sim_id is None and not incoming_sim_id:
            self._pre_bind_drops += 1
            if self._pre_bind_drops <= 3 or self._pre_bind_drops % 64 == 0:
                logger.info(
                    "[GTX] dropping pre-bind tick (no sim_id yet); drops=%d",
                    self._pre_bind_drops,
                )
            return
        if (
            self._sim_id is not None
            and incoming_sim_id
            and incoming_sim_id != self._sim_id
        ):
            # Sim transition without an ESS marker; rebind below.
            self._sim_epoch += 1
            self._reset_sim_buffers()
            self._data_cleanup_pending = True
            logger.info(
                "sim_id changed via tick (%s → %s), advancing sim_epoch to %d",
                self._sim_id,
                incoming_sim_id,
                self._sim_epoch,
            )
            self._sim_id = None

        tick_ts = int(tick.get("ts", 0) or 0)

        # INGEST COUNTER. There was no way to tell a tick that carried nothing from a tick that was
        # never sent: /data-status only reports books that already have PARQUETS, so an aggregator
        # receiving a steady stream of event-free ticks looked identical to one receiving nothing at
        # all, for as long as you cared to watch. That ambiguity hid the exchange mechanism producing
        # zero training rows across several hour-long runs. Cheap counters, logged once a minute.
        self._ingest_ticks += 1
        _bk = tick.get("books") or {}
        self._ingest_books += len(_bk)
        self._ingest_events += sum(len((_b or {}).get("events") or []) for _b in _bk.values())
        _now_log = time.time()
        if _now_log - self._last_ingest_log >= 60.0:
            self._last_ingest_log = _now_log
            logger.info(
                "[GTX] ingest: %d tick(s), %d book-slot(s), %d event(s), last tick_ts=%d",
                self._ingest_ticks, self._ingest_books, self._ingest_events, tick_ts,
            )
            self._ingest_ticks = self._ingest_books = self._ingest_events = 0

        # Drop exact retry duplicates (sim_ts == last_seen). Strictly stale
        # arrivals (sim_ts < last_seen) are legitimate reorder-buffer drains
        # and must still process. The sim-swap path below handles sim_id changes.
        pkt_sim_id = incoming_sim_id
        same_sim = (
            pkt_sim_id is None or self._sim_id is None or pkt_sim_id == self._sim_id
        )
        if (
            tick_ts
            and self._last_seen_sim_ts
            and same_sim
            and tick_ts == self._last_seen_sim_ts
        ):
            self._dedup_drops_total += 1
            if self._dedup_drops_total == 1 or self._dedup_drops_total % 64 == 0:
                logger.info(
                    "Dedup: dropped %d duplicate ticks (tick_ts=%d == last_seen)",
                    self._dedup_drops_total,
                    tick_ts,
                )
            return

        if tick_ts:
            self._last_seen_sim_ts = tick_ts

        def _pad(values, depth):
            return list(values[:depth]) + [0] * (depth - len(values))

        if self._price_scale is None:
            cfg = tick.get("config") or {}
            pd = cfg.get("priceDecimals")
            vd = cfg.get("volumeDecimals")
            if pd is None or vd is None:
                pd = DEFAULT_PRICE_DECIMALS
                vd = DEFAULT_VOLUME_DECIMALS
                logger.warning(
                    "[GTX] state tick carried no decimals; falling back to "
                    "canonical sim defaults pd=%d vd=%d. Scaling is wrong if "
                    "the live sim differs.",
                    pd,
                    vd,
                )
            self._price_scale = 10**pd
            self._vol_scale = 10**vd
            logger.info(
                "[GTX] bound price_scale=10^%d vol_scale=10^%d", pd, vd
            )

        # Bind sim_id from any packet that carries one.
        new_sim_id = incoming_sim_id
        if new_sim_id and new_sim_id != self._sim_id:
            # Decide a genuine sim transition by the most reliable signal: the
            # locally-restored marker (written by THIS server while it was
            # running the sim) takes precedence over the bucket marker, which
            # can lag a failed write and otherwise trip a false wipe on a normal
            # restart. Only fall back to the bucket marker on a full crash where
            # no local marker survived.
            if self._restored_sim_id is not None:
                transition = self._restored_sim_id != new_sim_id
            elif self._bucket_sim_id is not None:
                transition = self._bucket_sim_id != new_sim_id
            else:
                transition = False
            if self._sim_id is None and transition and self._no_startup_cleanup:
                # KEEPING THE BUCKET IS NOT THE SAME AS KEEPING THE CLOCK.
                #
                # --no-startup-cleanup says "do not wipe stored data/", which is about the BUCKET. The
                # in-memory tick buffers belong to the PREVIOUS sim's clock and are meaningless against
                # the new one: interval_start sat at 907s from the old epoch while
                # the live sim was at 687s, so `clock - interval_start` was NEGATIVE and no book could
                # flush until the sim ran another ~520 seconds -- past the end of the acceptance window.
                # Reset them regardless; stored data is untouched either way.
                self._reset_sim_buffers()
                logger.warning(
                    "sim_id transition on restart (restored=%s, bucket=%s → live=%s) "
                    "but --no-startup-cleanup set: keeping existing data/, NOT wiping. "
                    "A genuine rollover detected at runtime will still clean up.",
                    self._restored_sim_id, self._bucket_sim_id, new_sim_id,
                )
            elif self._sim_id is None and transition:
                self._sim_epoch += 1
                self._reset_sim_buffers()
                self._data_cleanup_pending = True
                logger.info(
                    "sim_id transition on restart (restored=%s, bucket=%s → live=%s), "
                    "advancing sim_epoch to %d and queueing data/ cleanup",
                    self._restored_sim_id,
                    self._bucket_sim_id,
                    new_sim_id,
                    self._sim_epoch,
                )
            self._restored_sim_id = None
            self._sim_id = new_sim_id
            self._bucket_sim_id = new_sim_id
            self._sim_marker_written = None   # force a re-assert for the new binding
            self._ensure_sim_marker()
            logger.info(
                "Bound to sim_id=%s (sim_epoch=%d)", self._sim_id, self._sim_epoch
            )
            self._log_event({"type": "sim_bind"})

        books = tick.get("books", {})

        for book_id, book_data in books.items():
            book_id = int(book_id)
            bids = book_data.get("bids", [])
            asks = book_data.get("asks", [])
            events = book_data.get("events", [])

            if book_id not in self._engines:
                self._engines[book_id] = MatchingEngine()
                self._order_sides[book_id] = {}
                self._last_ts[book_id] = 0
                self._session_open_mid[book_id] = None
                self._pending_rows[book_id] = []
                self._pending_interval_start[book_id] = 0

            engine = self._engines[book_id]
            engine.reset()
            for level in reversed(bids):
                p = round(level[0] * self._price_scale)
                v = max(1, round(level[1] * self._vol_scale))
                if p > 0:
                    engine.process_order(BID, p, v, is_buy=True)
            for level in reversed(asks):
                p = round(level[0] * self._price_scale)
                v = max(1, round(level[1] * self._vol_scale))
                if p > 0:
                    engine.process_order(ASK, p, v, is_buy=False)

            taker_fill_qty: dict[int, float] = {}
            for ev in events:
                if ev.get("y") == "t":
                    tid = ev.get("Ti", 0)
                    taker_fill_qty[tid] = taker_fill_qty.get(tid, 0.0) + float(
                        ev.get("q", 0)
                    )

            # Executions are emitted as a distinct signed type at the true fill
            # price and do NOT drive the engine (the aggressing order does).
            # Orders emit only their resting remainder so a crossing order's
            # volume is not double-counted: fills (execution rows) + remainder
            # (this row) = original size, with the engine driven at full size.
            for ev in events:
                y = ev.get("y", "o")
                side = ev.get("s", 0)
                is_buy = side == 0
                price = float(ev.get("p") or 0)
                evt = int(ev.get("t", 0))
                price_ticks = round(price * self._price_scale)

                if y == "t":
                    order_type = EXEC_BUY if is_buy else EXEC_SELL
                    qty = float(ev.get("q", 0))
                    vol_ticks = 0
                    emit_row, drive_engine = True, False
                elif y == "c":
                    order_type = CANCEL
                    qty = float(ev.get("q", 0))
                    vol_ticks = max(1, round(qty * self._vol_scale))
                    emit_row, drive_engine = True, True
                else:
                    eid = ev.get("i", 0)
                    order_type = BID if is_buy else ASK
                    self._order_sides[book_id][eid] = is_buy
                    remaining = float(ev.get("q", 0))
                    full_qty = remaining + taker_fill_qty.get(eid, 0.0)
                    qty = remaining
                    vol_ticks = max(1, round(full_qty * self._vol_scale))
                    emit_row, drive_engine = remaining > 0, True

                snap = engine.snapshot()
                mid = snap.mid_price
                if self._session_open_mid[book_id] is None and mid > 0:
                    self._session_open_mid[book_id] = mid

                if emit_row:
                    ask_vols = _pad(snap.ask_volumes, LOB_DEPTH)
                    bid_vols = _pad(snap.bid_volumes, LOB_DEPTH)

                    row = {
                        "timestamp": evt,
                        "order_type": order_type,
                        "rel_price": price_ticks - mid if mid > 0 else 0,
                        "volume_int": int(qty),
                        "volume_dec": qty - int(qty),
                        "interval_ns": (
                            evt - self._last_ts[book_id]
                            if self._last_ts[book_id] > 0
                            else 0
                        ),
                        "mid_price": mid,
                        "time_of_day_s": int((evt // 1_000_000_000) % 86400),
                        "mid_price_delta": (
                            int(mid - self._session_open_mid[book_id])
                            if self._session_open_mid[book_id]
                            else 0
                        ),
                    }
                    for i in range(LOB_DEPTH):
                        row[f"lob_ask_vol_{i + 1}"] = float(ask_vols[i]) / self._vol_scale
                        row[f"lob_bid_vol_{i + 1}"] = float(bid_vols[i]) / self._vol_scale

                    self._pending_rows[book_id].append(row)
                    self._adv_events += 1
                    if order_type in (EXEC_BUY, EXEC_SELL):
                        self._adv_trades += 1
                    if self._pending_interval_start[book_id] == 0:
                        self._pending_interval_start[book_id] = evt

                if drive_engine:
                    engine.process_order(order_type, price_ticks, vol_ticks, is_buy)
                self._last_ts[book_id] = evt

                if evt > self._max_timestamp_ns:
                    self._max_timestamp_ns = evt

                # Fixed-row pages: flush once the staging buffer reaches the
                # page size, so active books emit uniform pages. With the cap
                # disabled the per-tick path is off and the aggregation-loop
                # tail flush is the sole trigger.
                _cap = self._max_pending_rows_per_book
                if _cap and emit_row and len(self._pending_rows[book_id]) >= _cap:
                    self._flush_book_parquet(book_id)

    def _flush_book_parquet(self, book_id: int) -> None:
        """Flush accumulated rows for a book to a parquet file on S3.

        Uses a per-book non-blocking lock to prevent duplicate flushes when
        cap-triggered _process_tick calls race for the same book. If another
        flush is already in flight, return immediately — the staging buffer
        keeps accumulating and the next cap/interval trigger will re-flush
        with the latest rows.
        """
        import io as _io

        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        from GenTRX.src.util.schema import LOB_DEPTH, order_stream_schema

        lock = self._book_flush_locks.setdefault(book_id, threading.Lock())
        if not lock.acquire(blocking=False):
            return
        try:
            self._flush_book_parquet_locked(
                book_id, _io, np, pa, pq, LOB_DEPTH, order_stream_schema
            )
        finally:
            lock.release()

    def _flush_book_parquet_locked(
        self, book_id, _io, np, pa, pq, LOB_DEPTH, order_stream_schema
    ):
        rows = list(self._pending_rows.get(book_id, []))
        if not rows:
            return
        rows.sort(key=lambda r: r["timestamp"])

        ts_min = rows[0]["timestamp"]
        ts_max = rows[-1]["timestamp"]
        tag_start = _ts_to_tag(ts_min)
        tag_end = _ts_to_tag(ts_max)
        pq_filename = f"{tag_start}-{tag_end}.parquet"
        # _ts_to_tag is second-resolution (ddHHMMSS), so cap-triggered flushes
        # within the same sim-second collide on (tag_start, tag_end) → identical
        # filename → S3 PUT overwrites the prior flush. Disambiguate with a
        # sequence suffix on the end tag; _tag_to_ns strips `_NNNN` before
        # parsing so _filter_by_timestamp / glob behavior is unchanged.
        _existing = self._written_parquets.get(book_id, [])
        if any(e[0] == pq_filename for e in _existing):
            _n = 1
            while any(
                e[0] == f"{tag_start}-{tag_end}_{_n:04d}.parquet" for e in _existing
            ):
                _n += 1
            pq_filename = f"{tag_start}-{tag_end}_{_n:04d}.parquet"

        try:
            columns = {
                "timestamp": pa.array(
                    [r["timestamp"] for r in rows], type=pa.timestamp("ns")
                ),
                "order_type": np.array([r["order_type"] for r in rows], dtype=np.int8),
                "rel_price": np.array([r["rel_price"] for r in rows], dtype=np.int64),
                "volume_int": np.array([r["volume_int"] for r in rows], dtype=np.int32),
                "volume_dec": np.array(
                    [r["volume_dec"] for r in rows], dtype=np.float32
                ),
                "interval_ns": np.array(
                    [r["interval_ns"] for r in rows], dtype=np.int64
                ),
                "mid_price": np.array([r["mid_price"] for r in rows], dtype=np.int64),
                "time_of_day_s": np.array(
                    [r["time_of_day_s"] for r in rows], dtype=np.int32
                ),
                "mid_price_delta": np.array(
                    [r["mid_price_delta"] for r in rows], dtype=np.int64
                ),
            }
            for i in range(LOB_DEPTH):
                k_ask = f"lob_ask_vol_{i + 1}"
                k_bid = f"lob_bid_vol_{i + 1}"
                columns[k_ask] = np.array([r[k_ask] for r in rows], dtype=np.float64)
                columns[k_bid] = np.array([r[k_bid] for r in rows], dtype=np.float64)

            table = pa.table(columns, schema=order_stream_schema())
            buf = _io.BytesIO()
            pq.write_table(table, buf)
            parquet_bytes = buf.getvalue()
            # Marker must be current before data lands, so a restart binding to
            # this sim sees a matching marker and resumes instead of wiping.
            self._ensure_sim_marker()
            # Write parquets to data bucket (read by miners + validator)
            self.validator_store.put_data(
                self._validator_uid,
                book_id=book_id,
                filename=pq_filename,
                data=parquet_bytes,
            )
            # Per-book, per-flush data-plane line: at ~128 books on a flush cadence
            # it dominates the log (~94% of lines) and buries round/score/aggregate
            # signal. Debug-level — the per-round buffer summaries and /data-status
            # already report that data is flowing; only useful when chasing a
            # specific book's missing data.
            logger.debug(
                "[GTX] Parquet flushed: book %d, %d rows, %s",
                book_id,
                len(rows),
                pq_filename,
            )
            # Mirror to local cache so scoring reads hit disk instead of
            # round-tripping back through S3 for data this process just wrote.
            try:
                cache_dir = self._get_s3_cache_dir()
                local_path = cache_dir / str(book_id) / "intervals" / pq_filename
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(parquet_bytes)
                self._s3_cached_files[f"{book_id}/{pq_filename}"] = local_path
            except Exception as exc:
                logger.debug(
                    "Local parquet mirror failed (book %d, %s): %s; "
                    "scoring will fall back to S3 download",
                    book_id,
                    pq_filename,
                    exc,
                )
            # Register and clear only on success — failed writes leave rows
            # intact so the next interval flush retries with the accumulated data.
            if book_id not in self._written_parquets:
                self._written_parquets[book_id] = []
            self._written_parquets[book_id].append((pq_filename, ts_min, ts_max))
            del self._pending_rows[book_id][: len(rows)]
            if self._pending_rows[book_id]:
                self._pending_interval_start[book_id] = self._pending_rows[book_id][0][
                    "timestamp"
                ]
            else:
                self._pending_interval_start[book_id] = 0
        except Exception as exc:
            logger.error(
                "Failed to write parquet for book %d: %s — rows retained for next flush",
                book_id,
                exc,
            )


def add_api_key_middleware(app, api_key: str) -> None:
    """Install X-API-Key gate on every request if `api_key` is truthy.

    Requests without a matching `X-API-Key` header get a 401 before any
    route handler runs. No-op when `api_key` is empty — callers that need
    auth-off semantics rely on binding to 127.0.0.1 instead.
    """
    if not api_key:
        return

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class APIKeyMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.headers.get("X-API-Key") != api_key:
                return JSONResponse(
                    {"error": "Invalid or missing X-API-Key"},
                    status_code=401,
                )
            return await call_next(request)

    app.add_middleware(APIKeyMiddleware)


def create_gradient_router(
    checkpoint_path: str,
    val_data_path: str,
    output_path: str | None = None,
    **kwargs,
) -> tuple[APIRouter, GradientAggregator]:
    """Create a FastAPI router + aggregator for gradient exchange.

    Data collection is handled by the validator/proxy — it saves
    mvtrx state updates directly to S3. The gradient server
    reads from S3 for scoring.

    Returns (router, aggregator) — caller must call aggregator.start().
    """
    aggregator = GradientAggregator(
        checkpoint_path=checkpoint_path,
        val_data_path=val_data_path,
        output_path=output_path,
        **kwargs,
    )

    router = APIRouter(prefix="/gentrx", tags=["gentrx"])

    @router.post("/state")
    async def receive_state(request: Request):
        from fastapi import HTTPException
        from starlette.requests import ClientDisconnect

        try:
            body = await request.body()
        except ClientDisconnect:
            # 503 triggers client retry; sim_ts dedup absorbs duplicates.
            raise HTTPException(status_code=503, detail="client_disconnect_during_read")
        aggregator.receive_state(body)
        return {"status": "ok"}

    @router.get("/assignment")
    async def get_assignment(miner_uid: int = 0):
        """Get book/interval assignment for a miner."""
        return aggregator.get_assignment(miner_uid)

    @router.get("/health")
    async def health():
        return {"status": "ok", "version": aggregator.version}

    @router.get("/version")
    async def get_version():
        # Used by the validator service to detect new aggregation rounds.
        return {"version": aggregator.version}

    @router.get("/scores")
    async def get_scores(since_round: int = -1):
        """Get latest scores. Returns 204 if no new scores since since_round."""
        from fastapi.responses import Response as _Resp

        scores = aggregator._latest_scores
        if scores is None:
            return _Resp(status_code=204)
        if scores.get("round", -1) <= since_round:
            return _Resp(status_code=204)  # no new scores
        return scores

    @router.get("/metrics")
    async def prometheus_metrics():
        """Prometheus metrics for Grafana scraping (pull model).

        Exposes training round progress, gradient acceptance rates, per-miner
        scores, loss improvement, and aggregation timing.  Scrape at:
            http://<host>:<port>/gentrx/metrics
        """
        from fastapi.responses import Response as _Resp

        try:
            from prometheus_client import (
                CONTENT_TYPE_LATEST,
                CollectorRegistry,
                Counter,
                Gauge,
                generate_latest,
            )
        except ImportError:
            return _Resp(status_code=503, content="prometheus_client not installed")

        reg = CollectorRegistry()
        labels = ["netuid", "validator_uid"]
        lv = [
            str(getattr(aggregator._chain, "netuid", "") if aggregator._chain else ""),
            str(aggregator._validator_uid),
        ]

        def _g(name, doc, extra_labels=()):
            return Gauge(name, doc, labels + list(extra_labels), registry=reg)

        # Round / version
        g_round = _g("gentrx_agg_round", "Current aggregation round")
        g_version = _g(
            "gentrx_checkpoint_version",
            "Model checkpoint version (increments on accepted round)",
        )
        g_rounds_total = _g(
            "gentrx_rounds_aggregated_total", "Cumulative rounds aggregated"
        )
        g_rollbacks = _g(
            "gentrx_rollbacks_total",
            "Cumulative aggregations rolled back (no improvement)",
        )

        # Last aggregation
        g_n_scored = _g("gentrx_last_n_scored", "Gradients scored in last round")
        g_n_accepted = _g("gentrx_last_n_accepted", "Gradients accepted in last round")
        g_loss_before = _g(
            "gentrx_last_loss_before", "Model loss before last aggregation"
        )
        g_loss_after = _g("gentrx_last_loss_after", "Model loss after last aggregation")
        g_loss_delta = _g(
            "gentrx_last_loss_improvement",
            "Loss reduction in last round (before − after)",
        )
        g_t_score = _g(
            "gentrx_last_score_duration_s",
            "Gradient scoring duration (s) in last round",
        )
        g_t_agg = _g(
            "gentrx_last_aggregate_duration_s", "Aggregation duration (s) in last round"
        )
        g_t_total = _g(
            "gentrx_last_total_duration_s", "Total round duration (s) in last round"
        )

        # Active miners
        g_active = _g(
            "gentrx_active_miners", "Miners with non-zero gradient score this round"
        )
        g_benchmark = _g(
            "gentrx_benchmark_miners",
            "Benchmark (off-chain) miners registered with gradient server",
        )

        # Per-miner scores — is_benchmark=1 for UIDs registered via /register_bucket
        # (benchmark agents tracked for scoring but excluded from on-chain rewards)
        g_score_own = _g(
            "gentrx_miner_score_own",
            "Miner own-data score last round",
            ("miner_uid", "is_benchmark"),
        )
        g_score_held = _g(
            "gentrx_miner_score_held",
            "Miner held-out score last round",
            ("miner_uid", "is_benchmark"),
        )
        g_score = _g(
            "gentrx_miner_score",
            "Miner combined score last round",
            ("miner_uid", "is_benchmark"),
        )
        g_accepted = _g(
            "gentrx_miner_accepted",
            "1 if miner gradient accepted last round",
            ("miner_uid", "is_benchmark"),
        )
        g_miner_grad_norm = _g(
            "gentrx_miner_grad_norm",
            "Miner gradient L2 norm last round (0 if not submitted)",
            ("miner_uid", "is_benchmark"),
        )
        g_rollback_winner = _g(
            "gentrx_miner_was_rollback_winner",
            "1 if this miner's gradient was the chosen-but-reverted delta last round",
            ("miner_uid", "is_benchmark"),
        )

        g_round.labels(*lv).set(aggregator._agg_round)
        g_version.labels(*lv).set(aggregator._version)
        g_rounds_total.labels(*lv).set(aggregator._rounds_aggregated_total)
        g_rollbacks.labels(*lv).set(aggregator._rollbacks_total)

        agg = aggregator._last_aggregation
        g_n_scored.labels(*lv).set(agg.get("n_scored", 0))
        g_n_accepted.labels(*lv).set(agg.get("n_accepted", 0))
        lb = agg.get("loss_before")
        la = agg.get("loss_after")
        if lb is not None:
            g_loss_before.labels(*lv).set(lb)
        if la is not None:
            g_loss_after.labels(*lv).set(la)
        if lb is not None and la is not None:
            g_loss_delta.labels(*lv).set(lb - la)
        g_t_score.labels(*lv).set(agg.get("t_score_s", 0))
        g_t_agg.labels(*lv).set(agg.get("t_aggregate_s", 0))
        g_t_total.labels(*lv).set(agg.get("t_total_s", 0))

        benchmark_uids = set(aggregator._static_buckets.keys())
        g_benchmark.labels(*lv).set(len(benchmark_uids))

        scores = aggregator._latest_scores
        if scores:
            per_miner = scores.get("scores", {})
            g_active.labels(*lv).set(
                sum(1 for v in per_miner.values() if v.get("accepted"))
            )
            for uid_str, info in per_miner.items():
                is_bm = "1" if int(uid_str) in benchmark_uids else "0"
                mlv = lv + [uid_str, is_bm]
                g_score_own.labels(*mlv).set(float(info.get("score_own") or 0.0))
                g_score_held.labels(*mlv).set(float(info.get("score_held") or 0.0))
                g_score.labels(*mlv).set(float(info.get("score", 0.0) or 0.0))
                g_accepted.labels(*mlv).set(1 if info.get("accepted") else 0)
                g_miner_grad_norm.labels(*mlv).set(float(info.get("grad_norm") or 0.0))
                g_rollback_winner.labels(*mlv).set(
                    1 if info.get("was_rollback_winner") else 0
                )
        else:
            g_active.labels(*lv).set(0)

        return _Resp(content=generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    @router.post("/register_bucket/{uid}")
    async def register_bucket(uid: int, request: Request):
        """Register a static bucket for a UID not on-chain (e.g. benchmark miners).

        Body (JSON): {
            "endpoint_url": "http://...",
            "bucket_name": "agent-0",
            "access_key_id": "...",
            "secret_access_key": "..."
        }
        The entry is merged into _miner_buckets and survives subsequent chain
        refreshes because _refresh_miner_buckets only overwrites the dict when
        the chain returns a non-empty result, but the merge happens after.
        """
        from GenTRX.src.chain import BucketInfo

        data = await request.json()
        bi = BucketInfo(
            account_id=data.get("account_id", data.get("bucket_name", "")),
            access_key_id=data["access_key_id"],
            secret_access_key=data["secret_access_key"],
            _endpoint_override=data.get("endpoint_url"),
            _bucket_override=data.get("bucket_name"),
        )
        aggregator._miner_buckets[uid] = bi
        aggregator._static_buckets[uid] = bi
        logger.info(
            "[GTX] registered static bucket for uid=%d bucket=%s", uid, bi.bucket_name
        )
        return {"status": "ok", "uid": uid, "bucket": bi.bucket_name}

    @router.get("/data-status")
    async def get_data_status():
        """Return available data ranges per book.

        Used by the validator to create assignments from what actually exists.
        Returns:
            {
                "max_ts": int,        # global max sim timestamp (ns)
                "version": int,       # current model version
                "round": int,         # current aggregation round
                "books": {
                    "0": {"parquets": ["00000000-00000300.parquet", ...], "max_ts": int},
                    ...
                }
            }
        """
        books = {}
        for bid, plist in aggregator._written_parquets.items():
            bmax = max((end for _, _, end in plist), default=0)
            # Include f_start/f_end so the validator can filter to each miner's
            # training window instead of sending every parquet key.
            books[str(bid)] = {
                "parquets": [
                    [fname, f_start, f_end] for fname, f_start, f_end in plist
                ],
                "max_ts": bmax,
            }
        return {
            "max_ts": aggregator._max_timestamp_ns,
            "version": aggregator._version,
            "round": aggregator._agg_round,
            "books": books,
            # THE SHARD THIS AGGREGATOR READS. The validator stamps it on every assignment it creates,
            # so a miner learns where its gradient must land instead of deriving it from a per-tick
            # mechanism flag -- which oscillates whenever one axon serves both mechanisms, filing
            # gradients on whichever shard the last tick happened to name.
            "bucket_prefix": aggregator._bucket_prefix,
        }

    @router.post("/round")
    async def push_round(request: Request):
        """Accept an assignment plan from the validator.

        The validator drives round scheduling (block-based or timer-based),
        creates assignments from available data, and pushes the plan here.
        The gradient server records the assignments and manages the lifecycle
        (gradient collection, scoring, aggregation).

        Body (JSON):
            {
                "round": int,
                "assignments": {
                    "uid": {round, model_version, books, ts_start, ts_end, data, ...},
                    ...
                }
            }
        """
        import json as _json_mod

        body = await request.body()
        payload = _json_mod.loads(body)

        round_id = payload.get("round", aggregator._agg_round)
        assignments = payload.get("assignments", {})
        push_block = payload.get("block")
        if push_block is not None:
            aggregator._last_push_block = int(push_block)

        # Record the validator's held-out split for this round so scoring holds
        # out exactly the books training excluded. Prune stale rounds.
        pushed_val = payload.get("val_books")
        if pushed_val is not None:
            aggregator._pushed_val_books[round_id] = frozenset(str(b) for b in pushed_val)
            cutoff = round_id - 8
            for r in [r for r in aggregator._pushed_val_books if r < cutoff]:
                aggregator._pushed_val_books.pop(r, None)

        # The validator drives round cadence. A pushed round BELOW ours means
        # our _agg_round is stale-ahead (restored from a prior run, or seeded
        # off a since-reset chain). Left alone it deadlocks aggregation forever,
        # since the advance+close path below only fires on round_id > _agg_round.
        # Resync down to the validator's authoritative round and drop the
        # now-future pending-aggregation entries.
        if round_id < aggregator._agg_round:
            logger.warning(
                "[GTX] round resync: validator round %d < server _agg_round %d "
                "(stale restore/seed) — adopting validator round",
                round_id, aggregator._agg_round,
            )
            aggregator._agg_round = round_id
            aggregator._pending_aggregation_rounds = {
                r for r in aggregator._pending_aggregation_rounds if r < round_id
            }

        if round_id > aggregator._agg_round:
            prior_round = aggregator._agg_round
            # Preserve the closing round's assignments in _prev_round_assignments
            # so the loop can still find them for collection + aggregation.
            # Without this, the new round's installation below overwrites
            # them and round N's gradients never get scored.
            closing = {
                uid: a
                for uid, a in list(aggregator._assignments.items())
                if a.get("round") == prior_round
            }
            if closing:
                aggregator._prev_round_assignments.update(closing)
                # Remove from live _assignments to keep it round-N+1-only.
                for uid in closing:
                    aggregator._assignments.pop(uid, None)

            # Skip-stale policy: if the validator jumped multiple rounds ahead,
            # drop assignments older than prior_round (both live and in history).
            gap = round_id - aggregator._agg_round
            if gap > 1:
                stale_live = [
                    uid
                    for uid, a in list(aggregator._assignments.items())
                    if a.get("round", -1) < prior_round
                ]
                for uid in stale_live:
                    aggregator._assignments.pop(uid, None)
                stale_prev = [
                    uid
                    for uid, a in list(aggregator._prev_round_assignments.items())
                    if a.get("round", -1) < prior_round
                ]
                for uid in stale_prev:
                    aggregator._prev_round_assignments.pop(uid, None)
                logger.warning(
                    "[GTX] round jump %d -> %d: dropped %d stale assignments",
                    prior_round,
                    round_id,
                    len(stale_live) + len(stale_prev),
                )
            if closing:
                aggregator._pending_aggregation_rounds.add(prior_round)
                aggregator._pending_aggregation_at.setdefault(prior_round, time.time())
            aggregator._agg_round = round_id

        # Install the validator's assignments as DELIVERED. The validator
        # has already sent them to miners via dendrite, so the
        # heartbeat-loss fallback timer starts now.
        now = time.time()
        for uid_str, a in assignments.items():
            uid = int(uid_str)
            a["_state"] = "DELIVERED"
            a["_created_at"] = now
            a["_delivered_at"] = now
            a["_gradient_data"] = None
            a["_score"] = None
            aggregator._assignments[uid] = a

        aggregator._last_round_log_ts = 0.0  # emit status on next loop tick
        logger.info(
            "[GTX] round=%d accepted from validator: %d assignments",
            round_id,
            len(assignments),
        )
        aggregator._log_event(
            {
                "type": "round_delivered",
                "round": round_id,
                "n_assignments": len(assignments),
            }
        )
        return {"status": "ok", "round": round_id, "n_assignments": len(assignments)}

    return router, aggregator


# ---------------------------------------------------------------------------
# Standalone server for testing
# ---------------------------------------------------------------------------


def _resolve_validator_uid(args, chain, validator_store) -> str:
    """Best-effort: derive this gradient server's UID from on-chain commitments.

    Matches the validator_store's (bucket, endpoint) against each UID's
    committed BucketInfo. Returns the operator's --validator-uid if set,
    otherwise the discovered UID, otherwise "0" (proxy mode, no chain,
    or pre-commit bootstrap).
    """
    import bittensor as bt

    if args.validator_uid:
        return str(args.validator_uid)
    if chain is None or validator_store is None:
        return "0"
    target_bucket = (validator_store.bucket or "").lower()
    target_endpoint = (validator_store.endpoint_url or "").lower()
    if not target_bucket:
        return "0"
    try:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            buckets = loop.run_until_complete(chain.get_miner_buckets())
        finally:
            loop.close()
    except Exception as exc:
        bt.logging.warning(f"Validator UID discovery: chain query failed: {exc}")
        return "0"
    for uid, bi in buckets.items():
        if (bi.bucket_name or "").lower() == target_bucket and (
            (bi.endpoint_url or "").lower() == target_endpoint
        ):
            bt.logging.info(f"Validator UID resolved from chain commitment: {uid}")
            return str(uid)
    bt.logging.warning(
        f"Validator UID not found in {len(buckets)} chain commitments; "
        "defaulting to '0'. Pass --validator-uid explicitly if the validator "
        "process has not committed its bucket yet."
    )
    return "0"


if __name__ == "__main__":
    import argparse

    import uvicorn
    from fastapi import FastAPI

    parser = argparse.ArgumentParser(description="GenTRX gradient exchange server")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--val-data", required=True, help="Data directory (collected parquets)"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--log-path",
        default=None,
        help="JSONL log for dashboard (default: next to output)",
    )
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1 = localhost only. "
        "Use 0.0.0.0 for remote access, but set --api-key for security).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GENTRX_API_KEY", ""),
        help="Shared secret for validator↔gradient server auth. "
        "When set, all requests must include X-API-Key header. "
        "Also reads from GENTRX_API_KEY env var.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Aggregation interval in seconds (default: 30)",
    )
    parser.add_argument("--min-score", type=float, default=-0.1)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Reject a miner gradient whose L2 norm exceeds this (default: off). "
        "Set from the per-miner grad_norm log once the legitimate range is known.",
    )
    parser.add_argument(
        "--max-agg-norm",
        type=float,
        default=None,
        help="Cap the aggregated delta's L2 norm by scaling it down (default: off). "
        "Direction-preserving safety net for outlier rounds where one large gradient "
        "dominates the top-k re-compression. The rollback check itself catches "
        "catastrophic applies; this is belt-and-braces. Set to a number to engage.",
    )
    parser.add_argument(
        "--max-agg-element",
        type=float,
        default=None,
        help="Clamp each entry of the aggregated delta to ±this value (default: off). "
        "Catches single-weight overshoots the norm cap misses. Set to a number to engage.",
    )
    parser.add_argument(
        "--reject-untagged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject gradients with no model-version tag (default: ON). Every "
        "current miner stamps model_v_trained via _make_window_config, so an "
        "untagged gradient post-warmup is a stale/replayed/pre-patch submission "
        "and must not be paid. Warmup bypasses this (bootstrap accepts all), so "
        "it only bites after the warmup window. Use --no-reject-untagged to "
        "accept untagged gradients (legacy pre-tagging behaviour).",
    )
    parser.add_argument(
        "--enable-startup-cleanup",
        action="store_true",
        help="Allow wiping data/ on the startup sim-id bind. OFF by default: a "
        "pm2 restart resumes existing data instead of risking a false wipe from "
        "a stale marker. A genuine rollover detected at runtime always cleans up.",
    )
    parser.add_argument(
        "--label-smooth-sigma",
        type=float,
        default=1.0,
        help="Ordinal-aware soft-CE width in bins used for scoring (0 = strict "
        "CE). MUST match the miners' --gtx-label-smooth-sigma, or gradient "
        "scores compare losses the miners never trained against.",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=5,
        help="Rounds with no min_score/rollback after fresh model init (default: 5)",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=10,
        help="Max batches of validation data to evaluate each miner's gradient "
        "on. Default 10 — loss signal saturates fast, compute scales linearly. "
        "Bump to 30-50 only if you suspect noisy scores; drop to 5 for faster "
        "per-round aggregation on CPU-only gradient servers.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for scoring loaders. Must match miners' "
        "gtx_train_seq_len so the model is scored at its training context.",
    )
    parser.add_argument(
        "--max-loader-files",
        type=int,
        default=0,
        help="Cap parquet files the val loader scans per book (0 = no cap).",
    )
    parser.add_argument(
        "--agg-top-k-frac",
        type=float,
        default=1.0,
        help="Top-k fraction kept when re-compressing the aggregated delta "
             "applied to the canonical model and published as the version delta "
             "(1.0 = dense). Below ~0.1 the applied update is a sparse subset "
             "biased toward per-miner page overfit; dense applies the full mean.",
    )
    parser.add_argument(
        "--books-per-miner",
        type=int,
        default=3,
        help="Page files assigned per miner per round (one fixed-row page per "
             "book). Miners train them incrementally within the round budget, so "
             "weak hardware trains fewer and strong hardware trains all of them.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Fraction of books held for validation per interval",
    )
    parser.add_argument(
        "--parquet-interval-ns",
        type=int,
        default=300_000_000_000,
        help="Sim time per parquet file (default: 5min = 300_000_000_000 ns, matches training window)",
    )
    parser.add_argument(
        "--max-pending-rows-per-book",
        type=int,
        default=30_000,
        help="Page size: flush a book's staging buffer to parquet once it reaches "
             "this many rows. This is the primary flush trigger, so active books "
             "emit uniform fixed-row pages; the sim-time interval only tail-flushes "
             "a stalled book's partial page. 0 disables the row trigger "
             "(sim-time interval flush only).",
    )
    parser.add_argument(
        "--loop-sleep-s",
        type=float,
        default=5.0,
        help="Sleep between aggregation loop iterations (default: 5s)",
    )
    parser.add_argument(
        "--blocks-per-round",
        type=int,
        default=25,
        help="Server-side estimate of how many blocks make up one round. "
        "Used only by the heartbeat-loss fallback in _round_complete. "
        "Should match the validator's --gentrx.blocks_per_round.",
    )
    parser.add_argument(
        "--block-time-s",
        type=float,
        default=12.0,
        help="Assumed seconds per block on the target chain (default: 12s, "
        "finney). Combined with --blocks-per-round to estimate the "
        "round duration for the heartbeat-loss fallback.",
    )
    parser.add_argument(
        "--round-grace-s",
        type=float,
        default=30.0,
        help="Grace seconds added to the heartbeat-loss fallback estimate "
        "before force-closing a round (default: 30s). Only fires if "
        "the validator stops pushing POST /gentrx/round.",
    )
    parser.add_argument(
        "--max-gradient-bytes",
        type=int,
        default=20 * 1024 * 1024,
        help="Reject gradients larger than this (default: 20 MB). The n_types=5 "
        "full_final model produces ~14.5 MB top-k-compressed gradients; 20 MB "
        "leaves headroom. Raise further only if a larger model is deployed.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=10,
        help="Keep newest N checkpoints in the validator bucket (default: 10). "
        "Older checkpoints are deleted after a new one is published. "
        "Set 0 to disable pruning (you handle cleanup yourself — "
        "consider this if you mirror checkpoints to cold storage).",
    )
    parser.add_argument(
        "--keep-proposals",
        type=int,
        default=10,
        help="Keep newest N proposals in the validator bucket (default: 10). "
        "Older proposals are deleted after a new one is published. "
        "Set 0 to disable pruning.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=6,
        help="Upload a full checkpoint every N model versions (default: 6). "
        "In between, only the small per-version delta is published; miners "
        "advance by applying it. Lower = safer cold-start, more bandwidth.",
    )
    parser.add_argument(
        "--keep-version-deltas",
        type=int,
        default=24,
        help="Keep newest N per-version deltas in the validator bucket "
        "(default: 24). Must comfortably exceed checkpoint-interval so a "
        "cold-starter can replay from the latest baseline.",
    )
    parser.add_argument(
        "--publish-state-hash",
        action="store_true",
        help="Publish a per-version model state hash for the optional miner "
        "drift check. Off by default.",
    )
    parser.add_argument(
        "--s3-cache-retention-hours",
        type=float,
        default=24.0,
        help="Evict local parquet mirror (under <output>/s3_cache/) files "
        "older than this. Default 24h. Set 0 to disable rolling eviction "
        "(the sim-end wipe still runs).",
    )
    parser.add_argument(
        "--window-ns",
        type=int,
        default=None,
        help="Training window per assignment in ns (default: 5min = 300_000_000_000 ns)",
    )
    parser.add_argument(
        "--miner-buckets",
        default=None,
        help="JSON file with per-miner bucket config (LocalBucketConfig). "
        "Use only for proxy test (no chain). Localnet/production use --subtensor-network.",
    )
    parser.add_argument(
        "--subtensor-network",
        default=None,
        help="Bittensor network endpoint for chain-based bucket discovery "
        "(e.g. 'local', 'finney', 'wss://...'). Mutually exclusive with --miner-buckets.",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=None,
        help="Subnet netuid for chain commitment lookup (required with --subtensor-network).",
    )
    parser.add_argument(
        "--mode",
        default="simulation",
        choices=["simulation", "exchange"],
        help="Training mode shard for bucket keys (default: simulation). "
        "All keys live under gentrx/<network>/<mode>/. "
        "All keys live under gentrx/<network>/<mode>/, so the two modes are separate "
        "shards of the same store and never mix. BOTH MODES ARE SUPPORTED: mode selects "
        "the prefix and nothing else branches on it, so an aggregator and the validators "
        "pushing to it need only agree on the value. Run ONE SERVER PER MODE when both "
        "mechanisms are active, so a gradient is attributable to the mechanism that "
        "produced it.",
    )
    parser.add_argument(
        "--network",
        default=None,
        # Accept the same aliases network_from_subtensor() normalises, so
        # GENTRX_NETWORK=local (what run_gradients.sh forwards as
        # `--network local`) is valid — not just the canonical "localnet".
        choices=["mainnet", "main", "testnet", "test", "localnet", "local"],
        help="Explicit network shard for bucket keys (mainnet/main, "
        "testnet/test, or localnet/local). Overrides the heuristic derived "
        "from --subtensor-network / --netuid. Required when connecting via a "
        "custom wss:// endpoint that is not automatically recognised. "
        "Equivalent to setting GENTRX_NETWORK in the environment.",
    )
    parser.add_argument(
        "--endpoint-override",
        default=os.environ.get("GENTRX_CHAIN_ENDPOINT_OVERRIDE") or None,
        help="Override S3 endpoint for all miner buckets (useful for MinIO localnet "
        "where on-chain commitments don't include the endpoint URL). "
        "Also reads from GENTRX_CHAIN_ENDPOINT_OVERRIDE env var.",
    )
    parser.add_argument(
        "--is-aggregator",
        action="store_true",
        default=True,
        help="This server is the uid-0 aggregator: aggregates gradients and "
        "publishes canonical checkpoints. Use --no-is-aggregator for sibling "
        "validators that only score and publish scores.",
    )
    parser.add_argument(
        "--no-is-aggregator",
        dest="is_aggregator",
        action="store_false",
        help="Run as a sibling validator: score miners and publish scores only, "
        "no gradient aggregation or checkpoint publish.",
    )
    parser.add_argument(
        "--validator-uid",
        default="",
        help="Validator UID on the subnet. Used for both Prometheus metric "
        "labels and the proposals/<uid>/ path scoping. Optional — when "
        "omitted, the gradient server self-discovers it by matching its "
        "validator_store bucket against on-chain commitments. Falls back to "
        "'0' if discovery fails (proxy mode, pre-commit bootstrap).",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", ""),
        help="Enable wandb mirroring of aggregation events to the named project. "
        "Requires WANDB_API_KEY. Empty (default) = wandb disabled.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_RUN_NAME", ""),
        help="Optional wandb run name (default: wandb auto-generates).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log verbosity for GenTRX loggers (default: info).",
    )
    args = parser.parse_args()

    # CPU core affinity — pin to the last GRAD_CORES_COUNT cores (same pattern as
    # the validator via taos/im/utils/affinity.py).  The validator excludes these
    # cores from its own affinity so the two processes don't contend.
    _grad_cores_count = int(os.environ.get("GRAD_CORES_COUNT", "0"))
    if _grad_cores_count > 0:
        import multiprocessing as _mp

        _total_cpu = _mp.cpu_count()
        if _total_cpu > _grad_cores_count:
            _grad_cores = list(range(_total_cpu - _grad_cores_count, _total_cpu))
            try:
                os.sched_setaffinity(0, set(_grad_cores))
            except (AttributeError, OSError):
                pass  # not available on macOS / non-Linux
            import torch as _torch

            _torch.set_num_threads(_grad_cores_count)

    import bittensor as bt

    # Configure bt.logging before bt.Subtensor() so bt never defaults to
    # Warning level and suppresses our INFO messages.
    if args.log_level == "debug":
        bt.logging.set_debug()
    else:
        bt.logging.set_info()
    # enable_third_party_loggers is called once after bt.Subtensor so that all
    # loggers (including ones bt.Subtensor creates) are covered in a single pass.
    # Calling it here too would add a second QueueHandler to each logger and
    # cause duplicate log entries.

    # Bucket layout: gentrx/<network>/<mode>/...
    # network is derived from the connected subtensor (finney → mainnet,
    # everything else → testnet). mode is operator-selected via --mode.
    from GenTRX.src.gradient_store import (
        create_validator_store_from_env,
        gentrx_prefix,
        network_from_subtensor,
    )

    if args.network:
        os.environ["GENTRX_NETWORK"] = args.network
    # Pass netuid so a custom endpoint (e.g. a localnet ws://<ip> that isn't
    # finney/loopback) resolves via the deterministic netuid map (79→mainnet,
    # 366→testnet, else→localnet) instead of the "unknown wss → mainnet"
    # fallback. Without this the aggregator landed on `mainnet` for a netuid-2
    # localnet while the validator (which passes netuid) used `localnet`, so
    # their S3 prefixes diverged and the aggregator never saw miner gradients.
    network = network_from_subtensor(args.subtensor_network, netuid=args.netuid)
    bucket_prefix = gentrx_prefix(network, args.mode)
    bt.logging.info(
        f"Bucket prefix: {bucket_prefix} (network={network}, mode={args.mode})"
    )

    # Single validator bucket: checkpoints/ + data/ + proposals/ all live
    # under the (network, mode) prefix. Committed to chain so miners +
    # aggregator discover it without pre-config.
    validator_store = create_validator_store_from_env(
        mode="write", prefix=bucket_prefix
    )
    if validator_store:
        bt.logging.info(
            f"S3 validator store (write): {validator_store.endpoint_url}/{validator_store.bucket}"
        )
    else:
        parser.error("GENTRX_VALIDATOR_S3_* env vars required for gradient server")

    # Bucket discovery: chain (production/localnet) or local JSON (proxy test)
    chain = None
    if args.subtensor_network:
        if args.netuid is None:
            parser.error("--netuid is required when --subtensor-network is set")
        from GenTRX.src.chain import GenTRXChain

        sub = bt.Subtensor(network=args.subtensor_network)
        meta = sub.metagraph(args.netuid)
        chain = GenTRXChain(sub, args.netuid, meta)
        chain._endpoint_override = args.endpoint_override  # for MinIO localnet
        bt.logging.info(
            f"Chain bucket discovery: network={args.subtensor_network} "
            f"netuid={args.netuid} (override={args.endpoint_override or 'none'})"
        )
    elif args.miner_buckets:
        from GenTRX.src.chain import LocalBucketConfig

        chain = LocalBucketConfig(args.miner_buckets)
        bt.logging.info(
            f"Miner buckets loaded from {args.miner_buckets} (proxy test mode)"
        )

    args.validator_uid = _resolve_validator_uid(args, chain, validator_store)

    # Route all non-bt loggers (GenTRX.src.*, root, …) through bt.logging so
    # gradient server log format matches the validator. Called once here, after
    # bt.Subtensor, so every logger bt creates during init is covered in a
    # single pass. Calling it earlier too would add duplicate QueueHandlers.
    bt.logging.enable_third_party_loggers()
    # Suppress noisy uvicorn access logs — must come after enable_third_party
    # since that call resets all levels to bt's current level.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    app = FastAPI(title="GenTRX Gradient Server")
    router, aggregator = create_gradient_router(
        checkpoint_path=args.checkpoint,
        val_data_path=args.val_data,
        output_path=args.output,
        log_path=args.log_path,
        interval=args.interval,
        min_score=args.min_score,
        max_grad_norm=args.max_grad_norm,
        max_agg_norm=args.max_agg_norm,
        max_agg_element=args.max_agg_element,
        seq_len=args.seq_len,
        no_startup_cleanup=not args.enable_startup_cleanup,
        warmup_rounds=args.warmup_rounds,
        label_smooth_sigma=args.label_smooth_sigma,
        max_val_batches=args.max_val_batches,
        max_loader_files=args.max_loader_files,
        agg_top_k_frac=args.agg_top_k_frac,
        books_per_miner=args.books_per_miner,
        val_fraction=args.val_fraction,
        validator_store=validator_store,
        chain=chain,
        is_aggregator=args.is_aggregator,
        parquet_interval_ns=args.parquet_interval_ns,
        max_pending_rows_per_book=args.max_pending_rows_per_book,
        loop_sleep_s=args.loop_sleep_s,
        round_grace_s=args.round_grace_s,
        max_gradient_bytes=args.max_gradient_bytes,
        reject_untagged=args.reject_untagged,
        window_ns=args.window_ns,
        keep_checkpoints=args.keep_checkpoints,
        keep_proposals=args.keep_proposals,
        checkpoint_interval=args.checkpoint_interval,
        keep_version_deltas=args.keep_version_deltas,
        publish_state_hash=args.publish_state_hash,
        s3_cache_retention_hours=args.s3_cache_retention_hours,
        blocks_per_round=args.blocks_per_round,
        block_time_s=args.block_time_s,
        validator_uid=args.validator_uid,
        bucket_prefix=bucket_prefix,
    )
    # API key middleware — reject unauthenticated requests when key is set
    api_key = args.api_key
    add_api_key_middleware(app, api_key)
    if api_key:
        bt.logging.info("API key auth enabled (X-API-Key header required)")
    elif args.bind != "127.0.0.1":
        bt.logging.warning(
            f"Gradient server bound to {args.bind} WITHOUT API key — "
            "anyone can push state and read scores!"
        )

    # Optional wandb dashboard. Soft-dep: no-op when wandb isn't installed
    # or WANDB_PROJECT / --wandb-project isn't set.
    try:
        from GenTRX.src import wandb_ops

        wandb_ops.init_wandb(
            project=args.wandb_project,
            run_name=args.wandb_run_name or None,
            config={
                "netuid": args.netuid,
                "interval": args.interval,
                "min_score": args.min_score,
                "books_per_miner": args.books_per_miner,
                "val_fraction": args.val_fraction,
                "is_aggregator": args.is_aggregator,
                "parquet_interval_ns": args.parquet_interval_ns,
                "max_pending_rows_per_book": args.max_pending_rows_per_book,
            },
            tags=["aggregator" if args.is_aggregator else "sibling"],
        )
        import atexit

        atexit.register(wandb_ops.finish_wandb)
    except Exception:
        pass

    app.include_router(router)
    aggregator.start()

    # Single-line runtime version snapshot so bug reports carry exact
    # versions even though the Python pin is now open-ended (>=3.10).
    try:
        from GenTRX.src.service import _log_runtime_versions

        _log_runtime_versions()
    except Exception:
        pass

    bt.logging.info(f"Listening on {args.bind}:{args.port}")
    # timeout_graceful_shutdown=2 — the validator keeps a busy keep-alive
    # connection (state pushes every tick, scores + assignment polls). With
    # uvicorn's default `None`, SIGINT waits forever for "in-flight" requests
    # that never quiesce because new ones arrive faster than they finish.
    uvicorn.run(app, host=args.bind, port=args.port, timeout_graceful_shutdown=2)
