# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""GenTRXAgent
    taos FinanceAgent for data collection, inference, and distributed training.

All GenTRX-owned --agent.params keys carry a `gtx_` prefix to avoid colliding
with strategy-owned keys when this agent is subclassed by a third-party
trader (e.g. another model-using strategy already using `quantity`,
`temperature`, `train_steps`, etc.). Strategy keys on subclasses stay
unprefixed.

Three capabilities, controlled by config:

  Data collection (active when gtx_collect_data=true):
    Collects events from book.events, replays through an
    internal MatchingEngine for per-event LOB state, flushes to parquet.
    Output schema matches the tokenizer's expected fields (see tokenizer.py).

  Inference mode (active when gtx_n_trajectories > 0 and a checkpoint is loaded):
    Seeds a fresh engine from the authoritative L2 snapshot each tick, runs
    N closed-loop generation trajectories, converts the resulting price
    distribution to a scalar signal, and submits market orders.
    EXPERIMENTAL: may not be feasible in parallel with training
                  and serves as skeleton for how to use the model.

  Training mode (active when gtx_training_enabled=true):
    Receives assignments from the validator via POST /gentrx/assignment.
    Downloads assigned parquets from S3, trains in a background thread,
    uploads compressed gradient deltas to S3 for aggregation.
    Model version is pulled on-demand from the assignment; no background poll.

Data collection params:
    gtx_output_dir         (str):   Parquet output root. Default:
                                    <repo>/agents/data/<uid>/ (resolved
                                    via __file__-anchored REPO_ROOT).
                                    Override with the GENTRX_AGENT_OUTPUT_DIR
                                    env var for an absolute path.
    gtx_collect_data       (bool):  Write parquets locally. Set false for pure
                                    training agents that read data from S3.
                                    In-memory event processing still runs (needed
                                    for inference buffer). Default: true
    gtx_flush_interval_ns  (int):   Nanoseconds of sim time per parquet file.
                                    Default: 300_000_000_000 (5 min, matches the
                                    gradient server interval and training window)
    gtx_chunk_size         (int):   Rows per in-memory columnar chunk. Bounds
                                    the live list[dict] footprint. Default: 10_000
    gtx_max_pending_rows_per_book (int): Row cap that triggers an early
                                    partial-interval flush. 0 = time-only
                                    (no row cap). Default: 50_000 (matches the
                                    gradient server)

Inference params:
    gtx_checkpoint        (str):   Path to GenTRX .pt checkpoint.
    gtx_n_trajectories    (int):   Forecast trajectories per book. Default: 0 (disabled)
    gtx_n_gen_orders      (int):   Orders generated per trajectory. Default: 50
    gtx_temperature       (float): Sampling temperature. Default: 1.0
    gtx_signal_threshold  (float): |signal| above which we trade. Default: 0.001
    gtx_quantity          (float): Order size in base units. Default: 1.0

Training params:
    gtx_training_enabled  (bool):  Enable assignment-driven training (opt-in; needs
                                   the [gentrx] extra). Default: false
    gtx_train_steps       (int):   Optional fixed-step cap per window. Default
                                   0 = one pass over the assigned pages.
    gtx_round_budget_s    (float): Wall-clock training budget per round, in
                                   seconds. Default: 240.
    gtx_train_batch_size  (int):   Batch size. Default: 16 on cuda, 4 on cpu
                                   (attention is quadratic in seq×batch;
                                   smaller batch cuts CPU memory + per-step
                                   cost without changing data semantics).
    gtx_train_seq_len     (int):   Sequence length (also min observations). Default: 256
    gtx_train_lr          (float): Learning rate. Default: 1e-4
    gtx_top_k_frac        (float): Gradient compression ratio. Default: 0.10
    gtx_label_smooth_sigma (float): Soft-CE width in bins (0 = strict). Must
                                    match the gradient server. Default: 1.0
    gtx_device            (str):   Device override ("cpu", "cuda", "cuda:0", or
                                   "auto"). Default: "auto" (cuda if available
                                   else cpu). Force cpu to bypass GPU for
                                   debugging or shared-host scenarios.
    gtx_gradient_dir      (str):   Where to save gradients. Default: <gtx_output_dir>/gradients/
    gtx_aggregator_uid    (int):   UID of the canonical-checkpoint aggregator. Default: 0.
    gtx_mode              (str):   Training mode shard for bucket keys ("simulation"
                                   or "exchange"). Default: "simulation". Combined
                                   with the subtensor network ("finney" → mainnet,
                                   else testnet) to form the bucket prefix
                                   gentrx/<network>/<mode>/. Leave at "simulation"
                                   unless instructed otherwise; "exchange" is
                                   reserved for future exchange-data training.
    gtx_network           (str):   Explicit network shard: "mainnet" or "testnet". Leave
                                   empty to auto-detect from the connected subtensor
                                   (finney + standard wss endpoints → mainnet, else testnet).
                                   Required for private finney node operators.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import bittensor as bt

from fastapi import Request
from taos.im.agents import FinanceAgent, UnifiedAgentResponse, ExchangeStateUpdate
from taos.im.protocol import MarketSimulationStateUpdate
from taos.im.protocol.instructions import OrderDirection
from taos.im.protocol.events import SimulationEndEvent

# Light GenTRX helpers (no heavy deps) — safe on a core install; default_output_dir
# is used in initialize() (the basic, non-training path).
from GenTRX.src.bt_log import gtx_log
from GenTRX.src.util.paths import default_output_dir

# GenTRX distributed-training deps are the OPT-IN [gentrx] extra (polars, pyarrow;
# GenTRX.src.util.schema imports pyarrow). GenTRXAgent and basic trading import on
# core deps alone — only opt-in training/inference (gtx_training_enabled /
# gtx_n_trajectories>0) touch these, and every such path is gated on
# training_enabled/inference_enabled. Bind to None when absent so the module still
# imports; initialize() raises a clear "install .[gentrx]" error if the agent is
# actually configured for training/inference without them.
try:
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq
    from GenTRX.src.orderbook import MatchingEngine, LobSnapshot
    from GenTRX.src.util.schema import (
        BID,
        ASK,
        CANCEL,
        EXEC_BUY,
        EXEC_SELL,
        LOB_DEPTH,
        DEFAULT_PRICE_DECIMALS,
        DEFAULT_VOLUME_DECIMALS,
        order_stream_schema,
    )
    _GENTRX_DEPS_AVAILABLE = True
    _GENTRX_IMPORT_ERROR = None
except ImportError as _exc:
    pl = pa = pq = None
    MatchingEngine = LobSnapshot = order_stream_schema = None
    BID = ASK = CANCEL = EXEC_BUY = EXEC_SELL = LOB_DEPTH = None
    DEFAULT_PRICE_DECIMALS = DEFAULT_VOLUME_DECIMALS = None
    _GENTRX_DEPS_AVAILABLE = False
    _GENTRX_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# Per-book state
# ---------------------------------------------------------------------------


@dataclass
class BookBuffer:
    """All mutable state for one book (one market realization)."""

    engine: MatchingEngine
    book_id: int = 0
    # order_id → is_buy: needed to route Cancellations to the correct side.
    # Entries removed on cancel or full fill; only resting limit orders remain.
    order_sides: dict[int, bool] = field(default_factory=dict)
    # Live dict chunk: bounded to chunk_size rows under collection. Sealed
    # chunks move to batches as columnar RecordBatches (~10× smaller).
    events: list[dict[str, Any]] = field(default_factory=list)
    batches: list[pa.RecordBatch] = field(default_factory=list)
    last_ts: int = 0
    session_open_mid: int | None = None
    current_interval_start: int = 0
    # Flushes already written for current_interval_start; >0 only when a row
    # cap forces multiple partials in one interval. Resets when the interval
    # advances. Names the `_NNNN` suffix without touching the filesystem.
    flush_seq: int = 0


# ---------------------------------------------------------------------------
# Internal state container
# ---------------------------------------------------------------------------


class _GenTRXState:
    """Container for all GenTRXAgent-owned attributes.

    The parent owns this entire namespace; subclasses must not assign to
    or read into `self._gtx.*`. Putting all GenTRX state under one
    attribute keeps the inheritance contract simple: as long as the
    subclass does not name an attribute `_gtx`, it cannot collide with
    parent state.

    Fields are populated by ``GenTRXAgent.initialize()``; this class
    only declares them.
    """

    # SAFE DEFAULTS, so a bare state object never raises AttributeError.
    #
    # These are only annotations below; without class-level values, reading any field before
    # initialize() populates it is an AttributeError that kills the MINER, not just the agent. That is
    # exactly what gtx_enabled=false produces: initialize() returns early by design, leaving the object
    # bare, while miner.py:99 still calls _ensure_model_version() on every startup. Guarding each read
    # site was whack-a-mole -- model, then gentrx_inited, then store, each found only by crashing.
    # Defaults fix the class of bug instead of its instances. initialize() overwrites all of them, so
    # an enabled agent is unaffected.
    enabled: bool = True
    gentrx_inited: bool = False
    model = None
    store = None

    # ---- Data collection config (gtx_* params) ----
    output_dir: Path
    flush_interval_ns: int
    collect_data: bool
    chunk_size: int
    max_pending_rows_per_book: int

    # ---- Inference config ----
    n_trajectories: int
    n_gen_orders: int
    temperature: float
    signal_threshold: float
    order_qty: float
    inference_enabled: bool

    # ---- Training config ----
    training_enabled: bool
    training_url: str
    training_api_key: str
    train_steps: int
    train_batch_size: int
    train_seq_len: int
    top_k_frac: float
    keep_gradients: int
    train_lr: float
    label_smooth_sigma: float
    gradient_dir: Path

    # ---- Model state ----
    model: Any
    tokenizer: Any
    model_cfg: Any
    tokenizer_cfg: Any
    device: str
    model_version: int

    # ---- Per-book state ----
    books: dict[int, BookBuffer]
    flush_counts: dict[int, int]
    price_scale: int | None
    vol_scale: int | None

    # ---- Training queue / thread ----
    pending_assignments: list[dict]
    training_thread: threading.Thread | None
    training_in_progress: bool
    last_gradient_path: Path | None
    train_window_id: int

    # ---- S3 wiring ----
    store: Any
    data_store: Any
    write_store: Any
    discovered_aggregator_store: Any
    discovered_aggregator_uid: int
    s3_cache_dir: Path | None
    s3_cached_files: dict[str, Path]
    bucket_prefix: str

    # ---- Training logger + retry backoff ----
    tlog: logging.Logger
    retry_last_at: float
    retry_cooldown: float

    # ---- Lazy first-tick init guard ----
    # Mode-dependent setup (gtx_mode → bucket_prefix → S3 stores → checkpoint
    # bootstrap) cannot run in initialize() because _exchange_mode is only set
    # after the first update(). _ensure_gentrx_inited() flips this on the first
    # respond() call and short-circuits subsequent ticks.
    gentrx_inited: bool

    # ---- Co-base signal cache ----
    # When a composed agent extends both ComposedAgentBase AND GenTRXAgent, the
    # composer's engines.gentrx in_process module reads the latest per-book
    # signal via :meth:`GenTRXAgent.gentrx_signal`. The cache is populated each
    # tick by ``_execute_signal`` regardless of co-base status (the standalone
    # path also places orders; the co-base path only reads the cache and lets
    # the composer's weapons module decide what to do).
    last_signals: dict[int, float]

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class GenTRXAgent(FinanceAgent):
    """GenTRX data collection + optional inference + optional distributed training.

    Mode-agnostic: runs under both ``MarketSimulationStateUpdate`` and
    ``ExchangeStateUpdate``. ``gtx_mode`` (bucket shard) is derived from the
    actual run mode on the first ``respond()`` call. GenTRX runs simulation-only
    for now (``exchange_mode=False``); the interface supports exchange.
    """

    def initialize(self) -> None:
        """Set up the agent: strategy state, per-book records and the training-server link when configured."""
        bt.logging.set_info()

        # All GenTRX-owned state lives under self._gtx; subclasses see one
        # reserved attribute on self instead of every internal name.
        self._gtx = _GenTRXState()
        g = self._gtx

        # gtx_enabled=false: FULL off switch, not just "do not train".
        #
        # gtx_collect_data and gtx_training_enabled gate collection and training, but the setup BELOW
        # runs regardless: bucket prefix, S3 stores, checkpoint resolution. That work is expensive in
        # both memory and startup latency
        # start and the miner serving its axon, and a miner that cannot answer for that long is not
        # usable. It also holds substantial RSS on a box where RAM is the binding constraint.
        #
        # A caller that wants GenTRX genuinely off had no way to say so. This is that way. Every gtx_*
        # feature flag is forced off and setup is skipped, so the agent behaves as a pure trader: the
        # event hooks below all short-circuit on collect_data/training_enabled/inference_enabled.
        # GenTRX IS OPT-IN: with no gtx_* settings at all, it must not initialise. (Operator decision,
        # 2026-08-09.) Defaulting `enabled` to True meant every agent inheriting GenTRXAgent paid the
        # full setup cost -- bucket prefix, S3 stores, checkpoint resolution -- whether or not anyone
        # asked for it, which is what put 563s to >944s between process start and the axon serving.
        #
        # Explicit gtx_enabled always wins. Otherwise presence of ANY other gtx_* setting is read as
        # intent to use GenTRX, so existing configs that set gtx_training_enabled/gtx_collect_data keep
        # working without also having to learn about this flag.
        # Read intent from the VALUE, not the presence of the key: several agents set
        # gtx_training_enabled/gtx_collect_data to False in their own initialize() to opt out, and
        # treating that as "a gtx_* setting exists" would switch GenTRX back ON for exactly the agents
        # trying to turn it off.
        _gtx_asked = any(
            _cfg_bool(self.config, _k, False)
            for _k in ("gtx_training_enabled", "gtx_collect_data", "gtx_inference_enabled")
        )
        g.enabled = _cfg_bool(self.config, "gtx_enabled", _gtx_asked)
        if not g.enabled:
            g.collect_data = False
            g.training_enabled = False
            g.inference_enabled = False
            # miner.py:99 reads _gtx.model unconditionally when the agent exposes _gtx, so the state
            # object must still answer for it or the MINER dies at startup with AttributeError. Leaving
            # a half-built state object behind is how a "disable" flag turns into an outage.
            g.model = None
            bt.logging.info("[GTX] disabled via gtx_enabled=false; skipping all GenTRX setup")
            return

        # ---- Data collection config (gtx_* params) ----
        # Default resolves to <repo>/agents/data/<uid>/ via __file__-anchored
        # repo root (see GenTRX/src/util/paths.py). Override with the
        # gtx_output_dir agent param or GENTRX_AGENT_OUTPUT_DIR env var.
        g.output_dir = Path(
            getattr(self.config, "gtx_output_dir", "")
            or default_output_dir(self.uid)
        )
        # Interval-aligned flush: 5 min sim time, matching the gradient server's
        # parquet_interval_ns (and the training window).
        g.flush_interval_ns = int(
            getattr(self.config, "gtx_flush_interval_ns", 300_000_000_000)
        )
        # Set gtx_collect_data=false on pure training agents that read data from
        # S3. In-memory event processing still runs to keep the inference buffer
        # live.
        g.collect_data = _cfg_bool(self.config, "gtx_collect_data", False)  # opt-in, like training
        # Live dicts are sealed into columnar batches every chunk_size rows so
        # the in-memory list[dict] cost stays bounded to one chunk.
        g.chunk_size = max(1, int(getattr(self.config, "gtx_chunk_size", 10_000)))
        # Row cap that forces an early partial-interval flush, matching the
        # gradient server default. 0 disables the cap (time-only flush).
        g.max_pending_rows_per_book = max(
            0, int(getattr(self.config, "gtx_max_pending_rows_per_book", 50_000))
        )
        # Per-book on-disk retention. After each flush, the oldest parquets
        # under <output_dir>/<book>/intervals/ are deleted until at most this
        # many remain. The miner never reads these locally — they're either
        # uploaded (validator) or terminal write (benchmark/standalone) — so
        # bounding count is safe. 0 disables retention (unbounded growth).
        # Default 12 ≈ 1h at the 5-min flush interval.
        g.keep_intervals_per_book = max(
            0, int(getattr(self.config, "gtx_keep_intervals_per_book", 12))
        )

        # ---- Inference config ----
        g.n_trajectories = int(getattr(self.config, "gtx_n_trajectories", 0))
        g.n_gen_orders = int(getattr(self.config, "gtx_n_gen_orders", 50))
        g.temperature = float(getattr(self.config, "gtx_temperature", 1.0))
        g.signal_threshold = float(getattr(self.config, "gtx_signal_threshold", 0.001))
        g.order_qty = float(getattr(self.config, "gtx_quantity", 1.0))
        # Inference is experimental: may not be feasible in parallel with
        # training (GPU contention, memory pressure).
        g.inference_enabled = g.n_trajectories > 0

        # ---- Device selection (set once; _load_model picks it up later) ----
        # Override precedence:
        #   1. gtx_device agent param (explicit: "cpu", "cuda", "cuda:0", "auto")
        #   2. "auto" or unset → torch.cuda.is_available() ? "cuda" : "cpu"
        # Resolved here (before the training-config block) because several
        # training defaults below depend on whether we're on GPU or CPU.
        _device_cfg = str(getattr(self.config, "gtx_device", "auto") or "auto").strip().lower()
        if _device_cfg in ("", "auto"):
            try:
                import torch as _torch
                g.device = "cuda" if _torch.cuda.is_available() else "cpu"
            except Exception:
                g.device = "cpu"
        else:
            g.device = _device_cfg
        _is_cuda = g.device.startswith("cuda")

        # ---- Training config ----
        # OPT-IN: GenTRX distributed training is off unless explicitly enabled, so a
        # plain miner runs basic trading on core deps without the [gentrx] extra.
        g.training_enabled = _cfg_bool(self.config, "gtx_training_enabled", False)

        # Training/inference need the opt-in [gentrx] extra (polars, pyarrow). If the
        # agent is configured for either but the deps are absent, fail fast with a
        # clear message instead of a cryptic NoneType error deep in a collection path.
        if (g.training_enabled or g.inference_enabled) and not _GENTRX_DEPS_AVAILABLE:
            raise RuntimeError(
                "GenTRX training/inference is enabled (gtx_training_enabled / "
                "gtx_n_trajectories>0) but the optional [gentrx] extra is not installed. "
                "Install it with `pip install -e '.[gentrx]'`, or run basic trading only "
                "with gtx_training_enabled=false."
            ) from _GENTRX_IMPORT_ERROR
        # When set, the agent forwards each /gentrx/assignment payload to a
        # standalone miner_training_server instead of training inline.
        g.training_url = getattr(self.config, "gtx_training_url", "") or ""
        g.training_api_key = (
            getattr(self.config, "gtx_training_api_key", "")
            or os.environ.get("GENTRX_MINER_API_KEY", "")
        )
        # One pass over the assigned pages per round (gtx_train_steps=0); a small
        # incremental gradient — the big batch comes from aggregating across
        # miners. Batch stays device-aware (quadratic-attention memory at
        # batch=16/seq=256 is uncomfortable on CPU).
        g.is_cuda = _is_cuda
        g.train_steps = int(getattr(self.config, "gtx_train_steps", 0) or 0)
        g.round_budget_s = float(getattr(self.config, "gtx_round_budget_s", 240.0))
        # Optional drift check: verify the delta-advanced model against the
        # server's published per-version hash, resync on mismatch. Off unless
        # the server runs with --publish-state-hash.
        g.verify_drift = _cfg_bool(self.config, "gtx_verify_drift", False)
        g.train_batch_size = int(getattr(self.config, "gtx_train_batch_size", 16 if _is_cuda else 4))
        g.train_seq_len = int(getattr(self.config, "gtx_train_seq_len", 512))
        g.top_k_frac = float(getattr(self.config, "gtx_top_k_frac", 0.10))
        # Retention on the per-miner write bucket. 0 disables pruning entirely
        # (gradients accumulate; operator handles cleanup). Default 50 ≈ ~4h
        # of history at the standard round cadence.
        g.keep_gradients = int(getattr(self.config, "gtx_keep_gradients", 50))
        g.train_lr = float(getattr(self.config, "gtx_train_lr", 1e-4))
        g.label_smooth_sigma = float(
            getattr(self.config, "gtx_label_smooth_sigma", 1.0)
        )
        _gdir = getattr(self.config, "gtx_gradient_dir", None)
        g.gradient_dir = Path(_gdir) if _gdir else g.output_dir / "gradients"

        # ---- Pending assignments queue ----
        # Multiple validators may deliver assignments at the same block
        # boundary; the miner merges all data and trains once.
        g.pending_assignments = []

        # ---- Retry backoff for failed gradient uploads ----
        g.retry_last_at = 0.0
        g.retry_cooldown = 30.0  # seconds, doubles on failure up to 300s

        # ---- S3 wiring ----
        # Checkpoints come from the aggregator bucket (uid 0 or
        # gtx_aggregator_uid), resolved via chain in
        # _get_aggregator_store_for_assignment. data_store is overridden
        # per-assignment from the payload. write_store is the miner's
        # own bucket built from GENTRX_AGENT_S3_*.
        g.store = None
        g.data_store = None
        g.write_store = None
        g.discovered_aggregator_store = None
        g.discovered_aggregator_uid = int(getattr(self.config, "gtx_aggregator_uid", 0))
        g.bucket_prefix = ""  # populated by _ensure_gentrx_inited() on first respond()
        g.gentrx_inited = False
        g.last_signals = {}  # book_id → last signal float; read by composer co-base

        # ---- Training logger ----
        # logging.getLogger returns the same module-global Logger by name on
        # every initialize() call. Guard against duplicate FileHandlers if
        # this method ever re-runs in the same process — without the check,
        # each call would silently double the log fanout and open another fd.
        g.tlog = logging.getLogger(f"GenTRX.train.{self.uid}")
        g.tlog.setLevel(logging.INFO)
        g.tlog.propagate = False
        g.gradient_dir.mkdir(parents=True, exist_ok=True)
        if not g.tlog.handlers:
            _fh = logging.FileHandler(g.gradient_dir / "train.log")
            _fh.setFormatter(
                logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
            )
            g.tlog.addHandler(_fh)
        g.tlog.info("Training logger initialized (uid=%d)", self.uid)

        # ---- Per-book state ----
        g.price_scale = None
        g.vol_scale = None
        g.books = {}
        g.flush_counts = {}

        # ---- Training state ----
        g.training_thread = None
        g.training_in_progress = False
        g.last_gradient_path = None
        g.model_version = 0
        g.train_window_id = 0

        # ---- S3 download cache ----
        g.s3_cache_dir = None
        g.s3_cached_files = {}

        # ---- Model ----
        g.model = None
        g.tokenizer = None
        g.model_cfg = None
        g.tokenizer_cfg = None
        # g.device was set above (before the training-config block) so the
        # device-dependent training defaults could reference it. _load_model
        # picks up g.device when a checkpoint actually loads.
        checkpoint = getattr(self.config, "gtx_checkpoint", None)
        if checkpoint and Path(checkpoint).exists():
            try:
                self._load_model(checkpoint)
            except Exception as exc:
                gtx_log.error(f"Failed to load local checkpoint: {exc}")
        elif checkpoint:
            gtx_log.warning(f"Checkpoint not found locally: {checkpoint}")
        # S3 bootstrap (latest published checkpoint) is deferred to the first
        # respond(), see _ensure_gentrx_inited(). The local-file checkpoint above
        # is mode-independent so it stays here; the S3 path needs gtx_mode,
        # which derives from _exchange_mode, which is only known after the
        # first update() runs.

        # Apply retention to any data already on disk from a previous run.
        # Without this, a long-running miner restarted with a tighter cap (or
        # an existing miner pulling this change) would only prune-on-next-
        # flush, leaving the historical backlog to sit forever.
        self._prune_existing_intervals()

        # Assignment delivery endpoint. Validator POSTs assignments here
        # (separate from /handle, mirrors dendrite pattern). Two modes:
        #   - inline (default): append to local queue, train in-process.
        #   - forwarding (gtx_training_url set): POST the payload to the
        #     configured miner_training_server and return its response.
        if g.training_url:
            gtx_log.info(
                f"GenTRX training forwarding enabled: {g.training_url}"
                f"{' (with API key)' if g.training_api_key else ''}"
            )

        @self.router.post("/gentrx/assignment")
        async def receive_assignment(request: Request):
            """Accept a training assignment pushed by the miner's training server.

            Args:
                request (Request): The HTTP request carrying the assignment.
            """
            payload = await request.json()
            gtx_log.info(
                "[GTX] assignment received: round=%s model_v=%s data=%d files",
                payload.get("round", "?"),
                payload.get("model_version", "?"),
                len(payload.get("data", [])),
            )
            if self._gtx.training_url:
                return await self._forward_assignment(payload)
            self._gtx.pending_assignments.append(payload)
            return {"status": "ok"}

    # ------------------------------------------------------------------
    # Lazy first-tick init (mode-dependent setup)
    # ------------------------------------------------------------------

    def _ensure_gentrx_inited(self) -> None:
        """Resolve mode-dependent GenTRX state on the first tick.

        ``self._exchange_mode`` is unknown at ``initialize()`` time — it's only
        set per-state in ``FinanceAgent.update()``. So the bucket-prefix
        resolution + S3 store creation + checkpoint bootstrap must wait until
        we've seen the first state update. The ``gentrx_inited`` guard
        short-circuits subsequent calls.
        """
        g = self._gtx
        # gtx_enabled=false skips initialize() entirely, so this state object is intentionally bare.
        # miner.py:99 still calls _ensure_model_version() -> here on every startup, so without this
        # guard a disabled agent dies reading a field that was never meant to exist. Check `enabled`
        # first, before any other field is touched.
        if not getattr(g, "enabled", True):
            return
        if g.gentrx_inited:
            return
        g.gentrx_inited = True

        from GenTRX.src.gradient_store import gentrx_prefix, network_from_config

        # gtx_mode override: explicit param wins, else derive from actual run mode.
        _explicit_mode = str(getattr(self.config, "gtx_mode", "") or "")
        _mode = _explicit_mode or ("exchange" if self._exchange_mode else "simulation")

        # gtx_network operator override → env var; network_from_subtensor
        # (called via network_from_config) reads GENTRX_NETWORK first.
        _network_override = str(getattr(self.config, "gtx_network", "") or "")
        if _network_override:
            os.environ["GENTRX_NETWORK"] = _network_override

        # Pass netuid so network_from_config falls back to the deterministic
        # mapping (79 → mainnet, 366 → testnet, else → localnet) when
        # subtensor.network is "unknown" — happens when operator passes only
        # --subtensor.chain_endpoint without --subtensor.network.
        _netuid = getattr(self.config, "netuid", None)
        try:
            _netuid = int(_netuid) if _netuid is not None else None
        except (TypeError, ValueError):
            _netuid = None
        _network = network_from_config(
            getattr(self.config, "subtensor", None), netuid=_netuid
        )
        g.bucket_prefix = gentrx_prefix(_network, _mode)
        gtx_log.info(
            f"GenTRX bucket prefix: {g.bucket_prefix} "
            f"(network={_network}, mode={_mode})"
        )

        try:
            from GenTRX.src.gradient_store import (
                create_aggregator_store_from_env,
                GradientStore,
            )

            g.store = create_aggregator_store_from_env(prefix=g.bucket_prefix)
            if g.store:
                gtx_log.info(
                    f"S3 aggregator bucket fallback: {g.store.endpoint_url}/{g.store.bucket}"
                )
            g.data_store = g.store

            agent_bucket = os.environ.get("GENTRX_AGENT_S3_BUCKET")
            if agent_bucket:
                g.write_store = GradientStore(
                    endpoint_url=os.environ.get(
                        "GENTRX_AGENT_S3_ENDPOINT_URL",
                        g.store.endpoint_url if g.store else "",
                    ),
                    bucket=agent_bucket,
                    access_key=os.environ.get("GENTRX_AGENT_S3_ACCESS_KEY", ""),
                    secret_key=os.environ.get("GENTRX_AGENT_S3_SECRET_KEY", ""),
                    region=os.environ.get("GENTRX_AGENT_S3_REGION", "auto"),
                    prefix=g.bucket_prefix,
                )
                gtx_log.info(
                    f"S3 write (gradients): {g.write_store.endpoint_url}/{g.write_store.bucket}"
                )
        except ImportError:
            pass

        # Bootstrap from S3 only if no local checkpoint was loaded in initialize()
        # AND the miner actually needs a model (training or inference on). A plain
        # trading miner (gtx_training_enabled=false, gtx_n_trajectories=0) needs no
        # model, so skip the bootstrap entirely — otherwise _ensure_model_version does
        # on-chain aggregator-bucket discovery + an S3 checkpoint scan, which hangs/errors
        # on a bare localnet (no S3/aggregator) and prevents the agent from ever serving.
        # This is why sample agents lacking an _ensure_model_version no-op placed no
        # orders even with training disabled: the bootstrap ran regardless.
        if g.model is None and (g.training_enabled or g.inference_enabled):
            self._ensure_model_version()

        mode_parts = []
        if g.model:
            mode_parts.append("inference")
        if g.training_enabled:
            mode_parts.append("training")
        if g.collect_data:
            mode_parts.append("collect")
        train_desc = (
            f"assignment-driven training from S3, {g.train_steps} steps"
            if g.training_enabled else ""
        )
        gtx_log.info(
            f"GenTRXAgent ready (mode={_mode}) | output={g.output_dir} "
            f"| roles={'+'.join(mode_parts) or 'idle'}"
            f" | device={g.device}"
            f" | write_store={'configured' if g.write_store else 'NOT SET (gradient upload disabled)'}"
            f"{'| gtx_collect_data=false (S3 only)' if not g.collect_data else ''}"
            f"{f' | {train_desc}' if train_desc else ''}"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def handle(
        self,
        state: MarketSimulationStateUpdate | ExchangeStateUpdate,
    ) -> "FinanceAgentResponse | ExchangeAgentResponse":
        """Run the normal agent cycle, then drive assignment-driven training.

        The training trigger lives HERE, not in respond(). FinanceAgent.handle
        dispatches to respond_simulation()/respond_exchange(), and only the
        DEFAULT respond_simulation delegates back to respond() — so a subclass
        that overrides respond_simulation/respond_exchange (or a composed agent
        whose runtime supplies its own respond) never reaches respond() and
        would silently stop training while the queue grows unbounded. Anchoring
        the drain at handle() makes it independent of any customized respond.
        """
        response = super().handle(state)
        # The presence of _gtx is not the same question as whether GenTRX is ON. A disabled agent
        # (gtx_enabled=false) still HAS the state object, so this drained the training queue on every
        # state update and raised on fields initialize() deliberately never set -- once per query, which
        # is what turned a working agent into nine tracebacks in one dwell.
        if getattr(self, "_gtx", None) is not None and getattr(self._gtx, "enabled", True):
            self._drive_training()
        return response

    def _drive_training(self) -> None:
        """Drain queued assignments into training (fast, non-blocking).

        Forwards to a remote training service when gtx_training_url is set,
        else spawns the in-process download+train background thread via
        _maybe_train(). Guarded so nothing fires while a window is in flight.
        """
        if not (self._gtx.training_enabled and self._gtx.pending_assignments):
            return
        if self._gtx.training_url:
            # Split mode: drain queue and forward each assignment to the
            # standalone miner_training_server. Production dendrite delivery
            # writes directly to self._gtx.pending_assignments
            # (taos/im/neurons/miner.py forward_gentrx_assignment), bypassing
            # the FastAPI route, so this drain is the only forwarding path that
            # fires for live network traffic.
            self._drain_and_forward_assignments()
        elif not self._gtx.training_in_progress:
            try:
                self._maybe_train()
            except Exception as exc:
                self._gtx.tlog.error(f"_maybe_train failed: {exc}")
                import traceback

                self._gtx.tlog.error(traceback.format_exc())

    def respond(
        self,
        state: MarketSimulationStateUpdate | ExchangeStateUpdate,
    ) -> UnifiedAgentResponse:
        # First-tick lazy init: _exchange_mode is only set after update() runs
        # (FinanceAgent.handle dispatches update() before respond() each tick),
        # so this is the earliest the actual run mode is known. Subsequent
        # ticks short-circuit on the guard flag.
        """Produce this block's response by delegating to the mode-specific branch.

        Returns:
            The response carrying whatever instructions the strategy queued.
        """
        self._ensure_gentrx_inited()

        response = self.make_response()

        # gtx_enabled=false: behave as a plain trader. Everything below is GenTRX inference and reads
        # state that initialize() deliberately never populated, so without this the agent raised on
        # EVERY query (38 AttributeErrors on price_scale in one dwell) -- it served, was queried on both
        # mechanisms, and then threw instead of responding. Returning the empty response here is what
        # makes "GenTRX off" mean "no GenTRX behaviour" rather than "broken agent"; subclasses that add
        # their own order logic after super().respond() still run normally.
        if not getattr(self._gtx, "enabled", True):
            return response

        if self._gtx.price_scale is None:
            # Exchange mode: ExchangeConfig has volumeDecimals but NO
            # priceDecimals, so state.config.priceDecimals used to AttributeError
            # and 500 every ExchangeStateUpdate (blocking training). Mirror the
            # aggregator's exact rule (gradient_server _process_tick): if EITHER
            # decimal is absent, fall back to BOTH canonical defaults (pd=2,vd=4)
            # so the miner's price/vol scale can never diverge from the
            # aggregator's — a mismatch would silently corrupt tokenization.
            _pd = getattr(state.config, "priceDecimals", None)
            _vd = getattr(state.config, "volumeDecimals", None)
            if _pd is None or _vd is None:
                _pd = DEFAULT_PRICE_DECIMALS
                _vd = DEFAULT_VOLUME_DECIMALS
            self._gtx.price_scale = 10 ** int(_pd)
            self._gtx.vol_scale = 10 ** int(_vd)

        # Download the model_version named in the assignment when _maybe_train
        # runs (one download per round, naturally staggered by dendrite delivery).
        self._retry_pending_gradients()

        # In-memory event processing runs unconditionally: inference buffer
        # needs per-tick LOB state regardless of whether we're writing parquets.
        for book_id, book in state.books.items():
            try:
                self._ensure_book(book_id, book, state.timestamp)
                self._process_events(book_id, book)
                self._resync_engine(book_id, book)

                if self._gtx.collect_data:
                    # Parquet flush gated by collect_data.
                    buf = self._gtx.books[book_id]
                    pending = sum(b.num_rows for b in buf.batches) + len(buf.events)
                    cap = self._gtx.max_pending_rows_per_book
                    if buf.last_ts > 0 and pending:
                        event_interval = buf.last_ts // self._gtx.flush_interval_ns
                        buf_interval = (
                            buf.current_interval_start // self._gtx.flush_interval_ns
                        )
                        if event_interval > buf_interval or (cap and pending >= cap):
                            self._flush_book(book_id)
                    # Safety cap: the interval-boundary check above never
                    # fires if sim-time stalls inside one flush window
                    # (e.g. transition gaps), so events can accumulate
                    # unbounded. Force a flush past a hard cap to keep
                    # per-book RAM bounded.
                    if len(buf.events) > 50000:
                        self._flush_book(book_id)

                if self._gtx.model is not None and self._gtx.inference_enabled:
                    signal = self._infer(book_id, book)
                    response = self._execute_signal(response, book_id, signal)

                if not self._gtx.collect_data:
                    # Not flushing to parquet — cap buffer so memory stays bounded.
                    # Keep the inference context window; clear entirely if inference is off.
                    buf = self._gtx.books[book_id]
                    if self._gtx.inference_enabled and self._gtx.model is not None:
                        cap = self._gtx.max_pending_rows_per_book
                        ctx = self._gtx.model.config.max_seq_len
                        keep = min(ctx, cap) if cap else ctx
                        if len(buf.events) > keep:
                            buf.events = buf.events[-keep:]
                    else:
                        buf.events.clear()

            except Exception as exc:
                gtx_log.error(f"Book {book_id}: {exc}")
        # Assignment-driven training is driven by handle() (see _drive_training),
        # not here, so it survives subclasses that override respond_simulation/
        # respond_exchange and never delegate back to this respond().

        return response

    async def _forward_assignment(self, payload: dict) -> dict:
        """Forward an assignment payload to a remote miner_training_server.

        Async variant used by the FastAPI `/gentrx/assignment` route (proxy
        test path). The dendrite path uses the sync `_drain_and_forward_assignments`
        on respond() because it runs on the miner's request thread, not the
        agent's uvicorn loop.
        """
        url = f"{self._gtx.training_url.rstrip('/')}/miner/assignment"
        headers = {}
        if self._gtx.training_api_key:
            headers["X-API-Key"] = self._gtx.training_api_key
        # Tag with this agent's UID so a shared (multi-UID) training service
        # routes the gradient to the right bucket path.
        payload.setdefault("miner_uid", self.uid)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code >= 400:
                    self._gtx.tlog.warning(
                        f"forward to {url}: HTTP {resp.status_code} {resp.text[:200]}"
                    )
                    return {"status": "forwarded_error", "code": resp.status_code}
                return resp.json()
        except Exception as exc:
            self._gtx.tlog.warning(f"forward to {url} failed: {exc}")
            return {"status": "forward_failed", "error": str(exc)}

    def _drain_and_forward_assignments(self) -> None:
        """Sync forward of every queued assignment to the training service.

        Drains _pending_assignments and POSTs each to gtx_training_url's
        /miner/assignment endpoint. Any failure drops that assignment for
        the round — never blocks the trading loop, never retries.
        """
        # Snapshot then clear so axon-thread appends during forwarding are
        # picked up next tick rather than lost.
        assignments = self._gtx.pending_assignments
        self._gtx.pending_assignments = []
        if not assignments:
            return

        url = f"{self._gtx.training_url.rstrip('/')}/miner/assignment"
        headers = {"Content-Type": "application/json"}
        if self._gtx.training_api_key:
            headers["X-API-Key"] = self._gtx.training_api_key

        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                for payload in assignments:
                    # Tag with this agent's UID for shared multi-UID services.
                    payload.setdefault("miner_uid", self.uid)
                    try:
                        resp = client.post(url, json=payload, headers=headers)
                        if resp.status_code >= 400:
                            self._gtx.tlog.warning(
                                f"forward round={payload.get('round','?')} → {url}: "
                                f"HTTP {resp.status_code} {resp.text[:200]}"
                            )
                        else:
                            self._gtx.tlog.info(
                                f"forwarded round={payload.get('round','?')} → "
                                f"{resp.json()}"
                            )
                    except Exception as exc:
                        self._gtx.tlog.warning(
                            f"forward round={payload.get('round','?')} → {url} failed: {exc}"
                        )
        except Exception as exc:
            self._gtx.tlog.error(f"forward client setup failed: {exc}")

    def onEnd(self, event: SimulationEndEvent) -> None:
        """Handle simulation end: flush per-run state so the next run starts clean."""
        gtx_log.info("Simulation ended — flushing all book buffers.")
        for book_id in list(self._gtx.books):
            self._flush_book(book_id)
        # Wait for any in-progress training to finish
        if self._gtx.training_thread and self._gtx.training_thread.is_alive():
            gtx_log.info("Waiting for training thread to finish...")
            self._gtx.training_thread.join(timeout=60)

    # ------------------------------------------------------------------
    # Book initialisation and engine management
    # ------------------------------------------------------------------

    def _ensure_book(self, book_id: int, book: Any, timestamp: int) -> None:
        if book_id in self._gtx.books:
            return
        interval_start = (timestamp // self._gtx.flush_interval_ns) * self._gtx.flush_interval_ns
        buf = BookBuffer(
            engine=MatchingEngine(),
            book_id=book_id,
            last_ts=timestamp,
            current_interval_start=interval_start,
        )
        self._gtx.books[book_id] = buf
        self._gtx.flush_counts[book_id] = 0
        self._seed_engine_from_l2(buf.engine, book)

    def _seed_engine_from_l2(self, engine: MatchingEngine, book: Any) -> None:
        """Populate engine levels from L2 snapshot without crossing.

        Bids inserted first (no asks yet → all rest). Asks inserted second
        (best bid < best ask by invariant → no crossing).
        Levels inserted worst-to-best so the best ends up at the front.
        """
        engine.reset()
        for level in reversed(book.bids):
            p = round(level.price * self._gtx.price_scale)
            v = max(1, round(level.quantity * self._gtx.vol_scale))
            if p > 0:
                engine.process_order(BID, p, v, is_buy=True)
        for level in reversed(book.asks):
            p = round(level.price * self._gtx.price_scale)
            v = max(1, round(level.quantity * self._gtx.vol_scale))
            if p > 0:
                engine.process_order(ASK, p, v, is_buy=False)

    def _resync_engine(self, book_id: int, book: Any) -> None:
        """Reseed engine to correct drift without clearing order_sides."""
        self._seed_engine_from_l2(self._gtx.books[book_id].engine, book)

    # ------------------------------------------------------------------
    # Event collection
    # ------------------------------------------------------------------

    def _process_events(self, book_id: int, book: Any) -> None:
        """Process events from a tick into GenTRX training rows.

        Events are emitted in log order with executions as a distinct, signed
        type (EXEC_BUY/EXEC_SELL), not collapsed into the aggressing order.
        Order.quantity is the REMAINING size after any fills in this tick, so a
        crossing order of original size Q that fills F and rests R=Q-F produces
        a signed execution row per fill plus a BID/ASK row for the remainder R
        (omitted when R==0, a fully filled market order). The maker-side
        reduction is represented by the execution rows. The matching engine is
        still driven with the full original size (remaining + filled) so the
        reconstructed book stays correct; only the emitted rows differ.

        Sanity check: Order.timestamp == Trade.timestamp for same-tick fills.
        """
        buf = self._gtx.books[book_id]
        if not book.events:
            return

        # Index trade fill volumes by taker_id so the aggressing order can be
        # driven through the engine at its full original size.
        taker_fill_qty: dict[int, float] = {}
        for event in book.events:
            if _is_trade(event):
                tid = event.taker_id
                taker_fill_qty[tid] = taker_fill_qty.get(tid, 0.0) + float(
                    event.quantity
                )

        for event in book.events:
            if _is_trade(event):
                self._collect_trade(event, buf)
            elif _is_cancellation(event):
                self._collect_cancel(event, buf)
            else:
                self._collect_order(event, buf, taker_fill_qty)

    def _collect_trade(self, event: Any, buf: BookBuffer) -> None:
        # Signed execution at the true fill price. The aggressing order drives
        # the engine; this row does not (the maker reduction it represents is
        # applied when that order is processed), so emit row only.
        is_buy = event.side == 0
        order_type = EXEC_BUY if is_buy else EXEC_SELL
        qty = float(event.quantity)
        price_ticks = round(event.price * self._gtx.price_scale)
        ts = int(event.timestamp)

        snap = buf.engine.snapshot()
        self._append_row(buf, ts, order_type, price_ticks, qty, snap)
        buf.last_ts = ts

    def _collect_order(
        self, event: Any, buf: BookBuffer, taker_fill_qty: dict[int, float]
    ) -> None:
        is_buy = event.side == 0
        order_type = BID if is_buy else ASK
        buf.order_sides[event.id] = is_buy

        # Order.quantity is the remaining (resting) size after any immediate
        # fills; the fills are emitted separately as execution rows. The engine
        # is driven with the full original size so matching reproduces the book.
        remaining = float(event.quantity)
        filled = taker_fill_qty.get(event.id, 0.0)
        full_qty = remaining + filled

        price_ticks = round(event.price * self._gtx.price_scale)
        vol_ticks = max(1, round(full_qty * self._gtx.vol_scale))
        ts = int(event.timestamp)

        snap = buf.engine.snapshot()
        if remaining > 0:
            self._append_row(buf, ts, order_type, price_ticks, remaining, snap)
        buf.engine.process_order(order_type, price_ticks, vol_ticks, is_buy)
        buf.last_ts = ts
        # Order fully consumed (market order or crossing limit) — won't rest on
        # book so no future cancel is possible; free the order_sides entry.
        if remaining == 0:
            buf.order_sides.pop(event.id, None)

    def _collect_cancel(self, event: Any, buf: BookBuffer) -> None:
        order_id = event.orderId
        if order_id not in buf.order_sides:
            return  # placed before our session started — skip

        is_buy = buf.order_sides[order_id]
        price_ticks = round(event.price * self._gtx.price_scale)
        qty = float(event.quantity) if event.quantity is not None else 0.0
        vol_ticks = max(1, round(qty * self._gtx.vol_scale))
        ts = int(event.timestamp)

        snap = buf.engine.snapshot()
        self._append_row(buf, ts, CANCEL, price_ticks, qty, snap)
        buf.engine.process_order(CANCEL, price_ticks, vol_ticks, is_buy)
        buf.last_ts = ts
        del buf.order_sides[order_id]  # order is gone after cancel

    def _append_row(
        self,
        buf: BookBuffer,
        ts: int,
        order_type: int,
        price_ticks: int,
        qty: float,
        snap: LobSnapshot,
    ) -> None:
        """Thin wrapper — override ``collect_row()`` instead."""
        mid = snap.mid_price
        if buf.session_open_mid is None and mid > 0:
            buf.session_open_mid = mid
        row = self.collect_row(
            buf.book_id, ts, order_type, price_ticks, qty, snap, buf.session_open_mid,
            interval_ns=ts - buf.last_ts if buf.last_ts > 0 else 0,
        )
        if row is not None:
            buf.events.append(row)
            if self._gtx.collect_data and len(buf.events) >= self._gtx.chunk_size:
                buf.batches.append(_events_to_batch(buf.events))
                buf.events = []

    def collect_row(
        self,
        book_id: int,
        ts: int,
        order_type: int,
        price_ticks: int,
        qty: float,
        snap: LobSnapshot,
        session_open_mid: int | None,
        *,
        interval_ns: int = 0,
    ) -> dict[str, Any] | None:
        """Build the row dict for one order event before it is buffered.

        This is the primary override point for custom data collection.
        Called once per order, cancellation or execution event while
        ``gtx_collect_data=true``.  Return the dict to append to the
        local parquet buffer, or ``None`` to drop the event entirely.

        Args:
            book_id:          Which market this event belongs to.
            ts:               Event timestamp in nanoseconds.
            order_type:       BID, ASK, CANCEL, EXEC_BUY or EXEC_SELL
                              (integer constants from ``GenTRX.src.util.schema``).
            price_ticks:      Price in integer ticks (raw price × price_scale).
            qty:              Reconstructed original order size.
            snap:             LOB snapshot immediately before this event.
                              Contains mid_price, ask_volumes, bid_volumes.
            session_open_mid: First non-zero mid price of this session,
                              used for the mid_price_delta feature.  None
                              until the first order with a valid mid arrives.
            interval_ns:      Nanoseconds elapsed since the previous event
                              in this book (0 for the first event).

        The returned dict must be compatible with ``order_stream_schema()``
        (the schema written to parquet and consumed by ``OrderDataset``).
        Extra columns are silently ignored by the writer; missing required
        columns raise at flush time.
        """
        mid = snap.mid_price
        ask_vols = _pad(snap.ask_volumes, LOB_DEPTH)
        bid_vols = _pad(snap.bid_volumes, LOB_DEPTH)

        row: dict[str, Any] = {
            "timestamp": ts,
            "order_type": order_type,
            "rel_price": price_ticks - mid if mid > 0 else 0,
            "volume_int": int(qty),
            "volume_dec": qty - int(qty),
            "interval_ns": interval_ns,
            "mid_price": mid,
            "time_of_day_s": int((ts // 1_000_000_000) % 86400),
            "mid_price_delta": (
                int(mid - session_open_mid) if session_open_mid else 0
            ),
        }
        for i in range(LOB_DEPTH):
            row[f"lob_ask_vol_{i + 1}"] = float(ask_vols[i]) / self._gtx.vol_scale
            row[f"lob_bid_vol_{i + 1}"] = float(bid_vols[i]) / self._gtx.vol_scale

        return row

    # ------------------------------------------------------------------
    # Parquet flush
    # ------------------------------------------------------------------

    def _flush_book(self, book_id: int) -> None:
        buf = self._gtx.books[book_id]
        if not buf.events and not buf.batches:
            return

        batches, buf.batches = buf.batches, []
        if buf.events:
            batches.append(_events_to_batch(buf.events))
            buf.events = []
        table = pa.Table.from_batches(batches, schema=order_stream_schema())
        n = table.num_rows

        out_dir = self._gtx.output_dir / str(book_id) / "intervals"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Interval-aligned filename: boundaries, not event timestamps
        interval_start = buf.current_interval_start
        interval_end = interval_start + self._gtx.flush_interval_ns
        tag_start = _ts_to_tag(interval_start)
        tag_end = _ts_to_tag(interval_end)
        # Row-cap early flushes write multiple partials per interval; the seq
        # suffix keeps each unique. First flush of an interval has no suffix.
        # The validator's _tag_to_ns strips `_NNNN` when parsing timestamps and
        # dataset readers glob *.parquet without parsing filenames, so the seq
        # suffix leaves downstream paths unaffected.
        suffix = f"_{buf.flush_seq:04d}" if buf.flush_seq else ""
        out_path = out_dir / f"{tag_start}-{tag_end}{suffix}.parquet"

        # Advance interval pointer (resets the partial counter for the next
        # interval); otherwise this interval saw another partial.
        if buf.last_ts >= interval_end:
            buf.current_interval_start = (
                buf.last_ts // self._gtx.flush_interval_ns
            ) * self._gtx.flush_interval_ns
            buf.flush_seq = 0
        else:
            buf.flush_seq += 1
        self._gtx.flush_counts[book_id] = self._gtx.flush_counts.get(book_id, 0) + 1

        pq.write_table(table, out_path)

        manifest_path = out_dir.parent / "manifest.json"
        existing = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {
                "source": "taos_live",
                "book_id": book_id,
                "n_intervals": 0,
                "total_orders": 0,
            }
        )
        existing["n_intervals"] += 1
        existing["total_orders"] += n
        manifest_path.write_text(json.dumps(existing, indent=2))

        gtx_log.info(f"Book {book_id}: flushed {n} events → {out_path.name}")

        self._prune_intervals(out_dir)

    def _prune_intervals(self, intervals_dir: Path) -> int:
        """Keep at most `keep_intervals_per_book` parquets in `intervals_dir`,
        deleting the oldest by mtime. No-op when retention is disabled (=0)
        or the directory holds <= the cap. Returns the number removed."""
        keep = self._gtx.keep_intervals_per_book
        if keep <= 0:
            return 0
        try:
            files = sorted(
                intervals_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime
            )
        except OSError:
            return 0
        if len(files) <= keep:
            return 0
        removed = 0
        for stale in files[: len(files) - keep]:
            try:
                stale.unlink()
                removed += 1
            except OSError as exc:
                gtx_log.debug(f"prune skip {stale.name}: {exc}")
        return removed

    def _prune_existing_intervals(self) -> None:
        """Walk every <output_dir>/<book>/intervals/ at startup and apply the
        retention cap to whatever's already on disk. Called once from
        `initialize()` after `keep_intervals_per_book` is resolved."""
        if self._gtx.keep_intervals_per_book <= 0:
            return
        root = self._gtx.output_dir
        if not root.exists():
            return
        total = 0
        dirs = 0
        for book_dir in root.iterdir():
            if not book_dir.is_dir() or not book_dir.name.isdigit():
                continue
            intervals = book_dir / "intervals"
            if intervals.exists():
                removed = self._prune_intervals(intervals)
                if removed:
                    total += removed
                    dirs += 1
        if total:
            gtx_log.info(
                f"startup retention: pruned {total} stale parquet(s) across "
                f"{dirs} book dir(s) (keep={self._gtx.keep_intervals_per_book})"
            )

    # ------------------------------------------------------------------
    # Distributed training
    # ------------------------------------------------------------------

    def _get_aggregator_store_for_assignment(self, assignment: dict):
        """Return a GradientStore pointing at a validator bucket that has a
        canonical checkpoint published.

        Tries candidates in order:
          1. configured aggregator_uid (operator override via --agent.params).
          2. assignment["validator_uid"] — the validator that sent this assignment.
          3. metagraph scan — handles aggregator uid-drift across registration orders.
          4. self._gtx.store (env-var operator override, last resort).

        A candidate "wins" only if its bucket has at least one published
        checkpoint (`get_latest_version() > 0`). Result is cached for the
        session — restart the miner if the aggregator topology changes.
        """
        if self._gtx.discovered_aggregator_store is not None:
            return self._gtx.discovered_aggregator_store

        from GenTRX.src.gradient_store import GradientStore
        import os

        def _build_store(bi) -> GradientStore:
            return GradientStore(
                endpoint_url=bi.endpoint_url,
                bucket=bi.bucket_name,
                access_key=bi.access_key_id,
                secret_key=bi.secret_access_key,
                region=os.environ.get("GENTRX_VALIDATOR_S3_REGION", "auto"),
                prefix=self._gtx.bucket_prefix,
            )

        try:
            subtensor = getattr(self, "subtensor", None)
            metagraph = getattr(self, "metagraph", None)
            netuid = getattr(getattr(self, "config", None), "netuid", None)

            if subtensor is None or metagraph is None or netuid is None:
                self._gtx.tlog.info(
                    f"chain discovery skipped — subtensor={subtensor is not None} "
                    f"metagraph={metagraph is not None} netuid={netuid}"
                )
            else:
                from GenTRX.src.chain import GenTRXChain
                gtx_chain = GenTRXChain(subtensor, netuid, metagraph)
                # Apply local endpoint override so MinIO buckets resolve correctly.
                # Chain commitments don't encode endpoint URLs; GENTRX_CHAIN_ENDPOINT_OVERRIDE
                # tells the miner where to find buckets (e.g. http://localhost:9000 for MinIO).
                _ep_override = os.environ.get("GENTRX_CHAIN_ENDPOINT_OVERRIDE", "")
                if _ep_override:
                    gtx_chain._endpoint_override = _ep_override

                try:
                    configured_uid = int(getattr(self.config, "gtx_aggregator_uid", 0))
                except (TypeError, ValueError):
                    configured_uid = 0

                self._gtx.tlog.info(
                    f"chain discovery: netuid={netuid} aggregator_uid={configured_uid}"
                )

                try:
                    bucket_info = gtx_chain.get_bucket(configured_uid)
                    if bucket_info is not None:
                        store = _build_store(bucket_info)
                        latest = store.get_latest_version(configured_uid)
                        if latest > 0:
                            self._gtx.tlog.info(
                                f"Aggregator bucket discovered: uid={configured_uid} "
                                f"{bucket_info.endpoint_url}/{bucket_info.bucket_name} "
                                f"(latest v{latest})"
                            )
                            self._gtx.discovered_aggregator_store = store
                            self._gtx.discovered_aggregator_uid = configured_uid
                            return store
                        else:
                            self._gtx.tlog.info(
                                f"uid={configured_uid} bucket found "
                                f"({bucket_info.endpoint_url}/{bucket_info.bucket_name}) "
                                f"but no checkpoint yet (latest=0)"
                            )
                    else:
                        self._gtx.tlog.info(f"uid={configured_uid} no chain commitment found")
                except Exception as exc:
                    self._gtx.tlog.info(f"uid={configured_uid} bucket probe failed: {exc}")

                scan_uids: list[int] = []
                sender = (assignment or {}).get("validator_uid")
                if sender is not None:
                    scan_uids.append(int(sender))
                try:
                    n = int(metagraph.n.item())
                except Exception:
                    n = 0
                for uid in range(n):
                    if uid != configured_uid and uid not in scan_uids:
                        scan_uids.append(uid)

                for uid in scan_uids:
                    try:
                        bucket_info = gtx_chain.get_bucket(uid)
                        if bucket_info is None:
                            continue
                        store = _build_store(bucket_info)
                        latest = store.get_latest_version(uid)
                        if latest > 0:
                            self._gtx.tlog.info(
                                f"Aggregator bucket discovered: uid={uid} "
                                f"{bucket_info.endpoint_url}/{bucket_info.bucket_name} "
                                f"(latest v{latest})"
                            )
                            self._gtx.discovered_aggregator_store = store
                            self._gtx.discovered_aggregator_uid = uid
                            return store
                    except Exception as exc:
                        self._gtx.tlog.info(f"uid={uid} bucket probe failed: {exc}")
                        continue

                self._gtx.tlog.info(
                    f"chain discovery exhausted {len(scan_uids)} uids — no aggregator with checkpoint found"
                )

                if self._gtx.store is not None:
                    try:
                        if self._gtx.store.get_latest_version(configured_uid) > 0:
                            self._gtx.tlog.info(
                                f"Aggregator bucket from env override: "
                                f"{self._gtx.store.endpoint_url}/{self._gtx.store.bucket}"
                            )
                            self._gtx.discovered_aggregator_store = self._gtx.store
                            self._gtx.discovered_aggregator_uid = configured_uid
                            return self._gtx.store
                    except Exception as exc:
                        self._gtx.tlog.info(f"env-var store probe failed: {exc}")
        except Exception as exc:
            self._gtx.tlog.warning(f"Chain aggregator discovery failed: {exc}")
            import traceback as _tb
            self._gtx.tlog.warning(_tb.format_exc())

        return self._gtx.store

    def _assignment_data_store(self, assignment: dict):
        """Return a GradientStore for downloading training parquets.

        Note: S3 credentials reach miners via chain commitment, not the dendrite
        assignment payload.  Chain commitments are the trust anchor — any miner
        can verify independently which bucket a validator owns.  Dendrite is a
        point-to-point synapse channel for small messages; routing credentials
        through it would bypass the on-chain trust model and require miners to
        trust the validator's word directly rather than the chain.

        The assignment payload fallback below (data_bucket / data_access_key
        fields) is an older path retained for compatibility.
        """
        bucket = assignment.get("data_bucket", "")
        if bucket:
            try:
                from GenTRX.src.gradient_store import GradientStore
                import os
                return GradientStore(
                    endpoint_url=assignment.get("data_endpoint", ""),
                    bucket=bucket,
                    access_key=assignment.get("data_access_key", ""),
                    secret_key=assignment.get("data_secret_key", ""),
                    region=os.environ.get("GENTRX_VALIDATOR_S3_REGION", "auto"),
                    prefix=self._gtx.bucket_prefix,
                )
            except Exception as exc:
                self._gtx.tlog.warning(f"Failed to build data store from assignment fields: {exc}")
        # env-var store first, then chain-discovered aggregator store (training
        # data lives in the same validator bucket as the checkpoint).
        return self._gtx.data_store or self._gtx.discovered_aggregator_store

    def _download_assignment_data(self, assignment: dict) -> list[Path]:
        """Download pre-resolved data files from the assignment.

        The server pre-resolves S3 keys or local paths in the assignment's
        "data" field. The agent just downloads what it's told — no discovery,
        no timestamp filtering, no book enumeration.

        Files are cached locally — only downloaded once per session.
        """
        data_keys = assignment.get("data", [])
        data_source = assignment.get("data_source", "local")

        if not data_keys:
            return []

        if data_source == "local":
            # Local paths — already resolved by server, just return as Path
            return [Path(k) for k in data_keys if Path(k).exists()]

        # S3 mode: download each key to local cache from the data bucket.
        # Prefer bucket credentials carried in the assignment (set by the validator
        # when it has its own data bucket); fall back to env-var-configured store.
        data_store = self._assignment_data_store(assignment)
        if data_store is None:
            self._gtx.tlog.warning("assignment has S3 data but no data_store configured")
            return []

        if self._gtx.s3_cache_dir is None:
            self._gtx.s3_cache_dir = self._gtx.output_dir / "_s3_cache"

        # Namespace cache by validator bucket to avoid book ID collisions
        # when multiple validators have different data for the same book IDs.
        import hashlib as _hashlib
        bucket_id = assignment.get("data_bucket", "") or "default"
        bucket_hash = _hashlib.md5(bucket_id.encode()).hexdigest()[:8]
        cache_base = self._gtx.s3_cache_dir / bucket_hash

        local_files = []
        for key in data_keys:
            # Cache check: use bucket_hash + key as the unique cache key
            cache_key = f"{bucket_hash}/{key}"
            if cache_key in self._gtx.s3_cached_files:
                local_files.append(self._gtx.s3_cached_files[cache_key])
                continue

            # Parse key. Expected: "data/{validator_uid}/{book_id}/intervals/{filename}"
            parts = key.split("/")
            if len(parts) < 5 or parts[0] != "data" or parts[3] != "intervals":
                self._gtx.tlog.warning(f"  unexpected S3 key format: {key}")
                continue
            data_uid = parts[1]
            book_id = parts[2]
            filename = parts[-1]

            local_path = cache_base / book_id / "intervals" / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = data_store.get_data(
                    int(data_uid), book_id=int(book_id), filename=filename
                )
                local_path.write_bytes(data)
                self._gtx.s3_cached_files[cache_key] = local_path
                local_files.append(local_path)
            except Exception as exc:
                self._gtx.tlog.warning(f"  S3 download failed from data bucket: {key}: {exc}")

        return local_files

    def _maybe_train(self) -> None:
        """Queue a background training window from pending assignments.

        Returns FAST — all S3 downloads (model checkpoint, training parquets)
        and training itself happen in a background thread, so the caller
        (typically the dendrite state-update handler) is never blocked on
        network I/O. On production S3 the model is ~50MB; blocking here
        would cause the validator's POST /gentrx/state to time out.

        Consumes all queued assignments.
        """
        if self._gtx.training_in_progress:
            now = time.time()
            if now - getattr(self._gtx, "_last_progress_log_ts", 0) > 30:
                gtx_log.info("training in progress...")
                self._gtx._last_progress_log_ts = now
            return

        assignments = self._gtx.pending_assignments
        self._gtx.pending_assignments = []

        if not assignments:
            return

        # Keep only the freshest round: the merge is for same-round multi-validator
        # consolidation, but a backlog accumulating across rounds balloons one window.
        latest_round = max(int(a.get("round", 0) or 0) for a in assignments)
        stale = [a for a in assignments if int(a.get("round", 0) or 0) != latest_round]
        if stale:
            self._gtx.tlog.warning(
                f"dropping {len(stale)} stale queued assignment(s) from earlier "
                f"rounds (training slower than round cadence), keeping round {latest_round}"
            )
            assignments = [
                a for a in assignments if int(a.get("round", 0) or 0) == latest_round
            ]

        # Merge data keys + compute target model version across validators
        all_data_keys: list[str] = []
        all_books: list[str] = []
        target_v = 0
        primary = assignments[0]
        for a in assignments:
            all_data_keys.extend(a.get("data", []))
            all_books.extend(a.get("books", []))
            v = int(a.get("model_version", 0) or 0)
            if v > target_v:
                target_v = v

        self._gtx.tlog.info(
            f"assignments: {len(assignments)} validator(s), "
            f"round={primary.get('round', '?')} "
            f"model_v={target_v} "
            f"books={all_books} "
            f"data={len(all_data_keys)} files total"
        )

        if not all_data_keys:
            self._gtx.tlog.info("no data keys in any assignment — skipping")
            return
        if self._gtx.model is None and target_v <= 0:
            # First run + no version in assignment → can't bootstrap. Preserve.
            self._gtx.tlog.info(
                "no model loaded and assignment has no model_version — skipping (assignments preserved)"
            )
            self._gtx.pending_assignments = assignments
            return

        # Mark busy + spawn the download+train background thread. This method
        # RETURNS IMMEDIATELY; the heavy work runs off the caller's path.
        gtx_log.info(
            "[GTX] training started: round=%s model_v=%d data=%d files",
            primary.get("round", "?"), target_v, len(all_data_keys),
        )
        self._gtx.training_in_progress = True
        self._gtx.training_thread = threading.Thread(
            target=self._download_and_train_background,
            args=(assignments, target_v, primary, all_books),
            daemon=True,
        )
        self._gtx.training_thread.start()

    def _download_and_train_background(
        self,
        assignments: list[dict],
        target_v: int,
        primary: dict,
        all_books: list[str],
    ) -> None:
        """Runs on a background thread: download model + data, then train.

        The foreground _maybe_train returned before this function started,
        unblocking the state-update handler. All network I/O lives here.
        """
        train_model = None
        try:
            # Bootstrap the model if we don't have one yet (first run or
            # previous download failed). Cheaper than putting this in the
            # parallel block when we haven't downloaded anything yet — we
            # NEED this model locally before training can happen at all.
            if self._gtx.model is None:
                self._gtx.tlog.info(
                    f"no local model — bootstrapping from assignment (v{target_v})"
                )
                if not self._ensure_model_version(target_v, primary) or self._gtx.model is None:
                    self._gtx.tlog.info("model bootstrap failed — skipping (assignments preserved)")
                    self._gtx.pending_assignments = assignments
                    return

            # Download model rollover (if needed) + training data in parallel.
            need_model = target_v > self._gtx.model_version and target_v > 0
            if need_model:
                self._gtx.tlog.info(
                    f"model rollover: v{self._gtx.model_version} → v{target_v}, downloading..."
                )

            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

            # Per-future timeouts bound the worst case when S3 retries silently
            # stall (mis-scoped bucket creds, stuck DNS, etc.). The training
                # window is ~5 min at defaults, so a 120s download budget still
            # leaves ~3 min for training.
            MODEL_DL_TIMEOUT_S = 60.0
            DATA_DL_TIMEOUT_S = 90.0
            with ThreadPoolExecutor(max_workers=1 + len(assignments)) as pool:
                f_model = (
                    pool.submit(self._ensure_model_version, target_v, primary)
                    if need_model else None
                )
                f_data = [pool.submit(self._download_assignment_data, a) for a in assignments]

                if f_model is not None:
                    try:
                        model_ok = f_model.result(timeout=MODEL_DL_TIMEOUT_S)
                    except FutTimeout:
                        model_ok = False
                        self._gtx.tlog.warning(
                            f"model v{target_v} download timed out after "
                            f"{MODEL_DL_TIMEOUT_S:.0f}s — training on v{self._gtx.model_version}"
                        )
                    if not model_ok:
                        self._gtx.tlog.warning(
                            f"model v{target_v} download failed — training on v{self._gtx.model_version}"
                        )
                parquet_files = []
                for f in f_data:
                    try:
                        parquet_files.extend(f.result(timeout=DATA_DL_TIMEOUT_S))
                    except FutTimeout:
                        self._gtx.tlog.warning(
                            f"data download timed out after {DATA_DL_TIMEOUT_S:.0f}s"
                        )

            if not parquet_files:
                self._gtx.tlog.warning("download failed for all files — skipping")
                return

            import copy
            self._gtx.tlog.info(f"{len(parquet_files)} files ready, deepcopy model...")
            train_model = copy.deepcopy(self._gtx.model)

            self._gtx.tlog.info(
                f"window {self._gtx.train_window_id} STARTED | "
                f"{len(parquet_files)} files from {len(assignments)} validator(s) | "
                f"books={all_books} | "
                f"budget={self._gtx.round_budget_s:.0f}s | "
                f"model_v={target_v} | "
                f"device={self._gtx.device}"
            )
            # _train_background runs inline here (we're already on the
            # background thread); it handles compress + upload at the end.
            # Heartbeat thread: logs every 30s while training blocks so the
            # console shows the process is alive without flooding with step logs.
            _t0 = time.time()
            _hb_stop = threading.Event()

            def _heartbeat():
                while not _hb_stop.wait(30.0):
                    elapsed = time.time() - _t0
                    bt.logging.info(
                        f"[GTX] training in progress (uid={self.uid}): "
                        f"{elapsed:.0f}s elapsed"
                    )

            _hb = threading.Thread(target=_heartbeat, daemon=True)
            _hb.start()
            try:
                self._train_background(parquet_files, train_model, primary)
            finally:
                _hb_stop.set()
                _hb.join(timeout=1.0)
        except Exception as exc:
            self._gtx.tlog.error(f"_download_and_train_background failed: {exc}")
            import traceback
            self._gtx.tlog.error(traceback.format_exc())
        finally:
            self._gtx.training_in_progress = False
            gtx_log.info("[GTX] training thread finished")
            self._gtx.tlog.info("thread finished")
            self._clear_s3_cache()
            del train_model
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _train_background(
        self, parquet_files: list[Path], train_model, assignment: dict | None = None
    ) -> None:
        """Thin wrapper — override ``train()`` instead."""
        try:
            self.train(parquet_files, train_model, assignment)
        except Exception as exc:
            self._gtx.tlog.error(f"FAILED: {exc}")
            import traceback
            self._gtx.tlog.error(traceback.format_exc())

    def select_training_files(
        self, parquet_files: list[Path], assignment: dict | None
    ) -> list[Path]:
        """Return the subset of downloaded parquet files to train on.

        Called by ``train()`` before the DataLoader is built.  The default
        passes all files through unchanged.

        Override to filter by book, time range, file size, or any other
        criterion.  Returning an empty list skips the training window.

        Args:
            parquet_files: All local files downloaded for this round.
            assignment:    The primary assignment dict (keys: round, books,
                           ts_start, ts_end, validator_uid, …).
        """
        return parquet_files

    def _make_window_config(self, **overrides):
        """Build a WindowConfig with the correctness invariants filled from agent
        state — the version actually trained (so the aggregator's mismatch filter
        works) and the soft-CE width the server scores with (or the score grades a
        loss never trained). Callers override only the tunable knobs (n_steps, lr).
        Keeping these here means a custom train() cannot silently drop them."""
        from GenTRX.src.distributed import WindowConfig

        params = dict(
            n_steps=self._gtx.train_steps,
            lr=self._gtx.train_lr,
            window_id=self._gtx.train_window_id,
            miner_uid=self.uid,
            model_version=int(self._gtx.model_version or 0),
            label_smooth_sigma=self._gtx.label_smooth_sigma,
            budget_s=self._gtx.round_budget_s,
        )
        params.update(overrides)
        return WindowConfig(**params)

    def _submit_gradient(self, delta, assignment: dict | None = None) -> None:
        """Compress, upload, prune, finalize — the publish surface shared by every
        train() implementation. Stamps the trained version (belt-and-braces for
        custom loops), honours the validator's per-round top_k advice, retries to a
        local pending file on S3 failure, and advances the window counter."""
        from GenTRX.src.gradient import compress, serialize

        md = getattr(delta, "metadata", None)
        if md is not None and not getattr(md, "model_v_trained", 0):
            md.model_v_trained = int(self._gtx.model_version or 0)

        # Data-starved window: train_incremental returns a zero delta when no
        # loader yielded a batch (empty/too-short pages). Don't upload it — an
        # empty gradient carries no signal, only wastes an S3 round-trip and a
        # scoring slot, and (untagged edge cases aside) muddies aggregation.
        steps = int(getattr(md, "steps_trained", 0) or 0) if md is not None else 0
        norm = float(getattr(delta, "norm", 0.0) or 0.0)
        if steps <= 0 or norm <= 0.0:
            self._gtx.tlog.info(
                f"skipping gradient upload: nothing trained this window "
                f"(steps={steps}, norm={norm:.3g}) — data-starved round"
            )
            self._gtx.train_window_id += 1
            return

        _adv = (assignment or {}).get("advice") or {}
        _adv_tk = _adv.get("top_k_frac")
        top_k = float(_adv_tk) if _adv_tk else self._gtx.top_k_frac
        comp = compress(delta, top_k_frac=top_k)
        data = serialize(comp)

        if self._gtx.write_store is None:
            raise RuntimeError(
                "No S3 store configured. Set GENTRX_S3_* env vars to enable gradient upload."
            )
        round_id = (assignment or {}).get("round", self._gtx.train_window_id)
        try:
            self._gtx.write_store.put_gradient(miner_uid=self.uid, round_id=round_id, data=data)
            self._gtx.tlog.info(f"gradient uploaded to S3 (round={round_id})")
            bt.logging.info(
                f"[GTX] gradient uploaded (uid={self.uid}, round={round_id}, "
                f"{len(data)/1024:.1f} KB)"
            )
            if self._gtx.keep_gradients > 0:
                try:
                    n = self._gtx.write_store.prune_keep_latest(
                        f"gradients/{self.uid}/", keep=self._gtx.keep_gradients, suffix=".grad",
                    )
                    if n:
                        self._gtx.tlog.info(
                            f"pruned {n} old gradient(s), keeping latest {self._gtx.keep_gradients}"
                        )
                except Exception as prune_exc:
                    self._gtx.tlog.debug(f"gradient prune failed: {prune_exc}")
        except Exception as exc:
            self._gtx.tlog.warning(f"S3 upload failed: {exc} — saving for retry")
            pending_dir = self._gtx.gradient_dir / "pending"
            pending_dir.mkdir(parents=True, exist_ok=True)
            pending_path = pending_dir / f"block_{round_id:08d}_miner_{self.uid}.grad"
            pending_path.write_bytes(data)
            self._gtx.last_gradient_path = pending_path

        self._gtx.tlog.info(
            f"window {self._gtx.train_window_id} COMPLETE | "
            f"loss {delta.metadata.loss_before:.4f} → {delta.metadata.loss_after:.4f} | "
            f"gradient {len(data)/1024:.1f} KB"
        )
        self._gtx.train_window_id += 1

    def train(
        self, parquet_files: list[Path], train_model, assignment: dict | None = None
    ) -> None:
        """Run one training window and upload the compressed gradient.

        This is the primary override point for custom training logic.
        Called on a background thread after model and data have been
        downloaded; the foreground respond() path is already unblocked.

        Args:
            parquet_files: Local .parquet files covering the assigned round,
                           after ``select_training_files()`` filtering.
            train_model:   A deep copy of self._gtx.model — safe to mutate.
            assignment:    Primary assignment dict (round, validator_uid, …).

        The default implementation:
          1. Calls ``select_training_files()`` to filter the file list.
          2. Builds one ``DataLoader`` per page (recent first).
          3. Calls ``train_incremental(train_model, loaders, win_cfg, device)``
             from ``GenTRX.src.distributed`` (budget-governed).
          4. Compresses the resulting ``TrainingDelta`` and uploads to S3.
        """
        from GenTRX.src.dataloader import OrderDataset, InterleaveSampler
        from GenTRX.src.distributed import train_incremental
        from torch.utils.data import DataLoader

        parquet_files = self.select_training_files(parquet_files, assignment)
        if not parquet_files:
            self._gtx.tlog.info("select_training_files returned empty list — skipping")
            return

        # One interleaved dataset over ALL assigned files: block-shuffle so a
        # budget-capped round SPANS the assignment instead of front-loading the
        # freshest pages, recency-weighted (recency_alpha) so recent data is
        # over-sampled. Files sorted oldest->newest so the weighting favours the
        # newest. gtx_train_steps (>0) stays an optional total-step cap.
        import pyarrow.parquet as _pq
        good = []
        for f in sorted(parquet_files):
            try:
                if _pq.read_metadata(f).num_rows > self._gtx.train_seq_len:
                    good.append(f)
            except Exception as exc:
                self._gtx.tlog.debug(f"skip page {f}: {exc}")
        if not good:
            self._gtx.tlog.info("no trainable pages, skipping")
            return
        block = int(getattr(self._gtx, "shuffle_block", 1024))
        alpha = float(getattr(self._gtx, "recency_alpha", 1.0))
        max_cached = min(len(good), int(getattr(self._gtx, "train_max_cached", 4)))
        try:
            ds = OrderDataset(
                good, seq_len=self._gtx.train_seq_len,
                tokenizer=self._gtx.tokenizer, max_cached=max_cached,
            )
        except Exception as exc:
            self._gtx.tlog.info(f"dataset build failed ({exc}), skipping")
            return
        loaders = [DataLoader(
            ds, batch_size=self._gtx.train_batch_size,
            sampler=InterleaveSampler(ds, shuffle=True, block=block, recency_alpha=alpha),
            num_workers=0,
        )]

        self._gtx.tlog.info(
            f"{len(good)} files, block={block} recency_alpha={alpha} "
            f"max_cached={max_cached}, budget={self._gtx.round_budget_s:.0f}s..."
        )
        win_cfg = self._make_window_config()
        delta = train_incremental(train_model, loaders, win_cfg, self._gtx.device)
        self._gtx.tlog.info(
            f"training done: loss {delta.metadata.loss_before:.4f} → {delta.metadata.loss_after:.4f}"
        )
        bt.logging.info(
            f"[GTX] training done (uid={self.uid}): "
            f"loss {delta.metadata.loss_before:.4f} → {delta.metadata.loss_after:.4f} "
            f"({delta.metadata.steps_trained} steps)"
        )
        self._submit_gradient(delta, assignment)

    def _ensure_model_version(
        self, target: int | None = None, assignment: dict | None = None
    ) -> bool:
        """Download a checkpoint from the aggregator bucket if needed.

        Args:
            target: Minimum version to load. If None, queries uid-0's bucket
                    for the latest (used at bootstrap). When set, also checks
                    uid-0's latest — uses whichever is newer so miners stay
                    synced with the canonical model even when their assigning
                    validator is a sibling with a stale version.
            assignment: If provided, uses chain-based discovery (always uid-0).

        Returns True if the local model is at the requested version after the
        call (already current or freshly downloaded). Returns False on failure.
        """
        # Pre-respond bootstrap path: miner.py / benchmark.py both call this
        # before any state has arrived (so `_exchange_mode` is still the
        # default False = simulation). The env-var store + bucket_prefix are
        # populated lazily in _ensure_gentrx_inited, trigger it now so the
        # fast env-var path is available below. Idempotent: the gentrx_inited
        # flag is set BEFORE the inner _ensure_model_version recursion bottoms
        # out, so this never loops.
        self._ensure_gentrx_inited()
        # gtx_enabled=false leaves the state object bare by design, and _ensure_gentrx_inited returns
        # immediately in that case -- so every field read below (store, model, ...) is still unset.
        # miner.py:99 calls this on every startup regardless of the flag, so bail here too or a disabled
        # agent takes the miner down with it.
        if not getattr(self._gtx, "enabled", True):
            return False
        if assignment is not None:
            store = self._get_aggregator_store_for_assignment(assignment)
        elif self._gtx.store is not None:
            store = self._gtx.store
        else:
            # No env-var store and no assignment — try chain discovery anyway
            # (subtensor may be available even at init time after benchmark/miner
            # wires it in post-construction).
            store = self._get_aggregator_store_for_assignment({})
        if store is None:
            self._gtx.tlog.warning(
                "no aggregator store — chain discovery for uid-0 failed; "
                "cannot download checkpoint"
            )
            return self._gtx.model is not None

        try:
            agg_uid = self._gtx.discovered_aggregator_uid
            # Check aggregator's latest version — may be newer than assignment says
            latest = store.get_latest_version(agg_uid)
            if target is None:
                target = latest
            elif latest > 0 and latest > target:
                self._gtx.tlog.info(f"aggregator has v{latest} (assignment says v{target}) — using latest")
                target = latest

            if target <= 0:
                return self._gtx.model is not None
            if target <= self._gtx.model_version:
                return True

            from GenTRX.src.distributed import apply_version_deltas

            # Fast path: advance an existing model by applying canonical deltas
            # (usually a single delta, v(n-1) → v(n)).
            if self._gtx.model is not None and self._gtx.model_version > 0:
                expected = {n: p.shape for n, p in self._gtx.model.named_parameters()}
                reached = apply_version_deltas(
                    self._gtx.model, store, agg_uid, self._gtx.model_version, target, expected
                )
                if reached >= target:
                    self._gtx.model_version = target
                    gtx_log.info(f"Advanced model via deltas to v{target}")
                    if self._gtx.verify_drift:
                        self._verify_drift(store, agg_uid, target)
                    return True
                gtx_log.info(f"Delta gap after v{reached}; reloading baseline")

            # Cold start / delta gap: download the latest baseline checkpoint,
            # then replay deltas up to the head. latest.json points at the
            # baseline (full checkpoints are uploaded only every interval).
            baseline_v = latest if latest > 0 else target
            gtx_log.info(f"Downloading baseline checkpoint v{baseline_v} from aggregator bucket")
            ckpt_bytes = store.get_checkpoint(agg_uid, baseline_v)
            # Stage under the agent's output dir (survives /tmp cleaners).
            stage_dir = self._gtx.output_dir / "ckpt_cache"
            stage_dir.mkdir(parents=True, exist_ok=True)
            tmp = stage_dir / f"gentrx_ckpt_{self.uid}.pt"
            tmp.write_bytes(ckpt_bytes)
            del ckpt_bytes
            self._load_model(str(tmp))
            self._gtx.model_version = baseline_v
            if baseline_v < target:
                expected = {n: p.shape for n, p in self._gtx.model.named_parameters()}
                self._gtx.model_version = apply_version_deltas(
                    self._gtx.model, store, agg_uid, baseline_v, target, expected
                )
            gtx_log.info(f"Model at v{self._gtx.model_version} (target v{target})")
            if self._gtx.verify_drift and self._gtx.model_version >= target:
                self._verify_drift(store, agg_uid, target)
            return self._gtx.model_version >= target
        except Exception as exc:
            gtx_log.warning(f"Model sync to v{target} failed: {exc}")
            self._gtx.tlog.warning(f"model sync to v{target} failed: {exc}")
            return False

    def _verify_drift(self, store, agg_uid: int, version: int) -> None:
        """Optional: compare the delta-advanced model to the server's published
        hash and force a baseline reload on mismatch. No-op if hashes absent."""
        from GenTRX.src.distributed import model_state_hash

        meta = store.get_head_meta(agg_uid)
        want = meta.get("state_hash")
        if not want or int(meta.get("version", -1)) != version:
            return
        if model_state_hash(self._gtx.model) != want:
            self._gtx.tlog.warning(
                f"drift detected at v{version}; forcing baseline reload next sync"
            )
            self._gtx.model_version = 0  # next _ensure_model_version reloads baseline

    def _retry_pending_gradients(self) -> None:
        """Retry uploading gradients that failed to reach S3 on a previous cycle.

        Uses exponential backoff (30s → 60s → 120s → 300s cap) to avoid
        hammering S3 when it's down. Resets to 30s on success.
        """
        if self._gtx.write_store is None:
            return
        pending_dir = self._gtx.gradient_dir / "pending"
        if not pending_dir.exists():
            return

        now = time.time()
        if now - self._gtx.retry_last_at < self._gtx.retry_cooldown:
            return
        self._gtx.retry_last_at = now

        any_failed = False
        for grad_path in sorted(pending_dir.glob("*.grad")):
            try:
                # Filename: block_NNNNNNNN_miner_MM.grad
                parts = grad_path.stem.split("_")
                round_id = int(parts[1])
                data = grad_path.read_bytes()
                self._gtx.write_store.put_gradient(
                    miner_uid=self.uid, round_id=round_id, data=data
                )
                grad_path.unlink()
                self._gtx.tlog.info(f"Retried pending gradient: round {round_id}")
            except Exception as exc:
                self._gtx.tlog.debug(f"Retry failed for {grad_path.name}: {exc}")
                any_failed = True

        if any_failed:
            self._gtx.retry_cooldown = min(self._gtx.retry_cooldown * 2, 300.0)
        else:
            self._gtx.retry_cooldown = 30.0

    def _clear_s3_cache(self) -> None:
        """Delete locally cached training parquets after a round completes."""
        import shutil
        if self._gtx.s3_cache_dir and self._gtx.s3_cache_dir.exists():
            try:
                shutil.rmtree(self._gtx.s3_cache_dir)
                self._gtx.tlog.debug("S3 cache cleared")
            except Exception as exc:
                self._gtx.tlog.debug(f"S3 cache clear failed: {exc}")
        self._gtx.s3_cached_files.clear()
        self._gtx.s3_cache_dir = None

    # ------------------------------------------------------------------
    # Inference mode
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint: str) -> None:
        import gc
        import torch
        from GenTRX.src.gradient import load_checkpoint_safely, validate_state_dict
        from GenTRX.src.model import OrderModel, ModelConfig
        from GenTRX.src.tokenizer import OrderTokenizer, TokenizerConfig

        ckpt = load_checkpoint_safely(checkpoint)
        raw_cfg = ckpt.get("model_config", ckpt.get("config", {}))
        new_cfg = ModelConfig(
            **{
                k: v
                for k, v in raw_cfg.items()
                if k in ModelConfig.__dataclass_fields__
            }
        )
        tok_dict = ckpt.get("tokenizer_config")
        self._gtx.tokenizer_cfg = (
            TokenizerConfig.from_dict(tok_dict) if tok_dict else TokenizerConfig()
        )
        # Reuse the existing nn.Module when config is unchanged — every
        # rollover otherwise allocates a fresh OrderModel plus transient
        # CPU weights before .to(cuda), inflating per-cycle peak RSS.
        # When config DOES change, drop the prior model first so the swap
        # doesn't transiently hold two full models resident.
        if self._gtx.model is None or new_cfg != getattr(self._gtx, "model_cfg", None):
            if self._gtx.model is not None:
                self._gtx.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self._gtx.model = OrderModel(new_cfg)
        self._gtx.model_cfg = new_cfg
        validate_state_dict(ckpt["model_state_dict"], self._gtx.model.state_dict())
        self._gtx.model.load_state_dict(ckpt["model_state_dict"])
        self._gtx.model.eval()
        self._gtx.tokenizer = OrderTokenizer(self._gtx.tokenizer_cfg)
        # self._gtx.device is set in initialize() (respects gtx_device override).
        # Don't reassign here — would silently override an operator's explicit
        # "cpu" choice on hosts where cuda is available.
        self._gtx.model.to(self._gtx.device)

        # Drop the source state_dict explicitly so PyTorch releases the
        # CPU tensor buffers before this function returns; then trim the
        # glibc arena to give back what we can.
        del ckpt
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        gtx_log.info(f"GenTRX model loaded from {checkpoint} on {self._gtx.device}")

    def _infer(self, book_id: int, book: Any) -> float:
        """Run N forecast trajectories from current LOB state.

        Returns a fractional-return signal in (-inf, +inf):
            > +signal_threshold  → BUY
            < -signal_threshold  → SELL
            otherwise            → 0.0 (no trade)
        """
        from GenTRX.src.inference import generate_with_engine

        buf = self._gtx.books[book_id]
        if len(buf.events) < 32:
            return 0.0  # not enough context yet

        prompt = self._build_prompt(buf)
        if prompt is None:
            return 0.0

        current_mid = buf.engine.snapshot().mid_price
        if current_mid <= 0:
            return 0.0

        final_mids: list[float] = []
        for _ in range(self._gtx.n_trajectories):
            eng = MatchingEngine()
            self._seed_engine_from_l2(eng, book)

            prompt_copy = {k: v.clone() for k, v in prompt.items()}
            generated = generate_with_engine(
                self._gtx.model,
                self._gtx.tokenizer,
                eng,
                prompt_copy,
                n_orders=self._gtx.n_gen_orders,
                temperature=self._gtx.temperature,
                device=self._gtx.device,
            )
            if generated:
                final_mids.append(generated[-1].mid_price)

        if not final_mids:
            return 0.0

        median_final = float(np.median(final_mids))
        return (median_final - current_mid) / current_mid

    def _build_prompt(self, buf: BookBuffer) -> dict[str, Any] | None:
        """Build tokenized prompt tensors from recent buffered events."""
        import torch

        if self._gtx.tokenizer is None:
            return None

        ctx = self._gtx.model.config.max_seq_len
        events = buf.events[-ctx:]
        if not events:
            return None

        # Build Polars DataFrame (tokenizer.encode expects a pl.DataFrame)
        lob_ask = [
            [e[f"lob_ask_vol_{i + 1}"] for i in range(LOB_DEPTH)] for e in events
        ]
        lob_bid = [
            [e[f"lob_bid_vol_{i + 1}"] for i in range(LOB_DEPTH)] for e in events
        ]

        df_data: dict[str, list] = {
            "order_type": [e["order_type"] for e in events],
            "rel_price": [e["rel_price"] for e in events],
            "volume_int": [e["volume_int"] for e in events],
            "volume_dec": [e["volume_dec"] for e in events],
            "interval_ns": [e["interval_ns"] for e in events],
            "mid_price": [e["mid_price"] for e in events],
            "time_of_day_s": [e["time_of_day_s"] for e in events],
            "mid_price_delta": [e["mid_price_delta"] for e in events],
        }
        for i in range(LOB_DEPTH):
            df_data[f"lob_ask_vol_{i + 1}"] = [row[i] for row in lob_ask]
            df_data[f"lob_bid_vol_{i + 1}"] = [row[i] for row in lob_bid]

        df = pl.DataFrame(df_data)
        enc = self._gtx.tokenizer.encode(df)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr, dtype=torch.long).unsqueeze(0).to(self._gtx.device)

        return {
            "order_types": _t(enc["order_types"]),
            "price_bins": _t(enc["price_bins"]),
            "vol_int_bins": _t(enc["vol_int_bins"]),
            "vol_dec_bins": _t(enc["vol_dec_bins"]),
            "interval_bins": _t(enc["interval_bins"]),
            "lob_volumes": torch.tensor(enc["lob_volumes"], dtype=torch.float32)
            .unsqueeze(0)
            .to(self._gtx.device),
            "time_of_day": _t(enc["time_of_day"]),
            "mid_deltas": _t(enc["mid_deltas"]),
        }

    # Co-base order-suppression flag. The composer's codegen sets this to True
    # on the generated ComposedAgent class when engines.gentrx (in_process) is
    # in the manifest, so the composer's weapons module owns execution and
    # GenTRX only contributes the signal via :meth:`gentrx_signal`.
    _gtx_signal_only: bool = False

    def gentrx_signal(self, book_id: int) -> float | None:
        """Latest cached per-book GenTRX signal (sign = direction, magnitude
        compared against ``gtx_signal_threshold``).  ``None`` until the first
        ``respond()`` populates the cache for that book.  Read by the composer
        co-base bridge (engines.gentrx in_process module).
        """
        return self._gtx.last_signals.get(book_id)

    def _execute_signal(
        self, response: UnifiedAgentResponse, book_id: int, signal: float
    ) -> UnifiedAgentResponse:
        # Always cache so the composer co-base bridge can read the latest
        # value regardless of whether the standalone order path is suppressed.
        self._gtx.last_signals[book_id] = float(signal)
        if self._gtx_signal_only:
            return response
        if signal > self._gtx.signal_threshold:
            response.market_order(book_id, OrderDirection.BUY, self._gtx.order_qty)
        elif signal < -self._gtx.signal_threshold:
            response.market_order(book_id, OrderDirection.SELL, self._gtx.order_qty)
        return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg_bool(config: Any, key: str, default: bool) -> bool:
    """Read a boolean param from agent config, accepting multiple input forms.

    Handles the string encoding that --agent.params key=value produces:
        bool   True / False
        int    1 / 0
        str    "true" / "false" / "1" / "0" / "yes" / "no"  (case-insensitive)
    Falls back to default if the key is absent or the value is unrecognisable.
    """
    val = getattr(config, key, None)
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    # FLOATS ARRIVE HERE ROUTINELY, because ParseKwargs (taos/common/config/__init__.py) calls float()
    # on every --agent.params value it can. So `gtx_enabled=0` reaches this as 0.0, which is NOT an int,
    # and str(0.0) is "0.0" -- not in the string table below, so it fell through to `default` and the
    # flag silently kept its default value.
    # A flag that silently ignores a caller's explicit 0 is worse than one that rejects it.
    if isinstance(val, float):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return default


def _ts_to_tag(ts_ns: int) -> str:
    """Convert nanosecond timestamp to ddHHMMSS tag for filenames.

    E.g., 90061000000000 (day 1, 01:01:01) → "01010101"
    """
    total_secs = ts_ns // 1_000_000_000
    days = total_secs // 86400
    remainder = total_secs % 86400
    hours = remainder // 3600
    minutes = (remainder % 3600) // 60
    seconds = remainder % 60
    return f"{days:02d}{hours:02d}{minutes:02d}{seconds:02d}"


def _is_trade(event: Any) -> bool:
    """TradeInfo has taker/maker fields; Order and Cancellation do not."""
    return hasattr(event, "taker_id") or hasattr(event, "takerAgentId")


def _is_cancellation(event: Any) -> bool:
    """Cancellations reference an orderId; placements use id."""
    return hasattr(event, "orderId")


def _pad(values: list[int], depth: int) -> list[int]:
    return list(values[:depth]) + [0] * (depth - len(values))


def _events_to_batch(events: list[dict[str, Any]]) -> pa.RecordBatch:
    """Convert a list of collect_row() dicts to a columnar RecordBatch."""
    columns: dict[str, Any] = {
        "timestamp": pa.array(
            [e["timestamp"] for e in events], type=pa.timestamp("ns")
        ),
        "order_type": np.array([e["order_type"] for e in events], dtype=np.int8),
        "rel_price": np.array([e["rel_price"] for e in events], dtype=np.int32),
        "volume_int": np.array([e["volume_int"] for e in events], dtype=np.int32),
        "volume_dec": np.array([e["volume_dec"] for e in events], dtype=np.float32),
        "interval_ns": np.array([e["interval_ns"] for e in events], dtype=np.int64),
        "mid_price": np.array([e["mid_price"] for e in events], dtype=np.int64),
        "time_of_day_s": np.array(
            [e["time_of_day_s"] for e in events], dtype=np.int32
        ),
        "mid_price_delta": np.array(
            [e["mid_price_delta"] for e in events], dtype=np.int32
        ),
    }
    for i in range(LOB_DEPTH):
        k_ask = f"lob_ask_vol_{i + 1}"
        k_bid = f"lob_bid_vol_{i + 1}"
        columns[k_ask] = np.array([e[k_ask] for e in events], dtype=np.float64)
        columns[k_bid] = np.array([e[k_bid] for e in events], dtype=np.float64)
    return pa.RecordBatch.from_pydict(columns, schema=order_stream_schema())


# ---------------------------------------------------------------------------
# GenTRXAgent is importable as:
#   from taos.im.agents import GenTRXAgent
# or directly:
#   from taos.im.agents.gentrx import GenTRXAgent
