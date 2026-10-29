# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Miner-side training service.

Mirrors the validator's `gradient_server` topology on the miner side:
the trading agent stays thin and protocol-aware, the heavy work
(checkpoint download, train_window, gradient compress + upload)
runs in this service. Two deployment options:

- Same machine as the trading agent: loopback HTTP, no auth.
- Separate GPU host: set GENTRX_MINER_API_KEY on both sides.

This module is the in-process service class. The `__main__` driver
that exposes it over FastAPI is `GenTRX.src.miner_training_server`.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from GenTRX.src.gradient_store import GradientStore
    from GenTRX.src.model import ModelConfig, OrderModel
    from GenTRX.src.tokenizer import OrderTokenizer, TokenizerConfig

from GenTRX.src.bt_log import gtx_log
from GenTRX.src.util.paths import default_output_dir


logger = logging.getLogger("GenTRX.miner_training_service")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MinerTrainingConfig:
    """All gtx_-namespaced training knobs in one place.

    Field names drop the `gtx_` prefix here because the prefix is only
    needed when the keys travel through `--agent.params` and risk
    colliding with a strategy's own keys. Inside this module the names
    are unambiguous.
    """

    uid: int
    # Extra UIDs this instance also trains for (besides `uid`), sharing the
    # in-memory base model + GPU. Gradients still go to each UID's own path.
    uids: list[int] = field(default_factory=list)
    output_dir: Path = field(default_factory=default_output_dir)
    gradient_dir: Path | None = None  # defaults to output_dir / "gradients"
    # Wall-clock training budget per round (seconds). Split across the UIDs
    # served this round; train_incremental stops each UID when its slice is up.
    # Keep below the round wallclock minus model/data download + upload headroom.
    round_budget_s: float = 240.0
    # Optional fixed total-step cap per UID (0 = budget-governed, the default).
    train_steps: int = 0
    train_batch_size: int = 16
    train_seq_len: int = 256
    train_lr: float = 1e-4
    top_k_frac: float = 0.10
    label_smooth_sigma: float = 1.0
    aggregator_uid: int = 0
    # Training mode shard: "simulation" (default) or "exchange". Combined
    # with the connected subtensor network to form the gentrx/<network>/<mode>/
    # bucket prefix. Leave at "simulation" unless instructed otherwise.
    mode: str = "simulation"
    # Explicit network shard override: "mainnet" or "testnet". When set, takes
    # precedence over the heuristic. Required for private finney node operators.
    # Equivalent to GENTRX_NETWORK env var.
    network_override: str = ""
    # Optional bootstrap: load this checkpoint at startup if it exists
    initial_checkpoint: Path | None = None
    # Retention: keep newest N gradient files under gradients/ in the
    # write bucket. 0 disables pruning (gradients accumulate forever
    # — operator handles cleanup themselves). Default 50 ≈ ~4 hours
    # of history at the standard round cadence.
    keep_gradients: int = 50
    # Age-based local-disk eviction for <output_dir>/_s3_cache/. Files
    # older than this are deleted after each successful upload. 0
    # disables pruning (cache grows until shutdown or disk fill).
    s3_cache_retention_hours: int = 24


@dataclass
class _TrainingState:
    """Runtime state — not part of the public config."""

    pending_assignments: list[dict] = field(default_factory=list)
    registered_uids: set[int] = field(default_factory=set)
    training_thread: threading.Thread | None = None
    training_in_progress: bool = False
    train_window_id: int = 0
    model_version: int = 0
    last_uploaded_round: int | None = None
    last_loss_before: float | None = None
    last_loss_after: float | None = None
    retry_last_at: float = 0.0
    retry_cooldown: float = 30.0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MinerTrainingService:
    """Owns the miner-side training loop.

    Lifecycle:
        svc = MinerTrainingService(MinerTrainingConfig(uid=7, ...))
        svc.attach_subtensor(subtensor, metagraph, netuid)  # for chain discovery
        svc.bootstrap_model()                               # pull latest checkpoint
        # then per assignment:
        svc.submit_assignment(payload)
        # query state:
        svc.get_status()
        # at shutdown:
        svc.shutdown(timeout=60)

    Thread safety:
        - submit_assignment() is safe from any thread.
        - All training work runs on a single dedicated background thread.
          Only one training window can be in flight at a time.
        - get_status() is read-only and safe from any thread.
    """

    def __init__(self, config: MinerTrainingConfig):
        self.cfg = config
        self.cfg.output_dir = Path(self.cfg.output_dir)
        if self.cfg.gradient_dir is None:
            self.cfg.gradient_dir = self.cfg.output_dir / "gradients"
        else:
            self.cfg.gradient_dir = Path(self.cfg.gradient_dir)
        self.cfg.gradient_dir.mkdir(parents=True, exist_ok=True)

        self.state = _TrainingState()
        # UIDs budgeted up front each round; dynamically-pushed UIDs get added
        # here after their first (leftover-time) round.
        self.state.registered_uids = {self.cfg.uid, *self.cfg.uids}
        # gradient_dir is resolved to a concrete Path above; expose it non-optional
        # so downstream path joins don't trip the Optional field type.
        self.gradient_dir: Path = self.cfg.gradient_dir

        self._lock = threading.Lock()  # protects pending_assignments + training_in_progress

        # Subtensor (set later via attach_subtensor for chain-based discovery)
        self._subtensor = None
        self._metagraph = None
        self._netuid: int | None = None

        # gentrx/<network>/<mode>/ prefix. network is finalized once
        # attach_subtensor() runs (subtensor.network → network_from_subtensor);
        # we start out with a testnet placeholder so unit-style tests that
        # never attach a subtensor still produce a well-formed prefix.
        from GenTRX.src.gradient_store import (
            gentrx_prefix,
            network_from_subtensor,
        )
        if self.cfg.network_override:
            import os as _os
            _os.environ.setdefault("GENTRX_NETWORK", self.cfg.network_override)
        self._bucket_prefix: str = gentrx_prefix(
            network_from_subtensor(None), self.cfg.mode
        )

        # Model / tokenizer (loaded lazily)
        self.model: "OrderModel | None" = None
        self.tokenizer: "OrderTokenizer | None" = None
        self.model_cfg: "ModelConfig | None" = None
        self.tokenizer_cfg: "TokenizerConfig | None" = None
        # Set at init from torch's view of CUDA so the startup banner reflects
        # the device that _load_model_from_checkpoint will pick. Re-checked at
        # load time so dynamic CUDA visibility changes still work.
        try:
            import torch as _torch
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"

        # S3 stores
        self._store: "GradientStore | None" = None  # aggregator-bucket fallback (env-var)
        self._data_store: "GradientStore | None" = None  # default data store (== _store unless overridden)
        self._write_store: "GradientStore | None" = None  # per-miner write bucket
        self._write_stores: "dict[int, GradientStore]" = {}  # per-UID overrides of _write_store
        self._discovered_aggregator_store: "GradientStore | None" = None  # cached chain-based discovery
        # Guards the discovery cache so concurrent training threads don't both
        # walk the metagraph + probe every uid's bucket. In practice the
        # training loop is serialized, but the public API surface allows
        # external callers (status endpoints, ad-hoc scripts) to land here too.
        self._discovered_aggregator_store_lock = threading.Lock()
        self._discovered_aggregator_uid: int = self.cfg.aggregator_uid
        self._s3_cache_dir: Path | None = None
        self._s3_cached_files: dict[str, Path] = {}

        self._tlog = self._build_logger()
        self._init_s3_stores()

        if self.cfg.initial_checkpoint and Path(self.cfg.initial_checkpoint).exists():
            try:
                self._load_model(str(self.cfg.initial_checkpoint))
            except Exception as exc:
                gtx_log.error(f"Failed to load initial checkpoint: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach_subtensor(self, subtensor, metagraph, netuid: int) -> None:
        """Wire chain-based aggregator discovery.

        Also recomputes the bucket prefix from the connected subtensor's
        network identifier so all stores built after this call land in
        the right gentrx/<network>/<mode>/ shard.
        """
        from GenTRX.src.gradient_store import (
            gentrx_prefix,
            network_from_subtensor,
        )
        self._subtensor = subtensor
        self._metagraph = metagraph
        self._netuid = netuid
        if self.cfg.network_override:
            import os as _os
            _os.environ["GENTRX_NETWORK"] = self.cfg.network_override
        network_name = getattr(subtensor, "network", None)
        new_prefix = gentrx_prefix(
            network_from_subtensor(network_name), self.cfg.mode
        )
        if new_prefix != self._bucket_prefix:
            self._bucket_prefix = new_prefix
            for store in (self._store, self._data_store, self._write_store):
                if store is not None:
                    store.prefix = new_prefix
            gtx_log.info(
                f"GenTRX bucket prefix: {new_prefix} "
                f"(network={network_from_subtensor(network_name)}, mode={self.cfg.mode})"
            )

    def bootstrap_model(self) -> bool:
        """Pull the latest published checkpoint from the aggregator bucket.

        Returns True if a model is loaded after the call.
        """
        if self.model is not None:
            return True
        return self._ensure_model_version()

    def submit_assignment(self, payload: dict) -> dict:
        """Queue an assignment for training.

        Returns a status dict the HTTP layer can serialize as JSON.
        Never blocks on network or training — kicks off the training
        thread if one isn't already running.
        """
        if not isinstance(payload, dict) or "round" not in payload:
            return {"status": "rejected", "reason": "invalid payload"}

        # The forwarding agent tags the assignment with its own UID. Fall back
        # to this instance's primary UID for single-UID / legacy callers.
        payload.setdefault("miner_uid", self.cfg.uid)

        with self._lock:
            self.state.pending_assignments.append(payload)
            in_progress = self.state.training_in_progress

        round_id = payload.get("round", "?")
        self._tlog.info(
            f"assignment queued: round={round_id} uid={payload.get('miner_uid')} "
            f"validator_uid={payload.get('validator_uid', '?')} "
            f"books={payload.get('books', [])} "
            f"data_files={len(payload.get('data', []))} "
            f"in_progress={in_progress}"
        )

        if not in_progress:
            self._kick_training()
        return {"status": "queued", "round": round_id, "in_progress": in_progress}

    def get_status(self) -> dict:
        with self._lock:
            pending = len(self.state.pending_assignments)
            in_progress = self.state.training_in_progress
        return {
            "uid": self.cfg.uid,
            "training_in_progress": in_progress,
            "pending_assignments": pending,
            "model_version": self.state.model_version,
            "last_uploaded_round": self.state.last_uploaded_round,
            "last_loss_before": self.state.last_loss_before,
            "last_loss_after": self.state.last_loss_after,
            "train_window_id": self.state.train_window_id,
            "retry_cooldown_s": self.state.retry_cooldown,
            "pending_retry_count": self._pending_retry_count(),
        }

    def get_version_info(self) -> dict:
        return {
            "service": "miner_training_service",
            "uid": self.cfg.uid,
            "model_version": self.state.model_version,
        }

    def retry_pending_gradients(self) -> None:
        """Replay any locally-saved gradients from a previous failed S3 upload.

        Cooldown: 30 s → 300 s exponential. Reset to 30 s on success.
        Idempotent — call from any background tick.
        """
        if self._write_store is None:
            return
        pending_dir = self.cfg.gradient_dir / "pending"
        if not pending_dir.exists():
            return

        now = time.time()
        if now - self.state.retry_last_at < self.state.retry_cooldown:
            return
        self.state.retry_last_at = now

        any_failed = False
        for grad_path in sorted(pending_dir.glob("*.grad")):
            try:
                parts = grad_path.stem.split("_")
                round_id = int(parts[1])
                data = grad_path.read_bytes()
                self._write_store.put_gradient(
                    miner_uid=self.cfg.uid, round_id=round_id, data=data
                )
                grad_path.unlink()
                self._tlog.info(f"Retried pending gradient: round {round_id}")
            except Exception as exc:
                self._tlog.debug(f"Retry failed for {grad_path.name}: {exc}")
                any_failed = True

        if any_failed:
            self.state.retry_cooldown = min(self.state.retry_cooldown * 2, 300.0)
        else:
            self.state.retry_cooldown = 30.0

    def shutdown(self, timeout: float = 60.0) -> None:
        """Wait for any in-progress training thread to finish."""
        if self.state.training_thread and self.state.training_thread.is_alive():
            self._tlog.info("Shutdown: waiting for training thread...")
            self.state.training_thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal: setup
    # ------------------------------------------------------------------

    def _build_logger(self) -> logging.Logger:
        from logging.handlers import RotatingFileHandler

        log = logging.getLogger(f"GenTRX.miner_training_service.{self.cfg.uid}")
        log.setLevel(logging.INFO)
        log.propagate = False
        if not log.handlers:
            fh = RotatingFileHandler(
                self.cfg.gradient_dir / "train.log",
                maxBytes=50 * 1024 * 1024,
                backupCount=5,
            )
            fh.setFormatter(
                logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
            )
            log.addHandler(fh)
        log.info(f"Training service initialized (uid={self.cfg.uid})")
        return log

    def _init_s3_stores(self) -> None:
        """Build the three logical stores from environment.

        Same conventions as `taos.im.agents.GenTRXAgent`:
          _store        : aggregator bucket fallback (env-var)
          _data_store   : default data store (overridden per-assignment)
          _write_store  : per-miner write bucket
        """
        try:
            from GenTRX.src.gradient_store import (
                create_aggregator_store_from_env,
                GradientStore,
            )
        except ImportError:
            return

        self._store = create_aggregator_store_from_env(prefix=self._bucket_prefix)
        if self._store:
            gtx_log.info(
                f"S3 aggregator bucket fallback: {self._store.endpoint_url}/{self._store.bucket}"
            )
        self._data_store = self._store

        agent_bucket = os.environ.get("GENTRX_AGENT_S3_BUCKET")
        if agent_bucket:
            self._write_store = GradientStore(
                endpoint_url=os.environ.get(
                    "GENTRX_AGENT_S3_ENDPOINT_URL",
                    self._store.endpoint_url if self._store else "",
                ),
                bucket=agent_bucket,
                access_key=os.environ.get("GENTRX_AGENT_S3_ACCESS_KEY", ""),
                secret_key=os.environ.get("GENTRX_AGENT_S3_SECRET_KEY", ""),
                region=os.environ.get("GENTRX_AGENT_S3_REGION", "auto"),
                prefix=self._bucket_prefix,
            )
            gtx_log.info(
                f"S3 write (gradients): {self._write_store.endpoint_url}/{self._write_store.bucket}"
            )

    # ------------------------------------------------------------------
    # Internal: training trigger
    # ------------------------------------------------------------------

    def _kick_training(self) -> None:
        """Spawn the background thread to train all pending per-UID assignments."""
        with self._lock:
            if self.state.training_in_progress:
                return
            assignments = self.state.pending_assignments
            self.state.pending_assignments = []
            if not assignments:
                return
            self.state.training_in_progress = True

        by_uid: dict[int, list[dict]] = {}
        for a in assignments:
            by_uid.setdefault(int(a.get("miner_uid", self.cfg.uid)), []).append(a)
        target_v = max(
            (int(a.get("model_version", 0) or 0) for a in assignments), default=0
        )

        self._tlog.info(
            f"training kick: {len(by_uid)} uid(s) {sorted(by_uid)} "
            f"model_v={target_v} round_budget={self.cfg.round_budget_s:.0f}s"
        )

        if self.model is None and target_v <= 0:
            self._tlog.info(
                "no model and no model_version — skipping (assignments preserved)"
            )
            with self._lock:
                self.state.pending_assignments = assignments + self.state.pending_assignments
                self.state.training_in_progress = False
            return

        self.state.training_thread = threading.Thread(
            target=self._train_all_uids_background,
            args=(by_uid, target_v),
            daemon=True,
        )
        self.state.training_thread.start()

    def _train_all_uids_background(self, by_uid: dict, target_v: int) -> None:
        """Train every pending UID against one shared base model.

        Downloads the target checkpoint once, then for each UID deepcopies the
        base model, trains its pages incrementally within a budget slice, and
        uploads a distinct gradient. Registered UIDs get equal guaranteed
        slices; an unregistered (dynamically-pushed) UID trains on leftover time
        this round and is registered for the next one.
        """
        try:
            primary = next(iter(by_uid.values()))[0]
            need_model = self.model is None or (
                target_v > self.state.model_version and target_v > 0
            )
            if need_model:
                self._tlog.info(
                    f"ensuring model v{target_v} (have v{self.state.model_version})..."
                )
                ok = self._ensure_model_version(target_v, primary)
                if self.model is None:
                    self._tlog.info("model bootstrap failed — assignments preserved")
                    with self._lock:
                        for lst in by_uid.values():
                            self.state.pending_assignments = (
                                lst + self.state.pending_assignments
                            )
                    return
                if not ok:
                    self._tlog.warning(
                        f"model v{target_v} unavailable — training on v{self.state.model_version}"
                    )

            registered = [u for u in by_uid if u in self.state.registered_uids]
            if registered:
                per_uid = self.cfg.round_budget_s / len(registered)
                order = registered + [
                    u for u in by_uid if u not in self.state.registered_uids
                ]
            else:
                per_uid = self.cfg.round_budget_s / max(1, len(by_uid))
                order = list(by_uid)

            spent = 0.0
            for uid in order:
                if uid in self.state.registered_uids:
                    budget = per_uid
                else:
                    budget = min(per_uid, max(0.0, self.cfg.round_budget_s - spent))
                    self.state.registered_uids.add(uid)  # promote for next round
                if budget <= 0:
                    self._tlog.info(
                        f"uid {uid}: no budget left this round — deferred (registered next round)"
                    )
                    continue
                t_uid = time.perf_counter()
                try:
                    spent += self._train_one_uid(uid, by_uid[uid], target_v, budget)
                except Exception as exc:
                    # One UID's failure must not cost the remaining UIDs their round.
                    spent += time.perf_counter() - t_uid
                    self._tlog.error(f"uid {uid}: training failed: {exc}")
                    import traceback
                    self._tlog.error(traceback.format_exc())
        except Exception as exc:
            self._tlog.error(f"_train_all_uids_background failed: {exc}")
            import traceback
            self._tlog.error(traceback.format_exc())
        finally:
            with self._lock:
                self.state.training_in_progress = False
            try:
                self._prune_gradients()
            except Exception:
                pass
            self._tlog.info("thread finished")
            self._clear_s3_cache()

    def _build_page_loaders(self, files: list):
        """One DataLoader per page file, order preserved (recent pages first)."""
        from GenTRX.src.dataloader import OrderDataset, ChunkSampler
        from torch.utils.data import DataLoader

        loaders = []
        for f in files:
            try:
                ds = OrderDataset(
                    [f],
                    seq_len=self.cfg.train_seq_len,
                    tokenizer=self.tokenizer,
                    max_cached=1,
                )
            except Exception as exc:
                self._tlog.debug(f"skip page {f}: {exc}")
                continue
            loaders.append(
                DataLoader(
                    ds,
                    batch_size=self.cfg.train_batch_size,
                    sampler=ChunkSampler(ds, shuffle=True),
                    num_workers=0,
                )
            )
        return loaders

    def _write_store_for(self, uid: int):
        """Per-UID write store. Defaults to the shared store (one bucket, with
        `{uid}` in the path). Per-UID credential files are a future option."""
        return self._write_stores.get(uid, self._write_store)

    def _train_one_uid(
        self, uid: int, assignments: list[dict], target_v: int, budget_s: float
    ) -> float:
        """Train one UID's pages within budget_s and upload its gradient.

        Returns wall-clock seconds spent (download + train) so the caller can
        track the remaining round budget for any leftover UIDs.
        """
        import time as _time

        from GenTRX.src.distributed import train_incremental, WindowConfig
        from GenTRX.src.gradient import compress, serialize

        t0 = _time.perf_counter()
        files: list[Path] = []
        for a in assignments:
            files.extend(self._download_assignment_data(a))
        loaders = self._build_page_loaders(files)
        if not loaders:
            self._tlog.warning(f"uid {uid}: no trainable pages — skipping")
            return _time.perf_counter() - t0

        round_id = int(assignments[0].get("round", self.state.train_window_id))
        train_model = copy.deepcopy(self.model)
        self._tlog.info(
            f"uid {uid}: window {self.state.train_window_id} START | "
            f"{len(loaders)} pages | budget={budget_s:.0f}s | model_v={target_v}"
        )
        win_cfg = WindowConfig(
            n_steps=self.cfg.train_steps,   # 0 = budget-governed
            lr=self.cfg.train_lr,
            window_id=self.state.train_window_id,
            miner_uid=uid,
            model_version=target_v,
            label_smooth_sigma=self.cfg.label_smooth_sigma,
            budget_s=budget_s,
        )
        delta = train_incremental(train_model, loaders, win_cfg, self.device)
        data = serialize(compress(delta, top_k_frac=self.cfg.top_k_frac))

        store = self._write_store_for(uid)
        if store is not None:
            try:
                store.put_gradient(miner_uid=uid, round_id=round_id, data=data)
                self._tlog.info(
                    f"uid {uid}: gradient uploaded (round={round_id}, {len(data)/1024:.1f} KB)"
                )
                self.state.last_uploaded_round = round_id
            except Exception as exc:
                self._tlog.warning(f"uid {uid}: S3 upload failed: {exc} — saving for retry")
                pending_dir = self.cfg.gradient_dir / "pending"
                pending_dir.mkdir(parents=True, exist_ok=True)
                (pending_dir / f"block_{round_id:08d}_miner_{uid}.grad").write_bytes(data)
        else:
            self._tlog.warning(f"uid {uid}: no write store configured — gradient dropped")

        self.state.last_loss_before = float(delta.metadata.loss_before)
        self.state.last_loss_after = float(delta.metadata.loss_after)
        self.state.train_window_id += 1
        self._tlog.info(
            f"uid {uid}: COMPLETE | loss {delta.metadata.loss_before:.4f} → "
            f"{delta.metadata.loss_after:.4f} | {delta.metadata.steps_trained} steps"
        )
        return _time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Internal: S3 store discovery + downloads
    # ------------------------------------------------------------------

    def _get_aggregator_store_for_assignment(self, assignment: dict | None):
        """Return a GradientStore pointing at a validator bucket with a
        published checkpoint. See taos.im.agents.GenTRXAgent for the equivalent
        inline logic; this is a duplicate by design.
        """
        if self._discovered_aggregator_store is not None:
            return self._discovered_aggregator_store

        from GenTRX.src.gradient_store import GradientStore

        def _build_store(bi) -> GradientStore:
            return GradientStore(
                endpoint_url=bi.endpoint_url,
                bucket=bi.bucket_name,
                access_key=bi.access_key_id,
                secret_key=bi.secret_access_key,
                region=os.environ.get("GENTRX_VALIDATOR_S3_REGION", "auto"),
                prefix=self._bucket_prefix,
            )

        try:
            if (
                self._subtensor is not None
                and self._metagraph is not None
                and self._netuid is not None
            ):
                from GenTRX.src.chain import GenTRXChain
                gtx_chain = GenTRXChain(self._subtensor, self._netuid, self._metagraph)

                configured_uid = self.cfg.aggregator_uid

                # Step 1: configured uid via chain
                try:
                    bucket_info = gtx_chain.get_bucket(configured_uid)
                    if bucket_info is not None:
                        store = _build_store(bucket_info)
                        latest = store.get_latest_version(configured_uid)
                        if latest > 0:
                            self._tlog.info(
                                f"Aggregator bucket discovered: uid={configured_uid} "
                                f"{bucket_info.endpoint_url}/{bucket_info.bucket_name} "
                                f"(latest v{latest})"
                            )
                            self._discovered_aggregator_store = store
                            self._discovered_aggregator_uid = configured_uid
                            return store
                except Exception as exc:
                    self._tlog.debug(f"uid={configured_uid} bucket probe failed: {exc}")

                # Step 2: env-var store
                if self._store is not None:
                    try:
                        if self._store.get_latest_version(configured_uid) > 0:
                            self._tlog.info(
                                f"Aggregator bucket from env: "
                                f"{self._store.endpoint_url}/{self._store.bucket}"
                            )
                            self._discovered_aggregator_store = self._store
                            self._discovered_aggregator_uid = configured_uid
                            return self._store
                    except Exception as exc:
                        self._tlog.debug(f"env-var store probe failed: {exc}")

                # Steps 3+4: sender then remaining metagraph
                scan_uids: list[int] = []
                sender = (assignment or {}).get("validator_uid")
                if sender is not None:
                    scan_uids.append(int(sender))
                try:
                    n = int(self._metagraph.n.item())
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
                            self._tlog.info(
                                f"Aggregator bucket discovered: uid={uid} "
                                f"{bucket_info.endpoint_url}/{bucket_info.bucket_name} "
                                f"(latest v{latest})"
                            )
                            self._discovered_aggregator_store = store
                            self._discovered_aggregator_uid = uid
                            return store
                    except Exception as exc:
                        self._tlog.debug(f"uid={uid} bucket probe failed: {exc}")
                        continue
        except Exception as exc:
            self._tlog.warning(f"Chain aggregator discovery failed: {exc}")

        return self._store

    def _assignment_data_store(self, assignment: dict):
        """Return a GradientStore for downloading training parquets."""
        bucket = assignment.get("data_bucket", "")
        if bucket:
            try:
                from GenTRX.src.gradient_store import GradientStore
                return GradientStore(
                    endpoint_url=assignment.get("data_endpoint", ""),
                    bucket=bucket,
                    access_key=assignment.get("data_access_key", ""),
                    secret_key=assignment.get("data_secret_key", ""),
                    region=os.environ.get("GENTRX_VALIDATOR_S3_REGION", "auto"),
                    prefix=self._bucket_prefix,
                )
            except Exception as exc:
                self._tlog.warning(f"Failed to build data store from assignment fields: {exc}")
        return self._data_store

    def _download_assignment_data(self, assignment: dict) -> list[Path]:
        data_keys = assignment.get("data", [])
        data_source = assignment.get("data_source", "local")

        if not data_keys:
            return []

        if data_source == "local":
            return [Path(k) for k in data_keys if Path(k).exists()]

        data_store = self._assignment_data_store(assignment)
        if data_store is None:
            self._tlog.warning("assignment has S3 data but no data_store configured")
            return []

        if self._s3_cache_dir is None:
            self._s3_cache_dir = self.cfg.output_dir / "_s3_cache"

        bucket_id = assignment.get("data_bucket", "") or "default"
        bucket_hash = hashlib.md5(bucket_id.encode()).hexdigest()[:8]
        cache_base = self._s3_cache_dir / bucket_hash

        local_files: list[Path] = []
        for key in data_keys:
            cache_key = f"{bucket_hash}/{key}"
            if cache_key in self._s3_cached_files:
                local_files.append(self._s3_cached_files[cache_key])
                continue

            parts = key.split("/")
            if len(parts) < 5 or parts[0] != "data" or parts[3] != "intervals":
                self._tlog.warning(f"  unexpected S3 key format: {key}")
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
                self._s3_cached_files[cache_key] = local_path
                local_files.append(local_path)
            except Exception as exc:
                self._tlog.warning(f"  S3 download failed from data bucket: {key}: {exc}")

        return local_files

    def _ensure_model_version(
        self, target: int | None = None, assignment: dict | None = None
    ) -> bool:
        store = (
            self._get_aggregator_store_for_assignment(assignment)
            if assignment is not None
            else self._store
        )
        if store is None:
            return self.model is not None

        try:
            agg_uid = self._discovered_aggregator_uid
            latest = store.get_latest_version(agg_uid)
            if target is None:
                target = latest
            elif latest > 0 and latest > target:
                self._tlog.info(f"aggregator has v{latest} (assignment says v{target}) — using latest")
                target = latest

            if target <= 0:
                return self.model is not None
            if target <= self.state.model_version:
                return True

            from GenTRX.src.distributed import apply_version_deltas

            # Fast path: advance an existing model by applying canonical deltas.
            if self.model is not None and self.state.model_version > 0:
                expected = {n: p.shape for n, p in self.model.named_parameters()}
                reached = apply_version_deltas(
                    self.model, store, agg_uid, self.state.model_version, target, expected
                )
                if reached >= target:
                    self.state.model_version = target
                    gtx_log.info(f"Advanced model via deltas → v{target}")
                    return True
                gtx_log.info(f"Delta gap after v{reached}; reloading baseline")

            # Cold start / delta gap: download latest baseline, replay to head.
            baseline_v = latest if latest > 0 else target
            gtx_log.info(f"Downloading baseline checkpoint v{baseline_v} from aggregator bucket")
            ckpt_bytes = store.get_checkpoint(agg_uid, baseline_v)
            stage_dir = self.cfg.output_dir / "ckpt_cache"
            stage_dir.mkdir(parents=True, exist_ok=True)
            tmp = stage_dir / f"gentrx_ckpt_{self.cfg.uid}.pt"
            tmp.write_bytes(ckpt_bytes)
            self._load_model(str(tmp))
            self.state.model_version = baseline_v
            if baseline_v < target:
                expected = {n: p.shape for n, p in self.model.named_parameters()}
                self.state.model_version = apply_version_deltas(
                    self.model, store, agg_uid, baseline_v, target, expected
                )
            gtx_log.info(f"Model at v{self.state.model_version} (target v{target})")
            return self.state.model_version >= target
        except Exception as exc:
            gtx_log.warning(f"Model sync to v{target} failed: {exc}")
            return False

    def _load_model(self, checkpoint: str) -> None:
        from GenTRX.src.gradient import load_checkpoint_safely, validate_state_dict
        from GenTRX.src.model import OrderModel, ModelConfig
        from GenTRX.src.tokenizer import OrderTokenizer, TokenizerConfig
        import torch

        ckpt = load_checkpoint_safely(checkpoint)
        raw_cfg = ckpt.get("model_config", ckpt.get("config", {}))
        self.model_cfg = ModelConfig(
            **{k: v for k, v in raw_cfg.items() if k in ModelConfig.__dataclass_fields__}
        )
        tok_dict = ckpt.get("tokenizer_config")
        self.tokenizer_cfg = (
            TokenizerConfig.from_dict(tok_dict) if tok_dict else TokenizerConfig()
        )
        self.model = OrderModel(self.model_cfg)
        validate_state_dict(ckpt["model_state_dict"], self.model.state_dict())
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.tokenizer = OrderTokenizer(self.tokenizer_cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        gtx_log.info(f"GenTRX model loaded from {checkpoint} on {self.device}")

    def _clear_s3_cache(self) -> None:
        import shutil
        if self._s3_cache_dir and self._s3_cache_dir.exists():
            try:
                shutil.rmtree(self._s3_cache_dir)
                self._tlog.debug("S3 cache cleared")
            except Exception as exc:
                self._tlog.debug(f"S3 cache clear failed: {exc}")
        self._s3_cached_files.clear()
        self._s3_cache_dir = None

    def _pending_retry_count(self) -> int:
        pending_dir = self.cfg.gradient_dir / "pending"
        if not pending_dir.exists():
            return 0
        return len(list(pending_dir.glob("*.grad")))

    def _prune_gradients(self) -> None:
        """Trim each served UID's gradients/ prefix to the configured retention.

        Hot bucket only — operators wanting long history should pull
        objects to cold storage on their own cadence.
        """
        if self.cfg.keep_gradients <= 0:
            return
        for uid in sorted(self.state.registered_uids):
            store = self._write_store_for(uid)
            if store is None:
                continue
            try:
                n = store.prune_keep_latest(
                    f"gradients/{uid}/",
                    keep=self.cfg.keep_gradients,
                    suffix=".grad",
                )
                if n:
                    self._tlog.info(
                        f"uid {uid}: pruned {n} old gradient(s), "
                        f"keeping latest {self.cfg.keep_gradients}"
                    )
            except Exception as exc:
                self._tlog.debug(f"uid {uid}: gradient prune failed: {exc}")


    def _prune_s3_cache(self) -> None:
        """Age-evict files under <output_dir>/_s3_cache/.

        Files with mtime older than `s3_cache_retention_hours` are
        deleted. No-op when the cache directory does not exist or
        retention is set to 0.
        """
        if self.cfg.s3_cache_retention_hours <= 0:
            return
        cache_root = self._s3_cache_dir or (self.cfg.output_dir / "_s3_cache")
        if not cache_root.exists():
            return
        cutoff = time.time() - self.cfg.s3_cache_retention_hours * 3600
        n = 0
        try:
            for f in cache_root.rglob("*"):
                if not f.is_file():
                    continue
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        n += 1
                except OSError:
                    continue
            if n:
                self._tlog.info(
                    f"pruned {n} stale _s3_cache file(s) older than "
                    f"{self.cfg.s3_cache_retention_hours}h"
                )
        except Exception as exc:
            self._tlog.debug(f"_s3_cache prune failed: {exc}")
