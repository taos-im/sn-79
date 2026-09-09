# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""GenTRX service — validator-side orchestrator for distributed training.

The validator drives round scheduling (block-based or timer-based), creates
assignments from available data, pushes the assignment plan to the gradient
server, and delivers assignments to miners via dendrite.

The gradient server is passive for scheduling — it processes state ticks,
reports data availability, accepts assignment plans, and handles gradient
collection + scoring + aggregation.

HTTP contract with gradient server:
  POST /gentrx/state          — push sim state tick (msgpack, every tick)
  GET  /gentrx/data-status    — available data ranges per book
  POST /gentrx/round          — push assignment plan for a round
  GET  /gentrx/scores         — poll miner scores
  GET  /gentrx/version        — health check

All HTTP calls use the GENTRX_API_KEY shared secret (via X-API-Key header).
"""

from __future__ import annotations

import atexit
import hashlib
import os
import queue
import random
import signal
import struct
import threading
import time
from typing import Any

import msgpack

from GenTRX.src.bt_log import gtx_log


class _TxSpool:
    """Append-only WAL for outbound state packets; survives validator restart."""

    def __init__(self, path: str, max_bytes: int = 50_000_000) -> None:
        self.path = path
        self.offset_path = path + ".offset"
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._ack_offset = self._read_offset()
        self._f = open(path, "a+b")
        self.evictions_total = 0

    def append(self, data: bytes) -> int:
        evicted = 0
        with self._lock:
            self._f.seek(0, os.SEEK_END)
            self._f.write(struct.pack(">I", len(data)))
            self._f.write(data)
            self._f.flush()
            if self._f.tell() > self.max_bytes:
                evicted = self._evict_locked(self.max_bytes // 2)
        return evicted

    def replay_unacked(self) -> list[bytes]:
        with self._lock:
            self._f.seek(self._ack_offset)
            records: list[bytes] = []
            while True:
                header = self._f.read(4)
                if len(header) < 4:
                    break
                length = struct.unpack(">I", header)[0]
                data = self._f.read(length)
                if len(data) < length:
                    break
                records.append(data)
            self._f.seek(0, os.SEEK_END)
            return records

    def ack_one(self) -> None:
        with self._lock:
            saved = self._f.tell()
            self._f.seek(self._ack_offset)
            header = self._f.read(4)
            if len(header) >= 4:
                length = struct.unpack(">I", header)[0]
                self._ack_offset += 4 + length
                self._write_offset_locked()
            self._f.seek(saved)

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.close()

    def _read_offset(self) -> int:
        try:
            with open(self.offset_path, "rb") as f:
                buf = f.read(8)
                if len(buf) < 8:
                    return 0
                return struct.unpack(">Q", buf)[0]
        except FileNotFoundError:
            return 0

    def _write_offset_locked(self) -> None:
        tmp = self.offset_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(struct.pack(">Q", self._ack_offset))
        os.replace(tmp, self.offset_path)

    def _evict_locked(self, bytes_to_drop: int) -> int:
        self._f.seek(self._ack_offset)
        dropped_bytes = 0
        dropped_records = 0
        while dropped_bytes < bytes_to_drop:
            header = self._f.read(4)
            if len(header) < 4:
                break
            length = struct.unpack(">I", header)[0]
            self._f.seek(length, os.SEEK_CUR)
            dropped_bytes += 4 + length
            dropped_records += 1
        self._ack_offset += dropped_bytes
        self.evictions_total += dropped_records
        self._compact_locked()
        return dropped_records

    def _compact_locked(self) -> None:
        self._f.seek(self._ack_offset)
        remaining = self._f.read()
        self._f.close()
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(remaining)
        os.replace(tmp, self.path)
        self._ack_offset = 0
        self._write_offset_locked()
        self._f = open(self.path, "a+b")

# Bittensor mainnet block time (~12 s/block). Used only for human-readable
# "next round in ~N min" log messages; not used for any scheduling logic.
_BITTENSOR_BLOCK_TIME_S = 12.0


def _log_runtime_versions() -> None:
    """One-shot startup log: Python + key dep versions.

    With the Python version pin now open-ended (>=3.10) and no requirements
    lock file, bug reports need to carry the exact versions the operator is
    actually running. Cheap to compute, single line per dep.
    """
    import sys
    lines = [f"python={sys.version.split()[0]}"]
    for name in ("bittensor", "torch", "httpx", "msgpack", "boto3", "fastapi", "uvicorn"):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            lines.append(f"{name}={ver}")
        except Exception:
            lines.append(f"{name}=absent")
    gtx_log.info("runtime: %s", " ".join(lines))


class GenTRXService:
    """Validator-side orchestrator for GenTRX distributed training.

    Drives round scheduling, creates assignments from available data,
    pushes them to the gradient server, delivers to miners via dendrite.
    """

    # Assignment creation defaults (same as old gradient_server values)
    DEFAULT_BOOKS_PER_MINER = 3
    DEFAULT_WINDOW_NS = 300_000_000_000  # 5 min

    DEFAULT_TX_QUEUE_SIZE = 256
    DEFAULT_TX_MAX_ATTEMPTS = 3
    DEFAULT_TX_BACKOFF_BASE_S = 0.5
    DEFAULT_TX_SPOOL_MAX_BYTES = 50_000_000
    DEFAULT_TX_SUMMARY_INTERVAL_S = 60.0
    DEFAULT_TX_DRAIN_TIMEOUT_S = 10.0

    def __init__(
        self,
        packager: Any,
        gradient_server_url: str,
        api_key: str = "",
        poll_interval: float = 30.0,
        deliver_fn: Any | None = None,
        miner_uids: list[int] | None = None,
        miner_uids_fn: Any | None = None,
        log_path: str | None = None,
        # Round scheduling
        blocks_per_round: int = 0,
        get_block_fn: Any | None = None,
        # Assignment tunables
        books_per_miner: int = 3,
        window_ns: int = 0,
        val_fraction: float = 0.10,
        # Identity (used to scope data keys under data/<validator_uid>/)
        validator_uid: int | str = 0,
        # WAL spool for at-least-once state push across restart. None disables.
        tx_spool_path: str | None = None,
        tx_spool_max_bytes: int = 0,
    ) -> None:
        if not gradient_server_url:
            raise ValueError(
                "GenTRXService requires a gradient_server_url. Single-machine "
                "deployments use a loopback URL like http://127.0.0.1:8100/gentrx; "
                "there is no in-process mode."
            )
        self._packager = packager
        self._server_url = gradient_server_url.rstrip("/")
        self._api_key = api_key
        self._poll_interval = poll_interval
        self._deliver_fn = deliver_fn
        self._validator_uid = validator_uid
        self._miner_uids_static = miner_uids or []
        self._miner_uids_fn = miner_uids_fn

        # Round scheduling
        self._blocks_per_round = blocks_per_round
        self._get_block_fn = get_block_fn  # callable() -> int (current block number)
        self._current_round = 0
        self._last_round_push: float = 0.0
        self._last_known_block: int | None = None  # cached from last _should_advance_round query

        # Assignment tunables
        self._books_per_miner = books_per_miner or self.DEFAULT_BOOKS_PER_MINER
        self._window_ns = window_ns or self.DEFAULT_WINDOW_NS
        self._val_fraction = val_fraction
        self._val_books: set[str] | None = None  # current round's held-out split
        # Page-based assignment: pick one existing page per book (recency-biased,
        # with a stale-revisit tail so untouched pages still get covered) instead
        # of beta-sampling a time window and hoping parquets overlap it.
        self._page_last_round: dict[str, int] = {}
        self._stale_revisit_frac = 0.2
        self._recency_window = 4
        # IID assignment (flavor B): each miner gets a random SAMPLE of train
        # books (overlap allowed), not a disjoint slice — so gradients estimate
        # the same objective and are fairly comparable on the shared held-out
        # window. Flip to True for flavor A (every miner the identical sample).
        self._iid_shared_books = False

        # log_path kept as a no-op for API compatibility. GenTRX records now
        # flow through bt.logging via the `gtx_log` shim — no separate
        # handler chain, no duplicate emits. Operators grep bt's stream for
        # `[GTX]` to isolate GenTRX activity.
        _ = log_path

        # State
        self._scores: dict[int, dict] = {}
        self._last_aggregation_stats: dict = {}
        self._last_config: dict = {}
        self._last_poll: float = 0.0
        self._last_score_poll: float = 0.0
        self._last_score_round_seen: int = -1
        self._last_health_check: float = 0.0
        self._health_check_interval: float = 60.0  # seconds between /health pings
        self._last_health_ok: bool | None = None   # None = never checked
        # Max sim timestamp we've pushed to the gradient server — lets us
        # skip the /data-status + /round HTTP chatter during the warmup
        # window (sim time < window_ns) before any parquet could exist.
        self._max_sim_ts_pushed: int = 0
        self._last_warmup_log: float = 0.0

        spool_cap = tx_spool_max_bytes or self.DEFAULT_TX_SPOOL_MAX_BYTES
        self._tx_spool: _TxSpool | None = None
        if tx_spool_path:
            try:
                self._tx_spool = _TxSpool(tx_spool_path, max_bytes=spool_cap)
            except Exception as exc:
                gtx_log.warning("TX spool init failed, continuing without WAL: %s", exc)
                self._tx_spool = None
        # Spool is the durability bound, so memory queue can be unbounded
        # when spool is active. Without spool fall back to drop-oldest.
        qmax = 0 if self._tx_spool is not None else self.DEFAULT_TX_QUEUE_SIZE
        self._tx_queue: queue.Queue = queue.Queue(maxsize=qmax)
        self._tx_drops: int = 0
        self._tx_retries_total: int = 0
        self._tx_sends_total: int = 0
        self._tx_stop = threading.Event()
        if self._tx_spool is not None:
            replayed = self._tx_spool.replay_unacked()
            for data in replayed:
                self._tx_queue.put_nowait(data)
            if replayed:
                gtx_log.info(
                    "TX spool replay: re-enqueued %d packets from %s",
                    len(replayed), tx_spool_path,
                )
        self._tx_thread = threading.Thread(
            target=self._tx_worker, name="GenTRX-tx", daemon=True,
        )
        self._tx_thread.start()
        atexit.register(self._drain_on_exit)
        # SIGTERM (systemd, bare kill) bypasses atexit; SIGINT (pm2 default) does not.
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGTERM, self._handle_sigterm)
            except (ValueError, OSError) as exc:
                gtx_log.debug("SIGTERM handler not installed: %s", exc)

    @property
    def _miner_uids(self) -> list[int]:
        if self._miner_uids_fn is not None:
            try:
                return self._miner_uids_fn() or []
            except Exception:
                return []
        return self._miner_uids_static

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        deliver_fn: Any | None = None,
        miner_uids: list[int] | None = None,
        miner_uids_fn: Any | None = None,
        get_block_fn: Any | None = None,
        validator_uid: int | str = 0,
    ) -> GenTRXService | None:
        """Create from validator/proxy config.

        Returns None when GenTRX is disabled or gradient server URL is missing.
        """
        gentrx_cfg = getattr(config, "gentrx", None)
        if gentrx_cfg is None or not getattr(gentrx_cfg, "enabled", False):
            return None

        try:
            from GenTRX.src.state_packager import StatePackager
        except ImportError as exc:
            gtx_log.warning("import failed — disabled: %s", exc)
            return None

        server_url = getattr(gentrx_cfg, "gradient_server_url", "")
        if not server_url:
            gtx_log.error(
                "enabled but --gentrx.gradient_server_url is empty. "
                "Set it to the gradient server endpoint (e.g. "
                "http://127.0.0.1:8100/gentrx for single-machine setups)."
            )
            return None

        import os as _os
        api_key = getattr(gentrx_cfg, "api_key", "") or _os.environ.get("GENTRX_API_KEY", "")
        interval = getattr(gentrx_cfg, "interval", 30)
        log_path = getattr(gentrx_cfg, "log_path", "data/gentrx/gentrx_service.log")
        blocks_per_round = getattr(gentrx_cfg, "blocks_per_round", 0) or 0
        books_per_miner = getattr(gentrx_cfg, "books_per_miner", 0) or 0
        default_spool = f"data/gentrx/tx_spool_v{validator_uid}.bin"
        tx_spool_path = getattr(gentrx_cfg, "tx_spool_path", default_spool)
        tx_spool_max_bytes = int(getattr(gentrx_cfg, "tx_spool_max_bytes", 0) or 0)

        service = cls(
            packager=StatePackager(),
            gradient_server_url=server_url,
            api_key=api_key,
            poll_interval=interval,
            deliver_fn=deliver_fn,
            miner_uids=miner_uids,
            miner_uids_fn=miner_uids_fn,
            log_path=log_path,
            blocks_per_round=blocks_per_round,
            books_per_miner=books_per_miner,
            get_block_fn=get_block_fn,
            validator_uid=validator_uid,
            tx_spool_path=tx_spool_path,
            tx_spool_max_bytes=tx_spool_max_bytes,
        )
        gtx_log.info(
            "service init: server=%s, poll=%ds, blocks_per_round=%d",
            server_url, interval, blocks_per_round,
        )
        if blocks_per_round > 0 and get_block_fn is not None:
            try:
                block = get_block_fn()
                current_round = block // blocks_per_round
                next_block = (current_round + 1) * blocks_per_round
                blocks_until = next_block - block
                mins = blocks_until * _BITTENSOR_BLOCK_TIME_S / 60
                gtx_log.info(
                    "[GTX] current block=%d (round=%d); next round at block ~%d (~%.0f min)",
                    block, current_round, next_block, mins,
                )
            except Exception:
                pass
        else:
            gtx_log.info(
                "[GTX] round scheduling: timer mode, first round in ~%.0fs",
                interval,
            )
        _log_runtime_versions()
        return service

    # ------------------------------------------------------------------
    # State packaging (called every tick)
    # ------------------------------------------------------------------

    def push_state(self, state: Any) -> None:
        """Extract a tick packet and hand it to the background TX worker.

        Reverted to the 0.4.6 pattern: extract_state + msgpack.packb run
        SYNCHRONOUSLY on the caller's thread (the validator's hot path).
        The intermediate _pack_worker background thread was introduced in
        0.5.x to move the CPU work off-path, but on mainnet it competed
        with reward_executor for the GIL every tick and became the
        dominant driver of reward-cycle backpressure (the "Waiting for
        rewarding to catch up" queue). Doing the pack inline pays a
        50-200ms tick-latency cost that stays local to the hot path while
        freeing the GIL for reward_executor between ticks.
        """
        if self._packager is None:
            return
        try:
            packet = self._packager.extract_state(state)
            ts = packet.get("ts") if isinstance(packet, dict) else None
            if ts is not None and ts > self._max_sim_ts_pushed:
                self._max_sim_ts_pushed = int(ts)
            data = msgpack.packb(packet, use_bin_type=True)
        except Exception as exc:
            gtx_log.warning("push_state failed: %s", exc)
            return
        self._enqueue_state(data)

    def _enqueue_state(self, data: bytes) -> None:
        if self._tx_spool is not None:
            try:
                evicted = self._tx_spool.append(data)
                if evicted:
                    gtx_log.warning(
                        "TX spool evicted %d oldest record(s); total=%d",
                        evicted, self._tx_spool.evictions_total,
                    )
            except Exception as exc:
                gtx_log.warning("TX spool append failed: %s", exc)
        try:
            self._tx_queue.put_nowait(data)
            return
        except queue.Full:
            pass
        # No-spool mode only: drop oldest to make room.
        try:
            self._tx_queue.get_nowait()
            self._tx_drops += 1
            if self._tx_drops == 1 or self._tx_drops % 32 == 0:
                gtx_log.warning(
                    "state TX queue full (size=%d); dropped %d packets total",
                    self.DEFAULT_TX_QUEUE_SIZE, self._tx_drops,
                )
        except queue.Empty:
            pass
        try:
            self._tx_queue.put_nowait(data)
        except queue.Full:
            pass

    def _tx_worker(self) -> None:
        last_summary = time.time()
        last_retries = 0
        last_sends = 0
        while not self._tx_stop.is_set():
            now = time.time()
            if now - last_summary >= self.DEFAULT_TX_SUMMARY_INTERVAL_S:
                spool_evict = self._tx_spool.evictions_total if self._tx_spool else 0
                gtx_log.info(
                    "TX summary: tx_q=%d tx_drops=%d spool_evict=%d "
                    "retries=+%d sends=+%d (last %.0fs)",
                    self._tx_queue.qsize(), self._tx_drops, spool_evict,
                    self._tx_retries_total - last_retries,
                    self._tx_sends_total - last_sends,
                    now - last_summary,
                )
                last_retries = self._tx_retries_total
                last_sends = self._tx_sends_total
                last_summary = now
            try:
                data = self._tx_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            sent = False
            try:
                sent = self._post_state_with_retry(data)
            finally:
                if sent and self._tx_spool is not None:
                    try:
                        self._tx_spool.ack_one()
                    except Exception as exc:
                        gtx_log.warning("TX spool ack failed: %s", exc)
                self._tx_queue.task_done()

    def _post_state_with_retry(self, data: bytes) -> bool:
        """Return True iff the server accepted (2xx-4xx); False on retry exhaustion."""
        for attempt in range(self.DEFAULT_TX_MAX_ATTEMPTS):
            if attempt > 0:
                self._tx_retries_total += 1
                time.sleep(self.DEFAULT_TX_BACKOFF_BASE_S * (2 ** (attempt - 1)))
            t_start = time.time()
            try:
                resp = self._http_post_sync("/state", data)
                t = time.time() - t_start
                status = resp.status_code
                if status < 400:
                    self._tx_sends_total += 1
                    if t > 0.2:
                        gtx_log.info("state POST slow t=%.2fs bytes=%d", t, len(data))
                    return True
                if status < 500:
                    gtx_log.warning(
                        "state POST got %d (terminal): check API key / URL", status,
                    )
                    return True
                gtx_log.debug(
                    "state POST attempt %d/%d got %d after %.2fs",
                    attempt + 1, self.DEFAULT_TX_MAX_ATTEMPTS, status, t,
                )
            except Exception as exc:
                gtx_log.debug(
                    "state POST attempt %d/%d failed after %.2fs: %s",
                    attempt + 1, self.DEFAULT_TX_MAX_ATTEMPTS,
                    time.time() - t_start, exc,
                )
        return False

    def _drain_on_exit(self) -> None:
        """Flush in-flight packets on clean shutdown. SIGKILL is not covered.

        Drain order is pack queue first (so any in-flight states still
        get packed and forwarded to the TX queue), then TX queue, then
        stop both worker threads.
        """
        if not self._tx_thread.is_alive():
            return
        deadline = time.time() + self.DEFAULT_TX_DRAIN_TIMEOUT_S
        # Let the TX worker finish anything push_state produced.
        while time.time() < deadline and not self._tx_queue.empty():
            time.sleep(0.1)
        self._tx_stop.set()
        self._tx_thread.join(timeout=2.0)
        if self._tx_spool is not None:
            self._tx_spool.close()

    def _handle_sigterm(self, signum, frame) -> None:
        gtx_log.info("SIGTERM received; draining TX worker")
        self._drain_on_exit()
        prev = getattr(self, "_prev_sigterm", None)
        if callable(prev):
            # Hand off to the previously-installed handler (e.g. the validator's
            # graceful cleanup + sys.exit); it owns process teardown from here.
            prev(signum, frame)
            return
        # No chainable predecessor (SIG_DFL/SIG_IGN/None) — fall back to the
        # default disposition: reset to default and re-raise to terminate.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    # ------------------------------------------------------------------
    # Round scheduling + assignment creation (validator-driven)
    # ------------------------------------------------------------------

    def _should_advance_round(self) -> int | None:
        """Check if a new round should start. Returns new round number or None.

        Block-synced mode: round = block // blocks_per_round.
        Timer mode: advance after poll_interval since last round push.
        """
        if self._blocks_per_round > 0 and self._get_block_fn is not None:
            try:
                block = self._get_block_fn()
                self._last_known_block = block
                block_round = block // self._blocks_per_round
                if block_round > self._current_round:
                    return block_round
            except Exception as exc:
                gtx_log.debug("block query failed: %s", exc)
            return None

        # Timer mode
        now = time.time()
        if now - self._last_round_push >= self._poll_interval:
            return self._current_round + 1
        return None

    async def _fetch_data_status(self) -> dict | None:
        """GET /gentrx/data-status — available data ranges per book."""
        t_start = time.time()
        try:
            resp = await self._http_get_async("/data-status")
            t = time.time() - t_start
            if t > 0.5:
                gtx_log.info("data-status fetch t=%.2fs", t)
            return resp.json()
        except Exception as exc:
            gtx_log.debug("data-status fetch failed after %.2fs: %s", time.time() - t_start, exc)
            return None

    def _create_assignments(self, data_status: dict, round_id: int | None = None) -> dict[int, dict]:
        """Create assignments for all miners by PAGE (IID, shared held-out window).

        Each miner gets a random IID sample of train books; for each book we
        pick one existing fixed-row page from the data-status registry —
        recency-biased with a stale-revisit tail. This cannot sample into an
        empty time gap (the failure of the old beta-window approach) and handles
        quiet books naturally. All miners share one [ts_start, ts_end] window
        (the union span of chosen pages) so held-out forward scoring uses the
        same baseline for everyone.
        """
        books = data_status.get("books", {})
        if not books:
            return {}

        model_version = data_status.get("version", 0)

        # Tolerant sort: numeric ids by value first, any non-numeric id last
        # (lexical). A bare int(b) here raised ValueError on a malformed book
        # key and, since the caller swallows exceptions, silently skipped the
        # round every tick the bad key persisted (GenTRX training would stall
        # indefinitely with only a warning).
        def _book_sort_key(b):
            s = str(b)
            return (0, int(s)) if s.lstrip("-").isdigit() else (1, s)

        all_book_ids = sorted(books.keys(), key=_book_sort_key)

        # Bound the stale-revisit bookkeeping: drop entries for pages no longer
        # present in the data-status registry. Without this, _page_last_round
        # grows without limit as sim time advances and new parquets appear.
        live_pages = set()
        for _bid, _meta in books.items():
            for _item in _meta.get("parquets", []):
                _fname = _item[0] if isinstance(_item, (list, tuple)) else _item
                live_pages.add(f"{_bid}/{_fname}")
        for _stale in [k for k in self._page_last_round if k not in live_pages]:
            self._page_last_round.pop(_stale, None)

        # Seed every per-round RNG (val split, per-miner book sample, page pick)
        # and the "round" label on the round being ASSIGNED, not the last
        # successfully-pushed round. self._current_round only advances after a
        # push; in block mode new_round can jump by >1 (missed blocks), so keying
        # seeds on _current_round would make two validators with different push
        # histories derive a different train/val split for the same round_id.
        rid = round_id if round_id is not None else self._current_round

        # Val split rotates per round, disjoint from that round's train (every book
        # trains over time, none permanently excluded). Pushed to the server so it
        # holds out exactly what training excluded.
        n_val = max(1, int(len(all_book_ids) * self._val_fraction))
        val_rng = random.Random(
            hashlib.sha256(f"{rid}:val".encode()).hexdigest()
        )
        self._val_books = set(
            val_rng.sample(all_book_ids, min(n_val, len(all_book_ids)))
        )

        train_books = [b for b in all_book_ids if b not in self._val_books]
        if not train_books:
            # Every book landed in the val split (<=1 book, or val_fraction too
            # high). Falling back to training on val books would let miners train
            # AND be scored on the same book — the exact contamination the
            # held-out design removes. Skip the round instead.
            gtx_log.info(
                "no train books after val split (books=%d, val=%d) — skipping round",
                len(all_book_ids),
                len(self._val_books),
            )
            return {}

        miner_uids = self._miner_uids
        if not miner_uids:
            return {}

        def _norm_pages(book_id):
            """[(fname, f_start, f_end), ...] for a book, legacy strings tolerated."""
            out = []
            for item in books.get(book_id, {}).get("parquets", []):
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    out.append((item[0], item[1], item[2]))
                else:
                    out.append((item, 0, 0))
            return out

        # Resolve data bucket credentials from gradient server's validator store
        # (embedded in the assignment so miners don't need pre-configuration)
        import os
        data_endpoint = os.environ.get("GENTRX_VALIDATOR_S3_ENDPOINT_URL", "")
        data_bucket = os.environ.get("GENTRX_VALIDATOR_S3_BUCKET", "")
        data_access = os.environ.get("GENTRX_VALIDATOR_S3_READ_ACCESS_KEY", "")
        data_secret = os.environ.get("GENTRX_VALIDATOR_S3_READ_SECRET_KEY", "")

        # IID (flavor B): each miner gets a random overlapping SAMPLE of train
        # books, so gradients estimate the same objective; fairness comes from the
        # shared held-out window below. _iid_shared_books toggles flavor A.
        per_miner: dict[int, tuple[list, list, list]] = {}
        spans_all: list[tuple[int, int]] = []
        k = min(self._books_per_miner, len(train_books))
        for miner_uid in miner_uids:
            book_seed = (
                f"{rid}:books"
                if self._iid_shared_books
                else f"{rid}:{miner_uid}:books"
            )
            book_rng = random.Random(hashlib.sha256(book_seed.encode()).hexdigest())
            assigned_books = book_rng.sample(train_books, k)

            page_rng = random.Random(
                hashlib.sha256(f"{rid}:{miner_uid}:page".encode()).hexdigest()
            )
            data_keys, spans = [], []
            for book_id in assigned_books:
                pages = _norm_pages(book_id)
                if not pages:
                    continue
                if page_rng.random() < self._stale_revisit_frac:
                    # Least-recently-assigned page; never-assigned (-1) win.
                    idx = min(
                        range(len(pages)),
                        key=lambda i: self._page_last_round.get(
                            f"{book_id}/{pages[i][0]}", -1
                        ),
                    )
                else:
                    lo = max(0, len(pages) - self._recency_window)
                    idx = page_rng.randrange(lo, len(pages))
                fname, f_start, f_end = pages[idx]
                data_keys.append(f"data/{self._validator_uid}/{book_id}/intervals/{fname}")
                spans.append((f_start, f_end))
                self._page_last_round[f"{book_id}/{fname}"] = rid

            if not data_keys:
                continue
            per_miner[miner_uid] = (assigned_books, data_keys, spans)
            spans_all.extend(spans)

        if not per_miner:
            return {}

        # One shared [start, end] for all miners this round: forward-only held
        # scoring after a shared `end` gives every miner the same baseline (single
        # pass), and `end` = max trained page end so it never overlaps training.
        shared_start = min(s for s, _ in spans_all)
        shared_end = max(e for _, e in spans_all)

        assignments = {}
        for miner_uid, (assigned_books, data_keys, _spans) in per_miner.items():
            assignments[miner_uid] = {
                "round": rid,
                "model_version": model_version,
                "books": assigned_books,
                "ts_start": shared_start,
                "ts_end": shared_end,
                "data": data_keys,
                "data_source": "s3",
                # The shard OUR aggregator reads, straight from its /data-status. A miner that adopts this
                # files its gradient where the round will look for it, instead of guessing from a per-tick
                # mechanism flag that oscillates when one axon serves both mechanisms.
                "bucket_prefix": data_status.get("bucket_prefix", ""),
                "data_endpoint": data_endpoint,
                "data_bucket": data_bucket,
                "data_access_key": data_access,
                "data_secret_key": data_secret,
            }

        return assignments

    async def _push_round(self, round_id: int, assignments: dict[int, dict]) -> bool:
        """POST /gentrx/round — push assignment plan to gradient server."""
        t_start = time.time()
        try:
            import json as _json
            body: dict = {
                "round": round_id,
                "assignments": {str(uid): a for uid, a in assignments.items()},
            }
            if self._val_books is not None:
                # The held-out split for this round: the server scores against
                # exactly the books training excluded (no independent re-derive).
                body["val_books"] = sorted(self._val_books)
            if self._last_known_block is not None:
                body["block"] = self._last_known_block
            payload = _json.dumps(body).encode()
            await self._http_post_async(
                "/round", payload, content_type="application/json"
            )
            t = time.time() - t_start
            self._last_round_push = time.time()
            gtx_log.info(
                "push_round round=%d n_assignments=%d bytes=%d t=%.2fs",
                round_id, len(assignments), len(payload), t,
            )
            return True
        except Exception as exc:
            gtx_log.warning("round push failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Main loop entry point (called every tick)
    # ------------------------------------------------------------------

    async def poll_and_deliver(self) -> None:
        """Drive round scheduling, deliver assignments, poll scores.

        Called every tick by handle_state. Rate-limited internally — most
        calls are no-ops.
        """
        if self._deliver_fn is None or not self._miner_uids:
            return

        # Warmup gate: before the sim has produced a full training window,
        # there's no data for assignments anyway. Skip the /data-status +
        # /round HTTP chatter until max_sim_ts crosses window_ns.
        if self._max_sim_ts_pushed < self._window_ns:
            now = time.time()
            if now - self._last_warmup_log >= 300.0:
                self._last_warmup_log = now
                pct = 100.0 * self._max_sim_ts_pushed / self._window_ns if self._window_ns else 0
                ns_remaining = self._window_ns - self._max_sim_ts_pushed
                mins_remaining = ns_remaining / 1e9 / 60
                gtx_log.info(
                    "[GTX] warmup: sim window %.1f%% full (%.0f min of sim time remaining "
                    "before first round can be assigned)",
                    pct, mins_remaining,
                )
            return

        # Check if a new round should start
        new_round = self._should_advance_round()
        if new_round is not None:
            data_status = await self._fetch_data_status()
            if data_status is not None:
                assignments = self._create_assignments(data_status, round_id=new_round)
                if assignments:
                    if await self._push_round(new_round, assignments):
                        # Advance only after a confirmed push so a simultaneous
                        # restart (gradient server not yet up) retries next tick
                        # rather than silently dropping the round.
                        self._current_round = new_round
                        gtx_log.info(
                            "round=%d: created %d assignments, pushing to miners",
                            new_round, len(assignments),
                        )
                        if self._blocks_per_round > 0:
                            next_block = (new_round + 1) * self._blocks_per_round
                            mins = self._blocks_per_round * _BITTENSOR_BLOCK_TIME_S / 60
                            gtx_log.info(
                                "[GTX] next assignment round at block ~%d (~%.0f min)",
                                next_block, mins,
                            )
                        else:
                            gtx_log.info(
                                "[GTX] next assignment round in ~%.0fs (timer mode)",
                                self._poll_interval,
                            )
                        try:
                            await self._deliver_fn(assignments)
                        except Exception as exc:
                            gtx_log.warning("delivery failed: %s", exc)
                            import traceback
                            gtx_log.debug(traceback.format_exc())
                    # if push failed (server down), don't advance — retry next tick
                else:
                    # No data to assign yet — advance to avoid thrashing data-status
                    self._current_round = new_round
                    gtx_log.debug("round=%d: no assignments (insufficient data)", new_round)

        # Score polling — rate-limited independently
        now = time.time()
        if now - self._last_score_poll >= self._poll_interval:
            self._last_score_poll = now
            await self.poll_scores()

        await self._check_health()

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    async def poll_scores(self) -> None:
        """Poll the gradient server for new scores."""
        t_start = time.time()
        try:
            resp = await self._http_get_async(
                f"/scores?since_round={self._last_score_round_seen}"
            )
            if resp.status_code == 204:
                return
            payload = resp.json()
            t = time.time() - t_start
            if t > 0.5:
                gtx_log.info("poll_scores t=%.2fs", t)
            round_id = payload.get("round", -1)
            if round_id > self._last_score_round_seen:
                self._last_score_round_seen = round_id
                self.receive_scores(payload)
        except Exception as exc:
            gtx_log.debug("poll_scores failed: %s", exc)

    def receive_scores(self, payload: dict) -> None:
        """Update local score store from a gradient-server response.

        The payload now carries an `aggregation` block + counters (see
        gradient_server._deliver_scores) so downstream consumers — Prometheus
        via ReportingService IPC, dashboards, etc. — have everything they
        need without a second HTTP call.
        """
        self._scores = {
            int(uid_str): score_data
            for uid_str, score_data in payload.get("scores", {}).items()
        }
        agg = payload.get("aggregation")
        if agg:
            self._last_aggregation_stats = dict(agg)
        counters = payload.get("counters")
        if counters:
            self._last_aggregation_stats.update(counters)
        cfg = payload.get("config")
        if cfg:
            self._last_config = dict(cfg)
        scores_compact = {
            uid: f"{s['score']:.3f}{'✓' if s.get('accepted') else '✗'}"
            for uid, s in self._scores.items()
        }
        gtx_log.info(
            "scores: round=%s accepted=%s/%s scores=%s",
            payload.get("round"),
            payload.get("n_accepted"),
            payload.get("n_scored"),
            scores_compact,
        )

    def get_scores(self) -> dict[int, dict]:
        """Return current per-miner scores."""
        return dict(self._scores)

    def get_training_stats(self) -> dict:
        """Return last aggregation stats (loss, acceptance rate, version, timing)."""
        return dict(self._last_aggregation_stats)

    def get_config(self) -> dict:
        """Return the gradient server's effective hyperparameter config."""
        return dict(self._last_config)

    def register_benchmark_bucket(self, uid: int, bucket: dict) -> bool:
        """POST /gentrx/register_bucket/{uid} — inject static bucket for a benchmark miner.

        bucket keys: endpoint_url, bucket_name, access_key_id, secret_access_key
        """
        try:
            import httpx
            r = httpx.post(
                f"{self._server_url}/register_bucket/{uid}",
                json=bucket,
                headers=self._headers(),
                timeout=5.0,
            )
            return r.status_code == 200
        except Exception as exc:
            gtx_log.warning("register_benchmark_bucket uid=%d failed: %s", uid, exc)
            return False

    def is_healthy(self, timeout: float = 3.0) -> bool:
        """GET /health on the gradient server. Returns True if reachable and status=ok."""
        try:
            import httpx
            r = httpx.get(f"{self._server_url}/health", headers=self._headers(), timeout=timeout)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    async def _check_health(self) -> None:
        """Periodic health check — logs a warning on failure, info on recovery."""
        now = time.time()
        if now - self._last_health_check < self._health_check_interval:
            return
        self._last_health_check = now
        try:
            resp = await self._http_get_async("/health", timeout=3.0)
            ok = resp.status_code == 200 and resp.json().get("status") == "ok"
        except Exception as exc:
            ok = False
            gtx_log.warning("health check failed: %s", exc)
        if ok and self._last_health_ok is not True:
            gtx_log.info("gradient server healthy at %s", self._server_url)
        elif not ok and self._last_health_ok is not False:
            gtx_log.warning("gradient server unreachable at %s", self._server_url)
        self._last_health_ok = ok

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    # Async variants (httpx) for the tick path so the validator's event loop
    # never blocks on HTTP. A tiny sync variant remains for push_state, which
    # is called from the state-packager thread, not the event loop.

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key} if self._api_key else {}

    async def _http_get_async(self, path: str, timeout: float = 5.0):
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.get(
                f"{self._server_url}{path}", headers=self._headers()
            )

    async def _http_post_async(
        self, path: str, data: bytes,
        content_type: str = "application/octet-stream", timeout: float = 5.0,
    ):
        import httpx
        headers = {"Content-Type": content_type, **self._headers()}
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(
                f"{self._server_url}{path}", content=data, headers=headers
            )

    def _http_post_sync(
        self, path: str, data: bytes,
        content_type: str = "application/octet-stream", timeout: float = 5.0,
    ):
        import httpx
        headers = {"Content-Type": content_type, **self._headers()}
        return httpx.post(
            f"{self._server_url}{path}",
            content=data, headers=headers, timeout=timeout,
        )
