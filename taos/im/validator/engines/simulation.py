"""
validator/engines/simulation.py

SimulationEngine — owns every simulation-specific operation.

This class is populated by moving code that currently lives directly in
validator.py.  The validator is left with only mode-agnostic logic.

Methods and their source locations in the original validator.py:

    start()              ← __init__: load_simulation_config() call +
                           socket-open calls + _start_seed_service() call
    stop()               ← cleanup_ipc(): close _sim_req / _sim_res sockets
    receive()            ← _listen() / orderbook(): socket recv + msgpack
                           unpack + parse_dict + system notice dispatch
    _normalize()         ← handle_state(): field unpack from raw state object
    execute()            ← (no-op; simulator handles instructions internally)
    respond()            ← _listen() finally block: msgpack.packb + socket send
    on_start()           ← onStart(): simulation_id update + timestamp shift +
                           initial_balance reset + open_positions clear + save
    on_end()             ← onEnd(): clear simulation_id + save + update_repo
    collect_resets()     ← process_resets(): scan notices for RDRA/ERDRA

Nothing else moves.  The validator's IPC sockets for /taosim-req and
/taosim-res transfer to this class and are opened in start().
The validator's cleanup_ipc() calls self.engine.stop() to close them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import time
import traceback
import zipfile
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

import msgpack
import psutil
import torch

from taos import __spec_version__
from taos.im.validator.engines import (
    MarketEngine,
    NormalizedState,
    NormalizedTradeEvent,
)

from taos.im.protocol import MarketSimulationStateUpdate

if TYPE_CHECKING:
    from taos.im.neurons.validator import Validator

logger = logging.getLogger(__name__)

# Fresh-episode clock floor (ns). A SIMULATION_START whose clock is at/below
# max(grace_period, this) is a genuine new episode → re-derive simulation_id
# from the new logDir. Above it, the start is a mid-run resume/reconnect and the
# locked simulation_id is preserved, so the data service does not treat the
# still-running run as new and prune its history. Floored well above the
# per-run start (clock ~0) yet far below any multi-hour run.
_FRESH_EPISODE_FLOOR_NS = 30 * 60 * 1_000_000_000  # 30 sim-minutes


def _network_label(network: str) -> str:
    """Map subtensor network name to a filesystem-safe label."""
    _map = {'local': 'localnet', 'test': 'testnet', 'finney': 'mainnet', 'mainnet': 'mainnet'}
    return _map.get(str(network).lower(), str(network).lower().replace('/', '_'))


def _read_first_and_last_nonempty(path: str, tail_chunk: int = 131072) -> tuple[str, Optional[str]]:
    """Return (first line incl. newline, last non-empty line) of a file, reading
    only the head plus a bounded tail.

    Replaces full readlines() scans that only need these two lines: the
    fundamental CSVs grow unboundedly with sim runtime, and reading them whole
    made load_fundamental O(file) — ~9s/scoring round after 20h of sim, under
    both the GIL and _reward_lock. This is O(1) in file size. Reads backwards in
    tail_chunk steps so a line truncated at a chunk boundary is completed by the
    next step; decodes strict utf-8 so corrupt bytes raise (caller catches and
    skips the block, same as the old open('r') path).
    """
    with open(path, 'rb') as f:
        first = f.readline().decode('utf-8')
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        data = b''
        while pos > 0:
            step = min(pos, tail_chunk)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
            lines = data.split(b'\n')
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip():
                    # Complete line only if its start is within the data read so
                    # far (i > 0 ⇒ a newline precedes it; pos == 0 ⇒ file start).
                    if i > 0 or pos == 0:
                        return first, lines[i].decode('utf-8')
                    break
    return first, None


class SimulationEngine(MarketEngine):

    """The simulation-mechanism engine: drives the C++ simulator and consumes its events.

    Counterpart of the exchange engine, behind the same MarketEngine interface.
    """
    def __init__(self, config: Any, validator: "Validator") -> None:
        self.config    = config
        self.validator = validator

        self._sim_config  = None   # MarketSimulationConfig, populated in start()
        self._book_ids: list[int] = []

        # IPC sockets — opened in start(), closed in stop()
        # These replace self._sim_req / self._sim_res (or however named)
        # on the validator.  Remove those attributes from validator.__init__.
        self._req_socket = None
        self._res_socket = None

        # UI-submitted orders: fetched on a background poll task into
        # _external_buffer, drained into _external_instructions each tick (see
        # MarketEngine._external_orders_loop / _drain_external_orders).
        self._external_instructions: list[dict] = []
        self._external_buffer: list[dict] = []
        self._external_poll_task = None

        # Per-book trigger prices snapshot; populated by the optional SL/TP
        # extension when installed, and left empty otherwise.
        self._live_triggers: dict = {}
        self._sltp_changed: bool = False
        self._sltp_ext = None
        # Optional component; not part of this tree. Import is guarded.
        try:
            from taos.im.validator.engines.exetrx import SLTPInferenceExtension
            self._sltp_ext = SLTPInferenceExtension(self)
        except ImportError:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the engine's consumers and connect to the simulator."""
        v = self.validator

        # ── 1. Load simulation XML config ─────────────────────────────────────
        self._load_config()

        # ── 2. Benchmark agents + effective UID count ─────────────────────────
        v.benchmark_agents = []
        v.benchmark_start_uid = v.subnet_info.max_uids
        if v.config.benchmark.agents:
            self._load_benchmark_agents()
        v.effective_max_uids = v.subnet_info.max_uids + len(v.benchmark_agents)
        v.scores = torch.zeros(
            v.effective_max_uids, dtype=torch.float32, device=v.device
        )

        # ── 3. Initialize all state structures on the validator ────────────────
        self.initialize_structures()

        # ── 4. Open IPC request socket ─────────────────────────────────────────
        import posix_ipc as _posix_ipc
        self._req_socket = _posix_ipc.MessageQueue(
            "/taosim-req",
            flags=_posix_ipc.O_CREAT,
            max_messages=1,
            max_message_size=8,
        )
        self._res_socket = None

        # ── 5. Load persisted state ────────────────────────────────────────────
        v.load_state()

        # ── 6. Start seed service ──────────────────────────────────────────────
        v.seed_process = None
        self._start_seed_service()

        logger.info(
            "SimulationEngine started: book_count=%d sim_id=%s",
            self._sim_config.book_count,
            getattr(self._sim_config, 'simulation_id', 'unknown'),
        )

    def stop(self) -> None:
        """
        Called from validator.cleanup_ipc().

        Move the socket-close lines from cleanup_ipc() here verbatim, e.g.:
            self._sim_req.close()
            self._sim_res.close()
            posix_ipc.unlink_message_queue(path)  ← if applicable

        Replace with:
            self.engine.stop()
        in cleanup_ipc().
        """
        self._stop_external_poller()
        for sock in (self._req_socket, self._res_socket):
            try:
                if sock is not None:
                    sock.close()
            except Exception as exc:
                logger.debug(f"SimulationEngine.stop socket close: {exc}")
        logger.info("SimulationEngine stopped")

    # ─────────────────────────────────────────────────────────────────────────
    # receive()
    #
    # Source: validator._listen() inner loop, from socket.recv() down to
    # (but not including) the call to handle_state().
    # The finally block moves to respond() below.
    # ─────────────────────────────────────────────────────────────────────────

    async def receive(self) -> tuple[Any, Optional[NormalizedState], float]:
        """
        Read one message from /taosim-req, deserialise, and return a NormalizedState.
        """
        receive_start = time.time()
        raw_bytes = await self._recv_bytes()
        _q4_t_recv = time.time()
        # Retry unpack: SHM data may be partially written even after the size
        # poll passes (simulator signals MQ before completing the write), or the
        # buffer may be misframed (stale-MQ / sim-restart size mismatch). A
        # misframe makes msgpack decode a dict/list where a map key is expected
        # and raise TypeError ("unhashable type"), which is NOT a ValueError — so
        # it must be caught here too, otherwise it escapes to _listen, which
        # returns an EMPTY response and silently drops that step's miner orders.
        for _attempt in range(8):
            try:
                raw_dict = msgpack.unpackb(raw_bytes, raw=False, use_list=True, strict_map_key=False)
                break
            except (ValueError, TypeError, msgpack.UnpackValueError):
                if _attempt == 7:
                    raise
                await asyncio.sleep(0.005 * (2 ** _attempt))
                raw_bytes = await self._recv_bytes_shm_only(raw_bytes)
        _q4_t_unpack = time.time()
        state = self._parse_state(raw_dict)
        _q4_t_parse = time.time()
        normalized = self._normalize(state)
        _q4_t_norm = time.time()
        normalized.logDir = raw_dict.get('logDir')
        # External orders are polled on a background task; drain the buffer with
        # no await so the data-service fetch never sits on the round path.
        self._ensure_external_poller()
        self._external_instructions = self._drain_external_orders()
        # [Q4-PROFILE] splits recv_norm into recv (shm read), unpack
        # (msgpack.unpackb), parse (_parse_state; pydantic construction, the
        # suspected model_construct hotspot), norm (_normalize) and ext (fetch
        # external orders). Tells us whether parse_dict is fat before optimizing.
        # Emit via bt.logging (loguru): this module's stdlib `logger` is not
        # routed to the pm2 sink, so logger.info() here was silently dropped.
        import bittensor as bt
        bt.logging.info(
            f"[Q4-PROFILE] recv={_q4_t_recv - receive_start:.3f}s "
            f"unpack={_q4_t_unpack - _q4_t_recv:.3f}s "
            f"parse={_q4_t_parse - _q4_t_unpack:.3f}s "
            f"norm={_q4_t_norm - _q4_t_parse:.3f}s "
            f"ext={time.time() - _q4_t_norm:.3f}s "
            f"bytes={len(raw_bytes) / 1048576:.1f}MB"
        )
        if self._external_instructions:
            logger.info(
                "SimulationEngine.receive: %d external order(s) fetched",
                len(self._external_instructions),
            )
        return raw_bytes, normalized, receive_start

    def _normalize(self, state: Any) -> NormalizedState:
        """
        Wrap a MarketSimulationStateUpdate in NormalizedState.

        The simulation state fields (books, accounts, notices) are already in
        exactly the shape _update_trade_volumes() and friends expect — this is
        just a rename/envelope so the validator's handle_state() receives a
        type-stable object regardless of mode.

        Moved from the top of handle_state() where these lines appeared:
            self.simulation_timestamp = state.timestamp
            books    = state.books
            accounts = state.accounts
            notices  = state.notices
            ...
        """
        block_time_ns = getattr(self._sim_config, 'block_time_ns', 12_000_000_000)
        config = self._sim_config.model_copy()
        config.logDir = None  # don't expose log dir to miners

        # Build minimal pools dict from book prices for UI display
        pools: dict = {}
        for netuid_key, book in (state.books or {}).items():
            try:
                nid   = int(netuid_key)
                price = book.get('last_price') or book.get('price', 0)
                if price and price > 0:
                    pools[nid] = {'price': price}
            except Exception:
                pass

        # Cache active book IDs for external order validation (prevents C++ crash on invalid bookId)
        self._active_book_ids: set = {int(k) for k in (state.books or {}).keys()}

        return NormalizedState(
            timestamp = state.timestamp,
            block     = state.timestamp // block_time_ns,
            books     = state.books,
            accounts  = state.accounts,
            notices   = state.notices,
            pools     = pools or None,
            config    = config,
            version   = __spec_version__,
        )

    async def _fetch_external_orders(self) -> list[dict]:
        """Fetch UI-submitted pending orders from the data service and verify them.

        Requires the exchange engine for sr25519 verification and UI→LOB translation.
        The exchange engine is an optional component that is not part of this tree; when it is
        unavailable this becomes a no-op and sim runs continue without UI injection.
        """
        try:
            from taos.im.validator.engines.exchange import ExchangeEngine
        except ImportError:
            return []

        # Prefer simulation.data_service_url; fall back to exchange.data_service_url
        url = (getattr(getattr(self.config, 'simulation', None), 'data_service_url', '')
               or getattr(getattr(self.config, 'exchange', None), 'data_service_url', ''))
        if not url:
            return []
        fetch_url = f"{url.rstrip('/')}/api/v1/orders/pending"

        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=3.0)
            _headers = {}
            _secret = os.environ.get("INGEST_SECRET", "")
            if _secret:
                _headers["X-Ingest-Secret"] = _secret
            async with aiohttp.ClientSession() as session:
                async with session.get(fetch_url, headers=_headers, timeout=timeout) as resp:
                    if resp.status != 200:
                        logger.debug(
                            "_fetch_external_orders: HTTP %d from %s", resp.status, fetch_url
                        )
                        return []
                    orders = await resp.json()
        except Exception as exc:
            logger.debug(f"_fetch_external_orders: {exc}")
            return []

        if not orders:
            return []

        valid: list[dict] = []
        v = self.validator

        # Build uid→benchmark-agent map (simulation has no live metagraph)
        bm_by_uid: dict[int, dict] = {
            bm['uid']: bm for bm in getattr(v, 'benchmark_agents', [])
        }

        for order in orders:
            meta = order.pop('_meta', {})
            if not meta:
                logger.warning("_fetch_external_orders: order missing _meta, dropping")
                continue

            coldkey    = meta.get('coldkey', '')
            hotkey     = meta.get('hotkey', '')
            nonce      = meta.get('nonce', 0)
            signature  = meta.get('signature', '')
            order_hash = meta.get('order_hash', '')
            uid        = order.get('agentId')

            if uid is None or not coldkey or not hotkey or not signature:
                logger.warning("_fetch_external_orders: incomplete metadata, dropping")
                continue

            # Ownership check for benchmark agents (if known)
            bm = bm_by_uid.get(uid)
            if bm is not None:
                if bm.get('hotkey', '') != hotkey:
                    logger.warning(f"_fetch_external_orders: hotkey mismatch uid={uid}, dropping")
                    continue
                if bm.get('coldkey', '') != coldkey:
                    logger.warning(f"_fetch_external_orders: coldkey mismatch uid={uid}, dropping")
                    continue

            # Re-verify signature: hotkey signs "{nonce}.{hotkey}.{uid}.{order_hash}"
            message = f"{nonce}.{hotkey}.{uid}.{order_hash}"
            if not ExchangeEngine._verify_sr25519(message, signature, hotkey):
                logger.warning(
                    "_fetch_external_orders: invalid signature uid=%d, dropping", uid
                )
                continue

            # `self=None` is deliberate and must stay positional: the translator is an instance method
            # whose only use of `self` is a step simulation has no state for. Passing None skips that
            # step explicitly; omitting the argument would bind `raw` to `self` instead.
            translated = ExchangeEngine._translate_ui_order(None, order)
            if translated is None:
                logger.warning(
                    "_fetch_external_orders: unrecognised instruction type uid=%d type=%s, dropping",
                    uid, order.get('type'),
                )
                continue

            # Validate bookId against the simulator's active books to prevent C++ vector out-of-range crash
            book_id = translated.get('payload', {}).get('bookId')
            active_books = getattr(self, '_active_book_ids', set())
            if book_id is not None and active_books and int(book_id) not in active_books:
                logger.warning(
                    "_fetch_external_orders: bookId=%d not in active books %s, dropping",
                    int(book_id), sorted(active_books),
                )
                continue

            # Thread the exchange-API order UUID through the C++ round trip via
            # clientOrderId (UInt32, echoed back in RDPOL/RDPOM and ET events).
            # The data service uses the UUID ('xo' on notices/fills) to drive its
            # durable order-status store. Counter starts high so it can never
            # collide with miner-assigned clientOrderIds.
            ext_oid = order.get('_order_id')
            if ext_oid and 'payload' in translated and translated.get('type') != 'CANCEL_ORDERS':
                cloid = self._next_ext_cloid()
                translated['payload']['clientOrderId'] = cloid
                self._ext_cloid_to_uuid[cloid] = str(ext_oid)

            valid.append(translated)

        return valid

    def _next_ext_cloid(self) -> int:
        """Monotonic UInt32 clientOrderId for external (exchange-API) orders."""
        if not hasattr(self, '_ext_cloid_to_uuid'):
            self._ext_cloid_to_uuid: dict[int, str] = {}
            self._ext_engineoid_to_uuid: dict[int, str] = {}
        counter = getattr(self, '_ext_cloid_counter', 0) + 1
        self._ext_cloid_counter = counter
        return 3_000_000_000 + (counter % 1_000_000_000)

    # ─────────────────────────────────────────────────────────────────────────
    # execute() — no-op
    # ─────────────────────────────────────────────────────────────────────────

    async def execute(
        self,
        state: NormalizedState,
        miner_responses: list,
    ) -> list[NormalizedTradeEvent]:
        """
        No-op. The C++ simulator processes miner instructions itself and
        injects trade events into the next tick's state.notices.
        _update_trade_volumes() therefore gets pre-populated notices with no
        adapter needed.

        Args:
            state: The block's state update.
            miner_responses: The miners' responses (unused here).

        Returns:
            list: Always empty in simulation mode.
        """
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # respond()
    #
    # Source: validator._listen() finally block
    # ─────────────────────────────────────────────────────────────────────────

    def respond(self, raw_message: Any, response: dict) -> None:
        """
        Serialise and write the validator response to /taosim-res via SHM.

        Args:
            raw_message: The simulator request being answered.
            response: The validator response to serialize.
        """
        try:
            self.validator.last_response = response
            packed = msgpack.packb(response, use_bin_type=True)
            self._send_bytes(packed)
        except Exception as exc:
            logger.error(f"SimulationEngine.respond error: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # on_start()
    #
    # Source: validator.onStart()  — move the entire body here.
    # Delete validator.onStart() or leave it as a one-line delegation shim
    # if other code calls it directly (then remove the shim later).
    # ─────────────────────────────────────────────────────────────────────────

    def on_start(self, timestamp, event) -> None:
        """Handle the simulator's start event: reset per-run state.

        Args:
            timestamp: The simulator's start timestamp.
            event: The start event payload.
        """
        v = self.validator
        # Capture the locked simulation_id BEFORE _load_config() rebuilds
        # v.simulation from XML (which resets simulation_id + logDir to None).
        # Needed to preserve it across a mid-run resume (see the reset logic below).
        _prev_sim_id = (
            getattr(v.simulation, "simulation_id", None) if getattr(v, "simulation", None) else None
        )
        self._load_config()
        volume_decimals = v.simulation.volumeDecimals

        logger.info("Shifting timestamps for simulation restart...")

        old_simulation_timestamp = v.simulation_timestamp  # End time of old simulation
        new_simulation_timestamp = timestamp  # Start time of new simulation (0)

        # Tell the shadow scoring service to run the SAME shift on its own
        # structures. FIFO with the state tee: all old-sim frames precede this,
        # all new-sim frames follow it.
        _shadow = getattr(v, '_scoring_shadow', None)
        if _shadow is not None:
            _shadow.emit_sim_start(old_simulation_timestamp, new_simulation_timestamp)

        # The full history shift lives in trade.shift_simulation_histories —
        # shared verbatim with the shadow scoring service so both sides make the
        # identical transition.
        from taos.im.validator.trade import shift_simulation_histories
        shift_simulation_histories(
            v, old_simulation_timestamp, new_simulation_timestamp,
            book_count=v.simulation.book_count,
            volume_decimals=volume_decimals,
            lookback=v.config.scoring.kappa.lookback,
            volume_assessment_period=v.config.scoring.activity.trade_volume_assessment_period,
            miner_wealth=v.simulation.miner_wealth,
            effective_max_uids=v.effective_max_uids,
            log=logger.info,
        )

        v.start_time = time.time()
        v.simulation_timestamp = timestamp
        v.start_timestamp = v.simulation_timestamp
        v.last_state_time = None
        v.step_rates = []
        if event.logDir != v.simulation.logDir:
            logger.info(f"Simulation log directory changed: {v.simulation.logDir} -> {event.logDir}")
            self._notify_seed_log_dir_change(event.logDir)
            v.simulation.logDir = event.logDir
        # simulation_id lifecycle. A genuine new episode resets the clock to ~0
        # (start="0"), so on_tick should re-derive the id from the new logDir.
        # But a restart/reconnect that resumes an in-progress run fires this same
        # event with the clock already past the grace period AND (because the
        # simulator writes a wall-clock-named output dir) a rotated logDir —
        # re-deriving there mints a NEW id for the same run, which makes the data
        # service classify it as a new simulation and prune the entire in-progress
        # run's history. So: fresh episode (or no prior id) → reset + re-derive;
        # mid-run resume → preserve the locked id, decoupling it from the logDir.
        _grace = int(getattr(v.simulation, "grace_period", 0) or 0)
        _fresh_threshold = max(_grace, _FRESH_EPISODE_FLOOR_NS)
        if not _prev_sim_id or new_simulation_timestamp <= _fresh_threshold:
            v.simulation.simulation_id = None
            logger.info(
                f"Fresh episode (t={new_simulation_timestamp} <= {_fresh_threshold}ns) — "
                "simulation_id will re-derive from logDir"
            )
        else:
            v.simulation.simulation_id = _prev_sim_id
            logger.warning(
                f"Preserving simulation_id {_prev_sim_id!r} across restart "
                f"(t={new_simulation_timestamp} > grace {_fresh_threshold}ns) — mid-run resume, "
                "not re-deriving from logDir (avoids data-service prune of the in-progress run)"
            )
        # open_positions / initial_balances / recent_trades / recent_miner_trades
        # resets are handled inside shift_simulation_histories (shared with the
        # shadow scoring service).

        self.compress_outputs(start=True)

        logger.info("-"*40)
        logger.info("SIMULATION STARTED")
        logger.info("-"*40)
        logger.info(f"START TIME: {v.start_time}")
        logger.info(f"TIMESTAMP : {v.start_timestamp}")
        logger.info(f"OUT DIR   : {v.simulation.logDir}")
        logger.info("-"*40)

        self.load_fundamental()

        asyncio.run_coroutine_threadsafe(v._save_state_sync(), v.main_loop).result()
        logger.info("Simulation restart complete")

    # ─────────────────────────────────────────────────────────────────────────
    # on_end()
    #
    # Source: validator.onEnd() — move the entire body here.
    # ─────────────────────────────────────────────────────────────────────────

    def on_end(self) -> None:
        """Handle the simulator's end event: flush and close per-run state."""
        v = self.validator
        logger.info("SIMULATION ENDED")
        v.simulation.logDir = None
        # Clear the locked id on a clean end so a restart afterwards re-derives a
        # fresh id for the NEXT episode instead of restoring this (ended) run's id.
        v.simulation.simulation_id = None
        self._notify_seed_log_dir_change(None)
        v.fundamental_price = {bookId: None for bookId in range(v.simulation.book_count)}
        v.pending_notices = {uid: [] for uid in range(v.effective_max_uids)}
        asyncio.run_coroutine_threadsafe(
            v._save_state_sync(), v.main_loop
        ).result(timeout=30)
        v.update_repo(end=True)

    def on_tick(self, state: NormalizedState) -> None:
        """
        Per-tick simulation state updates — mirrors the block that lived in the
        original handle_state() between version-stamping and volume injection.

        Ordering preserved from original:
          1. Hourly update_repo check against OLD simulation_timestamp
          2. logDir change detection
          3. simulation_timestamp ← state.timestamp
          4. simulation_id ← basename(logDir)[:13]
          5. Periodic log compression
        """
        v = self.validator
        # 1. Periodic validator-update check (uses OLD simulation_timestamp)
        if v.simulation_timestamp % 3_600_000_000_000 == 0 and v.simulation_timestamp != 0:
            v.update_repo()
        # 2. Log-directory change detection
        if v.simulation.logDir != state.logDir:
            logger.info(
                f"Simulation log directory changed: {v.simulation.logDir} -> {state.logDir}"
            )
            v.simulation.logDir = state.logDir
            self._notify_seed_log_dir_change(state.logDir)
        # 3. Update simulation_timestamp
        v.simulation_timestamp = state.timestamp
        # 4. Derive simulation_id from logDir — set once and lock; mid-sim logDir
        #    blips must not change the ID or ingest.py will wipe the DB.
        if v.simulation and state.logDir and not v.simulation.simulation_id:
            v.simulation.simulation_id = os.path.basename(state.logDir)[:13]
            logger.info(f"simulation_id locked: {v.simulation.simulation_id!r}")
            # Update validator state file to the versioned path now that sim_id is known.
            # If the placeholder file was already written (e.g. on a fresh start),
            # rename it so continuity is preserved.
            network = _network_label(getattr(v.config.subtensor, 'network', 'local'))
            versioned = v.config.neuron.full_path + f"/validator.simulation.{network}.{v.simulation.simulation_id}.mp"
            placeholder = v.config.neuron.full_path + f"/validator.simulation.{network}.mp"
            if os.path.exists(placeholder) and not os.path.exists(versioned):
                try:
                    os.rename(placeholder, versioned)
                    logger.info(f"validator_state_file renamed: {versioned!r}")
                except OSError as _e:
                    logger.warning(f"Could not rename validator state file: {_e}")
            v.validator_state_file = versioned
            logger.info(f"validator_state_file set: {v.validator_state_file!r}")
        # 5. Periodic log compression
        if v.simulation_timestamp % v.simulation.log_window == v.simulation.publish_interval:
            self.compress_outputs()

    # ─────────────────────────────────────────────────────────────────────────
    # collect_resets()
    # ─────────────────────────────────────────────────────────────────────────

    def collect_resets(self, state: NormalizedState, pending: set) -> None:
        """
        Scan validator's own notices for RDRA/ERDRA and add miner UIDs to `pending`.
        Failed resets trigger a pagerduty alert.

        Args:
            state: The block's state update.
            pending: Set of uids to extend with reset targets.
        """
        v = self.validator
        from taos.im.validator.trade import collect_reset_uids
        ok, failed = collect_reset_uids(state, v.uid)
        pending.update(ok)
        for reset in failed:
            v.pagerduty_alert(
                f"Failed to Reset Agent {reset.get('a')} : {reset.get('m')}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation-specific helpers (moved from validator.py)
    # ─────────────────────────────────────────────────────────────────────────

    def load_fundamental(self) -> None:
        """Load fundamental price data from simulation output files."""
        v = self.validator
        if v.simulation.logDir:
            prices = {}
            for block in range(v.simulation.block_count):
                block_file = os.path.join(
                    v.simulation.logDir,
                    f'fundamental.{block * v.simulation.books_per_block}-'
                    f'{v.simulation.books_per_block * (block + 1) - 1}.csv'
                )
                try:
                    # Only the first line (book ids) and last non-empty line
                    # (latest price row) matter; tail-read them in O(1) instead
                    # of readlines() over the whole ever-growing file.
                    first_line, fp_line = _read_first_and_last_nonempty(block_file)
                    if first_line == '':
                        continue
                    book_ids = [int(col) for col in first_line.split(',') if col != "Timestamp\n"]
                    if fp_line is not None and book_ids:
                        prices = prices | {book_ids[i]: float(price) for i, price in enumerate(fp_line.strip().split(',')[:-1])}
                except FileNotFoundError:
                    logger.warning(f"load_fundamental: missing file {block_file} — skipping block {block}")
                except Exception as exc:
                    logger.warning(f"load_fundamental: error reading block {block} fundamental: {exc}")
        else:
            prices = {bookId: None for bookId in range(v.simulation.book_count)}
        v.fundamental_price = prices

    def _compress_outputs(self, start: bool = False) -> None:
        """Compress old simulator log outputs and perform disk cleanup."""
        v = self.validator
        v.compressing = True
        try:
            if v.simulation.logDir:
                log_root = Path(v.simulation.logDir).parent
                for output_dir in log_root.iterdir():
                    if output_dir.is_dir():
                        log_archives = {}
                        log_path = Path(output_dir)
                        for log_file in log_path.iterdir():
                            if log_file.is_file() and log_file.suffix == '.log':
                                log_period = log_file.name.split('.')[1]
                                if len(log_period) == 13:
                                    log_end = (int(log_period.split('-')[1][:2]) * 3600 + int(log_period.split('-')[1][2:4]) * 60 + int(log_period.split('-')[1][4:])) * 1_000_000_000
                                else:
                                    log_end = (int(log_period.split('-')[1][:2]) * 86400 + int(log_period.split('-')[1][2:4]) * 3600 + int(log_period.split('-')[1][4:6]) * 60 + int(log_period.split('-')[1][6:])) * 1_000_000_000
                                if log_end < v.simulation_timestamp or (start and str(output_dir.resolve()) != v.simulation.logDir):
                                    log_type = log_file.name.split('-')[0]
                                    label = f"{log_type}_{log_period}"
                                    if label not in log_archives:
                                        log_archives[label] = []
                                    log_archives[label].append(log_file)
                        for label, log_files in log_archives.items():
                            archive = log_path / f"{label}.zip"
                            logger.info(f"Compressing {label} files to {archive.name}...")
                            with zipfile.ZipFile(archive, "w" if not archive.exists() else "a", compression=zipfile.ZIP_DEFLATED) as zipf:
                                for log_file in log_files:
                                    try:
                                        zipf.write(log_file, arcname=Path(log_file).name)
                                        os.remove(log_file)
                                        logger.debug(f"Added {log_file.name} to {archive.name}")
                                    except Exception as ex:
                                        logger.error(f"Failed to add {log_file.name} to {archive.name} : {ex}")
                if psutil.disk_usage('/').percent > 85:
                    min_retention_date = int((datetime.today() - timedelta(days=7)).strftime("%Y%m%d"))
                    logger.warning("Disk usage > 85% - cleaning up old outputs...")
                    for output in sorted(log_root.iterdir(), key=lambda f: f.name[:13]):
                        try:
                            archive_date = int(output.name[:8])
                        except Exception:
                            continue
                        if archive_date < min_retention_date:
                            try:
                                if output.is_file() and output.name.endswith('.zip'):
                                    output.unlink()
                                elif output.is_dir():
                                    shutil.rmtree(output)
                                disk_usage = psutil.disk_usage('/').percent
                                logger.info(f"Deleted {output.name} ({disk_usage}% disk available).")
                                if disk_usage <= 85:
                                    break
                            except Exception as ex:
                                v.pagerduty_alert(
                                    f"Failed to remove output {output.name} : {ex}",
                                    details={"trace": traceback.format_exc()}
                                )
        except Exception as ex:
            v.pagerduty_alert(
                f"Failure during output compression : {ex}",
                details={"trace": traceback.format_exc()}
            )
        finally:
            v.compressing = False

    def compress_outputs(self, start: bool = False) -> None:
        """Launch asynchronous log compression in a background thread."""
        v = self.validator
        if not v.compressing:
            Thread(
                target=self._compress_outputs,
                args=(start,),
                daemon=True,
                name=f'compress_{v.step}',
            ).start()

    # ─────────────────────────────────────────────────────────────────────────
    # Config loading (moved from validator.load_simulation_config)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        """Load simulation XML config and set state file paths on the validator."""
        v = self.validator
        import xml.etree.ElementTree as ET
        from taos.im.protocol.models import MarketSimulationConfig
        v.xml_config = ET.parse(v.config.simulation.xml_config).getroot()
        v.simulation = MarketSimulationConfig.from_xml(v.xml_config)
        v.simulator_config_file = str(Path(v.config.simulation.xml_config).resolve())
        network = _network_label(getattr(v.config.subtensor, 'network', 'local'))
        # LOB state: long parameter-list label (unchanged)
        v.simulation_state_file = v.config.neuron.full_path + f"/{v.simulation.label()}.mp"
        # Validator state: pick most recently modified versioned file if one exists;
        # otherwise use the placeholder path (will be renamed when sim_id is locked).
        full_path = Path(v.config.neuron.full_path)
        versioned_files = sorted(
            full_path.glob(f"validator.simulation.{network}.*.mp"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        placeholder = str(full_path / f"validator.simulation.{network}.mp")
        if versioned_files:
            v.validator_state_file = str(versioned_files[0])
            logger.info(f"validator_state_file (startup): {v.validator_state_file!r}")
        else:
            v.validator_state_file = placeholder
        self._sim_config = v.simulation
        self._book_ids = list(range(self._sim_config.book_count))

    # ─────────────────────────────────────────────────────────────────────────
    # Benchmark agent handling (moved from validator)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_benchmark_agents(self) -> None:
        """Load benchmark agent configurations from JSON file."""
        v = self.validator
        import json
        try:
            with open(v.config.benchmark.agents, 'r') as f:
                benchmark_config = json.load(f)
            import bittensor as bt
            for idx, agent in enumerate(benchmark_config['agents']):
                uid = v.benchmark_start_uid + idx
                coldkey = agent.get('coldkey', 'benchmark')
                v.benchmark_agents.append({
                    'uid': uid,
                    'name': agent['name'],
                    'ip': agent['ip'],
                    'port': agent['port'],
                    'hotkey': agent['hotkey'],
                    'coldkey': coldkey,
                    'axon': bt.AxonInfo(
                        ip=agent['ip'],
                        port=agent['port'],
                        hotkey=agent['hotkey'],
                        coldkey=coldkey,
                        version=v.metagraph.axons[0].version,
                        ip_type=v.metagraph.axons[0].ip_type,
                        protocol=v.metagraph.axons[0].protocol,
                    )
                })
                logger.info(
                    f"Loaded benchmark agent: {agent['name']} (UID {uid}) "
                    f"at {agent['ip']}:{agent['port']}"
                )
        except Exception as ex:
            logger.error(f"Failed to load benchmark agents: {ex}")
            logger.error(traceback.format_exc())
            v.benchmark_agents = []

    def get_extended_metagraph(self):
        """Return metagraph extended with benchmark agent UIDs."""
        v = self.validator
        if not v.benchmark_agents:
            return v.metagraph

        import bittensor as bt
        import torch as _torch
        extended_axons = [None] * v.effective_max_uids
        extended_hotkeys = [None] * v.effective_max_uids

        for uid, axon in enumerate(v.metagraph.axons):
            extended_axons[uid] = axon
            extended_hotkeys[uid] = v.metagraph.hotkeys[uid]

        for i, agent in enumerate(v.benchmark_agents):
            uid = v.benchmark_start_uid + i
            extended_axons[uid] = agent['axon']
            extended_hotkeys[uid] = agent['hotkey']

        placeholder_axon = bt.AxonInfo(
            version=v.metagraph.axons[0].version if v.metagraph.axons else 0,
            hotkey="",
            coldkey="",
            ip="0.0.0.0",
            port=0,
            ip_type=4,
            protocol=4,
            placeholder1=0,
            placeholder2=0,
        )
        for uid in range(v.effective_max_uids):
            if extended_axons[uid] is None:
                extended_axons[uid] = placeholder_axon
                extended_hotkeys[uid] = placeholder_axon.hotkey

        class ExtendedMetagraph:
            """Metagraph wrapper carrying the benchmark uids beyond the chain's own."""
            def __init__(self, uids, hotkeys, axons):
                self.uids = _torch.tensor(uids)
                self.hotkeys = hotkeys
                self.axons = axons

        return ExtendedMetagraph(
            list(range(v.effective_max_uids)),
            extended_hotkeys,
            extended_axons,
        )

    def on_resync_metagraph(self, old_size: int, new_size: int) -> None:
        """Update benchmark agent UIDs when metagraph size changes.

        Benchmark slots are PINNED to subnet_info.max_uids (the subnet's
        configured capacity) — NOT to the current registered-neuron count
        from `len(metagraph.hotkeys)`. Otherwise the benchmark UID slides
        with each new registration and overwrites real chain UIDs in the
        meta:agents Redis key.

        Args:
            old_size: Metagraph size before the resync.
            new_size: Metagraph size after.
        """
        v = self.validator
        old_effective_max_uids = v.effective_max_uids
        v.benchmark_start_uid = v.subnet_info.max_uids
        v.effective_max_uids = v.subnet_info.max_uids + len(v.benchmark_agents)
        if old_effective_max_uids == v.effective_max_uids:
            return

        import torch as _torch
        logger.info(
            f"Resizing scores tensor: {old_effective_max_uids} -> {v.effective_max_uids} "
            f"(network: {new_size}, benchmarks: {len(v.benchmark_agents)})"
        )
        new_scores = _torch.zeros(
            v.effective_max_uids, dtype=_torch.float32, device=v.device
        )
        min_len = min(old_effective_max_uids, v.effective_max_uids)
        new_scores[:min_len] = v.scores[:min_len]
        v.scores = new_scores

        new_gentrx = _torch.zeros(
            v.effective_max_uids, dtype=_torch.float32, device=v.device
        )
        new_gentrx[:min_len] = v.gentrx_scores[:min_len]
        v.gentrx_scores = new_gentrx

        if v.benchmark_agents:
            logger.info("Updating benchmark agent UIDs...")
            for idx, agent in enumerate(v.benchmark_agents):
                old_uid = agent['uid']
                new_uid = v.benchmark_start_uid + idx
                if old_uid == new_uid:
                    continue
                logger.info(f"Benchmark agent {agent['name']}: UID {old_uid} -> {new_uid}")
                agent['uid'] = new_uid
                for data_dict in [
                    v.miner_stats, v.initial_balances, v.activity_factors,
                    v.pnl_factors, v.kappa_values, v.unnormalized_scores,
                    v.inventory_history, v.trade_volumes, v.realized_pnl_history,
                    v.open_positions,
                ]:
                    if old_uid in data_dict:
                        data_dict[new_uid] = data_dict.pop(old_uid)
                for volume_dict in [
                    v.volume_sums, v.maker_volume_sums, v.taker_volume_sums,
                    v.self_volume_sums, v.roundtrip_volume_sums,
                ]:
                    if old_uid in volume_dict:
                        volume_dict[new_uid] = volume_dict.pop(old_uid)
                if old_uid in v.roundtrip_volumes:
                    v.roundtrip_volumes[new_uid] = v.roundtrip_volumes.pop(old_uid)
                if old_uid < len(v.scores):
                    v.scores[new_uid] = v.scores[old_uid]
                    v.scores[old_uid] = 0.0
                if old_uid < len(v.gentrx_scores):
                    v.gentrx_scores[new_uid] = v.gentrx_scores[old_uid]
                    v.gentrx_scores[old_uid] = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Structure initialization (moved from validator._initialize_all_structures)
    # ─────────────────────────────────────────────────────────────────────────

    def initialize_structures(self) -> None:
        """Initialize all UID-indexed state structures on the validator."""
        v = self.validator
        book_count = self._sim_config.book_count
        logger.info(
            f"Initializing structures for {v.effective_max_uids} UIDs "
            f"(network: {v.subnet_info.max_uids}, benchmarks: {len(v.benchmark_agents)}) "
            f"with {book_count} books"
        )

        v.simulation_timestamp = 0
        v.initial_balances_published = {uid: False for uid in range(v.effective_max_uids)}

        if not hasattr(v, 'activity_factors') or len(v.activity_factors) < v.effective_max_uids:
            v.activity_factors = {
                uid: {bookId: 0.0 for bookId in range(book_count)}
                for uid in range(v.effective_max_uids)
            }
        if not hasattr(v, 'pnl_factors') or len(v.pnl_factors) < v.effective_max_uids:
            v.pnl_factors = {
                uid: {bookId: 1.0 for bookId in range(book_count)}
                for uid in range(v.effective_max_uids)
            }
        if not hasattr(v, 'kappa_values') or len(v.kappa_values) < v.effective_max_uids:
            v.kappa_values = {
                uid: {
                    'books': {bookId: None for bookId in range(book_count)},
                    'books_weighted': {bookId: 0.0 for bookId in range(book_count)},
                    'total': None, 'average': None, 'median': None,
                    'normalized_average': 0.0, 'normalized_median': 0.0,
                    'normalized_total': 0.0,
                    'activity_weighted_normalized_median': 0.0,
                    'penalty': 0.0, 'score': 0.0,
                }
                for uid in range(v.effective_max_uids)
            }
        if not hasattr(v, 'unnormalized_scores') or len(v.unnormalized_scores) < v.effective_max_uids:
            v.unnormalized_scores = {uid: 0.0 for uid in range(v.effective_max_uids)}
        if not hasattr(v, 'trade_volumes') or len(v.trade_volumes) < v.effective_max_uids:
            v.trade_volumes = {
                uid: {
                    bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                    for bookId in range(book_count)
                }
                for uid in range(v.effective_max_uids)
            }
        if not hasattr(v, 'initial_balances') or len(v.initial_balances) < v.effective_max_uids:
            v.initial_balances = {
                uid: {
                    bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                    for bookId in range(book_count)
                }
                for uid in range(v.effective_max_uids)
            }
        if not hasattr(v, 'recent_miner_trades') or len(v.recent_miner_trades) < v.effective_max_uids:
            v.recent_miner_trades = {
                uid: {bookId: [] for bookId in range(book_count)}
                for uid in range(v.effective_max_uids)
            }
        if not hasattr(v, 'miner_stats') or len(v.miner_stats) < v.effective_max_uids:
            v.miner_stats = {
                uid: {'requests': 0, 'timeouts': 0, 'failures': 0, 'rejections': 0, 'call_time': []}
                for uid in range(v.effective_max_uids)
            }
        if not hasattr(v, 'recent_trades'):
            v.recent_trades = {bookId: [] for bookId in range(book_count)}
        if not hasattr(v, 'fundamental_price'):
            v.fundamental_price = {bookId: None for bookId in range(book_count)}

        from collections import defaultdict, deque
        if not hasattr(v, 'volume_sums'):
            v.volume_sums = defaultdict(lambda: defaultdict(float))
        if not hasattr(v, 'maker_volume_sums'):
            v.maker_volume_sums = defaultdict(lambda: defaultdict(float))
        if not hasattr(v, 'taker_volume_sums'):
            v.taker_volume_sums = defaultdict(lambda: defaultdict(float))
        if not hasattr(v, 'self_volume_sums'):
            v.self_volume_sums = defaultdict(lambda: defaultdict(float))
        if not hasattr(v, 'open_positions'):
            v.open_positions = defaultdict(lambda: defaultdict(lambda: {
                'longs': deque(), 'shorts': deque()
            }))
        if not hasattr(v, 'inventory_history'):
            v.inventory_history = {uid: {} for uid in range(v.effective_max_uids)}
        if not hasattr(v, 'realized_pnl_history'):
            v.realized_pnl_history = defaultdict(lambda: defaultdict(dict))
        if not hasattr(v, 'roundtrip_volumes'):
            v.roundtrip_volumes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        if not hasattr(v, 'roundtrip_volume_sums'):
            v.roundtrip_volume_sums = defaultdict(lambda: defaultdict(float))
        if not hasattr(v, 'kappa_cache'):
            v.kappa_cache = {}
        if not hasattr(v, 'pending_notices'):
            v.pending_notices = {uid: [] for uid in range(v.effective_max_uids)}

        logger.info(f"All structures initialized for {v.effective_max_uids} UIDs")

    # ─────────────────────────────────────────────────────────────────────────
    # State persistence (moved from validator snapshot methods)
    # ─────────────────────────────────────────────────────────────────────────

    def build_simulation_state(self) -> dict:
        """Build simulation-specific state dict for persistence."""
        v = self.validator
        return {
            "start_time": v.start_time,
            "start_timestamp": v.start_timestamp,
            "step_rates": list(v.step_rates),
            "initial_balances": v.initial_balances,
            "recent_trades": {
                book_id: [t.model_dump(mode="json") for t in trades]
                for book_id, trades in v.recent_trades.items()
            },
            "recent_miner_trades": {
                uid: {
                    book_id: [[t.model_dump(mode="json"), r] for t, r in trades]
                    for book_id, trades in uid_trades.items()
                }
                for uid, uid_trades in v.recent_miner_trades.items()
            },
            "pending_notices": v.pending_notices,
            "simulation.logDir": v.simulation.logDir,
            # Persist the locked id so a process restart reloads it instead of
            # re-deriving from the (possibly rotated) logDir — which would mint a
            # new id for the same run and trigger a data-service prune.
            "simulation.simulation_id": v.simulation.simulation_id,
            # External-order UUID threading maps: resting exchange-API orders can
            # fill hours later and across restarts; losing these breaks the data
            # service's FILLED transitions (counter persists to avoid cloid reuse).
            "ext_cloid_to_uuid": dict(getattr(self, "_ext_cloid_to_uuid", {})),
            "ext_engineoid_to_uuid": dict(getattr(self, "_ext_engineoid_to_uuid", {})),
            "ext_cloid_counter": getattr(self, "_ext_cloid_counter", 0),
        }

    def restore_simulation_state(self, data: dict) -> None:
        """Restore simulation-specific state from disk."""
        v = self.validator
        v.start_time = data.get("start_time")
        v.start_timestamp = data.get("start_timestamp")
        v.step_rates = data.get("step_rates", [])
        v.initial_balances = data.get("initial_balances", v.initial_balances)
        if "simulation.logDir" in data and v.simulation:
            v.simulation.logDir = data["simulation.logDir"]
        # Restore the locked simulation_id (set after _load_config resets it) so
        # on_tick does not re-derive a new id from a rotated logDir on restart.
        if data.get("simulation.simulation_id") and v.simulation:
            v.simulation.simulation_id = data["simulation.simulation_id"]
            logger.info(f"Restored simulation_id: {v.simulation.simulation_id!r}")
        # External-order UUID maps (JSON round-trips stringify int keys).
        self._ext_cloid_to_uuid = {int(k): str(u) for k, u in (data.get("ext_cloid_to_uuid") or {}).items()}
        self._ext_engineoid_to_uuid = {int(k): str(u) for k, u in (data.get("ext_engineoid_to_uuid") or {}).items()}
        self._ext_cloid_counter = int(data.get("ext_cloid_counter") or 0)
        # recent_trades and recent_miner_trades are deserialized in load_state()
        # pending_notices are deserialized in load_state()

    # ─────────────────────────────────────────────────────────────────────────
    # Deregistration handling (moved from validator)
    # ─────────────────────────────────────────────────────────────────────────

    def handle_deregistration(self, uid: int, old_coldkey: str | None = None) -> None:
        """Flag UID for reset, zero its score.

        `old_coldkey` is accepted and unused here. The validator calls every engine the same way
        (taos/im/neurons/validator.py:1365 passes uid AND old_coldkey, because the exchange engine needs
        it to retire the departed miner's settlement-proxy wallet), and this signature took only uid. Every
        deregistration on the simulation mechanism therefore raised
        `TypeError: handle_deregistration() takes 2 positional arguments` out of the resync, which failed
        `_sync_and_check` whole: the departed miner was never flagged for reset, its score never zeroed,
        and `self.hotkeys` never updated from the metagraph, so the same deregistration re-raised on every
        subsequent resync and the slot's new occupant inherited the old standing indefinitely. Measured
        live on 2026-08-05 by the acceptance suite's deregistration stage.

        Simulation has no proxy wallet to retire, so the parameter is deliberately ignored rather than
        removed from the shared signature: one call site, one shape.

        Args:
            uid: The deregistered uid.
            old_coldkey: Accepted for interface parity; unused by this engine.
        """
        v = self.validator
        v.deregistered_uids.append(uid)
        v.scores[uid] = 0.0
        logger.debug(f"UID {uid} Deregistered - Scheduled for reset.")

    def apply_resets(self, pending: set) -> None:
        """Zero all scoring state for each UID in pending.

        The history/scoring zeroing lives in trade.reset_agent_histories —
        shared verbatim with the shadow scoring service (which receives a
        ("resets", uids) frame). Main-only bookkeeping stays here.
        """
        v = self.validator
        if not pending:
            return
        from taos.im.validator.trade import reset_agent_histories
        # NOTE: no frame to the scoring service — it derives the same resets
        # from the teed state (collect_reset_uids) at the same position.
        import bittensor as bt
        bt.logging.info(f"Applying agent resets: {sorted(pending)}")
        for uid in pending:
            reset_agent_histories(v, uid, self.book_ids)
            v.unnormalized_scores[uid] = 0.0
            v.initial_balances_published[uid] = False
            if uid in v.deregistered_uids:
                v.deregistered_uids.remove(uid)
            v.miner_stats[uid] = {
                'requests': 0, 'timeouts': 0, 'failures': 0,
                'rejections': 0, 'call_time': []
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Seed service (moved from validator)
    # ─────────────────────────────────────────────────────────────────────────

    def _start_seed_service(self) -> None:
        """Launch the seed service subprocess."""
        v = self.validator
        logger.info("Starting seed service from: ../validator/seed.py")
        cmd = [
            sys.executable, '-u', '../validator/seed.py',
            '--seed.fundamental.symbol.coinbase',
            v.config.simulation.seeding.fundamental.symbol.coinbase,
            '--seed.fundamental.symbol.binance',
            v.config.simulation.seeding.fundamental.symbol.binance,
            '--seed.external.symbol.coinbase',
            v.config.simulation.seeding.external.symbol.coinbase,
            '--seed.external.symbol.binance',
            v.config.simulation.seeding.external.symbol.binance,
            '--seed.external.sampling_seconds',
            str(v.config.simulation.seeding.external.sampling_seconds),
            '--logging.level',
            'INFO' if not (v.config.logging.debug or v.config.logging.trace) else 'DEBUG',
        ]
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        import subprocess
        v.seed_process = subprocess.Popen(
            cmd, stdout=sys.stdout, stderr=sys.stderr, env=env
        )
        logger.info(f"Seed service PID: {v.seed_process.pid}")
        time.sleep(2.0)
        if v.seed_process.poll() is not None:
            raise RuntimeError(
                f"Seed service died with exit code {v.seed_process.returncode}"
            )
        if v.simulation.logDir:
            self._notify_seed_log_dir_change(v.simulation.logDir)

    def _notify_seed_log_dir_change(self, new_log_dir: Optional[str]) -> None:
        """Notify seed service of log directory change via file + signal."""
        v = self.validator
        try:
            log_dir_file = '/tmp/validator_log_dir.txt'
            with open(log_dir_file, 'w') as f:
                f.write(new_log_dir or '')
            if not v.seed_process:
                return
            if v.seed_process.poll() is not None:
                logger.warning(
                    f"Seed service died (exit code {v.seed_process.returncode}), "
                    "cannot notify of log dir change"
                )
                return
            import signal as _signal
            v.seed_process.send_signal(_signal.SIGUSR1)
            logger.info(f"Notified seed service of log directory change: {new_log_dir}")
        except ProcessLookupError:
            logger.warning("Seed process not found when trying to notify log dir change")
        except Exception as ex:
            logger.warning(f"Failed to notify seed service of log dir change: {ex}")
        self._notify_sltp_sim_dir(new_log_dir)

    def _notify_sltp_sim_dir(self, sim_dir: Optional[str]) -> None:
        """Delegate the sim-dir push to the optional SL/TP extension when present."""
        if self._sltp_ext is None:
            return
        self._sltp_ext.notify_sim_dir(sim_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # Metadata
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def book_ids(self) -> list[int]:
        """Readable accessor for the internal field ``_book_ids``."""
        return self._book_ids

    @property
    def mode(self) -> str:
        """The engine's mechanism name: ``'simulation'``."""
        return 'simulation'

    @property
    def effective_max_uids(self) -> int:
        """Uid capacity including benchmark agents beyond the chain's registered set."""
        return getattr(self.validator, 'effective_max_uids', self.validator.subnet_info.max_uids)

    # ─────────────────────────────────────────────────────────────────────────
    # Transport helpers
    # Replace the bodies with the actual IPC calls from _listen().
    # ─────────────────────────────────────────────────────────────────────────

    async def _recv_bytes(self) -> bytes:
        """
        Read one full state update from the simulator via Posix MQ + SHM.

        Receives the 8-byte size pointer from /taosim-req, then reads the
        msgpack payload from the /state shared-memory segment.
        """
        import posix_ipc as _posix_ipc
        import mmap as _mmap

        loop = asyncio.get_event_loop()
        # [RECV-PROFILE] discriminates the scoring-round recv spike:
        #   mq     = sim-side: how long the simulator took to produce+signal the
        #            next state (includes sim checkpoint pauses / bandwidth slowdown)
        #   resume = validator-side: executor-thread done -> coroutine resumed on
        #            the event loop (pure GIL/event-loop starvation)
        #   poll   = SHM write-completion wait (sim write lag + poll-loop GIL churn)
        #   read   = mmap read of the payload
        _t0 = time.time()
        _mq_done = [0.0]

        def _mq_recv():
            _r = self._req_socket.receive()
            _mq_done[0] = time.time()
            return _r

        msg, _ = await loop.run_in_executor(None, _mq_recv)
        _t_resume = time.time()
        byte_size = int.from_bytes(msg, byteorder="little")
        shm_req = _posix_ipc.SharedMemory("/state")
        packed_data = None
        try:
            # Wait for the SHM /state write to be ready, then read it. The
            # simulator can signal via MQ slightly before the write lands, so a
            # brief poll is normal. Two ready conditions, whichever comes first:
            #   1. size reaches the announced byte_size (normal case), or
            #   2. size STABILIZES > 0 below the announced size — a stale/oversized
            #      MQ frame announced MORE than was actually written (the SHM was
            #      truncated smaller for the current, smaller state).
            # Condition 2 is essential: without it, polling `fstat >= announced`
            # can never succeed on a stale frame, and the old
            # `max(15, byte_size/100_000)` timeout (~80s for an 8 MB payload)
            # stalled the whole sim<->validator handshake ~80s before falling back
            # to the actual size. A partial write (size right, bytes not yet fully
            # copied) is caught downstream by the msgpack unpack-retry loop.
            import os as _os
            _deadline = time.monotonic() + 15.0
            _last = -1
            _stable_since = None
            _ready = None
            while time.monotonic() < _deadline:
                _sz = _os.fstat(shm_req.fd).st_size
                if _sz >= byte_size:
                    _ready = byte_size
                    break
                if _sz == _last and _sz > 0:
                    if _stable_since is None:
                        _stable_since = time.monotonic()
                    elif time.monotonic() - _stable_since >= 0.25:
                        logger.warning(
                            f"SHM /state size mismatch: announced {byte_size} B, "
                            f"actual {_sz} B — reading actual (stale MQ frame?)"
                        )
                        _ready = _sz
                        break
                else:
                    _last = _sz
                    _stable_since = None
                time.sleep(0.002)
            if _ready is None:
                _actual = _os.fstat(shm_req.fd).st_size
                if _actual <= 0:
                    raise RuntimeError(
                        f"SHM /state not ready after 15s: announced {byte_size} B, got {_actual} B"
                    )
                logger.warning(
                    f"SHM /state not ready after 15s; reading actual {_actual} B "
                    f"(announced {byte_size} B)"
                )
                _ready = _actual
            byte_size = _ready
            _t_poll = time.time()
            with _mmap.mmap(shm_req.fd, byte_size, _mmap.MAP_SHARED, _mmap.PROT_READ) as mm:
                packed_data = mm.read(byte_size)
            import bittensor as bt
            bt.logging.info(
                f"[RECV-PROFILE] mq={_mq_done[0] - _t0:.3f}s "
                f"resume={_t_resume - _mq_done[0]:.3f}s "
                f"poll={_t_poll - _t_resume:.3f}s "
                f"read={time.time() - _t_poll:.3f}s"
            )
        finally:
            shm_req.close_fd()
        return packed_data

    async def _recv_bytes_shm_only(self, prev_bytes: bytes) -> bytes:
        """Re-read the current SHM /state payload without consuming a new MQ message.
        Used to retry after a partial-write race on the first read."""
        import posix_ipc as _posix_ipc
        import mmap as _mmap
        import os as _os
        byte_size = len(prev_bytes)
        loop = asyncio.get_event_loop()
        def _read():
            shm = _posix_ipc.SharedMemory("/state")
            try:
                # Flat cap: the old byte_size/100_000 was ~80s for an 8 MB
                # payload. The loop breaks immediately when the SHM already holds
                # >= byte_size (the common retry case), and reads the actual size
                # otherwise, so a short ceiling is safe.
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    if _os.fstat(shm.fd).st_size >= byte_size:
                        break
                    time.sleep(0.002)
                # Use actual SHM size if it stabilised at a different value
                # (common on sim restart when new state is a different size).
                actual = _os.fstat(shm.fd).st_size
                read_size = actual if actual > 0 and actual != byte_size else byte_size
                if read_size != byte_size:
                    logger.warning(
                        "SHM /state retry: announced %d B, actual %d B — reading actual",
                        byte_size, read_size,
                    )
                with _mmap.mmap(shm.fd, read_size, _mmap.MAP_SHARED, _mmap.PROT_READ) as mm:
                    return mm.read(read_size)
            finally:
                shm.close_fd()
        return await loop.run_in_executor(None, _read)

    def _send_bytes(self, data: bytes) -> None:
        """
        Write the response to /responses SHM and signal via ephemeral /taosim-res MQ.
        """
        import posix_ipc as _posix_ipc
        import mmap as _mmap

        byte_size = len(data)
        mq_res = _posix_ipc.MessageQueue(
            "/taosim-res", flags=_posix_ipc.O_CREAT, max_messages=1, max_message_size=8
        )
        shm_res = _posix_ipc.SharedMemory("/responses", flags=_posix_ipc.O_CREAT, size=byte_size)
        with _mmap.mmap(shm_res.fd, byte_size, _mmap.MAP_SHARED, _mmap.PROT_WRITE | _mmap.PROT_READ) as mm:
            shm_res.close_fd()
            mm.write(data)
        mq_res.send(byte_size.to_bytes(8, byteorder="little"))
        mq_res.close()

    def _parse_state(self, raw_dict: dict) -> Any:
        """
        Parse deserialised dict → MarketSimulationStateUpdate.
        Verbatim from existing _listen():
            from taosim.protocol import MarketSimulationStateUpdate
            return MarketSimulationStateUpdate.parse_dict(raw_dict)
        """
        return MarketSimulationStateUpdate.parse_dict(raw_dict)

    # ─────────────────────────────────────────────────────────────────────────
    # SL/TP live trigger endpoints (called by validator HTTP server)
    # ─────────────────────────────────────────────────────────────────────────

    async def sltp_records(self, request) -> list:
        """Delegate the records POST to the optional SL/TP extension when present."""
        if self._sltp_ext is None:
            return []
        return await self._sltp_ext.handle_records(request)

    async def sltp_levels(self) -> dict:
        """Returns the latest trigger snapshot — populated by the optional extension."""
        return self._live_triggers