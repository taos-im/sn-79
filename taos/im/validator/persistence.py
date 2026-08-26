# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import gc
import math
import time
import shutil
import asyncio
import traceback
import msgpack
import aiofiles
import aiofiles.os
import torch

from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

import bittensor as bt

from taos.im.utils.save import save_state_worker
from taos.im.protocol.models import TradeInfo
from taos.im.validator.debeta import sum_hist_2level, sum_hist_1level
from taos.im.protocol.events import TradeEvent

if TYPE_CHECKING:
    from taos.im.neurons.validator import Validator


def sanitize_recent_trades(book_trades) -> tuple[list, int]:
    """Drop restored trades whose id is not the engine's integer id.

    A trade id must be the raw engine integer: it goes back into the engine as `ExecutedFill`, where
    `tradeId` is `std::optional<uint64_t>`, and a string there throws `std::bad_cast` and drops the
    whole batch. `TradeInfo.model_construct` skips validation (hot path), so a bad id passes silently
    on load and only bites later at `model_dump`, in a different module, as a serializer warning.

    Such an id cannot be repaired. The only known producer was a synthetic
    `x:<extrinsic>:<agent>:<direction>:<p|f>` fallback, since deleted; no id of that shape was ever
    written to the tape, so there is nothing to join it to and guessing one would recreate exactly the
    identity divergence the fallback caused.

    A missing or null id is KEPT: a pre-settlement trade legitimately has no id yet, and dropping
    those would lose real trades, which is worse than the warning.
    """
    kept: list = []
    dropped = 0
    for t in book_trades:
        tid = t.get("i") if isinstance(t, dict) else None
        if tid is None:
            kept.append(t)
            continue
        if isinstance(tid, bool) or isinstance(tid, float):
            dropped += 1
            continue
        if isinstance(tid, int):
            if tid < 0:
                dropped += 1
                continue
            kept.append(t)
            continue
        # A msgpack round-trip can stringify a number; that id IS joinable, so coerce rather than drop.
        try:
            coerced = int(str(tid).strip())
        except (TypeError, ValueError):
            dropped += 1
            continue
        if coerced < 0:
            dropped += 1
            continue
        fixed = dict(t)
        fixed["i"] = coerced
        kept.append(fixed)
    return kept, dropped


def sanitize_miner_trades(book_trades) -> tuple[list, int]:
    """Drop restored miner trades that cannot answer the questions their consumers ask of them.

    `recent_miner_trades` holds [notice_dict, role] pairs and is restored with
    `TradeEvent.model_construct(**t)`, which skips validation. `takerFee` is a property returning
    `self.Tf`, so an entry persisted without the fee fields comes back as an object with no such
    attribute and metrics publishing dies on it: 'TradeEvent' object has no attribute 'Tf', four times
    in the 40 minutes after the producer was fixed, because the buffer keeps five entries per uid per
    book and the old ones survive until fresh trades displace them.

    Not repaired, dropped. A fee cannot be reconstructed after the fact, and substituting zero would
    understate cost wherever fees are real, which is worse than losing one historical slot from a
    rolling five-entry buffer.
    """
    kept: list = []
    dropped = 0
    for entry in book_trades or []:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            dropped += 1
            continue
        notice = entry[0]
        if not isinstance(notice, dict) or "Mf" not in notice or "Tf" not in notice:
            dropped += 1
            continue
        # The id must satisfy the SAME rule as sanitize_recent_trades, through the same function.
        #
        # These two sanitisers drifted: this one validated fees and ignored the id, the other validated
        # the id and ignored fees. An entry with good fees and a synthetic
        # x:0x<extrinsic>:<uid>:<direction>:<p|f> id therefore passed both, and the stack capture caught
        # it still tripping the serializer at validator.py:1894 (_prepare_reporting_data) four minutes
        # after this sanitiser had run. Two implementations of one criterion is the defect; sharing the
        # implementation is the fix.
        id_ok, id_dropped = sanitize_recent_trades([notice])
        if id_dropped or not id_ok:
            dropped += 1
            continue
        kept.append([id_ok[0], entry[1]])
    return kept, dropped


def _snap_hist3(h):
    """Deep-copy a {k1:{k2:{ts:val}}} de-beta history for persistence (tolerate concurrent mutation)."""
    return {k1: {k2: dict(tsd) for k2, tsd in d2.items()} for k1, d2 in (h or {}).items()}


def _snap_hist2(h):
    """Deep-copy a {k:{ts:val}} de-beta history for persistence."""
    return {k: dict(tsd) for k, tsd in (h or {}).items()}


def build_validator_state(
    self: Validator,
    inventory_snapshot,
    realized_pnl_snapshot,
    volume_sums_snapshots,
    trade_volumes_snapshot,
    roundtrip_volumes_snapshot,
    open_positions_snapshot
):
    """Assemble validator state dict from all snapshots.

    This method is shared by both sync and async construction paths
    to ensure consistency and maintainability.

    Args:
        inventory_snapshot (dict): Snapshot of inventory_history.
        realized_pnl_snapshot (dict): Snapshot of realized_pnl_history.
        volume_sums_snapshots (dict): Dictionary containing all volume sum snapshots.
        trade_volumes_snapshot (dict): Snapshot of trade_volumes.
        roundtrip_volumes_snapshot (dict): Snapshot of roundtrip_volumes.
        open_positions_snapshot (dict): Snapshot of open_positions.

    Returns:
        dict: Complete validator state containing step, simulation_timestamp, hotkeys,
            scores, activity_factors, pnl_factors, and all snapshot data.
    """
    return {
        "step": self.step,
        "simulation_timestamp": self.simulation_timestamp,
        "hotkeys": self.hotkeys,
        "scores": [score.item() for score in self.scores],
        "gentrx_scores": [score.item() for score in self.gentrx_scores],
        "activity_factors": self.activity_factors,
        "pnl_factors": self.pnl_factors,
        "trading_score_ema": dict(getattr(self, '_trading_score_ema', {}) or {}),
        "trading_score_ema_ts": getattr(self, '_trading_score_ema_ts', None),
        "trading_score_ema_n": dict(getattr(self, '_trading_score_ema_n', {}) or {}),
        "inventory_history": inventory_snapshot,
        "kappa_values": self.kappa_values,
        "realized_pnl_history": realized_pnl_snapshot,
        "open_positions": open_positions_snapshot,
        "unnormalized_scores": self.unnormalized_scores,
        "deregistered_uids": self.deregistered_uids,
        "trade_volumes": trade_volumes_snapshot,
        "roundtrip_volumes": roundtrip_volumes_snapshot,
        "volume_sums": volume_sums_snapshots['volume_sums'],
        "maker_volume_sums": volume_sums_snapshots['maker_volume_sums'],
        "taker_volume_sums": volume_sums_snapshots['taker_volume_sums'],
        "self_volume_sums": volume_sums_snapshots['self_volume_sums'],
        "roundtrip_volume_sums": volume_sums_snapshots['roundtrip_volume_sums'],
        # De-beta (P8/E5/P11) fill-stream accumulators, windowed exactly like trade_volumes. Persist the
        # TIMESTAMPED HISTORIES (making window + skill window) + the reconstruction STATE (per-book
        # inventory + last price); the running sums are rebuilt from the histories on load, so
        # running == sum(history within window) holds by construction. Only populated when de-beta is
        # enabled; empty/absent otherwise.
        "debeta_capbuy_hist": _snap_hist3(getattr(self, 'debeta_capbuy_hist', {})),
        "debeta_capsell_hist": _snap_hist3(getattr(self, 'debeta_capsell_hist', {})),
        "debeta_mtm_hist": _snap_hist3(getattr(self, 'debeta_mtm_hist', {})),
        "debeta_invsum_hist": _snap_hist3(getattr(self, 'debeta_invsum_hist', {})),
        "debeta_cp_hist": _snap_hist3(getattr(self, 'debeta_cp_hist', {})),
        "debeta_invn_hist": _snap_hist2(getattr(self, 'debeta_invn_hist', {})),
        "debeta_drift_hist": _snap_hist2(getattr(self, 'debeta_drift_hist', {})),
        "debeta_inv": {b: dict(u) for b, u in getattr(self, 'debeta_inv', {}).items()},
        "debeta_plast": dict(getattr(self, 'debeta_plast', {})),
        # Persist the rolling request/timeout accumulators so a restart doesn't
        # reset them and force a fresh ~100-request (~12 min) blackout before the
        # miner_gauges requests/timeouts/call_time series reappear in Grafana.
        # Per-uid dict + call_time list are copied to tolerate concurrent
        # update_stats() mutation on the main loop (no await between here).
        "miner_stats": snapshot_miner_stats(self),
    }


def snapshot_miner_stats(self: Validator):
    """Serialize per-miner statistics for the saved-state file."""
    return {
        uid: {
            "requests": s.get("requests", 0),
            "timeouts": s.get("timeouts", 0),
            "failures": s.get("failures", 0),
            "rejections": s.get("rejections", 0),
            "call_time": list(s.get("call_time", [])),
        }
        for uid, s in dict(getattr(self, "miner_stats", {})).items()
        if isinstance(s, dict)
    }


def build_save_light_fields(self: Validator) -> dict:
    """The main-only fields of the validator save file, shipped to the scoring
    service when it writes the file from its replica (save offload). Cheap
    copies, built on the main loop so they are atomic wrt round handling."""
    return {
        "step": self.step,
        "simulation_timestamp": self.simulation_timestamp,
        "hotkeys": list(self.hotkeys),
        "scores": [score.item() for score in self.scores],
        "gentrx_scores": [score.item() for score in self.gentrx_scores],
        "unnormalized_scores": dict(self.unnormalized_scores),
        "deregistered_uids": list(self.deregistered_uids),
        "trading_score_ema": dict(getattr(self, '_trading_score_ema', {}) or {}),
        "trading_score_ema_ts": getattr(self, '_trading_score_ema_ts', None),
        "trading_score_ema_n": dict(getattr(self, '_trading_score_ema_n', {}) or {}),
        "miner_stats": snapshot_miner_stats(self),
    }


def snapshot_inventory_history(self: Validator):
    """Serialize the inventory history for the saved-state file."""
    result = {}
    for uid in range(self.effective_max_uids):
        if uid in self.inventory_history and self.inventory_history[uid]:
            result[uid] = {ts: dict(books) for ts, books in self.inventory_history[uid].items()}
    return result


def snapshot_realized_pnl_history(self: Validator):
    """Serialize the realized-PnL history for the saved-state file."""
    result = {}
    for uid in range(self.effective_max_uids):
        if uid in self.realized_pnl_history:
            result[uid] = {ts: dict(books) for ts, books in self.realized_pnl_history[uid].items()}
    return result


def snapshot_2_level_dict(self: Validator, source_dict):
    """Serialize a ``{uid: {book: value}}`` mapping for the saved-state file.

    Args:
        source_dict: The two-level mapping to serialize.
    """
    result = {}
    for uid in range(self.effective_max_uids):
        if uid in source_dict:
            result[uid] = dict(source_dict[uid])
    return result


def snapshot_volume_sums(self: Validator):
    """Serialize per-miner volume sums for the saved-state file."""
    return {
        'volume_sums':           snapshot_2_level_dict(self, self.volume_sums),
        'maker_volume_sums':     snapshot_2_level_dict(self, self.maker_volume_sums),
        'taker_volume_sums':     snapshot_2_level_dict(self, self.taker_volume_sums),
        'self_volume_sums':      snapshot_2_level_dict(self, self.self_volume_sums),
        'roundtrip_volume_sums': snapshot_2_level_dict(self, self.roundtrip_volume_sums),
    }


def snapshot_trade_volumes(self: Validator):
    """Serialize per-miner trade-volume windows for the saved-state file."""
    result = {}
    for uid in range(self.effective_max_uids):
        if uid not in self.trade_volumes:
            continue
        result[uid] = {
            book_id: {role: dict(volumes) for role, volumes in roles.items()}
            for book_id, roles in self.trade_volumes[uid].items()
        }
    return result


def snapshot_roundtrip_volumes(self: Validator):
    """Serialize per-miner roundtrip volumes for the saved-state file."""
    result = {}
    for uid in range(self.effective_max_uids):
        if uid not in self.roundtrip_volumes:
            continue
        result[uid] = {book_id: dict(volumes) for book_id, volumes in self.roundtrip_volumes[uid].items()}
    return result


def snapshot_open_positions(self: Validator):
    """Serialize per-miner open positions for the saved-state file."""
    result = {}
    for uid in range(self.effective_max_uids):
        if uid not in self.open_positions:
            continue
        result[uid] = {
            book_id: {
                'longs':  list(pos.get('longs',  [])),
                'shorts': list(pos.get('shorts', [])),
            }
            for book_id, pos in self.open_positions[uid].items()
        }
    return result


def construct_save_data_sync(self: Validator):
    """Synchronously build all state data for saving.

    Constructs both simulation and validator state by calling each step method
    directly in sequence. This is the synchronous alternative to the async
    construction path.

    Returns:
        tuple: Three-element tuple containing:
            - simulation_state_data (dict): Simulation state dictionary.
            - validator_state_data (dict): Validator state dictionary.
            - prep_time (float): Total time spent preparing the data in seconds.
    """
    start = time.time()
    bt.logging.debug("Preparing state for saving (sync)...")

    simulation_state_data = self.engine.build_simulation_state()

    bt.logging.debug("Creating snapshots...")
    snapshot_start = time.time()

    inventory_snapshot = snapshot_inventory_history(self)
    realized_pnl_snapshot = snapshot_realized_pnl_history(self)
    volume_sums_snapshots = snapshot_volume_sums(self)
    trade_volumes_snapshot = snapshot_trade_volumes(self)
    roundtrip_volumes_snapshot = snapshot_roundtrip_volumes(self)
    open_positions_snapshot = snapshot_open_positions(self)

    bt.logging.debug(f"Created snapshots ({time.time()-snapshot_start:.4f}s)")

    validator_state_data = build_validator_state(
        self,
        inventory_snapshot,
        realized_pnl_snapshot,
        volume_sums_snapshots,
        trade_volumes_snapshot,
        roundtrip_volumes_snapshot,
        open_positions_snapshot
    )

    prep_time = time.time() - start
    bt.logging.info(f"Prepared save data sync ({prep_time:.4f}s)")
    return simulation_state_data, validator_state_data, prep_time


async def construct_save_data_async(self: Validator):
    """Asynchronously build all state data for saving.

    Constructs both simulation and validator state by running each step method
    in an executor thread pool and yielding control between steps. This allows
    other async operations to continue while state preparation is in progress.

    Returns:
        tuple: Three-element tuple containing:
            - simulation_state_data (dict): Simulation state dictionary.
            - validator_state_data (dict): Validator state dictionary.
            - prep_time (float): Total time spent preparing the data in seconds.
    """
    start = time.time()
    bt.logging.debug("Preparing state for saving (async)...")
    loop = asyncio.get_event_loop()

    simulation_state_data = await loop.run_in_executor(
        self.save_state_executor,
        self.engine.build_simulation_state
    )
    await asyncio.sleep(0)

    validator_state_data = await construct_validator_state_async(self)

    prep_time = time.time() - start
    bt.logging.info(f"Prepared save data async ({prep_time:.4f}s)")
    return simulation_state_data, validator_state_data, prep_time


async def construct_validator_state_async(self: Validator):
    """Build only the validator-state dict (snapshot hops on the save executor,
    yielding between steps). Split out of construct_save_data_async so the
    save-offload fallback can construct it without re-building simulation state."""
    loop = asyncio.get_event_loop()

    bt.logging.debug("Creating snapshots...")
    snapshot_start = time.time()

    inventory_snapshot = await loop.run_in_executor(
        self.save_state_executor,
        lambda: snapshot_inventory_history(self)
    )
    await asyncio.sleep(0)

    realized_pnl_snapshot = await loop.run_in_executor(
        self.save_state_executor,
        lambda: snapshot_realized_pnl_history(self)
    )
    await asyncio.sleep(0)

    volume_sums_snapshots = await loop.run_in_executor(
        self.save_state_executor,
        lambda: snapshot_volume_sums(self)
    )
    await asyncio.sleep(0)

    trade_volumes_snapshot = await loop.run_in_executor(
        self.save_state_executor,
        lambda: snapshot_trade_volumes(self)
    )
    await asyncio.sleep(0)

    roundtrip_volumes_snapshot = await loop.run_in_executor(
        self.save_state_executor,
        lambda: snapshot_roundtrip_volumes(self)
    )
    await asyncio.sleep(0)

    open_positions_snapshot = await loop.run_in_executor(
        self.save_state_executor,
        lambda: snapshot_open_positions(self)
    )
    await asyncio.sleep(0)

    bt.logging.debug(f"Created snapshots ({time.time()-snapshot_start:.4f}s)")

    return build_validator_state(
        self,
        inventory_snapshot,
        realized_pnl_snapshot,
        volume_sums_snapshots,
        trade_volumes_snapshot,
        roundtrip_volumes_snapshot,
        open_positions_snapshot
    )


def defragment_histories(self: Validator):
    """Rebuild history dicts to eliminate memory fragmentation.

    Reconstructs all major history and volume tracking dictionaries to reduce
    memory fragmentation that accumulates over time from repeated insertions
    and deletions. This helps maintain consistent memory usage patterns.

    The following structures are defragmented:
        - inventory_history
        - realized_pnl_history
        - trade_volumes
        - roundtrip_volumes
        - volume_sums (all variants)

    Side Effects:
        Triggers garbage collection after rebuilding all structures.
    """
    start = time.time()
    bt.logging.info("Defragmenting history structures...")

    # Rebuild inventory_history
    new_inv = {}
    for uid, hist in self.inventory_history.items():
        if hist:
            new_inv[uid] = {ts: dict(books) for ts, books in hist.items()}
    self.inventory_history = new_inv

    # Rebuild realized_pnl_history
    new_pnl = defaultdict(lambda: defaultdict(dict))
    for uid, timestamps in self.realized_pnl_history.items():
        for ts, books in timestamps.items():
            new_pnl[uid][ts] = dict(books)
    self.realized_pnl_history = new_pnl

    # Rebuild trade_volumes
    new_volumes = {}
    for uid, books in self.trade_volumes.items():
        new_volumes[uid] = {}
        for book_id, roles in books.items():
            new_volumes[uid][book_id] = {
                role: dict(volumes) for role, volumes in roles.items()
            }
    self.trade_volumes = new_volumes

    # Rebuild roundtrip_volumes
    new_rt = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for uid, books in self.roundtrip_volumes.items():
        for book_id, volumes in books.items():
            for ts, vol in volumes.items():
                new_rt[uid][book_id][ts] = vol
    self.roundtrip_volumes = new_rt

    # Rebuild volume sums (convert defaultdict to regular dict then back)
    self.volume_sums = defaultdict(lambda: defaultdict(float), {
        uid: dict(books) for uid, books in self.volume_sums.items()
    })
    self.maker_volume_sums = defaultdict(lambda: defaultdict(float), {
        uid: dict(books) for uid, books in self.maker_volume_sums.items()
    })
    self.taker_volume_sums = defaultdict(lambda: defaultdict(float), {
        uid: dict(books) for uid, books in self.taker_volume_sums.items()
    })
    self.self_volume_sums = defaultdict(lambda: defaultdict(float), {
        uid: dict(books) for uid, books in self.self_volume_sums.items()
    })
    self.fee_sums = defaultdict(lambda: defaultdict(float), {
        uid: dict(books) for uid, books in getattr(self, 'fee_sums', {}).items()
    })
    self.roundtrip_volume_sums = defaultdict(lambda: defaultdict(float), {
        uid: dict(books) for uid, books in self.roundtrip_volume_sums.items()
    })

    gc.collect()
    bt.logging.info(f"Defragmentation complete ({time.time()-start:.4f}s)")


# How many levels _stream_pack recurses before packing a value whole. 2 =>
# top-level map + one level into each value (per-UID for the {uid: {...}} state
# dicts), so a huge value like realized_pnl_history is packed as ~N per-UID
# calls (~10-30ms each) instead of one multi-second call that holds the GIL
# solid and freezes the event loop for the whole save.
_SAVE_STREAM_DEPTH = 2

# Save every Nth scoring interval (default 2 = every 10th sim update at the
# 5s scoring interval). The ~480MB validator save spans 9-16s of prep+pack+io
# per firing; at every-interval cadence it was the largest single contributor
# to the background GIL/io load that stalls scoring-round receives. Trade-off:
# a crash loses up to N intervals of scoring history. Override via env.
_SAVE_EVERY_INTERVALS = max(1, int(os.environ.get("SAVE_EVERY_INTERVALS", "2")))


def _stream_pack(packer, write, obj, depth):
    """Write msgpack bytes for `obj` via `write(bytes)`, BYTE-IDENTICAL to
    packer.pack(obj) / msgpack.packb(obj, use_bin_type=True).

    For a dict/list at depth>0, emit the map/array header then stream each
    element (recursing with depth-1) instead of packing the container in one
    call. A msgpack map/array is exactly its header followed by its packed
    elements in iteration order, so the concatenated stream is identical to a
    single pack() — but it splits one huge C-level pack() (which holds the GIL
    for its entire duration) into many small ones with GIL-switch points
    between, so a large state value can't monopolise the GIL during a save.
    At depth 0, or for scalars, packs the object whole. Returns bytes written.
    """
    total = 0
    if depth > 0 and isinstance(obj, dict):
        chunk = packer.pack_map_header(len(obj))
        write(chunk)
        total += len(chunk)
        for k, v in obj.items():
            chunk = packer.pack(k)
            write(chunk)
            total += len(chunk)
            total += _stream_pack(packer, write, v, depth - 1)
    elif depth > 0 and isinstance(obj, (list, tuple)):
        chunk = packer.pack_array_header(len(obj))
        write(chunk)
        total += len(chunk)
        for item in obj:
            total += _stream_pack(packer, write, item, depth - 1)
    else:
        chunk = packer.pack(obj)
        write(chunk)
        total += len(chunk)
    return total


async def save_state_async(self: Validator) -> bool:
    """
    Saves simulation and validator state asynchronously via executor workers.
    Uses atomic file writes (temp + rename) to prevent corruption.
    Backups are organized by simulation ID and timestamp.

    Returns:
        bool: True if state saved successfully, False otherwise.
    """
    if self.shared_state_saving:
        bt.logging.warning(f"Skipping save at step {self.step} — previous save still running.")
        return False

    self.shared_state_saving = True
    if not hasattr(self, '_reward_lock'):
        self._reward_lock = asyncio.Lock()

    try:
        bt.logging.info(f"Starting state saving for step {self.step}...")
        total_start = time.time()

        save_step = self.step
        save_timestamp = self.simulation_timestamp
        sim_id = self.simulation.simulation_id
        # Backup marker: simulation timestamp (ns) for sim mode, block number for exchange
        _is_exchange_mode = (sim_id == "exchange")
        backup_marker = getattr(self, 'current_block', self.step) if _is_exchange_mode else save_timestamp

        # Defragment at the start of each simulation hour
        simulation_hour = save_timestamp // 3600_000_000_000
        last_defrag_hour = getattr(self, '_last_defrag_hour', -1)

        if simulation_hour != last_defrag_hour:
            bt.logging.info(f"Simulation hour {simulation_hour} - triggering defragmentation")
            async with self._reward_lock:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.save_state_executor,
                    lambda: defragment_histories(self)
                )
            self._last_defrag_hour = simulation_hour

        # Save offload (Stage B v2): when the cutover scoring service is live,
        # the child writes the heavy validator-state file from its parity-
        # verified replica — main ships only the light main-only fields and
        # skips the multi-second snapshot+pack GIL hold entirely. Any failure
        # (child dead, uninitialized, timeout, write error) falls back to the
        # in-process path below. SAVE_OFFLOAD=0 disables.
        _shadow = getattr(self, '_scoring_shadow', None)
        _offload_req = None
        if (
            getattr(self, '_scoring_proc_cutover', False)
            and _shadow is not None
            and not _is_exchange_mode
            and os.environ.get("SAVE_OFFLOAD", "1") != "0"
        ):
            _offload_req = _shadow.request_save(self.validator_state_file, build_save_light_fields(self))

        validator_state_data = None
        if _offload_req is not None:
            bt.logging.info(f"[SAVE-OFFLOAD] validator state delegated to scoring service (ts={_offload_req[0]})")
            prep_start = time.time()
            async with self._reward_lock:
                simulation_state_data = await asyncio.get_event_loop().run_in_executor(
                    self.save_state_executor,
                    self.engine.build_simulation_state
                )
            prep_time = time.time() - prep_start
        else:
            async with self._reward_lock:
                total_inv_entries = sum(len(hist) for hist in self.inventory_history.values())
                total_pnl_entries = sum(len(hist) for hist in self.realized_pnl_history.values())
                total_vol_entries = sum(
                    sum(len(roles.get('total', {})) for roles in books.values())
                    for books in self.trade_volumes.values()
                )

                bt.logging.info(
                    f"Save prep: {total_inv_entries} inventory entries, "
                    f"{total_pnl_entries} P&L entries, {total_vol_entries} volume entries"
                )

                simulation_state_data, validator_state_data, prep_time = await construct_save_data_async(self)

        async def wait_for_query_and_receive(process):
            """
            Wait for both query completion AND state reception before continuing.
            This prevents IO contention between mmap reads and msgpack serialization.
            """
            wait_start = time.time()
            waited = False
            while self.querying:
                if not waited:
                    bt.logging.debug(f"Waiting for query to complete before {process}...")
                    waited = True
                await asyncio.sleep(0.05)
            while self._receiving_state:
                if not waited:
                    bt.logging.debug(f"Waiting for state reception to complete before {process}...")
                    waited = True
                await asyncio.sleep(0.05)
            wait_time = time.time() - wait_start
            if wait_time > 0.01:
                bt.logging.info(f"Waited {wait_time:.4f}s for query/reception to complete before {process}")
            await asyncio.sleep(0)

        async def async_save_worker():
            """Non-blocking async file I/O with atomic writes."""
            try:
                loop = asyncio.get_event_loop()

                def _pack_stream_fsync(path, obj):
                    """Stream-pack `obj` straight into `path` (via _stream_pack)
                    and fsync.

                    Byte-identical to writing msgpack.packb(obj, use_bin_type=True)
                    — see _stream_pack. Streaming avoids materialising the full
                    multi-hundred-MB intermediate buffer, overlaps serialization
                    with the file write, and (via _stream_pack's recursion) keeps
                    any single C-level pack() call small so it can't hold the GIL
                    solid and freeze the event loop mid-save. fsync BEFORE the
                    atomic os.replace guarantees the temp file's bytes are durable
                    before it takes the state file's place — without it, a power
                    loss right after the rename could leave a truncated state with
                    the previous good file gone. Returns bytes written.
                    """
                    packer = msgpack.Packer(use_bin_type=True)
                    with open(path, 'wb', buffering=1024 * 1024) as f:
                        total = _stream_pack(packer, f.write, obj, _SAVE_STREAM_DEPTH)
                        f.flush()
                        os.fsync(f.fileno())
                    return total

                await wait_for_query_and_receive('serializing simulation state')
                sim_start = time.time()
                sim_temp = f"{self.simulation_state_file}.tmp.{save_timestamp}"
                try:
                    sim_total = await loop.run_in_executor(
                        self.save_state_executor,
                        _pack_stream_fsync, sim_temp, simulation_state_data)
                    await aiofiles.os.replace(sim_temp, self.simulation_state_file)
                except Exception as ex:
                    if os.path.exists(sim_temp):
                        try:
                            await aiofiles.os.remove(sim_temp)
                        except Exception:
                            pass
                    raise ex
                sim_time = time.time() - sim_start

                _val_data = validator_state_data
                _child_result = None
                if _offload_req is not None:
                    val_start = time.time()

                    def _wait_child(fut=_offload_req[1]):
                        try:
                            return fut.result(timeout=180.0)
                        except Exception:
                            return None

                    _child_result = await loop.run_in_executor(self.save_state_executor, _wait_child)
                    if _child_result is not None:
                        val_total, _child_io = _child_result
                        val_time = time.time() - val_start
                        bt.logging.info(
                            f"[SAVE-OFFLOAD] validator state written by scoring service "
                            f"({val_total/1024/1024:.2f}MB, child io={_child_io:.2f}s, waited {val_time:.2f}s)"
                        )
                    else:
                        bt.logging.warning(
                            "[SAVE-OFFLOAD] scoring service save failed — in-process fallback"
                        )
                        async with self._reward_lock:
                            _val_data = await construct_validator_state_async(self)

                if _child_result is None:
                    await wait_for_query_and_receive('serializing validator state')
                    val_start = time.time()
                    val_temp = f"{self.validator_state_file}.tmp.{save_timestamp}"
                    try:
                        val_total = await loop.run_in_executor(
                            self.save_state_executor,
                            _pack_stream_fsync, val_temp, _val_data)
                        await aiofiles.os.replace(val_temp, self.validator_state_file)
                    except Exception as ex:
                        if os.path.exists(val_temp):
                            try:
                                await aiofiles.os.remove(val_temp)
                            except Exception:
                                pass
                        raise ex
                    val_time = time.time() - val_start

                sim_mb = sim_total / 1024 / 1024
                val_mb = val_total / 1024 / 1024
                bt.logging.info(
                    f"Serialized: sim={sim_mb:.2f}MB, val={val_mb:.2f}MB "
                    f"({sim_time + val_time:.4f}s, streamed)"
                )

                return {
                    'success': True,
                    'simulation_save_time': sim_time,
                    'validator_save_time': val_time,
                    'io_time': sim_time + val_time,
                    'sim_size_mb': sim_mb,
                    'val_size_mb': val_mb
                }
            except Exception as ex:
                return {
                    'success': False,
                    'error': str(ex),
                    'traceback': traceback.format_exc()
                }

        result = await async_save_worker()
        total_time = time.time() - total_start

        if result['success']:
            bt.logging.success(
                f"State saved: simulation ({result['simulation_save_time']:.4f}s, {result['sim_size_mb']:.2f}MB), "
                f"validator ({result['validator_save_time']:.4f}s, {result['val_size_mb']:.2f}MB), "
                f"total ({total_time:.4f}s, prep={prep_time:.4f}s, io={result['io_time']:.4f}s)"
            )

            await wait_for_query_and_receive('creating state backups')
            max_backups = 5
            for state_file in [self.simulation_state_file, self.validator_state_file]:
                if os.path.exists(state_file):
                    backup = f"{state_file}.{backup_marker}"
                    try:
                        # Hardlink instead of a byte copy: saves never modify the
                        # state file in place (always tmp + os.replace, i.e. a
                        # fresh inode each save), so a linked backup keeps the
                        # old bytes immutably. This avoids re-reading + re-
                        # writing the full state (~2x file size of I/O) after
                        # every save. os.link is a metadata-only syscall.
                        os.link(state_file, backup)
                        bt.logging.debug(f"Created backup (hardlink): {backup}")
                    except FileExistsError:
                        bt.logging.debug(f"Backup already exists: {backup}")
                    except OSError as ex:
                        # Filesystem without hardlink support — fall back to a
                        # full copy so backups never silently stop.
                        bt.logging.warning(
                            f"Hardlink backup failed ({ex}); falling back to copy")
                        try:
                            async with aiofiles.open(state_file, 'rb') as src:
                                content = await src.read()
                            async with aiofiles.open(backup, 'wb') as dst:
                                await dst.write(content)
                            bt.logging.debug(f"Created backup (copy): {backup}")
                        except Exception as ex2:
                            bt.logging.warning(
                                f"Failed to create backup {backup}: {ex2}")
                    state_path = Path(state_file)
                    try:
                        all_backups = []
                        for backup_file in state_path.parent.glob(f"{state_path.name}.*"):
                            try:
                                marker = int(backup_file.name.split('.')[-1])
                                all_backups.append((backup_file, marker))
                            except (ValueError, IndexError):
                                continue
                        all_backups.sort(key=lambda x: x[1], reverse=True)
                        for old_backup, _ in all_backups[max_backups:]:
                            try:
                                await aiofiles.os.remove(old_backup)
                                bt.logging.debug(f"Removed old backup: {old_backup.name}")
                            except Exception as ex:
                                bt.logging.warning(f"Failed to remove {old_backup.name}: {ex}")
                    except Exception as ex:
                        bt.logging.warning(f"Failed to clean old backups: {ex}")
        else:
            self.pagerduty_alert(
                f"Failed to save state for step {save_step}: {result['error']}",
                details={"trace": result.get('traceback')}
            )
        return result['success']

    except Exception as ex:
        self.pagerduty_alert(
            f"Failed to prepare state for save: {ex}",
            details={"trace": traceback.format_exc()}
        )
        return False
    finally:
        self.shared_state_saving = False
        bt.logging.debug(f"Released save lock for step {save_step}")


def schedule_save(self: Validator) -> None:
    """
    Schedules the asynchronous state-saving coroutine on the main event loop.

    Behavior:
        - Executes only at specific scoring intervals.
        - Ensures no previous save task is still running.
        - Dispatches `save_state_async()` thread-safely from the maintenance thread.

    Returns:
        None
    """
    if not self.last_state or self.last_state.timestamp % (_SAVE_EVERY_INTERVALS * self.config.scoring.interval) != 4_000_000_000:
        return
    if self.shared_state_saving:
        bt.logging.warning(f"Skipping save at step {self.step} — previous save still running.")
        return
    if self.querying:
        bt.logging.warning(f"Skipping save at step {self.step} — query in progress")
        return
    import threading
    bt.logging.debug(f"[SAVE] Scheduling from thread: {threading.current_thread().name}")
    bt.logging.debug(f"[SAVE] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
    self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(save_state_async(self)))


async def save_state_sync(self: Validator):
    """Save validator state synchronously from an async context.

    The method performs the following steps:
        1. Constructs simulation and validator state data synchronously
        2. Writes both state files to disk via save_state_worker
        3. Sends PagerDuty alert if save operation fails

    Raises:
        Exception: If state construction or save operation fails. Exceptions
            are caught, logged, and trigger PagerDuty alerts.

    Side Effects:
        - Writes simulation_state_file and validator_state_file to disk
        - Sends PagerDuty alerts on failure
        - Logs save operation status and timing
    """
    try:
        bt.logging.info("Saving state (sync)...")
        simulation_state_data, validator_state_data, prep_time = construct_save_data_sync(self)
        result = save_state_worker(
            simulation_state_data,
            validator_state_data,
            self.simulation_state_file,
            self.validator_state_file
        )
        if result['success']:
            bt.logging.success("State saved directly!")
        else:
            self.pagerduty_alert(f"Direct save failed: {result['error']}")
    except Exception as ex:
        self.pagerduty_alert(f"Error in direct save: {ex}", details={"trace": traceback.format_exc()})


def migrate_sampling_interval(self: Validator, old_interval: int, new_interval: int):
    """
    Re-align trade volumes from old sampling interval to new interval.

    Uses floor bucketing: trades are assigned to the start of their interval bucket.

    Args:
        old_interval: Previous sampling interval (e.g., 600_000_000_000 for 10 min)
        new_interval: New sampling interval (e.g., 3600_000_000_000 for 1 hour)
    """
    bt.logging.info(
        f"Migrating sampling intervals: "
        f"{old_interval}ns ({old_interval / 1e9:.0f}s) → "
        f"{new_interval}ns ({new_interval / 1e9:.0f}s)"
    )

    start_time = time.time()
    volume_decimals = self.simulation.volumeDecimals

    migrated_trade_entries = 0
    migrated_rt_entries = 0

    bt.logging.info("Migrating trade_volumes...")
    for uid in range(self.effective_max_uids):
        if uid not in self.trade_volumes:
            continue
        for book_id in list(self.trade_volumes[uid].keys()):
            for role in ['total', 'maker', 'taker', 'self']:
                old_volumes = self.trade_volumes[uid][book_id][role]
                if not old_volumes:
                    continue
                new_volumes = {}
                for old_ts, volume in old_volumes.items():
                    new_ts = (old_ts // new_interval) * new_interval
                    if new_ts not in new_volumes:
                        new_volumes[new_ts] = 0.0
                    new_volumes[new_ts] += volume
                    migrated_trade_entries += 1
                for ts in new_volumes:
                    new_volumes[ts] = round(new_volumes[ts], volume_decimals)
                self.trade_volumes[uid][book_id][role] = new_volumes
    bt.logging.info(f"Migrated {migrated_trade_entries} trade_volume entries")

    bt.logging.info("Migrating roundtrip_volumes...")
    new_roundtrip_volumes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for uid in range(self.effective_max_uids):
        if uid not in self.roundtrip_volumes:
            continue
        for book_id in list(self.roundtrip_volumes[uid].keys()):
            old_rt_volumes = self.roundtrip_volumes[uid][book_id]
            if not old_rt_volumes:
                continue
            new_rt_volumes = {}
            for old_ts, volume in old_rt_volumes.items():
                new_ts = (old_ts // new_interval) * new_interval
                if new_ts not in new_rt_volumes:
                    new_rt_volumes[new_ts] = 0.0
                new_rt_volumes[new_ts] += volume
                migrated_rt_entries += 1
            for ts in new_rt_volumes:
                new_rt_volumes[ts] = round(new_rt_volumes[ts], volume_decimals)
            for ts, vol in new_rt_volumes.items():
                new_roundtrip_volumes[uid][book_id][ts] = vol

    self.roundtrip_volumes = new_roundtrip_volumes
    bt.logging.info(f"Migrated {migrated_rt_entries} roundtrip_volume entries")

    elapsed = time.time() - start_time
    bt.logging.success(
        f"Sampling interval migration complete ({elapsed:.4f}s): "
        f"{migrated_trade_entries + migrated_rt_entries} total entries migrated"
    )




def _try_load_state_file(filepath, file_type="simulation"):
    """
    Attempt to load a state file, trying backups if the primary fails.
    Backups are organized by simulation ID and timestamp.
    Format: <filename>.<sim_id>.<timestamp>

    Args:
        filepath: Path to the primary state file
        file_type: "simulation" or "validator" for logging

    Returns:
        Unpacked state dict or None if all attempts fail
    """
    if os.path.exists(filepath):
        try:
            bt.logging.info(f"Loading {file_type} state from {filepath}...")
            with open(filepath, 'rb') as file:
                byte_data = file.read()
            use_list = (file_type == "simulation")
            state = msgpack.unpackb(byte_data, use_list=use_list, strict_map_key=False)
            bt.logging.success(f"Loaded {file_type} state from primary file")
            return state
        except Exception as ex:
            bt.logging.error(f"Failed to load {file_type} state from {filepath}: {ex}")
            bt.logging.warning("Attempting to load from backups...")
    state_path = Path(filepath)
    # Collect both new-format (single marker suffix) and legacy (two-part) backups
    candidate_files = set(state_path.parent.glob(f"{state_path.name}.*"))

    parsed_backups = []
    for backup_file in candidate_files:
        try:
            suffix = backup_file.name[len(state_path.name) + 1:]  # strip "basename."
            if not suffix:
                continue
            parts = suffix.split('.')
            # New format: <marker> — numeric timestamp or block number
            if len(parts) == 1 and parts[0].isdigit():
                parsed_backups.append({'path': backup_file, 'marker': int(parts[0])})
            # Legacy format: <sim_id>.<timestamp> — keep compatible
            elif len(parts) == 2 and parts[1].isdigit():
                parsed_backups.append({'path': backup_file, 'marker': int(parts[1])})
        except Exception:
            bt.logging.debug(f"Skipping malformed backup filename: {backup_file.name}")
            continue

    if not parsed_backups:
        bt.logging.warning(f"No backup files found for {filepath}")
        return None

    sorted_backups = sorted(parsed_backups, key=lambda x: x['marker'], reverse=True)
    bt.logging.info(f"Found {len(sorted_backups)} valid backups, trying most recent first…")

    for backup_info in sorted_backups:
        backup_file = backup_info['path']
        try:
            bt.logging.info(f"Attempting backup: {backup_file.name} (marker={backup_info['marker']})")
            with open(backup_file, 'rb') as file:
                byte_data = file.read()
            use_list = (file_type == "simulation")
            state = msgpack.unpackb(byte_data, use_list=use_list, strict_map_key=False)
            bt.logging.success(f"Successfully loaded from backup: {backup_file.name}")

            if os.path.exists(filepath):
                corrupted_backup = f"{filepath}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.corrupted"
                try:
                    shutil.copy2(filepath, corrupted_backup)
                    bt.logging.info(f"Saved corrupted primary file as: {corrupted_backup}")
                except Exception as ex:
                    bt.logging.warning(f"Failed to backup corrupted file: {ex}")
            bt.logging.info(f"Restoring {backup_file.name} as primary state file…")
            shutil.copy2(backup_file, filepath)
            bt.logging.success(f"Restored backup to {filepath}")
            return state
        except Exception as ex:
            bt.logging.warning(f"Backup {backup_file.name} failed: {ex}")
            continue
    bt.logging.error(f"All {len(sorted_backups)} backups failed for {file_type} state")
    return None





def _restore_trade_volumes(self, validator_state, book_ids, book_ids_set):
    """Reconstruct trade_volumes and per-role volume sums from saved validator
    state, migrate the sampling interval when it changed, and downsample
    inventory_history. Pure extraction from _load_validator_state; logic unchanged.
    """
    bt.logging.info("Processing trade volumes...")
    loaded_trade_vols = validator_state.get("trade_volumes", {})
    self.trade_volumes = {
        uid: {bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
            for bookId in book_ids}
        for uid in range(self.effective_max_uids)
    }
    for uid, books in loaded_trade_vols.items():
        if uid < self.effective_max_uids:
            self.trade_volumes[uid] = books
        else:
            bt.logging.debug(f"Skipping trade_volumes for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

    bt.logging.info("Processing trade volume histories...")
    reorg = False
    for uid in range(self.effective_max_uids):
        if uid not in self.trade_volumes:
            self.trade_volumes[uid] = {
                bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                for bookId in book_ids
            }

        for bookId in list(self.trade_volumes[uid].keys()):
            if 'total' not in self.trade_volumes[uid][bookId]:
                if not reorg:
                    bt.logging.info("Optimizing miner volume history structures...")
                    reorg = True
                volumes = {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                for time_key, role_volume in self.trade_volumes[uid][bookId].items():
                    sampled_time = math.ceil(time_key / self.config.scoring.activity.trade_volume_sampling_interval) * self.config.scoring.activity.trade_volume_sampling_interval
                    for role, volume in role_volume.items():
                        if sampled_time not in volumes[role]:
                            volumes[role][sampled_time] = 0.0
                        volumes[role][sampled_time] += volume
                self.trade_volumes[uid][bookId] = {
                    role: {time_key: round(volumes[role][time_key], self.simulation.volumeDecimals) for time_key in volumes[role]}
                    for role in volumes
                }

        for bookId in book_ids:
            if bookId not in self.trade_volumes[uid]:
                self.trade_volumes[uid][bookId] = {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
        self.trade_volumes[uid] = {k: v for k, v in self.trade_volumes[uid].items() if k in book_ids_set}

        if uid not in self.activity_factors:
            self.activity_factors[uid] = {bookId: 0.0 for bookId in book_ids}
        for bookId in book_ids:
            if bookId not in self.activity_factors[uid]:
                self.activity_factors[uid][bookId] = 0.0
        self.activity_factors[uid] = {k: v for k, v in self.activity_factors[uid].items() if k in book_ids_set}

        if uid not in self.pnl_factors:
            self.pnl_factors[uid] = {bookId: 1.0 for bookId in book_ids}
        for bookId in book_ids:
            if bookId not in self.pnl_factors[uid]:
                self.pnl_factors[uid][bookId] = 1.0
        self.pnl_factors[uid] = {k: v for k, v in self.pnl_factors[uid].items() if k in book_ids_set}

    def load_volume_sums(data, name, valid_ids):
        """Load volume sums and prune books not in valid_ids.

        Args:
            data: The saved-state payload.
            name: The field to load.
            valid_ids: Book ids to keep; others are pruned.
        """
        result = defaultdict(lambda: defaultdict(float))

        if name not in data:
            bt.logging.info(f"No {name} in saved state, initializing empty")
            return result

        volume_data = data[name]

        if volume_data:
            first_key = next(iter(volume_data.keys()))

            if isinstance(first_key, (tuple, list)) and len(first_key) == 2:
                bt.logging.info(f"Converting {name} from old tuple-key format to nested dict...")
                for key, vol in volume_data.items():
                    uid, book_id = key
                    if book_id in valid_ids and uid < self.effective_max_uids:
                        result[uid][book_id] = vol
                bt.logging.debug(f"Converted {len(volume_data)} entries in {name}")

            elif isinstance(first_key, int):
                first_value = volume_data[first_key]

                if isinstance(first_value, dict):
                    bt.logging.debug(f"Loading {name} in nested dict format...")
                    for uid, books in volume_data.items():
                        if uid < self.effective_max_uids:
                            for book_id, vol in books.items():
                                if book_id in valid_ids:
                                    result[uid][book_id] = vol
                        else:
                            bt.logging.debug(f"Skipping {name} for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
                else:
                    bt.logging.warning(f"Unexpected format for {name}: single-level dict")
                    if valid_ids and first_key < self.effective_max_uids:
                        result[first_key][next(iter(valid_ids))] = first_value
            else:
                bt.logging.warning(f"Unknown format for {name}, initializing empty")

        return result

    bt.logging.info("Processing volume sums...")
    self.volume_sums = load_volume_sums(validator_state, "volume_sums", book_ids_set)
    self.maker_volume_sums = load_volume_sums(validator_state, "maker_volume_sums", book_ids_set)
    self.taker_volume_sums = load_volume_sums(validator_state, "taker_volume_sums", book_ids_set)
    self.self_volume_sums = load_volume_sums(validator_state, "self_volume_sums", book_ids_set)
    self.fee_sums = load_volume_sums(validator_state, "fee_sums", book_ids_set)
    self.roundtrip_volume_sums = load_volume_sums(validator_state, "roundtrip_volume_sums", book_ids_set)
    # De-beta fill-stream accumulators — restore the TIMESTAMPED HISTORIES so a restart continues the
    # windowed making/skill/counterparty assessment instead of zeroing it (empty when de-beta was off;
    # harmless). Running sums are rebuilt from the histories (running == sum(history within window)), so a
    # restart cannot desync them. The reconstruction STATE (per-book inventory + last price) is restored
    # directly so the MTM path continues correctly.
    def _load_hist3(name):
        return {int(k1): {int(k2): {int(ts): float(v) for ts, v in (tsd or {}).items()}
                          for k2, tsd in (d2 or {}).items()}
                for k1, d2 in (validator_state.get(name) or {}).items()}

    def _load_hist2(name):
        return {int(k): {int(ts): float(v) for ts, v in (tsd or {}).items()}
                for k, tsd in (validator_state.get(name) or {}).items()}

    def _dd2(sums):
        return defaultdict(lambda: defaultdict(float),
                           {u: defaultdict(float, b) for u, b in sums.items()})

    self.debeta_capbuy_hist = _load_hist3("debeta_capbuy_hist")
    self.debeta_capsell_hist = _load_hist3("debeta_capsell_hist")
    self.debeta_mtm_hist = _load_hist3("debeta_mtm_hist")
    self.debeta_invsum_hist = _load_hist3("debeta_invsum_hist")
    self.debeta_cp_hist = _load_hist3("debeta_cp_hist")
    self.debeta_invn_hist = _load_hist2("debeta_invn_hist")
    self.debeta_drift_hist = _load_hist2("debeta_drift_hist")
    # rebuild running sums from the histories
    self.capture_buy_sums = _dd2(sum_hist_2level(self.debeta_capbuy_hist))
    self.capture_sell_sums = _dd2(sum_hist_2level(self.debeta_capsell_hist))
    self.debeta_mtm = _dd2(sum_hist_2level(self.debeta_mtm_hist))
    self.debeta_invsum = _dd2(sum_hist_2level(self.debeta_invsum_hist))
    self.debeta_cp = sum_hist_2level(self.debeta_cp_hist)
    self.debeta_invn = sum_hist_1level(self.debeta_invn_hist)
    self.debeta_drift = sum_hist_1level(self.debeta_drift_hist)
    # reconstruction STATE (pfirst is vestigial in the windowed path — drift replaces it)
    self.debeta_pfirst = {}
    self.debeta_plast = {int(b): float(v) for b, v in (validator_state.get("debeta_plast") or {}).items()}
    self.debeta_inv = defaultdict(lambda: defaultdict(float), {
        int(b): defaultdict(float, {int(u): float(v) for u, v in (us or {}).items()})
        for b, us in (validator_state.get("debeta_inv") or {}).items()
    })

    bt.logging.info("Processing roundtrip volumes...")
    self.roundtrip_volumes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    if "roundtrip_volumes" in validator_state:
        for uid, books in validator_state["roundtrip_volumes"].items():
            if uid < self.effective_max_uids:
                for book_id, volumes in books.items():
                    if book_id not in book_ids_set:
                        continue
                    for timestamp, volume in volumes.items():
                        self.roundtrip_volumes[uid][book_id][timestamp] = volume
            else:
                bt.logging.debug(f"Skipping roundtrip_volumes for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

    bt.logging.info("Processing realized PnL history...")
    self.realized_pnl_history = defaultdict(lambda: defaultdict(dict))
    # Running totals over realized_pnl_history for the MVTRX push payload's
    # agent_pnl / agent_pnl_book fields. Maintained incrementally by
    # trade.py at every write/prune so the push builder doesn't have to
    # re-walk the entire O(N*T*B) history on every state cycle. Rebuilt
    # from realized_pnl_history via bootstrap_pnl_totals after load.
    self.agent_pnl_by_book = defaultdict(lambda: defaultdict(float))
    self.agent_pnl_total = defaultdict(float)
    if "realized_pnl_history" in validator_state:
        for uid, hist in validator_state["realized_pnl_history"].items():
            if uid < self.effective_max_uids:
                for timestamp, books in hist.items():
                    ts_pnl = {
                        book_id: round(pnl, self.simulation.volumeDecimals)
                        for book_id, pnl in books.items()
                        if book_id in book_ids_set and round(pnl, self.simulation.volumeDecimals) != 0.0
                    }
                    self.realized_pnl_history[uid][timestamp] = ts_pnl
            else:
                bt.logging.debug(f"Skipping realized_pnl_history for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

    bt.logging.info("Processing open positions...")
    self.open_positions = defaultdict(lambda: defaultdict(lambda: {
        'longs': deque(),
        'shorts': deque()
    }))
    if "open_positions" in validator_state:
        legacy_count = 0
        for uid, books in validator_state["open_positions"].items():
            if uid < self.effective_max_uids:
                for book_id, pos in books.items():
                    if book_id not in book_ids_set:
                        continue
                    longs = []
                    for p in pos['longs']:
                        if len(p) == 4:
                            longs.append(tuple(p))
                        elif len(p) == 3:
                            longs.append((*p, 0.0))
                            legacy_count += 1
                    shorts = []
                    for p in pos['shorts']:
                        if len(p) == 4:
                            shorts.append(tuple(p))
                        elif len(p) == 3:
                            shorts.append((*p, 0.0))
                            legacy_count += 1
                    self.open_positions[uid][book_id]['longs'] = deque(longs)
                    self.open_positions[uid][book_id]['shorts'] = deque(shorts)
            else:
                bt.logging.debug(f"Skipping open_positions for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
        if legacy_count > 0:
            bt.logging.info(f"Converted {legacy_count} legacy positions to new format")

    current_sampling_interval = self.config.scoring.activity.trade_volume_sampling_interval
    detected_old_interval = None
    bt.logging.info(f"Checking for sampling interval changes (current config: {current_sampling_interval}ns)...")
    for uid in self.trade_volumes:
        if detected_old_interval is not None:
            break
        for book_id in self.trade_volumes[uid]:
            if detected_old_interval is not None:
                break
            for role in ['total', 'maker', 'taker', 'self']:
                timestamps = sorted(self.trade_volumes[uid][book_id][role].keys())
                if len(timestamps) >= 3:
                    detected_old_interval = timestamps[2] - timestamps[1]
                    bt.logging.info(
                        f"Detected old sampling interval from trade_volumes[{uid}][{book_id}][{role}]: "
                        f"{detected_old_interval}ns"
                    )
                    break
    if detected_old_interval is None:
        for uid in self.roundtrip_volumes:
            if detected_old_interval is not None:
                break
            for book_id in self.roundtrip_volumes[uid]:
                timestamps = sorted(self.roundtrip_volumes[uid][book_id].keys())
                if len(timestamps) >= 2:
                    detected_old_interval = timestamps[1] - timestamps[0]
                    bt.logging.info(
                        f"Detected old sampling interval from roundtrip_volumes[{uid}][{book_id}]: "
                        f"{detected_old_interval}ns"
                    )
                    break
    if detected_old_interval is not None and detected_old_interval != current_sampling_interval:
        bt.logging.warning(
            f"Sampling interval mismatch detected! "
            f"Old: {detected_old_interval}ns ({detected_old_interval / 1e9:.0f}s), "
            f"New: {current_sampling_interval}ns ({current_sampling_interval / 1e9:.0f}s)"
        )
        migrate_sampling_interval(self, detected_old_interval, current_sampling_interval)
    elif detected_old_interval is None:
        bt.logging.info("No existing volume data found, no migration needed")
    else:
        bt.logging.info(f"Sampling interval unchanged ({current_sampling_interval}ns), no migration needed")

    bt.logging.info("Downsampling inventory_history to minimal storage...")
    downsampled_count = 0
    for uid in self.inventory_history:
        hist = self.inventory_history[uid]
        if not hist:
            continue
        if len(hist) <= 3:
            continue
        timestamps = sorted(hist.keys())
        minimal_hist = {
            timestamps[0]: hist[timestamps[0]],
            timestamps[-2]: hist[timestamps[-2]],
            timestamps[-1]: hist[timestamps[-1]]
        }
        old_size = len(hist)
        self.inventory_history[uid] = minimal_hist
        downsampled_count += (old_size - 3)
    bt.logging.info(
        f"Downsampled inventory_history: removed {downsampled_count} entries"
    )

def _load_validator_state(self):
    """Load and reconstruct validator state (scores, gentrx_scores, activity/pnl
    factors, kappa values, volume sums, histories) from the validator state file,
    handling .pt->msgpack conversion, effective_max_uids reshaping, and sampling-
    interval migration. Pure extraction of the validator half of load_state.
    """
    if os.path.exists(self.validator_state_file.replace('.mp', '.pt')):
        bt.logging.info("Pytorch validator state file exists - converting to msgpack...")
        pt_validator_state = torch.load(self.validator_state_file.replace('.mp', '.pt'), weights_only=False)
        pt_validator_state["scores"] = [score.item() for score in pt_validator_state['scores']]
        with open(self.validator_state_file, 'wb') as file:
            packed_data = msgpack.packb(
                pt_validator_state, use_bin_type=True
            )
            file.write(packed_data)
        os.rename(self.validator_state_file.replace('.mp', '.pt'), self.validator_state_file.replace('.mp', '.pt') + ".bak")
        bt.logging.info(f"Pytorch validator state file converted to msgpack at {self.validator_state_file}")

    book_ids     = list(self.engine.book_ids)
    book_ids_set = set(book_ids)

    def _default_kappa(bids):
        return {
            'books': {bookId: None for bookId in bids},
            'books_weighted': {bookId: 0.0 for bookId in bids},
            'total': None, 'average': None, 'median': None,
            'normalized_average': 0.0, 'normalized_median': 0.0,
            'normalized_total': 0.0,
            'activity_weighted_normalized_median': 0.0,
            'penalty': 0.0, 'score': 0.0,
        }

    if not self.config.neuron.reset:
        validator_state = _try_load_state_file(self.validator_state_file, "validator")
        if validator_state:
            bt.logging.info("Populating validator data...")
            self.step = validator_state["step"]
            self.simulation_timestamp = validator_state.get("simulation_timestamp", 0)
            self.hotkeys = validator_state["hotkeys"]
            self.deregistered_uids = list(validator_state.get("deregistered_uids", []))

            # Restore rolling request/timeout accumulators (persisted so a restart
            # doesn't wipe the ~100-request window and blackout the miner_gauges
            # requests/timeouts/call_time series in Grafana for ~12 min). Runs
            # after initialize_structures() seeded miner_stats for all UIDs, so
            # this overlays saved counts onto the fresh zeroed structure.
            _loaded_miner_stats = validator_state.get("miner_stats", {})
            if _loaded_miner_stats:
                if not isinstance(getattr(self, "miner_stats", None), dict):
                    self.miner_stats = {}
                _restored = 0
                for _uid_key, _st in _loaded_miner_stats.items():
                    if not isinstance(_st, dict):
                        continue
                    try:
                        _uid = int(_uid_key)
                    except (TypeError, ValueError):
                        continue
                    if _uid < 0 or _uid >= self.effective_max_uids:
                        continue
                    self.miner_stats[_uid] = {
                        "requests": int(_st.get("requests", 0)),
                        "timeouts": int(_st.get("timeouts", 0)),
                        "failures": int(_st.get("failures", 0)),
                        "rejections": int(_st.get("rejections", 0)),
                        "call_time": list(_st.get("call_time", [])),
                    }
                    _restored += 1
                bt.logging.info(f"Restored miner_stats for {_restored} UIDs")

            loaded_scores = validator_state["scores"]
            self.scores = torch.zeros(self.effective_max_uids, dtype=torch.float32, device=self.device)
            num_scores_to_copy = min(len(loaded_scores), self.effective_max_uids)
            self.scores[:num_scores_to_copy] = torch.tensor(loaded_scores[:num_scores_to_copy])

            if len(loaded_scores) > self.effective_max_uids:
                bt.logging.warning(
                    f"Loaded state has {len(loaded_scores)} scores but current effective_max_uids is {self.effective_max_uids}. "
                    f"Truncating scores for UIDs {self.effective_max_uids} onwards."
                )
            elif len(loaded_scores) < self.effective_max_uids:
                bt.logging.info(
                    f"Loaded state has {len(loaded_scores)} scores but current effective_max_uids is {self.effective_max_uids}. "
                    f"Initializing scores for UIDs {len(loaded_scores)} onwards as 0.0."
                )

            loaded_gentrx_scores = validator_state.get("gentrx_scores", [])
            self.gentrx_scores = torch.zeros(self.effective_max_uids, dtype=torch.float32, device=self.device)
            num_gentrx_to_copy = min(len(loaded_gentrx_scores), self.effective_max_uids)
            if num_gentrx_to_copy:
                self.gentrx_scores[:num_gentrx_to_copy] = torch.tensor(loaded_gentrx_scores[:num_gentrx_to_copy])

            _ema = validator_state.get("trading_score_ema", {}) or {}
            self._trading_score_ema = {int(u): float(v) for u, v in _ema.items()}
            self._trading_score_ema_ts = validator_state.get("trading_score_ema_ts", None)
            _ema_n = validator_state.get("trading_score_ema_n", {}) or {}
            self._trading_score_ema_n = {int(u): int(v) for u, v in _ema_n.items()}

            loaded_activity = validator_state.get("activity_factors", {})
            if loaded_activity and isinstance(list(loaded_activity.values())[0], float):
                loaded_activity = {
                    uid: {bookId: loaded_activity[uid] for bookId in book_ids}
                    for uid in loaded_activity
                }

            self.activity_factors = {
                uid: {bookId: 0.0 for bookId in book_ids}
                for uid in range(self.effective_max_uids)
            }
            for uid, books in loaded_activity.items():
                if uid < self.effective_max_uids:
                    self.activity_factors[uid] = books
                else:
                    bt.logging.debug(f"Skipping activity_factors for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

            loaded_pnl = validator_state.get("pnl_factors", {})
            if loaded_pnl and isinstance(list(loaded_pnl.values())[0], float):
                loaded_pnl = {
                    uid: {bookId: loaded_pnl[uid] for bookId in book_ids}
                    for uid in loaded_pnl
                }

            self.pnl_factors = {
                uid: {bookId: 1.0 for bookId in book_ids}
                for uid in range(self.effective_max_uids)
            }
            for uid, books in loaded_pnl.items():
                if uid < self.effective_max_uids:
                    self.pnl_factors[uid] = books
                else:
                    bt.logging.debug(f"Skipping pnl_factors for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

            loaded_inventory = validator_state.get("inventory_history", {})
            self.inventory_history = {uid: {} for uid in range(self.effective_max_uids)}
            for uid, hist in loaded_inventory.items():
                if uid < self.effective_max_uids:
                    self.inventory_history[uid] = hist
                else:
                    bt.logging.debug(f"Skipping inventory_history for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

            bt.logging.info("Processing inventory history...")
            for uid in self.inventory_history:
                for timestamp in self.inventory_history[uid]:
                    for bookId in book_ids:
                        if bookId not in self.inventory_history[uid][timestamp]:
                            self.inventory_history[uid][timestamp][bookId] = 0.0
                    self.inventory_history[uid][timestamp] = {
                        k: v for k, v in self.inventory_history[uid][timestamp].items()
                        if k in book_ids_set
                    }

            bt.logging.info("Processing kappa values...")
            if 'kappa_values' in validator_state:
                loaded_kappa = validator_state['kappa_values']
                self.kappa_values = {
                    uid: _default_kappa(book_ids)
                    for uid in range(self.effective_max_uids)
                }
                for uid, kappa_data in loaded_kappa.items():
                    if uid < self.effective_max_uids and kappa_data is not None:
                        kappa = kappa_data.copy()
                        if 'books' in kappa:
                            kappa['books'] = {
                                book_id: val for book_id, val in kappa['books'].items()
                                if book_id in book_ids_set
                            }
                            for book_id in book_ids:
                                if book_id not in kappa['books']:
                                    kappa['books'][book_id] = None
                        if 'books_weighted' in kappa:
                            kappa['books_weighted'] = {
                                book_id: val for book_id, val in kappa['books_weighted'].items()
                                if book_id in book_ids_set
                            }
                            for book_id in book_ids:
                                if book_id not in kappa['books_weighted']:
                                    kappa['books_weighted'][book_id] = 0.0
                        self.kappa_values[uid] = kappa
                    elif uid >= self.effective_max_uids:
                        bt.logging.debug(f"Skipping kappa_values for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
            elif 'sharpe_values' in validator_state:
                bt.logging.info("Converting sharpe_values to kappa_values format...")
                self.kappa_values = {
                    uid: _default_kappa(book_ids)
                    for uid in range(self.effective_max_uids)
                }
                for uid, sharpe_data in validator_state['sharpe_values'].items():
                    if uid < self.effective_max_uids and sharpe_data:
                        self.kappa_values[uid] = {
                            'books': sharpe_data.get('books_realized', {bookId: None for bookId in book_ids}),
                            'books_weighted': sharpe_data.get('books_weighted_realized', {bookId: 0.0 for bookId in book_ids}),
                            'total': sharpe_data.get('total_realized'),
                            'average': sharpe_data.get('average_realized'),
                            'median': sharpe_data.get('median_realized'),
                            'normalized_average': sharpe_data.get('normalized_average_realized', 0.0),
                            'normalized_median': sharpe_data.get('normalized_median_realized', 0.0),
                            'normalized_total': sharpe_data.get('normalized_total_realized', 0.0),
                            'activity_weighted_normalized_median': sharpe_data.get('activity_weighted_normalized_median_realized', 0.0),
                            'penalty': sharpe_data.get('penalty_realized', 0.0),
                            'score': sharpe_data.get('score_realized', 0.0),
                        }
                    elif uid >= self.effective_max_uids:
                        bt.logging.debug(f"Skipping sharpe_values conversion for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
            else:
                self.kappa_values = {
                    uid: _default_kappa(book_ids)
                    for uid in range(self.effective_max_uids)
                }

            loaded_unnorm = validator_state.get("unnormalized_scores", {})
            self.unnormalized_scores = {uid: 0.0 for uid in range(self.effective_max_uids)}
            for uid, score in loaded_unnorm.items():
                if uid < self.effective_max_uids:
                    self.unnormalized_scores[uid] = score
                else:
                    bt.logging.debug(f"Skipping unnormalized_scores for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

            _restore_trade_volumes(self, validator_state, book_ids, book_ids_set)

            # Rebuild the MVTRX push running totals from the freshly-loaded
            # realized_pnl_history. From here on, trade.py maintains them
            # incrementally — but at boot we need the full walk once.
            from taos.im.validator.trade import bootstrap_pnl_totals
            _bp_start = time.time()
            bootstrap_pnl_totals(self)
            bt.logging.info(
                f"Bootstrapped agent_pnl running totals for {len(self.agent_pnl_total)} UIDs "
                f"({time.time()-_bp_start:.3f}s)"
            )

            bt.logging.success(f"Loaded validator state for {self.effective_max_uids} UIDs")
        else:
            bt.logging.warning("All validator state files corrupted, initializing fresh state")
            validator_state = None
    else:
        validator_state = None

    if validator_state is None:
        if self.config.neuron.reset:
            bt.logging.warning("`neuron.reset is True, ignoring previous validator state")
        else:
            bt.logging.info("No valid validator state found, initializing new state")

        self.activity_factors = {
            uid: {bookId: 0.0 for bookId in book_ids}
            for uid in range(self.effective_max_uids)
        }
        self.pnl_factors = {
            uid: {bookId: 1.0 for bookId in book_ids}
            for uid in range(self.effective_max_uids)
        }
        self.inventory_history = {uid: {} for uid in range(self.effective_max_uids)}
        self.kappa_values = {
            uid: _default_kappa(book_ids)
            for uid in range(self.effective_max_uids)
        }
        self.unnormalized_scores = {uid: 0.0 for uid in range(self.effective_max_uids)}
        self.trade_volumes = {
            uid: {bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                for bookId in book_ids}
            for uid in range(self.effective_max_uids)
        }
        self.volume_sums = defaultdict(lambda: defaultdict(float))
        self.maker_volume_sums = defaultdict(lambda: defaultdict(float))
        self.taker_volume_sums = defaultdict(lambda: defaultdict(float))
        self.self_volume_sums = defaultdict(lambda: defaultdict(float))
        self.fee_sums = defaultdict(lambda: defaultdict(float))
        self.roundtrip_volumes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.roundtrip_volume_sums = defaultdict(lambda: defaultdict(float))
        self.realized_pnl_history = defaultdict(lambda: defaultdict(dict))
        # Running totals over realized_pnl_history for the MVTRX push payload's
        # agent_pnl / agent_pnl_book fields. Maintained incrementally by
        # trade.py at every write/prune so the push builder doesn't have to
        # re-walk the entire O(N*T*B) history on every state cycle. Rebuilt
        # from realized_pnl_history via bootstrap_pnl_totals after load.
        self.agent_pnl_by_book = defaultdict(lambda: defaultdict(float))
        self.agent_pnl_total = defaultdict(float)
        self.open_positions = defaultdict(lambda: defaultdict(lambda: {
            'longs': deque(),
            'shorts': deque()
        }))

def load_state(self: Validator) -> None:
    """
    Loads validator and simulation state from msgpack or legacy PyTorch files.

    Behavior:
        - Converts `.pt` state files to msgpack if detected.
        - Loads simulation variables (balances, trades, notices, timestamps).
        - Loads validator data (scores, activity, Kappa values, volumes).
        - Reconstructs missing fields or reshapes data when schema versions differ.
        - Handles changes in effective_max_uids (benchmark agent count changes).
        - Reinitializes state when `neuron.reset=True` or files are absent.

    Returns:
        None
    """
    if os.path.exists(self.simulation_state_file.replace('.mp', '.pt')):
        bt.logging.info("Pytorch simulation state file exists - converting to msgpack...")
        pt_simulation_state = torch.load(self.simulation_state_file.replace('.mp', '.pt'), weights_only=False)
        with open(self.simulation_state_file, 'wb') as file:
            packed_data = msgpack.packb(
                {
                    "start_time": pt_simulation_state['start_time'],
                    "start_timestamp": pt_simulation_state['start_timestamp'],
                    "step_rates": pt_simulation_state['step_rates'],
                    "initial_balances": pt_simulation_state['initial_balances'],
                    "recent_trades": {book_id : [t.model_dump(mode='json') for t in book_trades] for book_id, book_trades in pt_simulation_state['recent_trades'].items()},
                    "recent_miner_trades": {uid : {book_id : [[t.model_dump(mode='json'), r] for t, r in trades] for book_id, trades in uid_miner_trades.items()} for uid, uid_miner_trades in pt_simulation_state['recent_miner_trades'].items()},
                    "pending_notices": pt_simulation_state['pending_notices'],
                    "simulation.logDir": pt_simulation_state['simulation.logDir']
                }, use_bin_type=True
            )
            file.write(packed_data)
        os.rename(self.simulation_state_file.replace('.mp', '.pt'), self.simulation_state_file.replace('.mp', '.pt') + ".bak")
        bt.logging.info(f"Pytorch simulation state file converted to msgpack at {self.simulation_state_file}")

    if not self.config.neuron.reset and os.path.exists(self.simulation_state_file):
        bt.logging.info(f"Loading simulation state variables from {self.simulation_state_file}...")


    if not self.config.neuron.reset:
        simulation_state = _try_load_state_file(self.simulation_state_file, "simulation")

        if simulation_state:
            # Delegate engine-specific fields (start_time, start_timestamp, step_rates,
            # logDir for simulation; book_ids, last_block, step, etc. for exchange)
            self.engine.restore_simulation_state(simulation_state)

            loaded_pending_notices = simulation_state.get("pending_notices", {})
            self.pending_notices = {uid: [] for uid in range(self.effective_max_uids)}
            for uid, notices in loaded_pending_notices.items():
                if uid < self.effective_max_uids:
                    self.pending_notices[uid] = notices
                else:
                    bt.logging.debug(f"Skipping pending_notices for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
            loaded_initial_balances = simulation_state.get("initial_balances", {})
            self.initial_balances = {
                uid: {bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                    for bookId in self.engine.book_ids}
                for uid in range(self.effective_max_uids)
            }
            for uid, initial_balances in loaded_initial_balances.items():
                if uid < self.effective_max_uids:
                    first_entry = next(iter(initial_balances.values()), {})
                    if 'WEALTH' not in first_entry:
                        self.initial_balances[uid] = {
                            bookId: initial_balance | {'WEALTH': self.simulation.miner_wealth}
                            for bookId, initial_balance in initial_balances.items()
                        }
                    else:
                        self.initial_balances[uid] = initial_balances
                else:
                    bt.logging.debug(f"Skipping initial_balances for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

            self.recent_trades = {}
            _rt_dropped = 0
            for book_id, book_trades in simulation_state.get("recent_trades", {}).items():
                _clean, _n = sanitize_recent_trades(book_trades)
                _rt_dropped += _n
                self.recent_trades[book_id] = [TradeInfo.model_construct(**t) for t in _clean]
            if _rt_dropped:
                bt.logging.warning(
                    f"recent_trades: dropped {_rt_dropped} restored trade(s) whose id is not the "
                    f"engine's integer id (unjoinable to any recorded trade; see sanitize_recent_trades)"
                )
            loaded_recent_miner_trades = simulation_state.get("recent_miner_trades", {})
            self.recent_miner_trades = {
                uid: {bookId: [] for bookId in self.engine.book_ids}
                for uid in range(self.effective_max_uids)
            }
            _mt_dropped = [0]
            if loaded_recent_miner_trades:
                for uid, uid_miner_trades in loaded_recent_miner_trades.items():
                    if uid < self.effective_max_uids:
                        _mt_clean = {}
                        for book_id, trades in uid_miner_trades.items():
                            _c, _n = sanitize_miner_trades(trades)
                            _mt_dropped[0] += _n
                            _mt_clean[book_id] = [[TradeEvent.model_construct(**t), r] for t, r in _c]
                        self.recent_miner_trades[uid] = {
                            book_id: v
                            for book_id, v in _mt_clean.items()
                        }
                    else:
                        bt.logging.debug(f"Skipping recent_miner_trades for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
                if _mt_dropped[0]:
                    bt.logging.warning(
                        f"recent_miner_trades: dropped {_mt_dropped[0]} restored trade(s) with no fee "
                        f"fields (a fee cannot be reconstructed; metrics publishing read them as "
                        f"attributes and died: 'TradeEvent' object has no attribute 'Tf')"
                    )

            bt.logging.success("Loaded simulation state")
        else:
            bt.logging.warning("All simulation state files corrupted, initializing fresh state")
            simulation_state = None
    else:
        simulation_state = None

    if simulation_state is None:
        if self.config.neuron.reset:
            bt.logging.warning("`neuron.reset is True, ignoring previous state")
        else:
            bt.logging.info("No valid simulation state found, initializing new state")

        self.pending_notices = {uid: [] for uid in range(self.effective_max_uids)}
        self.initial_balances = {
            uid: {bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                for bookId in self.engine.book_ids}
            for uid in range(self.effective_max_uids)
        }
        self.recent_trades = {bookId: [] for bookId in self.engine.book_ids}
        self.recent_miner_trades = {
            uid: {bookId: [] for bookId in self.engine.book_ids}
            for uid in range(self.effective_max_uids)
        }
        self.fundamental_price = {bookId: None for bookId in self.engine.book_ids}

    _load_validator_state(self)
