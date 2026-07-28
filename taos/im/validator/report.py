# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Prometheus metrics reporting service: publishes validator, simulation, miner,
book, trade, and GenTRX gauges via a FastAPI endpoint over POSIX IPC.
"""
import os
import sys
import traceback
import time
import torch
import psutil
import asyncio
import bittensor as bt
import pandas as pd
import posix_ipc
import mmap
import struct
import msgpack
import argparse    

from typing import Dict
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from taos.im.protocol.models import TradeInfo, MarketSimulationConfig
from taos.im.protocol.events import TradeEvent

from taos.common.utils.prometheus import prometheus
from taos.im.utils import duration_from_timestamp
from prometheus_client import Counter, Gauge, Info, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import GaugeMetricFamily
from fastapi import FastAPI
from fastapi.responses import Response
import uvicorn
import threading

async def _push_to_mvtrx_data_service(state_dict: dict, url: str) -> None:
    """Delegate to the optional data-service push module when present."""
    try:
        from taos.im.validator.mvtrx_push import push_to_data_service as _push
    except ImportError:
        return
    await _push(state_dict, url)


class ReportingService:
    # Shared validator state pushed in via IPC before each report (not set in
    # __init__). Annotation-only declarations — no runtime effect — so static
    # analysis resolves these attributes.
    kappa_values: dict
    activity_factors: dict
    pnl_factors: dict
    scores: dict
    unnormalized_scores: dict
    miner_stats: dict
    initial_balances: dict
    initial_balances_published: dict
    step_rates: dict
    validator_config: dict
    current_block: int
    simulation_timestamp: int
    step: int
    uid: int
    fundamental_price: float
    shared_state_reporting: bool

    def __init__(self, config):
        """
        Initialise the reporting service, setting up IPC channels and Prometheus metrics.

        Creates POSIX message queues and shared memory segments for receiving
        publish requests from the validator, initialises all Prometheus registries,
        and starts the FastAPI metrics server in a background thread.

        Args:
            config: Configuration object with wallet, netuid, and prometheus settings.
        """
        self.config = config
        self.wallet = bt.Wallet(
            path=self.config.wallet.path,
            name=self.config.wallet.name,
            hotkey=self.config.wallet.hotkey
        )
        self.running = True
        self.prometheus_initialized = False
        self.current_sim_id = None
        
        _pfx = getattr(config, 'ipc_prefix', 'validator')
        bt.logging.info(f"Reporting service IPC prefix: {_pfx!r}")
        self.request_queue = posix_ipc.MessageQueue(
            f"/{_pfx}-report-req",
            flags=posix_ipc.O_CREAT,
            max_messages=2,
            max_message_size=1024
        )
        self.response_queue = posix_ipc.MessageQueue(
            f"/{_pfx}-report-res",
            flags=posix_ipc.O_CREAT,
            max_messages=2,
            max_message_size=1024
        )
        self.request_shm = posix_ipc.SharedMemory(
            f"/{_pfx}-report-data",
            flags=posix_ipc.O_CREAT,
            size=200 * 1024 * 1024
        )
        self.response_shm = posix_ipc.SharedMemory(
            f"/{_pfx}-report-response-data",
            flags=posix_ipc.O_CREAT,
            size=100 * 1024 * 1024
        )
        
        self.request_mem = mmap.mmap(self.request_shm.fd, self.request_shm.size)
        self.response_mem = mmap.mmap(self.response_shm.fd, self.response_shm.size)
        
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.report_executor = ThreadPoolExecutor(max_workers=1)        
        self._init_prometheus()
        
    def _start_metrics_server(self):
        """
        Start a FastAPI server exposing per-registry Prometheus metric endpoints.

        Binds to `config.prometheus.port` on all interfaces. Runs in a daemon
        thread so it shuts down automatically when the process exits.
        """
        app = FastAPI()

        @app.get("/metrics")
        def all_metrics():
            """All metrics combined (backwards compatibility)"""
            output = b''.join([generate_latest(r) for r in self.registries.values()])
            return Response(content=output, media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/validator")
        def validator_metrics():
            """Validator-specific metrics: counters, validator_gauges, neuron_info"""
            return Response(content=generate_latest(self.registry_validator), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/simulation")
        def simulation_metrics():
            """Simulation metrics: simulation_gauges"""
            return Response(content=generate_latest(self.registry_simulation), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/miner")
        def miner_metrics():
            """Miner metrics: miner_gauges, miners"""
            return Response(content=generate_latest(self.registry_miner), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/agent")
        def agent_metrics():
            """Miner metrics: agent_gauges"""
            return Response(content=generate_latest(self.registry_agent), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/books")
        def book_metrics():
            """Book metrics: book_gauges, books"""
            return Response(content=generate_latest(self.registry_books), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/trades")
        def trade_metrics():
            """Trade metrics: trades, miner_trades"""
            return Response(content=generate_latest(self.registry_trades), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/gentrx")
        def gentrx_metrics():
            """GenTRX distributed-training metrics: pool allocation, per-miner EMA scores."""
            return Response(content=generate_latest(self.registry_gentrx), media_type=CONTENT_TYPE_LATEST)

        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=self.config.prometheus.port, log_level="debug")

        self.metrics_server_thread = threading.Thread(target=run_server, daemon=True)
        self.metrics_server_thread.start()
        bt.logging.success(f"Prometheus metrics server started on port {self.config.prometheus.port}")
    
    def _init_prometheus(self):
        """
        Initialise all Prometheus collector registries and metric objects.

        Creates separate registries for validator, simulation, miner, agent, books,
        trades, and gentrx metrics, then starts the metrics HTTP server.
        """
        prometheus(
            config=self.config,
            port=self.config.prometheus.port,
            level=None,
            start_server=False
        )
        self.registry_validator = CollectorRegistry()
        self.registry_simulation = CollectorRegistry()
        self.registry_miner = CollectorRegistry()
        self.registry_agent = CollectorRegistry()
        self.registry_books = CollectorRegistry()
        self.registry_trades = CollectorRegistry()
        self.registry_gentrx = CollectorRegistry()

        self.registries = {
            'validator': self.registry_validator,
            'simulation': self.registry_simulation,
            'miner': self.registry_miner,
            'agent': self.registry_agent,
            'books': self.registry_books,
            'trades': self.registry_trades,
            'gentrx': self.registry_gentrx,
        }

        # Child-handle / last-value cache for the persistent gauge families
        # (agent_gauges, miner_gauges, book_gauges, simulation_gauges). Keyed by
        # (id(gauge), label_tuple) -> [child, last_value]. The apply loop resolves
        # the Prometheus child once and skips the lock+tuple-normalize that
        # gauge.labels() pays on every call, and skips the .set() entirely when
        # the value is unchanged from the prior cycle. This is the hot path at
        # full cardinality (~1.6M agent_gauges updates/cycle on mainnet's
        # 128 books x ~256 miners). Cleared families (trades/books/miner_trades)
        # are NOT cached — their children are destroyed each cycle. Entries are
        # evicted via _remove_and_evict on every .remove(), and the whole cache
        # is reset on simulation changeover (clear_all_metrics).
        self._child_cache = {}

        self.prometheus_counters = Counter('counters', 'Counter summaries for the running validator.', ['wallet', 'netuid', 'sim_id', 'timestamp', 'counter_name'], registry=self.registry_validator)
        self.prometheus_simulation_gauges = Gauge('simulation_gauges', 'Gauge summaries for global simulation metrics.', ['wallet', 'netuid', 'sim_id', 'simulation_gauge_name'], registry=self.registry_simulation)
        self.prometheus_validator_gauges = Gauge('validator_gauges', 'Gauge summaries for validator-related metrics.', ['wallet', 'netuid', 'sim_id', 'validator_gauge_name'], registry=self.registry_validator)
        self.prometheus_miner_gauges = Gauge('miner_gauges', 'Gauge summaries for miner-related metrics.', ['wallet', 'netuid', 'sim_id', 'agent_id', 'miner_gauge_name'], registry=self.registry_miner)
        self.prometheus_book_gauges = Gauge('book_gauges', 'Gauge summaries for book-related metrics.', ['wallet', 'netuid', 'sim_id', 'book_id', 'level', 'book_gauge_name'], registry=self.registry_books)
        # agent_gauges is the high-cardinality family (~1.8M series/cycle). It is
        # served via a snapshot collector instead of an eager Gauge so the per-cycle
        # apply cost (gauge.labels()+locked .set() x1.8M) collapses to a dict swap.
        self.prometheus_agent_gauges = _SnapshotCollector('agent_gauges', 'Gauge summaries for agent-related metrics.', ['wallet', 'netuid', 'sim_id', 'book_id', 'agent_id', 'agent_gauge_name'])
        self.registry_agent.register(self.prometheus_agent_gauges)
        # Bounded-slot shape: per-trade numeric fields live in the metric VALUE keyed
        # by `trade_gauge_name`, indexed by a fixed `slot` (rolling-buffer position).
        # This caps cardinality at books x buffer_len x fields instead of minting a
        # new series per trade (price/fee/volume/timestamp/id were previously labels).
        # trades / miner_trades / books are clear-and-rebuild families (rolling
        # slot buffers re-emitted in full each cycle). At mainnet cardinality the
        # eager path — gauge.clear() then a fresh labels()+set() per series (the
        # labels() recreates the child the clear() just destroyed) — was ~890K
        # ops/cycle and the dominant reporting-apply cost. Served instead via
        # replace-mode snapshot collectors: the per-cycle cost collapses to a
        # dict build + swap, and serialisation is lock-free on /metrics scrape.
        self.prometheus_trades = _SnapshotCollector('trades', 'Gauge summaries for trade metrics.',
            ['wallet', 'netuid', 'sim_id', 'book_id', 'slot', 'trade_gauge_name'],
            carry_forward=False)
        self.registry_trades.register(self.prometheus_trades)
        self.prometheus_miner_trades = _SnapshotCollector('miner_trades', 'Gauge summaries for agent trade metrics.',
            ['wallet', 'netuid', 'sim_id', 'book_id', 'uid', 'slot', 'miner_trade_gauge_name'],
            carry_forward=False)
        self.registry_trades.register(self.prometheus_miner_trades)
        self.prometheus_books = _SnapshotCollector('books', 'Gauge summaries for book snapshot metrics.', [
            'wallet', 'netuid', 'sim_id', 'timestamp', 'timestamp_str', 'book_id',
            'bid_5', 'bid_vol_5', 'bid_4', 'bid_vol_4', 'bid_3', 'bid_vol_3', 'bid_2', 'bid_vol_2', 'bid_1', 'bid_vol_1',
            'ask_5', 'ask_vol_5', 'ask_4', 'ask_vol_4', 'ask_3', 'ask_vol_3', 'ask_2', 'ask_vol_2', 'ask_1', 'ask_vol_1',
            'book_gauge_name'
        ], carry_forward=False)
        self.registry_books.register(self.prometheus_books)
        self.prometheus_miners = Gauge('miners', 'Gauge summaries for miner metrics.', [
            'wallet', 'netuid', 'sim_id', 'timestamp', 'timestamp_str', 'agent_id', 'hotkey', 'coldkey',
            'placement', 'base_balance', 'base_loan', 'base_collateral', 'quote_balance', 'quote_loan', 'quote_collateral',
            'inventory_value', 'inventory_value_change', 'pnl', 'pnl_change', 'total_realized_pnl',
            'total_daily_volume', 'min_daily_volume', 'average_daily_volume',
            'total_roundtrip_volume', 'min_roundtrip_volume', 'average_roundtrip_volume',
            'activity_factor', 'pnl_factor',
            'kappa', 'kappa_penalty', 'kappa_score',
            'pnl_score', 'combined_score', 'gentrx_score',
            'unnormalized_score', 'score',
            'miner_gauge_name'
        ], registry=self.registry_miner)
        self.prometheus_miner_identity = Gauge('miner_identity', 'Per-miner identity (hotkey/coldkey) for historical attribution; value always 1.0, re-series on re-registration.', ['wallet', 'netuid', 'sim_id', 'agent_id', 'hotkey', 'coldkey'], registry=self.registry_miner)
        self.prometheus_info = Info('neuron_info', "Info summaries for the running validator.", ['wallet', 'netuid', 'sim_id'], registry=self.registry_validator)
        self.prometheus_gentrx_gauges = Gauge('gentrx_gauges', 'GenTRX distributed-training validator metrics.', ['wallet', 'netuid', 'sim_id', 'gentrx_gauge_name'], registry=self.registry_gentrx)
        self.prometheus_gentrx_miner_scores = Gauge('gentrx_miner_scores', 'Per-miner GenTRX EMA score (validator-smoothed).', ['wallet', 'netuid', 'sim_id', 'uid'], registry=self.registry_gentrx)
        self.prometheus_gentrx_training = Gauge('gentrx_training', 'GenTRX model training statistics (loss, acceptance, timing).', ['wallet', 'netuid', 'sim_id', 'stat'], registry=self.registry_gentrx)
        self.prometheus_gentrx_miner_score_own = Gauge('gentrx_miner_score_own', 'Per-miner GenTRX own-data score (last round, pre-EMA).', ['wallet', 'netuid', 'sim_id', 'uid'], registry=self.registry_gentrx)
        self.prometheus_gentrx_miner_score_held = Gauge('gentrx_miner_score_held', 'Per-miner GenTRX held-out validation score (last round, pre-EMA).', ['wallet', 'netuid', 'sim_id', 'uid'], registry=self.registry_gentrx)
        self.prometheus_gentrx_miner_score = Gauge('gentrx_miner_score', 'Per-miner GenTRX combined score (last round, pre-EMA).', ['wallet', 'netuid', 'sim_id', 'uid'], registry=self.registry_gentrx)
        self.prometheus_gentrx_miner_accepted = Gauge('gentrx_miner_accepted', 'Per-miner GenTRX gradient accepted in last round (1=yes, 0=no).', ['wallet', 'netuid', 'sim_id', 'uid'], registry=self.registry_gentrx)
        self.prometheus_gentrx_miner_overfitting = Gauge('gentrx_miner_overfitting', 'Per-miner overfitting flag in last round (1=overfit detected, 0=ok).', ['wallet', 'netuid', 'sim_id', 'uid'], registry=self.registry_gentrx)
        self.prometheus_gentrx_miner_rollback_winner = Gauge('gentrx_miner_rollback_winner', 'Per-miner flag indicating gradient was selected during rollback (1=yes, 0=no).', ['wallet', 'netuid', 'sim_id', 'uid'], registry=self.registry_gentrx)
        self.prometheus_gentrx_miner_grad_norm = Gauge('gentrx_miner_grad_norm', 'Per-miner gradient L2 norm in last round.', ['wallet', 'netuid', 'sim_id', 'uid'], registry=self.registry_gentrx)
        self._start_metrics_server()
        self.prometheus_initialized = True
        
    def clear_all_metrics(self):
        """
        Clear all Prometheus metrics across all registries.
        
        This is called when a new simulation starts to prevent stale metrics
        from the previous simulation from persisting in graphs.
        """
        bt.logging.info("Clearing all metrics for simulation changeover...")
        start = time.time()
        
        try:
            # Clear all gauge metrics
            self.prometheus_simulation_gauges.clear()
            self.prometheus_validator_gauges.clear()
            self.prometheus_miner_gauges.clear()
            self.prometheus_book_gauges.clear()
            self.prometheus_agent_gauges.clear()
            self.prometheus_trades.clear()
            self.prometheus_miner_trades.clear()
            self.prometheus_books.clear()
            self.prometheus_miners.clear()
            self.prometheus_miner_identity.clear()
            self.prometheus_info.clear()
            # Cached children now point at removed series — drop the cache so the
            # next cycle re-resolves fresh handles.
            self._child_cache.clear()
            bt.logging.success(f"All metrics cleared ({time.time()-start:.4f}s)")
        except Exception as e:
            bt.logging.error(f"Error clearing metrics: {e}")
            bt.logging.error(traceback.format_exc())

    def _apply_snapshot_collectors(self, updates):
        """Route updates targeting snapshot-collector families into fresh
        per-cycle snapshots and swap them in; return (remaining, agent_count,
        cleared_count) where `remaining` is the non-collector updates for the
        caller to apply via its own path (_set_cached / _set_if_changed).

        agent_gauges carries forward; trades/books/miner_trades replace. Keeping
        the collector set here means both apply paths in publish_metrics stay in
        sync — a family added to one is handled by both.
        """
        agent_collector = self.prometheus_agent_gauges
        snapshots = {
            agent_collector: {},
            self.prometheus_trades: {},
            self.prometheus_miner_trades: {},
            self.prometheus_books: {},
        }
        remaining = []
        for update in updates:
            snap = snapshots.get(update[0])
            if snap is not None:
                snap[update[2:]] = update[1]
            else:
                remaining.append(update)
        for collector, snap in snapshots.items():
            # Fidelity with the old clear-and-rebuild: the eager path only
            # clear()'d a family when it had >=1 update this cycle, so a cycle
            # emitting none (e.g. no new trades) LEFT the prior series in place.
            # Skip the empty swap for replace families so trades/miner_trades
            # don't blank out during quiet cycles. agent_gauges (carry-forward)
            # always updates: an empty merge is a no-op that retains its series.
            if snap or collector is agent_collector:
                collector.update(snap)
        cleared_count = (
            len(snapshots[self.prometheus_trades])
            + len(snapshots[self.prometheus_miner_trades])
            + len(snapshots[self.prometheus_books])
        )
        return remaining, len(snapshots[agent_collector]), cleared_count

    async def run(self):
        """
        Main async event loop for the reporting service.

        Drains any stale IPC messages on startup, then waits for 'publish' or
        'shutdown' commands from the validator via the POSIX request queue,
        dispatching to `publish_metrics` and writing results back via shared memory.
        """
        bt.logging.info("Reporting service started")

        while True:
            try:
                self.request_queue.receive(timeout=0.0)
                bt.logging.warning("Drained stale message from reporting request queue")
            except posix_ipc.BusyError:
                break
        
        while self.running:
            try:
                message, _ = self.request_queue.receive(timeout=1.0)
                command = message.decode('utf-8')
                
                if command == 'publish':
                    read_start = time.time()
                    self.request_mem.seek(0)
                    size_bytes = self.request_mem.read(8)
                    data_size = struct.unpack('Q', size_bytes)[0]
                    request_bytes = self.request_mem.read(data_size)
                    
                    deserialize_start = time.time()
                    data = msgpack.unpackb(request_bytes, raw=False, strict_map_key=False)
                    deserialize_time = time.time() - deserialize_start
                    
                    bt.logging.info(f"Read reporting data ({time.time()-read_start:.4f}s, deserialize={deserialize_time:.4f}s)")

                    await self.publish_metrics(data)
                    
                    result = {
                        'step': data.get('step'),
                        'initial_balances_published': self.initial_balances_published,
                        'miner_stats': self.miner_stats
                    }
                    write_start = time.time()
                    
                    serialize_start = time.time()
                    result_bytes = msgpack.packb(result, use_bin_type=True)
                    serialize_time = time.time() - serialize_start
                    
                    self.response_mem.seek(0)
                    self.response_mem.write(struct.pack('Q', len(result_bytes)))
                    self.response_mem.write(result_bytes)
                    bt.logging.info(f"Wrote reporting response data ({time.time()-write_start:.4f}s, serialize={serialize_time:.4f}s)")

                    drain_start = time.time()
                    drained = 0
                    while True:
                        try:
                            self.response_queue.receive(timeout=0.0)
                            drained += 1
                        except posix_ipc.BusyError:
                            break
                    if drained > 0:
                        bt.logging.warning(f"Drained {drained} stale reporting response signals ({time.time()-drain_start:.4f}s)")
                    send_start = time.time()
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            self.response_queue.send(b'ready', timeout=1.0)
                            bt.logging.info(f"Reporting Response signal sent ({time.time()-send_start:.4f}s)")
                            break
                        except posix_ipc.BusyError:
                            bt.logging.warning(f"Reporting Response queue full, retry {attempt+1}/{max_retries}")
                            try:
                                self.response_queue.receive(timeout=0.0)
                                bt.logging.debug("Drained one more stale message from Reporting Response Queue")
                            except posix_ipc.BusyError:
                                pass
                            time.sleep(0.1)
                        except Exception as e:
                            bt.logging.error(f"Error sending Reporting response signal: {e}")
                            break
                    
                elif command == 'shutdown':
                    bt.logging.info("Shutdown command received")
                    self.running = False
                    
            except posix_ipc.BusyError:
                await asyncio.sleep(0.01)
            except Exception as e:
                bt.logging.error(f"Error in reporting loop: {e}")
                bt.logging.error(traceback.format_exc())
        
        self.cleanup()
    
    async def publish_metrics(self, data):
        """
        Deserialise a reporting payload and publish all Prometheus metrics.

        Clears stale metrics whenever the simulation ID changes, then delegates
        to `report()` to push updated gauges.

        Args:
            data (dict): Decoded msgpack payload from the validator containing
                simulation state, account balances, trade data, and scoring results.
        """
        new_sim_id = data['simulation']['simulation_id']
        
        if new_sim_id is None:
            # simulation_id is transiently None — on_start() resets it and on_tick()
            # re-derives it on the next state frame. Treat None as "not yet known":
            # skip this publish entirely rather than treating it as a sim change,
            # which would trigger a spurious clear and possibly a double-clear on recovery.
            bt.logging.debug("publish_metrics: simulation_id is None, deferring until valid")
            return

        if self.current_sim_id is not None and new_sim_id != self.current_sim_id:
            # Genuine simulation changeover: both IDs are real and they differ.
            # Clear so stale metrics from the old simulation don't bleed into the new one.
            bt.logging.warning(
                f"Simulation ID changed: {self.current_sim_id} → {new_sim_id}. "
                f"Clearing all metrics..."
            )
            self.clear_all_metrics()
        else:
            # First valid sim_id seen (fresh start, exchange mode, or reporting-service
            # restart mid-simulation). Do NOT clear: a mid-sim restart would wipe valid
            # accumulating metrics, and stale gauges from a prior instance are overwritten
            # naturally on the next publish cycle.
            if self.current_sim_id is None:
                bt.logging.info(
                    f"First valid metrics publish (sim_id={new_sim_id}). "
                    f"Resuming without clearing existing metrics."
                )

        self.current_sim_id = new_sim_id
        
        def deserialize_to_nested_dict(d):
            """Normalize volume-sum dicts to nested {uid: {book_id: float}} with
            int keys. Accepts the nested form the validator now sends directly
            (msgpack preserves int keys under strict_map_key=False) and the legacy
            flat 'uid:book_id' string-keyed form, so a mixed-version validator /
            reporting pair can't silently drop volume data."""
            result = defaultdict(lambda: defaultdict(float))
            for key, val in d.items():
                if isinstance(val, dict):
                    uid = int(key)
                    for book_id, vol in val.items():
                        result[uid][int(book_id)] = vol
                else:
                    uid, book_id = map(int, str(key).split(':'))
                    result[uid][book_id] = val
            return result

        self.recent_trades = {
            int(bookId): [TradeInfo.model_construct(**t) for t in trades]
            for bookId, trades in data['recent_trades'].items()
        }
        self.recent_miner_trades = {
            int(uid): {
                int(bookId): [(TradeEvent.model_construct(**item['trade']), item['role']) for item in trades]
                for bookId, trades in book_trades.items()
            }
            for uid, book_trades in data['recent_miner_trades'].items()
        }

        self.volume_sums = deserialize_to_nested_dict(data['volume_sums'])
        self.maker_volume_sums = deserialize_to_nested_dict(data['maker_volume_sums'])
        self.taker_volume_sums = deserialize_to_nested_dict(data['taker_volume_sums'])
        self.self_volume_sums = deserialize_to_nested_dict(data['self_volume_sums'])
        self.fee_sums = deserialize_to_nested_dict(data.get('fee_sums', {}))
        self.roundtrip_volume_sums = deserialize_to_nested_dict(data['roundtrip_volume_sums'])
        self.inventory_history = data['inventory_history']

        self.total_realized_pnl = {
            int(uid): pnl for uid, pnl in data['total_realized_pnl'].items()
        }
        self.realized_pnl_by_book = {
            int(uid): {
                int(book_id): pnl 
                for book_id, pnl in books.items()
            }
            for uid, books in data['realized_pnl_by_book'].items()
        }

        for key in ['activity_factors', 'pnl_factors', 'kappa_values',
                    'unnormalized_scores', 'scores', 'miner_stats', 'initial_balances',
                    'initial_balances_published', 'simulation_timestamp', 'step',
                    'step_rates', 'fundamental_price', 'shared_state_rewarding',
                    'current_block', 'uid', 'metagraph_data', 'validator_config']:
            setattr(self, key, data[key])
        self.gentrx_scores = data.get('gentrx_scores', {})
        self.gentrx_enabled = data.get('gentrx_enabled', False)
        self.gentrx_training = data.get('gentrx_training', {})
        self.gentrx_scores_detailed = data.get('gentrx_scores_detailed', {})
        self.gentrx_config = data.get('gentrx_config', {})
        
        class SimpleState:
            pass
        self.last_state = SimpleState()
        self.last_state.accounts = data['last_state']['accounts']
        self.last_state.books = data['last_state']['books']
        self.last_state.notices = data['last_state']['notices']
        
        class SimpleMetagraph:
            pass
        self.metagraph = SimpleMetagraph()
        for key, value in self.metagraph_data.items():
            setattr(self.metagraph, key, value)
        
        sim_data = data['simulation']
        if 'block_count' in sim_data:
            self.simulation = MarketSimulationConfig(**sim_data)
        else:
            try:
                from taos.im.protocol.exchange_config import ExchangeConfig
            except ImportError:
                # Exchange-engine sim_data only ever arrives when --engine exchange
                # is active. The public release excludes the exchange engine + its
                # config model; reaching this branch there means a misconfiguration.
                raise RuntimeError(
                    "Exchange engine mode is not supported in this build "
                    "(taos.im.protocol.exchange_config is excluded from the public release). "
                    "Run with --engine simulation (default)."
                )
            self.simulation = ExchangeConfig(**sim_data)
        
        if not self.prometheus_initialized:
            self._init_prometheus()

        # Build per-agent stats from scoring data available in this context.
        # volume_sums / maker_volume_sums are already deserialized as {uid: {book_id: float}}.
        def _sum_books(d):
            return {uid: sum(float(v) for v in books.values())
                    for uid, books in d.items()}

        # In simulation mode open_positions tracks longs/shorts counts.
        # In exchange mode the LOB accounts['o'] field holds resting orders directly.
        is_exchange = not hasattr(self.simulation, 'block_count')
        agent_open_orders = {}
        agent_orders_detail: dict = {}

        if is_exchange:
            # Build open-order counts and per-order detail from LOB accounts['o'].
            # accounts format: {uid: {netuid: {'o': [{i,s,p,q,...}], 'bb':..., ...}}}
            lob_accounts = data.get('last_state', {}).get('accounts') or {}
            for uid_key, uid_books in lob_accounts.items():
                try:
                    uid = int(uid_key)
                    if not isinstance(uid_books, dict):
                        continue
                    for nid_key, acct in uid_books.items():
                        if not isinstance(acct, dict):
                            continue
                        nid = int(nid_key)
                        for ord_ in (acct.get('o') or []):
                            if not isinstance(ord_, dict):
                                continue
                            lob_id = ord_.get('i', ord_.get('id'))
                            if lob_id is None:
                                continue
                            agent_open_orders[uid] = agent_open_orders.get(uid, 0) + 1
                            agent_orders_detail.setdefault(uid, []).append({
                                "order_id": int(lob_id),
                                "netuid":   nid,
                                "side":     int(ord_.get('s', ord_.get('side', 0))),
                                "price":    float(ord_.get('p', ord_.get('price', 0.0))),
                                "quantity": float(ord_.get('q', ord_.get('quantity', 0.0))),
                            })
                except Exception:
                    continue
            # Stale-Redis fix: explicitly write an empty list for UIDs with no orders
            # so Redis overwrites the key instead of leaving the previous value until TTL.
            for uid_key in lob_accounts:
                try:
                    uid = int(uid_key)
                    if uid not in agent_orders_detail:
                        agent_orders_detail[uid] = []
                except Exception:
                    pass
        else:
            for uid, book_pos in (data.get('open_positions') or {}).items():
                total = sum(
                    p.get('longs_count', 0) + p.get('shorts_count', 0)
                    for p in (book_pos.values() if isinstance(book_pos, dict) else [])
                )
                agent_open_orders[uid] = total

        _ingest_url = (
            getattr(getattr(self.config, 'exchange', None), 'data_service_url', '')
            if is_exchange else
            (getattr(getattr(self.config, 'simulation', None), 'data_service_url', '')
             or getattr(getattr(self.config, 'exchange', None), 'data_service_url', ''))
        )
        is_observe = bool((data.get('validator_config') or {}).get('observe', False))

        asyncio.create_task(_push_to_mvtrx_data_service({
            "mode":           "exchange" if is_exchange else "simulation",
            "network":        getattr(getattr(self.config, 'subtensor', None), 'network', ''),
            "timestamp":      int(time.time() * 1e9),
            "block":          data.get('current_block', 0),
            "books":          data['last_state']['books'],
            "accounts":       data['last_state']['accounts'],
            "pools":          data['last_state'].get('pools'),
            "reconciliation": data.get('reconciliation', {}),
            # per-agent scoring — used by data service to build agent:stats
            "agent_scores":          self.scores,
            "agent_kappa":           {uid: float((kv or {}).get('score', 0.0))
                                      for uid, kv in self.kappa_values.items()},
            "agent_volume":          _sum_books(self.volume_sums),
            "agent_maker_volume":    _sum_books(self.maker_volume_sums),
            "agent_roundtrip_volume":_sum_books(self.roundtrip_volume_sums),
            "agent_pnl":             {uid: float(pnl) for uid, pnl in self.total_realized_pnl.items()},
            "agent_open_orders":     agent_open_orders,
            "agent_orders_detail":   agent_orders_detail,
            # chain-level data forwarded from exchange engine's last chain state
            "block_events":   data.get('block_events', []),
            "delegates":      data.get('delegates', {}),
            # metagraph for identity + validator panel
            "metagraph":      self.metagraph_data,
            "validator_uid":  self.uid,
            # simulation only: fundamental prices
            "fundamental_price": self.fundamental_price,
        }, url=_ingest_url))

        if not is_observe:
            await report(self)
    
    def pagerduty_alert(self, message, details=None):
        """
        Log a critical alert message (stub — the reporting service has no PagerDuty hook).

        Args:
            message (str): Human-readable alert description.
            details (dict, optional): Additional context to log alongside the message.
        """
        bt.logging.error(f"ALERT: {message}")
        if details:
            bt.logging.error(f"Details: {details}")
    
    def cleanup(self):
        """
        Release all IPC and thread-pool resources held by the reporting service.
        """
        self.request_queue.close()
        self.response_queue.close()
        self.request_mem.close()
        self.request_shm.close_fd()
        self.thread_pool.shutdown(wait=True)
        self.report_executor.shutdown(wait=True)

def publish_validator_gauges(self: ReportingService):
    """
    Publishes validator-specific metrics to Prometheus gauges.
    
    Metrics include validator metagraph information (UID, stake, trust, dividends, emission, 
    last update, active status) and system resource usage (CPU, RAM, disk).
    
    Args:
        self (ReportingService): The intelligent markets simulation validator instance
        
    Returns:
        None
    """
    bt.logging.debug("Publishing validator metrics...")
    start = time.time()
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="uid").set( self.uid )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="stake").set( self.metagraph.stake[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="validator_trust").set( self.metagraph.validator_trust[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="dividends").set( self.metagraph.dividends[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="emission").set( self.metagraph.emission[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="last_update").set( self.current_block - self.metagraph.last_update[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="active").set( self.metagraph.active[self.uid] )
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    disk_info = psutil.disk_usage('/')
    disk_usage = disk_info.percent
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="cpu_usage_percent").set( cpu_usage )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="ram_usage_percent").set( memory_usage )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="disk_usage_percent").set( disk_usage )    
    bt.logging.debug(f"Validator metrics published ({time.time()-start:.4f}s).")

def publish_gentrx_gauges(self: ReportingService) -> None:
    """
    Publish GenTRX distributed-training metrics to the gentrx Prometheus registry.

    Exposes pool-level gauges (enabled flag, simulation share, active miner count),
    per-miner EMA-smoothed scores, and training statistics (loss, acceptance rate,
    checkpoint version, round count, rollback count, and timing).

    Args:
        self (ReportingService): The reporting service instance holding Prometheus
            gauge objects and the current gentrx_scores / gentrx_training state.
    """
    wallet_addr = self.wallet.hotkey.ss58_address
    netuid = self.config.netuid
    simid = self.simulation.simulation_id

    gentrx_scores = getattr(self, 'gentrx_scores', {}) or {}
    gentrx_enabled = getattr(self, 'gentrx_enabled', False)
    gentrx_share = self.validator_config.get('scoring', {}).get('gentrx_simulation_share', 0.0)
    active_miners = sum(1 for s in gentrx_scores.values() if s > 0)

    g = self.prometheus_gentrx_gauges
    g.labels(wallet=wallet_addr, netuid=netuid, sim_id=simid, gentrx_gauge_name="enabled").set(1 if gentrx_enabled else 0)
    g.labels(wallet=wallet_addr, netuid=netuid, sim_id=simid, gentrx_gauge_name="simulation_share").set(gentrx_share)
    g.labels(wallet=wallet_addr, netuid=netuid, sim_id=simid, gentrx_gauge_name="active_miners").set(active_miners)

    cfg = getattr(self, 'gentrx_config', {}) or {}
    for key in ('min_score', 'overfit_ratio', 'books_per_miner', 'val_fraction'):
        v = cfg.get(key)
        if v is not None:
            g.labels(wallet=wallet_addr, netuid=netuid, sim_id=simid, gentrx_gauge_name=key).set(float(v))
    ema_alpha = self.validator_config.get('scoring', {}).get('gentrx_ema_alpha')
    if ema_alpha is not None:
        g.labels(wallet=wallet_addr, netuid=netuid, sim_id=simid, gentrx_gauge_name="ema_alpha").set(float(ema_alpha))

    ms = self.prometheus_gentrx_miner_scores
    for uid_str, score in gentrx_scores.items():
        ms.labels(wallet=wallet_addr, netuid=netuid, sim_id=simid, uid=str(uid_str)).set(float(score))

    detailed = getattr(self, 'gentrx_scores_detailed', {}) or {}
    for uid_key, entry in detailed.items():
        if not isinstance(entry, dict):
            continue
        uid = str(uid_key)
        lv = dict(wallet=wallet_addr, netuid=netuid, sim_id=simid, uid=uid)
        s_own = entry.get('score_own')
        s_held = entry.get('score_held')
        if s_own is not None:
            self.prometheus_gentrx_miner_score_own.labels(**lv).set(float(s_own))
        if s_held is not None:
            self.prometheus_gentrx_miner_score_held.labels(**lv).set(float(s_held))
        self.prometheus_gentrx_miner_score.labels(**lv).set(float(entry.get('score', 0.0)))
        self.prometheus_gentrx_miner_accepted.labels(**lv).set(1.0 if entry.get('accepted') else 0.0)
        self.prometheus_gentrx_miner_overfitting.labels(**lv).set(1.0 if entry.get('overfitting') else 0.0)
        self.prometheus_gentrx_miner_rollback_winner.labels(**lv).set(1.0 if entry.get('was_rollback_winner') else 0.0)
        gn = entry.get('grad_norm')
        if gn is not None:
            self.prometheus_gentrx_miner_grad_norm.labels(**lv).set(float(gn))

    # Training stats from last aggregation round
    tr = getattr(self, 'gentrx_training', {}) or {}
    if tr:
        t = self.prometheus_gentrx_training
        lv = dict(wallet=wallet_addr, netuid=netuid, sim_id=simid)
        for stat, val in (
            ("loss_before",           tr.get("loss_before")),
            ("loss_after",            tr.get("loss_after")),
            ("loss_delta",            (tr["loss_before"] - tr["loss_after"])
                                      if tr.get("loss_before") is not None and tr.get("loss_after") is not None
                                      else None),
            ("loss_improvement_pct",  tr.get("loss_improvement_pct")),
            ("n_assigned",            tr.get("n_assigned")),
            ("n_delivered",           tr.get("n_delivered")),
            ("n_collected",           tr.get("n_collected")),
            ("n_version_mismatched",  tr.get("n_version_mismatched")),
            ("n_scored",              tr.get("n_scored")),
            ("n_accepted",            tr.get("n_accepted")),
            ("acceptance_rate",       (tr["n_accepted"] / tr["n_scored"])
                                      if tr.get("n_scored") else None),
            ("version",               tr.get("version")),
            ("agg_round",             tr.get("round")),
            ("rolled_back",           1 if tr.get("rolled_back") else 0),
            ("rollback_rate_10w",     tr.get("rollback_rate_10w")),
            ("rollback_rate_50w",     tr.get("rollback_rate_50w")),
            ("rounds_aggregated_total", tr.get("rounds_aggregated_total")),
            ("rollbacks_total",       tr.get("rollbacks_total")),
            ("t_score_s",             tr.get("t_score_s")),
            ("t_aggregate_s",         tr.get("t_aggregate_s")),
            ("t_total_s",             tr.get("t_total_s")),
            ("t_proposal_eval_s",     tr.get("t_proposal_eval_s")),
            ("t_save_ckpt_s",         tr.get("t_save_ckpt_s")),
            ("t_loader_build_s",      tr.get("t_loader_build_s")),
            ("grad_norm_mean",        tr.get("grad_norm_mean")),
            ("grad_norm_min",         tr.get("grad_norm_min")),
            ("grad_norm_max",         tr.get("grad_norm_max")),
            ("grad_norm_median",      tr.get("grad_norm_median")),
            ("grad_norm_std",         tr.get("grad_norm_std")),
            ("overlap_pairs_checked", tr.get("overlap_pairs_checked")),
            ("overlap_pairs_high",    tr.get("overlap_pairs_high")),
            ("overlap_mean",          tr.get("overlap_mean")),
            ("overlap_max",           tr.get("overlap_max")),
            ("loader_cache_hits",     tr.get("loader_cache_hits")),
            ("loader_cache_misses",   tr.get("loader_cache_misses")),
            ("loader_cache_hit_rate", tr.get("loader_cache_hit_rate")),
            ("proposals_evaluated",   tr.get("proposals_evaluated")),
            ("proposals_skipped",     tr.get("proposals_skipped")),
            *(
                (f"per_field_loss_before_{f}", tr.get(f"per_field_loss_before_{f}"))
                for f in ("order_type", "price", "vol_int", "vol_dec", "interval")
            ),
            *(
                (f"per_field_loss_after_{f}", tr.get(f"per_field_loss_after_{f}"))
                for f in ("order_type", "price", "vol_int", "vol_dec", "interval")
            ),
        ):
            if val is not None:
                t.labels(**lv, stat=stat).set(float(val))


def publish_info(self: ReportingService) -> None:
    """
    Publishes static simulation and validator information metrics

    Args:
        self (ReportingService): The intelligent markets simulation validator.
    Returns:
        None
    """
    prometheus_info = {
        'uid': str(self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )) if self.wallet.hotkey.ss58_address in self.metagraph.hotkeys else -1,
        'network': self.config.subtensor.network,
        'coldkey': str(self.wallet.coldkeypub.ss58_address),
        'coldkey_name': self.config.wallet.name,
        'hotkey': str(self.wallet.hotkey.ss58_address),
        'name': self.config.wallet.hotkey
    } | {
        f"config_scoring_{name}": str(value)
        for name, value in self.validator_config['scoring'].items()
    } | {
         f"simulation_{name}" : str(value) for name, value in self.simulation.model_dump().items() if name != 'logDir' and name != 'fee_policy'
    } | (self.simulation.fee_policy.to_prom_info() if getattr(self.simulation, 'fee_policy', None) else {})
    self.prometheus_info.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id ).info (prometheus_info)
    publish_validator_gauges(self)
    publish_gentrx_gauges(self)

class _SnapshotCollector:
    """Snapshot-backed collector replacing an eager high-cardinality Gauge.

    Removes the ~1.8M-per-cycle gauge.labels()+locked .set() apply loop (the
    reporting-timeout root cause): the per-cycle cost becomes plain dict ops, and
    serialisation happens lock-free on /metrics scrape via collect().

    Semantics are byte-for-byte identical to the eager Gauge, including
    persistence: a series **carries forward at its last value** until it is
    explicitly evicted (evict(), mirroring gauge.remove()) or cleared (clear(),
    mirroring gauge.clear()). It does NOT drop a series merely because the series
    was not re-emitted this cycle — matching the eager Gauge, where a child set
    once survives until removed. Each cycle, update() merges this cycle's emitted
    values over the retained snapshot and applies pending evictions.
    """

    def __init__(self, name, documentation, labelnames, carry_forward=True):
        self._name = name
        self._documentation = documentation
        self._labelnames = list(labelnames)
        self._snapshot = {}
        self._pending_evict = set()
        # carry_forward=True (agent_gauges): a series persists at its last value
        # until explicitly evicted, mirroring an eager Gauge that keeps a child
        # once set. carry_forward=False (the former clear-and-rebuild families
        # trades / books / miner_trades): each cycle FULLY replaces the snapshot,
        # mirroring gauge.clear() + rebuild — the new snapshot is the exposition.
        self._carry_forward = carry_forward

    def evict(self, labels):
        # Mark a series for removal at the next update() — mirrors gauge.remove().
        self._pending_evict.add(labels)

    def update(self, cycle_values):
        # copy-then-swap keeps collect() (running on the scrape thread) lock-free
        # and torn-read-safe: collect() only ever sees a fully-built snapshot.
        if not self._carry_forward:
            # Replace: the family is re-emitted in full every cycle, so this
            # cycle's values ARE the complete exposition — no merge/carry-forward
            # (which would leak stale rolling-buffer slots from prior cycles).
            self._snapshot = cycle_values
            self._pending_evict = set()
            return
        merged = dict(self._snapshot)
        merged.update(cycle_values)
        if self._pending_evict:
            for key in self._pending_evict:
                merged.pop(key, None)
            self._pending_evict = set()
        self._snapshot = merged

    def clear(self):
        self._snapshot = {}
        self._pending_evict = set()

    def collect(self):
        snapshot = self._snapshot
        family = GaugeMetricFamily(self._name, self._documentation, labels=self._labelnames)
        add_metric = family.add_metric
        for labels, value in snapshot.items():
            add_metric([str(label) for label in labels], value)
        yield family


def _set_if_changed(gauge, value, *labels):
    """
    Sets a Prometheus gauge value only if it differs from the current value.
    
    Args:
        gauge: Prometheus gauge object to update
        value: New value to set on the gauge
        *labels: Variable number of positional label values for the gauge
        
    Returns:
        None
    """
    # gauge.labels() is the dominant cost of the apply loop (acquires the
    # collector lock, normalizes the label tuple, dict-lookup/create the child)
    # and runs ~1.8M times per reporting cycle. Resolve the child ONCE and reuse
    # it for both the read and the write: the previous form called labels()
    # twice on every changed series, roughly doubling the per-update cost.
    # labels() creates the child if absent, so the old KeyError branch is moot.
    child = gauge.labels(*labels)
    if child._value.get() != value:
        child.set(value)

def _set_cached(cache, gauge, value, labels):
    """Set a gauge value via a persistent child-handle / last-value cache.

    For the persistent gauge families this is the hot path. On a cache hit the
    only work is a dict lookup and a float comparison; gauge.labels() (lock +
    label-tuple normalization + dict lookup) and a locked _value.get() are paid
    only on first sight of a series. Unchanged values short-circuit with no
    Prometheus interaction at all.

    Args:
        cache: dict keyed by (id(gauge), labels) -> [child, last_value].
        gauge: Prometheus gauge object.
        value: New value to set.
        labels: Tuple of positional label values.
    """
    ck = (id(gauge), labels)
    entry = cache.get(ck)
    if entry is None:
        child = gauge.labels(*labels)
        child.set(value)
        cache[ck] = [child, value]
        return
    if entry[1] != value:
        entry[0].set(value)
        entry[1] = value

def _remove_and_evict(cache, gauge, *labels):
    """Remove a gauge series and evict any cached handle for it.

    Eviction is required so that if the series reappears with the same value it
    was last set to, _set_cached re-creates it rather than short-circuiting on a
    stale last-value entry and leaving the series absent from the registry.

    For a _SnapshotCollector the carry-forward snapshot persists series across
    cycles, so eviction must be explicit (mirrors gauge.remove()).
    """
    if isinstance(gauge, _SnapshotCollector):
        gauge.evict(labels)
        return
    try:
        gauge.remove(*labels)
    except KeyError:
        pass
    cache.pop((id(gauge), labels), None)

def _set_if_changed_metric(gauge, value, **labels):
    """
    Sets a Prometheus gauge value only if it differs from the current value using keyword labels.
    
    Args:
        gauge: Prometheus gauge object to update
        value: New value to set on the gauge
        **labels: Variable number of keyword label-value pairs for the gauge
        
    Returns:
        None
    """
    # Single labels() lookup reused for read + write (see _set_if_changed).
    child = gauge.labels(**labels)
    if child._value.get() != value:
        child.set(value)

def report_worker(validator_data: Dict, state_data: Dict) -> Dict:
    """
    Compute per-miner and per-book metrics from a snapshot of validator state.

    Runs in a thread-pool executor to avoid blocking the reporting event loop.

    Args:
        validator_data (Dict): Snapshot of validator state including volume sums,
            inventory history, realized P&L, activity/pnl factors, kappa values,
            scores, and simulation config.
        state_data (Dict): Current simulator state containing accounts, books,
            and notices.

    Returns:
        Dict: Result dict with keys 'metrics' (computed miner/book data),
            'updated_stats' (unused), and 'error' (str or None on failure).
    """
    result = {
        'metrics': {},
        'updated_stats': {},
        'error': None
    }
    try:
        validator_data['simulation_timestamp']
        validator_data['step']
        accounts = state_data['accounts']
        books = state_data['books']
        if not accounts:
            return result
        
        total_realized_pnl = validator_data['total_realized_pnl']
        realized_pnl_by_book = validator_data['realized_pnl_by_book']
        fee_sums_by_book = validator_data.get('fee_sums', {})

        volume_sums = validator_data['volume_sums']
        maker_volume_sums = validator_data['maker_volume_sums']
        taker_volume_sums = validator_data['taker_volume_sums']
        self_volume_sums = validator_data['self_volume_sums']
        roundtrip_volume_sums = validator_data['roundtrip_volume_sums']
        volume_decimals = validator_data['simulation_config']['volumeDecimals']

        daily_volumes = {}
        for agentId in accounts.keys():
            daily_volumes[agentId] = {}
            for bookId in range(validator_data['book_count']):
                total_vol = volume_sums.get(agentId, {}).get(bookId, 0.0)
                total_maker_vol = maker_volume_sums.get(agentId, {}).get(bookId, 0.0)
                total_taker_vol = taker_volume_sums.get(agentId, {}).get(bookId, 0.0)
                total_self_vol = self_volume_sums.get(agentId, {}).get(bookId, 0.0)
                daily_volumes[agentId][bookId] = {
                    'total': total_vol,
                    'maker': total_maker_vol,
                    'taker': total_taker_vol,
                    'self': total_self_vol,
                }

        daily_roundtrip_volumes = {}
        for agentId in accounts.keys():
            daily_roundtrip_volumes[agentId] = {}
            for bookId in range(validator_data['book_count']):
                roundtrip_vol = roundtrip_volume_sums.get(agentId, {}).get(bookId, 0.0)
                daily_roundtrip_volumes[agentId][bookId] = roundtrip_vol
        
        inventory_history = validator_data['inventory_history']
        total_inventory_history = {}
        pnl = {}

        for agentId in accounts.keys():
            if agentId < 0:
                continue
            if agentId not in inventory_history or not inventory_history[agentId]:
                continue
            if len(inventory_history[agentId]) < 2:
                continue
            
            total_inventory_history[agentId] = [
                sum(list(inventory_value.values()))
                for inventory_value in list(inventory_history[agentId].values())
            ]
            pnl[agentId] = total_inventory_history[agentId][-1] - total_inventory_history[agentId][0]

        scores = torch.FloatTensor(list(validator_data['scores'].values()))
        indices = scores.argsort(dim=-1, descending=True)
        placements = torch.empty_like(indices).scatter_(
            -1, indices, torch.arange(scores.size(-1), device=scores.device)
        )

        miner_metrics = {}
        for agentId, accounts_data in accounts.items():
            if agentId < 0:
                continue
            if agentId not in total_inventory_history:
                continue

            base_decimals = validator_data['simulation_config']['baseDecimals']
            quote_decimals = validator_data['simulation_config']['quoteDecimals']

            total_base_balance = round(
                sum([accounts_data[bookId]['bb']['t'] for bookId in books]),
                base_decimals
            )
            total_base_loan = round(
                sum([accounts_data[bookId]['bl'] for bookId in books]),
                base_decimals
            )
            total_base_collateral = round(
                sum([accounts_data[bookId]['bc'] for bookId in books]),
                base_decimals
            )
            total_quote_balance = round(
                sum([accounts_data[bookId]['qb']['t'] for bookId in books]),
                quote_decimals
            )
            total_quote_loan = round(
                sum([accounts_data[bookId]['ql'] for bookId in books]),
                quote_decimals
            )
            total_quote_collateral = round(
                sum([accounts_data[bookId]['qc'] for bookId in books]),
                quote_decimals
            )

            total_daily_volume = {
                role: round(
                    sum([book_volume[role] for book_volume in daily_volumes[agentId].values()]),
                    volume_decimals
                )
                for role in ['total', 'maker', 'taker', 'self']
            }

            average_daily_volume = {
                role: round(
                    total_daily_volume[role] / len(daily_volumes[agentId]),
                    volume_decimals
                )
                for role in ['total', 'maker', 'taker', 'self']
            }

            min_daily_volume = {
                role: min([book_volume[role] for book_volume in daily_volumes[agentId].values()])
                for role in ['total', 'maker', 'taker', 'self']
            }

            total_roundtrip_volume = round(
                sum(daily_roundtrip_volumes[agentId].values()),
                volume_decimals
            )
            average_roundtrip_volume = round(
                total_roundtrip_volume / len(daily_roundtrip_volumes[agentId]),
                volume_decimals
            )
            min_roundtrip_volume = min(daily_roundtrip_volumes[agentId].values()) if daily_roundtrip_volumes[agentId] else 0.0

            activity_factor = (
                sum(validator_data['activity_factors'][agentId].values()) /
                len(validator_data['activity_factors'][agentId])
            ) if validator_data['activity_factors'][agentId] else 0.0
            pnl_factor = (
                sum(validator_data['pnl_factors'][agentId].values()) /
                len(validator_data['pnl_factors'][agentId])
            ) if validator_data['pnl_factors'][agentId] else 0.0
            kappa_values = validator_data['kappa_values'][agentId] if agentId in validator_data['kappa_values'] else None
            # EMA'd trading standing (0.5.5): reported as kappa_score/combined_score so the
            # dashboard shows the smoothed standing that drives reward, not the raw per-window
            # value. Falls back to the raw kappa_values when the EMA is off or not yet seeded.
            _ema_std = (validator_data.get('trading_score_ema') or {}).get(agentId)

            miner_metrics[agentId] = {
                'total_base_balance': total_base_balance,
                'total_base_loan': total_base_loan,
                'total_base_collateral': total_base_collateral,
                'total_quote_balance': total_quote_balance,
                'total_quote_loan': total_quote_loan,
                'total_quote_collateral': total_quote_collateral,
                'total_inventory_value': total_inventory_history[agentId][-1],
                'inventory_value_change': (
                    total_inventory_history[agentId][-1] - total_inventory_history[agentId][-2]
                    if len(total_inventory_history[agentId]) > 1 else 0.0
                ),
                'pnl': pnl[agentId],
                'pnl_change': (
                    pnl[agentId] - (total_inventory_history[agentId][-2] - total_inventory_history[agentId][0])
                    if len(total_inventory_history[agentId]) > 1 else 0.0
                ),
                'total_realized_pnl': total_realized_pnl.get(agentId, 0.0),
                'total_daily_volume': total_daily_volume,
                'average_daily_volume': average_daily_volume,
                'min_daily_volume': min_daily_volume,
                'total_roundtrip_volume': total_roundtrip_volume,
                'average_roundtrip_volume': average_roundtrip_volume,
                'min_roundtrip_volume': min_roundtrip_volume,
                'activity_factor': activity_factor,
                'pnl_factor': pnl_factor,
                'kappa': kappa_values['median'] if kappa_values else None,
                'kappa_penalty': kappa_values.get('penalty') if kappa_values else None,
                'activity_weighted_normalized_median': kappa_values.get('activity_weighted_normalized_median') if kappa_values else None,
                'kappa_score': kappa_values.get('score') if kappa_values else None,
                'pnl_score': kappa_values.get('pnl_score') if kappa_values else None,
                'combined_score': _ema_std if _ema_std is not None else (kappa_values.get('final_score') if kappa_values else None),
                'unnormalized_score': validator_data['unnormalized_scores'].get(agentId, 0.0),
                'score': scores[agentId].item() if agentId < len(scores) else 0.0,
                'gentrx_score': float(validator_data.get('gentrx_scores', {}).get(agentId, 0.0)),
                'placement': placements[agentId].item() if agentId < len(placements) else len(scores),
            }

        result['metrics'] = {
            'miner_metrics': miner_metrics,
            'daily_volumes': daily_volumes,
            'daily_roundtrip_volumes': daily_roundtrip_volumes,
            'total_inventory_history': total_inventory_history,
            'total_realized_pnl': total_realized_pnl,
            'realized_pnl_by_book': realized_pnl_by_book,
            'fee_sums_by_book': fee_sums_by_book,
            'pnl': pnl,
            'scores': scores.tolist(),
            'placements': placements.tolist(),
        }
    except Exception as ex:
        result['error'] = str(ex)
        result['traceback'] = traceback.format_exc()
    return result


async def report(self: ReportingService) -> None:
    """
    Calculates and publishes metrics related to simulation state, validator and agent performance.

    Args:
        self (ReportingService): The intelligent markets simulation validator.
    Returns:
        None
    """
    try:
        self.shared_state_reporting = True
        report_step = self.step
        simulation_duration = duration_from_timestamp(self.simulation_timestamp)
        bt.logging.info(f"Publishing Metrics at Step {self.step} ({simulation_duration})...")
        report_start = time.time()
        updates = deque()    
        bt.logging.debug("Collecting simulation metrics...")
        start = time.time()
        
        agent_gauges = self.prometheus_agent_gauges
        book_gauges = self.prometheus_book_gauges
        miner_gauges = self.prometheus_miner_gauges
        wallet_addr = self.wallet.hotkey.ss58_address
        netuid = self.config.netuid
        simid = self.simulation.simulation_id

        updates.append((
            self.prometheus_simulation_gauges,
            self.simulation_timestamp,
            wallet_addr,
            netuid,
            simid,
            "timestamp"
        ))

        updates.append((
            self.prometheus_simulation_gauges,
            sum(self.step_rates) / len(self.step_rates) if len(self.step_rates) > 0 else 0,
            wallet_addr,
            netuid,
            simid,
            "step_rate"
        ))
        bt.logging.debug(f"Simulation metrics collected ({time.time()-start:.4f}s).")

        has_new_trades = False
        has_new_miner_trades = False

        publish_info(self)

        bt.logging.debug("Collecting book metrics...")
        book_start = time.time()
        for bookId, book in self.last_state.books.items():
            if book['b']:
                bid_cumsum = 0
                for i, level in enumerate(book['b']):
                    updates.append((book_gauges, level['p'],
                        wallet_addr, netuid, simid, bookId, i, "bid"))
                    updates.append((book_gauges, level['q'],
                        wallet_addr, netuid, simid, bookId, i, "bid_vol"))
                    bid_cumsum += level['q']
                    updates.append((book_gauges, bid_cumsum,
                        wallet_addr, netuid, simid, bookId, i, "bid_vol_sum"))
                    if i == 20:
                        break
            if book['a']:
                ask_cumsum = 0
                for i, level in enumerate(book['a']):
                    updates.append((book_gauges, level['p'],
                        wallet_addr, netuid, simid, bookId, i, "ask"))
                    updates.append((book_gauges, level['q'],
                        wallet_addr, netuid, simid, bookId, i, "ask_vol"))
                    ask_cumsum += level['q']
                    updates.append((book_gauges, ask_cumsum,
                        wallet_addr, netuid, simid, bookId, i, "ask_vol_sum"))
                    if i == 20:
                        break
            if book['b'] and book['a']:
                mid = (book['b'][0]['p'] + book['a'][0]['p']) / 2
                updates.append((book_gauges, mid,
                    wallet_addr, netuid, simid, bookId, 0, "mid"))

                def get_price(side, idx):
                    if side == 'bid':
                        return book['b'][idx]['p'] if len(book['b']) > idx else 0
                    if side == 'ask':
                        return book['a'][idx]['p'] if len(book['a']) > idx else 0

                def get_vol(side, idx):
                    if side == 'bid':
                        return book['b'][idx]['q'] if len(book['b']) > idx else 0
                    if side == 'ask':
                        return book['a'][idx]['q'] if len(book['a']) > idx else 0

                updates.append((self.prometheus_books, 1.0,
                    wallet_addr, netuid, simid, self.simulation_timestamp, simulation_duration, bookId,
                    get_price('bid',4), get_vol('bid',4), get_price('bid',3), get_vol('bid',3), get_price('bid',2), get_vol('bid',2),
                    get_price('bid',1), get_vol('bid',1), get_price('bid',0), get_vol('bid',0),
                    get_price('ask',4), get_vol('ask',4), get_price('ask',3), get_vol('ask',3), get_price('ask',2), get_vol('ask',2),
                    get_price('ask',1), get_vol('ask',1), get_price('ask',0), get_vol('ask',0),
                    "books"
                ))
            if book['e']:
                trades = [event for event in book['e'] if event['y'] == 't']
                if trades:
                    last_trade = trades[-1]
                    fp = self.fundamental_price.get(bookId) if self.fundamental_price else None
                    if fp is not None:
                        if isinstance(fp, pd.Series):
                            updates.append((book_gauges,
                                fp.iloc[-1],
                                wallet_addr, netuid, simid, bookId, 0, "fundamental_price"))
                        elif fp:
                            updates.append((book_gauges,
                                fp,
                                wallet_addr, netuid, simid, bookId, 0, "fundamental_price"))
                        else:
                            _remove_and_evict(self._child_cache, book_gauges,
                                wallet_addr, netuid, simid, bookId, 0, "fundamental_price")

                    updates.append((book_gauges, last_trade['p'],
                        wallet_addr, netuid, simid, bookId, 0, "trade_price"))
                    updates.append((book_gauges, sum([trade['q'] for trade in trades]),
                        wallet_addr, netuid, simid, bookId, 0, "trade_volume"))
                    updates.append((book_gauges, sum([trade['q'] for trade in trades if trade['s'] == 0]),
                        wallet_addr, netuid, simid, bookId, 0, "trade_buy_volume"))
                    updates.append((book_gauges, sum([trade['q'] for trade in trades if trade['s'] == 1]),
                        wallet_addr, netuid, simid, bookId, 0, "trade_sell_volume"))

                    has_new_trades = True
            if getattr(self.simulation, 'fee_policy', None) and self.simulation.fee_policy.fee_type == 'dynamic':
                DISMTR = self.last_state.books[bookId].get('r', self.last_state.books[bookId].get('mtr', 0))
                DISmakerRate = self.last_state.accounts[0][bookId]['f']['m']
                DIStakerRate = self.last_state.accounts[0][bookId]['f']['t']
                updates.append((book_gauges, DISmakerRate,
                        wallet_addr, netuid, simid, bookId, 0, "dynamic_maker_rate"))
                updates.append((book_gauges, DIStakerRate,
                        wallet_addr, netuid, simid, bookId, 0, "dynamic_taker_rate"))
                updates.append((book_gauges, DISMTR,
                        wallet_addr, netuid, simid, bookId, 0, "maker_taker_ratio"))
        bt.logging.debug(f"Book metrics collected ({time.time()-book_start:.4f}s).")

        if has_new_trades:
            bt.logging.debug("Collecting trade metrics...")
            start = time.time()
            for bookId, trades in self.recent_trades.items():
                # slot=0 is the most recent trade; values that were labels become
                # the gauge value keyed by trade_gauge_name. timestamp is emitted in
                # seconds (ns/1e9) to stay float64 integer-exact; the Grafana table
                # formats it (the human-readable timestamp_str label is dropped).
                for slot, trade in enumerate(reversed(trades)):
                    for name, val in (
                        ("timestamp", trade.timestamp / 1e9),
                        ("price", trade.price),
                        ("volume", trade.quantity),
                        ("maker_fee", trade.maker_fee),
                        ("taker_fee", trade.taker_fee),
                        ("side", trade.side),
                        ("taker_agent_id", trade.taker_agent_id),
                        ("maker_agent_id", trade.maker_agent_id),
                        ("aggressing_order_id", trade.taker_id),
                        ("resting_order_id", trade.maker_id),
                        ("trade_id", trade.id),
                    ):
                        updates.append((self.prometheus_trades, val,
                            wallet_addr, netuid, simid, bookId, slot, name))

            bt.logging.debug(f"Trade metrics collected ({time.time()-start:.4f}s).")

        if not self.last_state.accounts:
            bt.logging.info(f"Applying {len(updates)} metric updates...")
            apply_start = time.time()
            remaining, _, _ = self._apply_snapshot_collectors(updates)
            for update in remaining:
                _set_if_changed(*update)
            bt.logging.info(f"Applied {len(updates)} updates in {time.time()-apply_start:.4f}s")
            bt.logging.info(f"Metrics Published for Step {report_step} ({time.time()-report_start}s).")
            return
            
        bt.logging.debug("Computing miner metrics in worker process...")
        computation_start = time.time()
        volume_sums_snapshot = {uid: dict(books) for uid, books in self.volume_sums.items()}
        maker_volume_sums_snapshot = {uid: dict(books) for uid, books in self.maker_volume_sums.items()}
        taker_volume_sums_snapshot = {uid: dict(books) for uid, books in self.taker_volume_sums.items()}
        self_volume_sums_snapshot = {uid: dict(books) for uid, books in self.self_volume_sums.items()}
    
        validator_data = {
            'simulation_timestamp': self.simulation_timestamp,
            'step': self.step,
            'volume_sums': volume_sums_snapshot,
            'maker_volume_sums': maker_volume_sums_snapshot,
            'taker_volume_sums': taker_volume_sums_snapshot,
            'self_volume_sums': self_volume_sums_snapshot,
            'roundtrip_volume_sums': {uid: dict(books) for uid, books in self.roundtrip_volume_sums.items()},
            'inventory_history': self.inventory_history,
            'total_realized_pnl': self.total_realized_pnl,
            'realized_pnl_by_book': self.realized_pnl_by_book,
            'fee_sums': {uid: dict(books) for uid, books in self.fee_sums.items()},
            'activity_factors': self.activity_factors,
            'pnl_factors': self.pnl_factors,
            'kappa_values': self.kappa_values,
            'unnormalized_scores': self.unnormalized_scores,
            'scores': self.scores,
            'gentrx_scores': self.gentrx_scores,
            'book_count': self.simulation.book_count,
            'simulation_config': {
                'volumeDecimals': self.simulation.volumeDecimals,
                'baseDecimals': getattr(self.simulation, 'baseDecimals', self.simulation.volumeDecimals),
                'quoteDecimals': getattr(self.simulation, 'quoteDecimals', self.simulation.volumeDecimals),
            }
        }

        state_data = {
            'accounts': self.last_state.accounts,
            'books': self.last_state.books,
            'notices': self.last_state.notices,
        }
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self.report_executor, report_worker, validator_data, state_data)
        while not future.done():
            await asyncio.sleep(0.001)
        result = future.result()

        if result['error']:
            bt.logging.error(f"Error in report worker: {result['error']}\n{result.get('traceback', 'N/A')}")
            return

        bt.logging.debug(f"Miner metrics computed ({time.time()-computation_start:.4f}s).")

        metrics = result['metrics']
        miner_metrics = metrics['miner_metrics']
        daily_volumes = metrics['daily_volumes']
        daily_roundtrip_volumes = metrics['daily_roundtrip_volumes']
        self.realized_pnl_by_book = metrics['realized_pnl_by_book']
        _fee_sums_by_book = metrics.get('fee_sums_by_book', {})

        bt.logging.debug("Collecting agent book metrics...")
        start = time.time()

        bt.logging.debug("Pre-extracting inventory/kappa data...")
        extract_start = time.time()

        start_inventories = {}
        last_inventories = {}
        kappa_data = {}
        for agentId in self.last_state.accounts.keys():
            if agentId < 0:
                continue
            if agentId not in self.inventory_history or not self.inventory_history[agentId]:
                continue
            if len(self.inventory_history[agentId]) < 2:
                continue
            inv_values = list(self.inventory_history[agentId].values())
            start_inventories[agentId] = [i for i in inv_values if len(i) > 0][0]
            last_inventories[agentId] = inv_values[-1]
            kappa_data[agentId] = self.kappa_values.get(agentId)
        bt.logging.debug(f"Pre-extraction complete ({time.time()-extract_start:.4f}s)")

        for agentId, accounts in self.last_state.accounts.items():
            initial_balance_publish_status = {bookId: False for bookId in range(self.simulation.book_count)}
            for bookId, _account in accounts.items():
                if agentId in self.initial_balances and self.initial_balances[agentId][bookId]['BASE'] is not None and not self.initial_balances_published.get(agentId, False):
                    updates.append((agent_gauges, self.initial_balances[agentId][bookId]['BASE'],
                        wallet_addr, netuid, simid, bookId, agentId, "base_balance_initial"))
                    updates.append((agent_gauges, self.initial_balances[agentId][bookId]['QUOTE'],
                        wallet_addr, netuid, simid, bookId, agentId, "quote_balance_initial"))
                    updates.append((agent_gauges, self.initial_balances[agentId][bookId]['WEALTH'],
                        wallet_addr, netuid, simid, bookId, agentId, "wealth_initial"))
                    initial_balance_publish_status[bookId] = True
            if all(initial_balance_publish_status.values()):
                self.initial_balances_published[agentId] = True

            if agentId not in start_inventories:
                continue

            start_inv = start_inventories[agentId]
            last_inv = last_inventories[agentId]
            kappas = kappa_data[agentId]

            for bookId, account in accounts.items():
                updates.append((agent_gauges, account['bb']['t'], wallet_addr, netuid, simid, bookId, agentId, "base_balance_total"))
                updates.append((agent_gauges, account['bb']['f'], wallet_addr, netuid, simid, bookId, agentId, "base_balance_free"))
                updates.append((agent_gauges, account['bb']['r'], wallet_addr, netuid, simid, bookId, agentId, "base_balance_reserved"))
                updates.append((agent_gauges, account['qb']['t'], wallet_addr, netuid, simid, bookId, agentId, "quote_balance_total"))
                updates.append((agent_gauges, account['qb']['f'], wallet_addr, netuid, simid, bookId, agentId, "quote_balance_free"))
                updates.append((agent_gauges, account['qb']['r'], wallet_addr, netuid, simid, bookId, agentId, "quote_balance_reserved"))
                updates.append((agent_gauges, account['bl'], wallet_addr, netuid, simid, bookId, agentId, "base_loan"))
                updates.append((agent_gauges, account['bc'], wallet_addr, netuid, simid, bookId, agentId, "base_collateral"))
                updates.append((agent_gauges, account['ql'], wallet_addr, netuid, simid, bookId, agentId, "quote_loan"))
                updates.append((agent_gauges, account['qc'], wallet_addr, netuid, simid, bookId, agentId, "quote_collateral"))
                if account['f']['v']:
                    updates.append((agent_gauges, account['f']['v'], wallet_addr, netuid, simid, bookId, agentId, "fees_traded_volume"))
                updates.append((agent_gauges, account['f']['m'], wallet_addr, netuid, simid, bookId, agentId, "fees_maker_rate"))
                updates.append((agent_gauges, account['f']['t'], wallet_addr, netuid, simid, bookId, agentId, "fees_taker_rate"))
                updates.append((agent_gauges, last_inv.get(bookId, 0.0), wallet_addr, netuid, simid, bookId, agentId, "inventory_value"))
                updates.append((agent_gauges, last_inv.get(bookId, 0.0) - start_inv.get(bookId, 0.0), wallet_addr, netuid, simid, bookId, agentId, "pnl"))
                if agentId in self.realized_pnl_by_book:
                    book_realized_pnl = self.realized_pnl_by_book[agentId].get(bookId, 0.0)
                    updates.append((agent_gauges, book_realized_pnl, wallet_addr, netuid, simid, bookId, agentId, "realized_pnl"))
                else:
                    updates.append((agent_gauges, 0.0, wallet_addr, netuid, simid, bookId, agentId, "realized_pnl"))
                _agent_fee_books = _fee_sums_by_book.get(agentId, _fee_sums_by_book.get(str(agentId), {}))
                updates.append((agent_gauges, _agent_fee_books.get(bookId, _agent_fee_books.get(str(bookId), 0.0)),
                                wallet_addr, netuid, simid, bookId, agentId, "net_fee"))
                updates.append((agent_gauges, daily_volumes[agentId][bookId]['total'], wallet_addr, netuid, simid, bookId, agentId, "daily_volume"))
                updates.append((agent_gauges, daily_volumes[agentId][bookId]['maker'], wallet_addr, netuid, simid, bookId, agentId, "daily_maker_volume"))
                updates.append((agent_gauges, daily_volumes[agentId][bookId]['taker'], wallet_addr, netuid, simid, bookId, agentId, "daily_taker_volume"))
                updates.append((agent_gauges, daily_volumes[agentId][bookId]['self'], wallet_addr, netuid, simid, bookId, agentId, "daily_self_volume"))
                updates.append((agent_gauges, daily_roundtrip_volumes[agentId][bookId], wallet_addr, netuid, simid, bookId, agentId, "daily_roundtrip_volume"))
                updates.append((agent_gauges, self.activity_factors.get(agentId, {}).get(bookId, 0.0), wallet_addr, netuid, simid, bookId, agentId, "activity_factor"))
                updates.append((agent_gauges, self.pnl_factors.get(agentId, {}).get(bookId, 1.0), wallet_addr, netuid, simid, bookId, agentId, "pnl_factor"))
                if kappas:
                    if kappas['books'][bookId] is not None:
                        updates.append((agent_gauges, kappas['books'][bookId], wallet_addr, netuid, simid, bookId, agentId, "kappa"))
                    else:
                        _remove_and_evict(self._child_cache, agent_gauges, wallet_addr, netuid, simid, bookId, agentId, "kappa")
                    if 'books_weighted' in kappas and kappas['books_weighted'][bookId] is not None:
                        updates.append((agent_gauges, kappas['books_weighted'][bookId], wallet_addr, netuid, simid, bookId, agentId, "weighted_kappa"))
                    else:
                        _remove_and_evict(self._child_cache, agent_gauges, wallet_addr, netuid, simid, bookId, agentId, "weighted_kappa")
                else:
                    _remove_and_evict(self._child_cache, agent_gauges, wallet_addr, netuid, simid, bookId, agentId, "kappa")
        bt.logging.debug(f"Agent book metrics collected ({time.time()-start:.4f}s).")

        bt.logging.debug("Collecting miner trade metrics...")
        start = time.time()
        for agentId, notices in self.last_state.notices.items():
            if agentId < 0:
                continue
            for notice in notices:
                if notice['y'] in ["EVENT_TRADE", "ET"]:
                    has_new_miner_trades = True
                    break
            if has_new_miner_trades:
                break
        if has_new_miner_trades:
            for uid, book_miner_trades in self.recent_miner_trades.items():
                for bookId, miner_trades in book_miner_trades.items():
                    if len(miner_trades) > 0:
                        last_maker_trade = None
                        last_taker_trade = None
                        # Bounded-slot shape (slot=0 most recent); numeric fields move
                        # from labels into the gauge value keyed by miner_trade_gauge_name.
                        # role is encoded 0=maker / 1=taker; timestamp emitted in seconds.
                        for slot, (miner_trade, role) in enumerate(reversed(self.recent_miner_trades[uid][bookId])):
                            side = miner_trade.side if role == 'taker' else int(not miner_trade.side)
                            fee = miner_trade.makerFee if role == 'maker' else miner_trade.takerFee
                            for name, val in (
                                ("timestamp", miner_trade.timestamp / 1e9),
                                ("price", miner_trade.price),
                                ("volume", miner_trade.quantity),
                                ("fee", fee),
                                ("side", side),
                                ("role", 1 if role == 'taker' else 0),
                            ):
                                updates.append((self.prometheus_miner_trades, val,
                                    wallet_addr, netuid, simid, miner_trade.bookId, uid, slot, name))
                            if role == 'maker':
                                last_maker_trade = miner_trade
                            if role == 'taker':
                                last_taker_trade = miner_trade
                        if last_maker_trade:
                            updates.append((agent_gauges, last_maker_trade.makerFeeRate, wallet_addr, netuid, simid, bookId, uid, "fees_last_maker_rate"))
                        if last_taker_trade:
                            updates.append((agent_gauges, last_taker_trade.takerFeeRate, wallet_addr, netuid, simid, bookId, uid, "fees_last_taker_rate"))
        bt.logging.debug(f"Miner trade metrics collected ({time.time()-start:.4f}s).")

        bt.logging.debug("Collecting miner metrics...")
        self.prometheus_miners.clear()
        self.prometheus_miner_identity.clear()
        start = time.time()
        for agentId in miner_metrics:
            m = miner_metrics[agentId]

            updates.append((miner_gauges, m['total_base_balance'], wallet_addr, netuid, simid, agentId, "total_base_balance"))
            updates.append((miner_gauges, m['total_base_loan'], wallet_addr, netuid, simid, agentId, "total_base_loan"))
            updates.append((miner_gauges, m['total_base_collateral'], wallet_addr, netuid, simid, agentId, "total_base_collateral"))
            updates.append((miner_gauges, m['total_quote_balance'], wallet_addr, netuid, simid, agentId, "total_quote_balance"))
            updates.append((miner_gauges, m['total_quote_loan'], wallet_addr, netuid, simid, agentId, "total_quote_loan"))
            updates.append((miner_gauges, m['total_quote_collateral'], wallet_addr, netuid, simid, agentId, "total_quote_collateral"))
            updates.append((miner_gauges, m['total_inventory_value'], wallet_addr, netuid, simid, agentId, "total_inventory_value"))
            updates.append((miner_gauges, m['pnl'], wallet_addr, netuid, simid, agentId, "pnl"))
            updates.append((miner_gauges, m['total_realized_pnl'], wallet_addr, netuid, simid, agentId, "total_realized_pnl"))

            updates.append((miner_gauges, m['total_daily_volume']['total'], wallet_addr, netuid, simid, agentId, "total_daily_volume"))
            updates.append((miner_gauges, m['total_daily_volume']['maker'], wallet_addr, netuid, simid, agentId, "total_daily_maker_volume"))
            updates.append((miner_gauges, m['total_daily_volume']['taker'], wallet_addr, netuid, simid, agentId, "total_daily_taker_volume"))
            updates.append((miner_gauges, m['total_daily_volume']['self'], wallet_addr, netuid, simid, agentId, "total_daily_self_volume"))

            updates.append((miner_gauges, m['average_daily_volume']['total'], wallet_addr, netuid, simid, agentId, "average_daily_volume"))
            updates.append((miner_gauges, m['average_daily_volume']['maker'], wallet_addr, netuid, simid, agentId, "average_daily_maker_volume"))
            updates.append((miner_gauges, m['average_daily_volume']['taker'], wallet_addr, netuid, simid, agentId, "average_daily_taker_volume"))
            updates.append((miner_gauges, m['average_daily_volume']['self'], wallet_addr, netuid, simid, agentId, "average_daily_self_volume"))

            updates.append((miner_gauges, m['min_daily_volume']['total'], wallet_addr, netuid, simid, agentId, "min_daily_volume"))
            updates.append((miner_gauges, m['min_daily_volume']['maker'], wallet_addr, netuid, simid, agentId, "min_daily_maker_volume"))
            updates.append((miner_gauges, m['min_daily_volume']['taker'], wallet_addr, netuid, simid, agentId, "min_daily_taker_volume"))
            updates.append((miner_gauges, m['min_daily_volume']['self'], wallet_addr, netuid, simid, agentId, "min_daily_self_volume"))
            
            updates.append((miner_gauges, m['total_roundtrip_volume'], wallet_addr, netuid, simid, agentId, "total_roundtrip_volume"))
            updates.append((miner_gauges, m['average_roundtrip_volume'], wallet_addr, netuid, simid, agentId, "average_roundtrip_volume"))
            updates.append((miner_gauges, m['min_roundtrip_volume'], wallet_addr, netuid, simid, agentId, "min_roundtrip_volume"))

            updates.append((miner_gauges, m['activity_factor'], wallet_addr, netuid, simid, agentId, "activity_factor"))
            updates.append((miner_gauges, m['pnl_factor'], wallet_addr, netuid, simid, agentId, "pnl_factor"))

            if m['kappa'] is not None:
                updates.append((miner_gauges, m['kappa'], wallet_addr, netuid, simid, agentId, "kappa"))
                if m['activity_weighted_normalized_median'] is not None:
                    updates.append((miner_gauges, m['activity_weighted_normalized_median'], wallet_addr, netuid, simid, agentId, "activity_weighted_normalized_median_kappa"))
                if m['kappa_penalty'] is not None:
                    updates.append((miner_gauges, m['kappa_penalty'], wallet_addr, netuid, simid, agentId, "kappa_penalty"))
                if m['kappa_score'] is not None:
                    updates.append((miner_gauges, m['kappa_score'], wallet_addr, netuid, simid, agentId, "kappa_score"))
                if m['pnl_score'] is not None:
                    updates.append((miner_gauges, m['pnl_score'], wallet_addr, netuid, simid, agentId, "pnl_score"))
                else:
                    _remove_and_evict(self._child_cache, miner_gauges, wallet_addr, netuid, simid, agentId, "pnl_score")
                if m['combined_score'] is not None:
                    updates.append((miner_gauges, m['combined_score'], wallet_addr, netuid, simid, agentId, "combined_score"))
                else:
                    _remove_and_evict(self._child_cache, miner_gauges, wallet_addr, netuid, simid, agentId, "combined_score")
            else:
                _remove_and_evict(self._child_cache, miner_gauges, wallet_addr, netuid, simid, agentId, "kappa")

            updates.append((miner_gauges, m['unnormalized_score'], wallet_addr, netuid, simid, agentId, "unnormalized_score"))
            updates.append((miner_gauges, m['score'], wallet_addr, netuid, simid, agentId, "score"))
            updates.append((miner_gauges, m['gentrx_score'], wallet_addr, netuid, simid, agentId, "gentrx_score"))
            updates.append((miner_gauges, m['placement'], wallet_addr, netuid, simid, agentId, "placement"))

            updates.append((miner_gauges, (self.metagraph.trust[agentId] if len(self.metagraph.trust) > agentId else 0.0), wallet_addr, netuid, simid, agentId, "trust"))
            updates.append((miner_gauges, (self.metagraph.consensus[agentId] if len(self.metagraph.consensus) > agentId else 0.0), wallet_addr, netuid, simid, agentId, "consensus"))
            updates.append((miner_gauges, (self.metagraph.incentive[agentId] if len(self.metagraph.incentive) > agentId else 0.0), wallet_addr, netuid, simid, agentId, "incentive"))
            updates.append((miner_gauges, (self.metagraph.emission[agentId] if len(self.metagraph.emission) > agentId else 0.0), wallet_addr, netuid, simid, agentId, "emission"))

            _ms = self.miner_stats.get(agentId)
            if _ms and _ms['requests'] >= 100:
                updates.append((miner_gauges, _ms['requests'], wallet_addr, netuid, simid, agentId, "requests"))
                updates.append((miner_gauges, _ms['requests'] - _ms['failures'] - _ms['timeouts'] - _ms['rejections'], wallet_addr, netuid, simid, agentId, "success"))
                updates.append((miner_gauges, _ms['failures'], wallet_addr, netuid, simid, agentId, "failures"))
                updates.append((miner_gauges, _ms['timeouts'], wallet_addr, netuid, simid, agentId, "timeouts"))
                updates.append((miner_gauges, _ms['rejections'], wallet_addr, netuid, simid, agentId, "rejections"))
                updates.append((miner_gauges, (sum(_ms['call_time']) / len(_ms['call_time']) if _ms['call_time'] else 0), wallet_addr, netuid, simid, agentId, "call_time"))
                self.miner_stats[agentId] = {'requests': 0, 'timeouts': 0, 'failures': 0, 'rejections': 0, 'call_time': []}

            _set_if_changed_metric(
                self.prometheus_miners,
                1.0,
                wallet=wallet_addr,
                netuid=netuid,
                sim_id=simid,
                agent_id=agentId,
                hotkey=(self.metagraph.hotkeys[agentId] if len(getattr(self.metagraph, 'hotkeys', [])) > agentId else ""),
                coldkey=(self.metagraph.coldkeys[agentId] if len(getattr(self.metagraph, 'coldkeys', [])) > agentId else ""),
                timestamp=self.simulation_timestamp,
                timestamp_str=duration_from_timestamp(self.simulation_timestamp),
                placement=m['placement'],
                base_balance=m['total_base_balance'],
                base_loan=m['total_base_loan'],
                base_collateral=m['total_base_collateral'],
                quote_balance=m['total_quote_balance'],
                quote_loan=m['total_quote_loan'],
                quote_collateral=m['total_quote_collateral'],
                inventory_value=m['total_inventory_value'],
                inventory_value_change=m['inventory_value_change'],
                pnl=m['pnl'],
                pnl_change=m['pnl_change'],
                total_realized_pnl=m['total_realized_pnl'],
                total_daily_volume=m['total_daily_volume']['total'],
                min_daily_volume=m['min_daily_volume']['total'],
                average_daily_volume=m['average_daily_volume']['total'],
                total_roundtrip_volume=m['total_roundtrip_volume'],
                min_roundtrip_volume=m['min_roundtrip_volume'], 
                average_roundtrip_volume=m['average_roundtrip_volume'],
                activity_factor=m['activity_factor'],
                pnl_factor=m['pnl_factor'],
                kappa=m['kappa'],
                kappa_penalty=m['kappa_penalty'],
                kappa_score=m['kappa_score'],
                pnl_score=m['pnl_score'],
                combined_score=m['combined_score'],
                gentrx_score=m['gentrx_score'],
                unnormalized_score=m['unnormalized_score'],
                score=m['score'],
                miner_gauge_name='miners'
            )
            _set_if_changed_metric(
                self.prometheus_miner_identity,
                1.0,
                wallet=wallet_addr,
                netuid=netuid,
                sim_id=simid,
                agent_id=agentId,
                hotkey=(self.metagraph.hotkeys[agentId] if len(getattr(self.metagraph, 'hotkeys', [])) > agentId else ""),
                coldkey=(self.metagraph.coldkeys[agentId] if len(getattr(self.metagraph, 'coldkeys', [])) > agentId else ""),
            )
        bt.logging.debug(f"Miner metrics collected ({time.time()-start:.4f}s).")
        
        # Diagnostic (env-gated, off by default): dump this cycle's agent series
        # (value, label-tuple) once, for the offline eager-vs-collector golden diff.
        # Works identically on the eager (0.4.6) and collector builds.
        _dump_path = os.environ.get('REPORT_DUMP_AGENT')
        if _dump_path and not getattr(self, '_agent_dumped', False):
            import pickle
            _ag_series = [(u[1], u[2:]) for u in updates if u[0] is self.prometheus_agent_gauges]
            try:
                with open(_dump_path, 'wb') as _f:
                    pickle.dump(_ag_series, _f)
                self._agent_dumped = True
                bt.logging.warning(f"REPORT_DUMP_AGENT: dumped {len(_ag_series)} agent series to {_dump_path}")
            except Exception as _e:
                bt.logging.warning(f"REPORT_DUMP_AGENT dump failed: {_e}")

        bt.logging.info(f"Applying {len(updates)} metric updates...")
        apply_start = time.time()
        cache = self._child_cache
        # agent_gauges (~1.8M-series bulk, carry-forward) + trades/books/
        # miner_trades (~890K, replace) all route into fresh per-cycle snapshots
        # swapped in at the end — no per-series labels()/.set() and no clear().
        # Everything else stays on the persistent-child cache (cheap on unchanged).
        remaining, agent_count, cleared_count = self._apply_snapshot_collectors(updates)
        for update in remaining:
            _set_cached(cache, update[0], update[1], update[2:])
        bt.logging.info(
            f"Applied {len(updates)} updates in {time.time()-apply_start:.4f}s "
            f"({agent_count} agent_gauges + {cleared_count} trades/books/miner_trades "
            f"via snapshot collectors)"
        )
        
        bt.logging.info(f"Metrics Published for Step {report_step} ({time.time()-report_start}s).")
    except Exception as ex:
        self.pagerduty_alert(f"Unable to publish metrics : {ex}", details={"traceback": traceback.format_exc()})
    finally:
        self.shared_state_reporting = False
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.logging.set_info()
    
    parser.add_argument('--netuid', type=int, default=1)
    parser.add_argument('--logging.level', type=str, default="info")
    parser.add_argument('--prometheus.port', type=int, default=9001)
    parser.add_argument('--prometheus.level', type=str, default='INFO')
    parser.add_argument('--cpu-cores', type=str, default=None)
    parser.add_argument('--ipc-prefix', type=str, default='validator',
                        help='Prefix for POSIX IPC resource names — "validator" for simulation, "exchange" for exchange mode')

    config = bt.Config(parser)
    bt.logging(config=config)

    if config.cpu_cores:
        cores = [int(c) for c in config.cpu_cores.split(',')]
        os.sched_setaffinity(0, set(cores))
        bt.logging.info(f"Reporting service assigned to cores: {cores}")
    
    service = ReportingService(config)
    
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        bt.logging.info("Reporting service stopped by user")
    except Exception as e:
        bt.logging.error(f"Reporting service crashed: {e}")
        bt.logging.error(traceback.format_exc())
        sys.exit(1)