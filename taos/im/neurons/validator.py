# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

if __name__ != "__mp_main__":
    import logging
    import os
    # Must precede `import bittensor` below: bittensor 10.3.2 builds its logging
    # singleton at import time and returns an empty config (disabling logging)
    # unless BT_NO_PARSE_CLI_ARGS is "false" in the env at that point. The run
    # scripts also export this, but set it here so a direct `python validator.py`
    # launch is equally safe. No-op on bittensor <10.3.2.
    os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "false")
    import json
    import signal
    import sys
    import platform
    import time
    from datetime import datetime
    import argparse
    import torch
    import traceback
    import msgspec
    import asyncio
    import posix_ipc
    import mmap
    import msgpack
    import atexit
    import subprocess
    import struct
    import select
    import copy
    from typing import TYPE_CHECKING, Any
    from ypyjson import YpyObject

    if TYPE_CHECKING:
        from taos.im.protocol.models import MarketSimulationConfig

    import bittensor as bt


    from GenTRX.src.bt_log import gtx_log

    import uvicorn
    from fastapi import FastAPI, APIRouter
    from fastapi import Request
    import threading
    from threading import Thread, Lock, Event
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    from git import Repo
    from pathlib import Path

    from taos.common.neurons.validator import BaseValidatorNeuron
    from taos.im.utils import duration_from_timestamp
    from taos.im.utils.affinity import get_core_allocation

    from taos.im.config import add_im_validator_args
    from taos.im.protocol.simulator import SimulatorResponseBatch
    from taos.im.protocol import MarketSimulationStateUpdate, FinanceEventNotification
    from taos.im.protocol.events import SimulationStartEvent
    from taos.im.validator.engines import NormalizedState, NormalizedTradeEvent, SimulationEngine

    # Optional exchange-mode component; not part of this tree. Import is guarded, so a validator without
    # it starts normally and simply does not surface those notices.
    try:
        from taos.im.validator import exchange_notices as _exchange_notices
    except ImportError:
        _exchange_notices = None

    async def _push_fill_notifications(trade_events: list, url: str) -> None:
        """Fire-and-forget: push fill notifications directly to the data service
        immediately after on-chain execution, bypassing the heavy ingest pipeline."""
        if not url or not trade_events:
            return
        _secret = os.environ.get("INGEST_SECRET", "")
        _headers = {"Content-Type": "application/json"}
        if _secret:
            _headers["x-ingest-secret"] = _secret

        async def _post_with_retry(client, endpoint: str, body: bytes, label: str) -> bool:
            """POST with up to 3 attempts on ConnectError (service restart window)."""
            _delays = [1.0, 3.0]
            for _attempt in range(3):
                try:
                    _resp = await client.post(endpoint, content=body, headers=_headers)
                    if _resp.status_code == 200:
                        bt.logging.info(f"_push_fill_notifications: {label}")
                    else:
                        bt.logging.warning(f"_push_fill_notifications: {label} HTTP {_resp.status_code}: {_resp.text[:120]}")
                    return True
                except httpx.ConnectError as _exc:
                    if _attempt < len(_delays):
                        bt.logging.warning(f"_push_fill_notifications: {label} ConnectError (attempt {_attempt+1}/3), retrying in {_delays[_attempt]}s: {_exc!r}")
                        await asyncio.sleep(_delays[_attempt])
                    else:
                        bt.logging.warning(f"_push_fill_notifications: {label} failed after 3 attempts: {_exc!r}")
                        return False
                except Exception as _exc:
                    bt.logging.warning(f"_push_fill_notifications: {label} failed: {_exc!r}")
                    return False
            return False

        try:
            import json as _json
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, write=5.0, read=5.0, pool=5.0)) as client:
                for _te in trade_events:
                    _taker_uid = getattr(_te, 'taker_uid', None)
                    _maker_uid = getattr(_te, 'maker_uid', None)
                    _nid      = getattr(_te, 'book_id', 0)
                    _p        = getattr(_te, 'price', 0.0)
                    _q        = getattr(_te, 'quantity', 0.0)
                    _side     = getattr(_te, 'side', 0)
                    _ts           = getattr(_te, 'timestamp', int(time.time() * 1e9))
                    _order_id     = getattr(_te, 'order_id', None)
                    _close_reason = getattr(_te, 'close_reason', None)
                    _linked_oid   = getattr(_te, 'linked_order_id', None)

                    # Push for taker
                    if _taker_uid is not None and _taker_uid != 0:
                        _notif = {
                            "type":         "fill",
                            "uid":          _taker_uid,
                            "netuid":       _nid,
                            "price":        _p,
                            "alpha_volume": _q,
                            "qty":          _q,
                            "direction":    _side,
                            "side":         "buy" if _side == 0 else "sell",
                            "role":         "taker",
                            "taker_uid":    _taker_uid,
                            "maker_uid":    _maker_uid,
                            "timestamp":    _ts,
                            "order_id":     _order_id,
                        }
                        if _close_reason is not None:
                            _notif["close_reason"]    = _close_reason
                        if _linked_oid is not None:
                            _notif["linked_order_id"] = _linked_oid
                        await _post_with_retry(
                            client,
                            f"{url}/api/v1/agents/{_taker_uid}/notify",
                            _json.dumps(_notif).encode('utf-8'),
                            f"taker uid={_taker_uid} netuid={_nid} side={_side} qty={_q:.4f} price={_p:.6f}",
                        )

                    # Push for maker when it's a distinct counterparty (not same as taker)
                    if _maker_uid is not None and _maker_uid != 0 and _maker_uid != _taker_uid:
                        _maker_side = 1 - _side  # maker is opposite of taker
                        _maker_notif = {
                            "type":         "fill",
                            "uid":          _maker_uid,
                            "netuid":       _nid,
                            "price":        _p,
                            "alpha_volume": _q,
                            "qty":          _q,
                            "direction":    _maker_side,
                            "side":         "buy" if _maker_side == 0 else "sell",
                            "role":         "maker",
                            "taker_uid":    _taker_uid,
                            "maker_uid":    _maker_uid,
                            "timestamp":    _ts,
                            "order_id":     _order_id,
                        }
                        if _close_reason is not None:
                            _maker_notif["close_reason"]    = _close_reason
                        if _linked_oid is not None:
                            _maker_notif["linked_order_id"] = _linked_oid
                        await _post_with_retry(
                            client,
                            f"{url}/api/v1/agents/{_maker_uid}/notify",
                            _json.dumps(_maker_notif).encode('utf-8'),
                            f"maker uid={_maker_uid} netuid={_nid} side={_maker_side} qty={_q:.4f} price={_p:.6f}",
                        )
        except Exception as _exc:
            bt.logging.warning(f"_push_fill_notifications: setup failed: {_exc!r}")

    async def _push_mvtrx(payload: dict, url: str) -> None:
        """Delegate to the optional data-service push module when present."""
        try:
            from taos.im.validator.mvtrx_push import push as _push
        except ImportError:
            return
        await _push(payload, url)

    class Validator(BaseValidatorNeuron):
        """
        Intelligent market simulation validator implementation.

        The validator is run as a FastAPI client in order to receive messages from the simulator engine for processing and forwarding to miners.
        Metagraph maintenance, weight setting, state persistence and other general bittensor routines are executed in a separate thread.
        The validator also handles publishing of metrics via Prometheus for visualization and analysis, as well as retrieval and recording of seed data for simulation price process generation.
        """

        # Instance state populated at init/load and by the delegated engine,
        # persistence, and trade modules (which operate on `self: Validator`).
        # Annotation-only declarations — no runtime effect; they document the
        # validator's state surface so static analysis resolves these attributes.
        simulation: "MarketSimulationConfig"
        roundtrip_volumes: dict
        roundtrip_volume_sums: dict
        volume_sums: dict
        maker_volume_sums: dict
        taker_volume_sums: dict
        self_volume_sums: dict
        fee_sums: dict
        inventory_history: dict
        realized_pnl_history: dict
        open_positions: dict
        pending_notices: dict
        kappa_cache: dict
        _gentrx_ema: dict
        seed_process: Any
        sltp_enabled: bool
        xml_config: str
        simulation_state_file: str
        validator_state_file: str
        simulator_config_file: str

        @classmethod
        def add_args(cls, parser: argparse.ArgumentParser) -> None:
            """
            Registers Intelligent-Markets-specific CLI configuration parameters.

            Args:
                parser (argparse.ArgumentParser): The main argument parser to extend.

            Returns:
                None
            """
            add_im_validator_args(cls, parser)

        def _setup_signal_handlers(self):
            """
            Registers OS signal handlers for graceful shutdown.

            Behavior:
                - Captures SIGINT, SIGTERM, and SIGHUP (if available).
                - Logs the received signal.
                - Triggers full validator cleanup.
                - Exits the process cleanly.

            Returns:
                None
            """
            def signal_handler(signum, frame):
                signal_name = signal.Signals(signum).name
                bt.logging.info(f"Received {signal_name}, initiating graceful shutdown...")
                self.cleanup()
                sys.exit(0)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, signal_handler)

        def _reap_orphaned_services(self):
            """Kill service subprocesses orphaned by a previous validator instance.

            pm2's kill timeout (~1.6s default) SIGKILLs the validator long before
            its graceful cleanup reaches the child services, so query/report/seed
            processes survive with PPID=1 — and the orphaned query service poaches
            the PERSISTENT posix MQs, producing the recurring 'Query service
            notification timeout' failure. PPID==1 + our service script in the
            cmdline is provably orphaned (a live validator's children carry its
            pid as PPID), so the kill is safe even with multiple validators on
            one box. Runs once at boot, before any IPC is opened.
            """
            import psutil
            _patterns = (
                'validator/query.py', 'validator/report.py', 'validator/seed.py',
            )
            reaped = 0
            for proc in psutil.process_iter(['pid', 'ppid', 'cmdline']):
                try:
                    if proc.info['ppid'] != 1:
                        continue
                    cmd = ' '.join(proc.info.get('cmdline') or [])
                    if any(p in cmd for p in _patterns):
                        bt.logging.warning(
                            f"Reaping orphaned service (pid {proc.info['pid']}): {cmd[:120]}"
                        )
                        proc.kill()
                        reaped += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if reaped:
                bt.logging.warning(f"Reaped {reaped} orphaned service process(es) from a previous instance")

        def _start_query_service(self):
            """
            Launches the validator's query service and initializes POSIX IPC resources.

            Responsibilities:
                - Spawns the query subprocess with correct wallet and network parameters.
                - Waits for creation of shared memory blocks and message queues.
                - Initializes memory maps for request/response communication.
                - Verifies that the query service is alive during startup.
                - Raises a RuntimeError if IPC initialization fails or service dies.

            Returns:
                None

            Raises:
                RuntimeError: If IPC endpoints are not ready within timeout
                            or if the subprocess exits unexpectedly.
            """
            import fcntl as _fcntl

            bt.logging.info("Starting query service from: ../validator/query.py")

            # Reap any existing query subprocess before spawning a new one.
            # Without this, every restart orphans the old process (reparented to
            # init); it keeps reading the shared named IPC queue, stealing
            # query/deliver messages and notifying its now-stale pipe fd — which
            # the live validator reads back as b'' / notification timeouts.
            _old_q = getattr(self, 'query_process', None)
            if _old_q is not None and _old_q.poll() is None:
                bt.logging.warning(f"Reaping existing query service (pid {_old_q.pid}) before restart")
                try:
                    _old_q.terminate()
                    try:
                        _old_q.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        _old_q.kill()
                        _old_q.wait(timeout=2.0)
                except Exception as _e:
                    bt.logging.warning(f"Failed to reap old query service: {_e}")

            # Recreate the notify pipe on every call (required on restarts because the
            # write end was closed after the previous start, and a fresh subprocess needs
            # a live write fd to pass via pass_fds).
            try:
                os.close(self.query_notify_read)
            except OSError:
                pass
            try:
                os.close(self.query_notify_write)
            except OSError:
                pass
            self.query_notify_read, self.query_notify_write = os.pipe()
            flags = _fcntl.fcntl(self.query_notify_read, _fcntl.F_GETFL)
            _fcntl.fcntl(self.query_notify_read, _fcntl.F_SETFL, flags | os.O_NONBLOCK)
            bt.logging.info(f"Notification pipe: read_fd={self.query_notify_read}, write_fd={self.query_notify_write}")

            core_allocation = get_core_allocation(
                sltp_cores_count=int(os.environ.get("SLTP_CORES_COUNT", "0")),
                grad_server_cores=int(os.environ.get("GRAD_CORES_COUNT", "0")),
                sim_cores_count=self._read_block_count(),
                os_headroom=int(os.environ.get("OS_HEADROOM_CORES", "4")),
            )
            _engine = getattr(self.config, 'engine', 'simulation')
            self._query_ipc_prefix = 'exchange' if _engine == 'exchange' else 'validator'
            cmd = [
                sys.executable,
                '-u',
                '../validator/query.py',
                '--logging.trace' if self.config.logging.trace else '--logging.debug' if self.config.logging.debug else '--logging.info',
                '--wallet.path', self.config.wallet.path,
                '--wallet.name', self.config.wallet.name,
                '--wallet.hotkey', self.config.wallet.hotkey,
                '--subtensor.network', self.config.subtensor.network,
                '--netuid', str(self.config.netuid),
                '--neuron.timeout', str(self.config.neuron.timeout),
                '--neuron.global_query_timeout', str(self.config.neuron.global_query_timeout),
                '--compression.level', str(self.config.compression.level),
                '--compression.engine', self.config.compression.engine,
                '--compression.parallel_workers', str(self.config.compression.parallel_workers),
                '--cpu-cores', ','.join(map(str, core_allocation['query'])),
                '--notify-fd', str(self.query_notify_write),
                '--ipc-prefix', self._query_ipc_prefix,
            ]
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            self.query_process = subprocess.Popen(
                cmd,
                stderr=sys.stderr,
                env=env,
                pass_fds=(self.query_notify_write,)
            )
            bt.logging.info(f"Query service PID: {self.query_process.pid}")
            os.close(self.query_notify_write)
            self.query_notify_write = -1  # mark as closed

            queue_name = f"/{self._query_ipc_prefix}_query_{self.config.wallet.hotkey}"

            # Wait for the ready signal BEFORE opening IPC resources.
            # The query subprocess sends b'R' only after setup_ipc() finishes — which
            # unlinks any stale queues/shm and creates fresh ones.  Connecting to IPC
            # before this signal risks grabbing resources left over from a previous
            # (still-running) query service that will be unlinked moments later.
            bt.logging.info("Waiting for query service ready signal...")
            ready = False
            for _attempt in range(150):  # 15s max (150 × 0.1s)
                if self.query_process.poll() is not None:
                    raise RuntimeError(f"Query service died with exit code {self.query_process.returncode}")
                readable, _, _ = select.select([self.query_notify_read], [], [], 0.1)
                if readable:
                    ready_signal = os.read(self.query_notify_read, 1)
                    if ready_signal == b'R':
                        bt.logging.success("Query service ready!")
                        ready = True
                        break
                    elif ready_signal != b'':
                        bt.logging.warning(f"Unexpected ready signal: {ready_signal!r}")
                        ready = True
                        break
            if not ready:
                raise RuntimeError("Query service did not send ready signal within 15s")

            # Now connect to the fresh IPC resources.
            bt.logging.info("Connecting to query service IPC resources...")
            max_retries = 30  # 30 seconds max
            for attempt in range(max_retries):
                try:
                    self.request_queue = posix_ipc.MessageQueue(f"{queue_name}_req")
                    self.response_queue = posix_ipc.MessageQueue(f"{queue_name}_res")
                    self.request_shm = posix_ipc.SharedMemory(f"{queue_name}_req_shm")
                    self.response_shm = posix_ipc.SharedMemory(f"{queue_name}_res_shm")

                    self.request_mem = mmap.mmap(self.request_shm.fd, self.request_shm.size)
                    self.response_mem = mmap.mmap(self.response_shm.fd, self.response_shm.size)

                    bt.logging.info(f"Query service IPC connected (request_shm: {self.request_shm.size / 1024 / 1024:.0f}MB, response_shm: {self.response_shm.size / 1024 / 1024:.0f}MB)")
                    return
                except posix_ipc.ExistentialError:
                    if attempt == 0:
                        bt.logging.debug("IPC resources not ready yet, waiting...")
                    time.sleep(1)
                    if self.query_process.poll() is not None:
                        raise RuntimeError(f"Query service died with exit code {self.query_process.returncode}")
            raise RuntimeError("Timeout waiting for query service IPC resources")

        def _start_reporting_service(self):
            bt.logging.info("Starting reporting service from: ../validator/report.py")

            # Reap any existing reporting subprocess before spawning a new one —
            # same orphan hazard as the query service (a stale report.py keeps
            # holding the prometheus port + shared report IPC; see _start_query_service).
            _old_r = getattr(self, 'reporting_process', None)
            if _old_r is not None and _old_r.poll() is None:
                bt.logging.warning(f"Reaping existing reporting service (pid {_old_r.pid}) before restart")
                try:
                    _old_r.terminate()
                    try:
                        _old_r.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        _old_r.kill()
                        _old_r.wait(timeout=2.0)
                except Exception as _e:
                    bt.logging.warning(f"Failed to reap old reporting service: {_e}")

            self._reporting = False
            core_allocation = get_core_allocation(
                sltp_cores_count=int(os.environ.get("SLTP_CORES_COUNT", "0")),
                grad_server_cores=int(os.environ.get("GRAD_CORES_COUNT", "0")),
                sim_cores_count=self._read_block_count(),
                os_headroom=int(os.environ.get("OS_HEADROOM_CORES", "4")),
            )
            # Use mode-specific IPC names so sim and exchange validators don't share queues.
            _engine = getattr(self.config, 'engine', 'simulation')
            self._reporting_ipc_prefix = 'exchange' if _engine == 'exchange' else 'validator'
            cmd = [
                sys.executable,
                '-u',
                '../validator/report.py',
                '--logging.trace' if self.config.logging.trace else '--logging.debug' if self.config.logging.debug else '--logging.info',
                '--wallet.path', self.config.wallet.path,
                '--wallet.name', self.config.wallet.name,
                '--wallet.hotkey', self.config.wallet.hotkey,
                '--subtensor.network', self.config.subtensor.network,
                '--netuid', str(self.config.netuid),
                '--prometheus.port', str(self.config.prometheus.port),
                '--prometheus.level', str(self.config.prometheus.level),
                '--cpu-cores', ','.join(map(str, core_allocation['reporting'])),
                '--ipc-prefix', self._reporting_ipc_prefix,
            ]
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            self.reporting_process = subprocess.Popen(cmd, stderr=sys.stderr, env=env)
            bt.logging.info(f"Reporting service PID: {self.reporting_process.pid}")

            _pfx = self._reporting_ipc_prefix
            bt.logging.info(f"Waiting for reporting service IPC resources (prefix={_pfx!r})...")
            max_retries = 30
            for attempt in range(max_retries):
                try:
                    self.reporting_request_queue = posix_ipc.MessageQueue(f"/{_pfx}-report-req")
                    self.reporting_response_queue = posix_ipc.MessageQueue(f"/{_pfx}-report-res")
                    self.reporting_request_shm = posix_ipc.SharedMemory(f"/{_pfx}-report-data")
                    self.reporting_response_shm = posix_ipc.SharedMemory(f"/{_pfx}-report-response-data")

                    self.reporting_request_mem = mmap.mmap(self.reporting_request_shm.fd, self.reporting_request_shm.size)
                    self.reporting_response_mem = mmap.mmap(self.reporting_response_shm.fd, self.reporting_response_shm.size)

                    bt.logging.info(f"Reporting service ready (shm: {self.reporting_request_shm.size / 1024 / 1024:.0f}MB)")
                    return

                except posix_ipc.ExistentialError:
                    if attempt == 0:
                        bt.logging.debug("IPC resources not ready yet, waiting...")
                    time.sleep(1)

                    if self.reporting_process.poll() is not None:
                        raise RuntimeError(f"Reporting service died with exit code {self.reporting_process.returncode}")

            raise RuntimeError("Timeout waiting for reporting service IPC resources")

        def monitor(self) -> None:
            """
            Periodically checks simulator health and restarts if needed with timeout protection.

            Runs in a blocking loop:
                - Sleeps 5 minutes between checks.
                - Logs simulator availability.
                - Handles and logs unexpected exceptions with timeout protection.

            Returns:
                None
            """
            last_restart_time = 0
            restart_cooldown = 60.0
            check_timeout = 30.0
            restart_timeout = 120.0
            observe = bool(getattr(getattr(self.config, 'neuron', None), 'observe', False))

            while True:
                try:
                    time.sleep(300)
                    if observe:
                        # Observe mode reads chain state via the provider — there's
                        # no local simulator/exchange (LOB) engine to health-check
                        # or restart. Skip to avoid false PagerDuty alerts and the
                        # no-op "restart" log noise. (Thread stays alive on purpose:
                        # returning would trip the post-launch is_alive() check.)
                        continue
                    bt.logging.info("Checking simulator state...")
                    check_start = time.time()
                    try:
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(check_simulator, self)
                            try:
                                is_healthy = future.result(timeout=check_timeout)
                            except concurrent.futures.TimeoutError:
                                bt.logging.error(f"Simulator health check timed out after {check_timeout}s")
                                is_healthy = False
                        check_elapsed = time.time() - check_start
                        if not is_healthy:
                            time_since_restart = time.time() - last_restart_time
                            if time_since_restart < restart_cooldown:
                                bt.logging.warning(
                                    f"Simulator unhealthy but restart on cooldown "
                                    f"({time_since_restart:.1f}s < {restart_cooldown}s)"
                                )
                                continue
                            bt.logging.warning(f"Simulator unhealthy (check took {check_elapsed:.1f}s), restarting...")
                            restart_start = time.time()
                            try:
                                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                    restart_future = executor.submit(restart_simulator, self)
                                    try:
                                        restart_future.result(timeout=restart_timeout)
                                        last_restart_time = time.time()
                                        restart_elapsed = time.time() - restart_start
                                        bt.logging.info(f"Simulator restarted successfully ({restart_elapsed:.1f}s)")
                                    except concurrent.futures.TimeoutError:
                                        bt.logging.error(f"Simulator restart timed out after {restart_timeout}s")
                                        self.pagerduty_alert(
                                            f"Simulator restart timeout after {restart_timeout}s"
                                        )
                                        last_restart_time = time.time()
                            except Exception as restart_ex:
                                bt.logging.error(f"Simulator restart failed: {restart_ex}")
                                bt.logging.error(traceback.format_exc())
                                self.pagerduty_alert(
                                    f"Simulator restart failed: {restart_ex}",
                                    details={"trace": traceback.format_exc()}
                                )
                                last_restart_time = time.time()
                        else:
                            bt.logging.info(f"Simulator online! (check took {check_elapsed:.1f}s)")

                    except Exception as check_ex:
                        bt.logging.error(f"Simulator health check failed: {check_ex}")
                        bt.logging.error(traceback.format_exc())

                except Exception:
                    bt.logging.error(f"Failure in simulator monitor : {traceback.format_exc()}")
                    time.sleep(10)

        def monitor_sltp(self) -> None:
            """Delegate to the optional SL/TP monitor loop when installed."""
            try:
                from taos.im.validator.exetrx import monitor_sltp as _monitor
            except ImportError:
                return
            _monitor(self)

        def update_repo(self, end=False) -> bool:
            """
            Checks for source or config changes in the repository and reloads components.

            Behavior:
                - Pulls latest remote changes.
                - Rebuilds simulator when C++ or Python sources change.
                - Restarts simulator on configuration changes.
                - Updates validator process when its own Python source changes.
                - Handles special behavior on simulation end.

            Args:
                end (bool): Whether the update is performed during simulation shutdown.

            Returns:
                bool: True if update steps completed successfully, False on error.
            """
            endpoint = getattr(getattr(self.config, "subtensor", None), "chain_endpoint", "") or ""
            if "localhost" in endpoint or "127.0.0.1" in endpoint:
                # Auto-update assumes pm2 supervision; localnet runs raw python.
                return True
            try:
                validator_py_files_changed, simulator_config_changed, simulator_py_files_changed, simulator_cpp_files_changed = check_repo(self)
                remote = self.repo.remotes[self.config.repo.remote]

                exchange_mode = getattr(getattr(self, 'engine', None), 'mode', 'simulation') == 'exchange'

                if not end:
                    if validator_py_files_changed and not (simulator_cpp_files_changed or simulator_py_files_changed):
                        bt.logging.warning("VALIDATOR LOGIC UPDATED - PULLING AND DEPLOYING.")
                        remote.pull()
                        update_validator(self)
                else:
                    try:
                        remote.pull()
                    except Exception as ex:
                        self.pagerduty_alert(f"Failed to pull changes from repo on simulation end : {ex}")
                    if not exchange_mode:
                        if simulator_cpp_files_changed or simulator_py_files_changed:
                            bt.logging.warning("SIMULATOR SOURCE CHANGED")
                            rebuild_simulator(self)
                        if simulator_config_changed:
                            bt.logging.warning("SIMULATOR CONFIG CHANGED")
                        restart_simulator(self, end)
                    if validator_py_files_changed:
                        update_validator(self)
                return True
            except Exception as ex:
                self.pagerduty_alert(f"Failed to update repo : {ex}", details={"traceback" : traceback.format_exc()})
                return False

        # NB: compress_outputs / load_simulation_config now live on the simulation engine
        # (taos/im/validator/engines/simulation.py).  Retained here as a comment to make
        # the move explicit during the GenTRX merge.

        def __init__(self, config=None) -> None:
            """
            Initializes the Intelligent Markets validator node.

            Responsibilities:
                - Loads simulation configuration from XML.
                - Initializes metagraph and subnet info.
                - Sets up executors, event loops, state locks, and signal handlers.
                - Loads prior simulation/validator state if available.
                - Initializes metrics, reporting, and query service.
                - Starts IPC-backed query subprocess.

            Args:
                config: Validator configuration object.

            Raises:
                Exception: If the simulation config XML file is missing.
            """
            super(Validator, self).__init__(config=config)

            kappa_w = self.config.scoring.kappa.weight
            pnl_w = self.config.scoring.pnl.weight
            gentrx_sim_share = getattr(getattr(self.config.scoring, 'gentrx', None), 'simulation_share', 0.0) or 0.0
            burn_ratio = getattr(self.config.neuron, 'burn_ratio', 0.0) or 0.0
            tolerance = 1e-6

            trading_sum = kappa_w + pnl_w
            if abs(trading_sum - 1.0) > tolerance:
                error_msg = (
                    f"Trading-pool weights must sum to 1.0, got "
                    f"kappa={kappa_w:.6f} + pnl={pnl_w:.6f} = {trading_sum:.6f}."
                )
                bt.logging.error(error_msg)
                raise ValueError(error_msg)

            if not (0.0 <= gentrx_sim_share <= 1.0):
                error_msg = (
                    f"--scoring.gentrx.simulation_share must be in [0, 1], "
                    f"got {gentrx_sim_share:.6f}."
                )
                bt.logging.error(error_msg)
                raise ValueError(error_msg)

            bt.logging.info(
                f"Scoring weights validated: trading=(kappa={kappa_w:.4f}, pnl={pnl_w:.4f}), "
                f"gentrx.simulation_share={gentrx_sim_share:.4f}, burn_ratio={burn_ratio:.4f}"
            )

            # Initialize subnet info and other basic validator/simulation properties
            self.subnet_info = self.subtensor.get_metagraph_info(self.config.netuid)

            self.benchmark_agents = []
            self.benchmark_start_uid = self.subnet_info.max_uids
            if self.config.benchmark.agents:
                self._load_benchmark_agents()
            max_bm_uid = max((a['uid'] for a in self.benchmark_agents), default=self.subnet_info.max_uids - 1)
            self.effective_max_uids = max(self.subnet_info.max_uids, max_bm_uid + 1)
            self.scores = torch.zeros(
                self.effective_max_uids, dtype=torch.float32, device=self.device
            )
            self.gentrx_scores = torch.zeros(
                self.effective_max_uids, dtype=torch.float32, device=self.device
            )
            self.last_state = None
            self.last_response = None
            self.msgpack_error_counter = 0
            self.simulation_timestamp = 0
            self.start_time = None
            self.start_timestamp = None
            self.last_state_time = None
            self.step_rates = []
            self._last_defrag_hour = -1
            self._last_prune_timestamp = None

            self.main_loop = asyncio.new_event_loop()
            self._main_loop_ready = Event()

            core_allocation = get_core_allocation(
                sltp_cores_count=int(os.environ.get("SLTP_CORES_COUNT", "0")),
                grad_server_cores=int(os.environ.get("GRAD_CORES_COUNT", "0")),
                sim_cores_count=self._read_block_count(),
                os_headroom=int(os.environ.get("OS_HEADROOM_CORES", "4")),
            )
            # Dedicated simulator core slice (blockCount-sized); pin the taosim
            # process to it so its worker threads don't contend with reward/save
            # on scoring rounds. Re-pinned each maintenance cycle (_sync_and_check).
            self.sim_cores = core_allocation.get('sim', [])
            if self.sim_cores:
                bt.logging.info(f"Simulator core slice: {self.sim_cores}")
                self._pin_simulator()
            ipc_cores = core_allocation['ipc']
            if len(ipc_cores) >= 2:
                mid = len(ipc_cores) // 2
                query_cores = ipc_cores[:mid]
                reporting_cores = ipc_cores[mid:]
            else:
                query_cores = ipc_cores
                reporting_cores = ipc_cores
            self.query_ipc_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix='query_ipc',
                initializer=lambda: os.sched_setaffinity(0, set(query_cores))
            )
            self.reporting_ipc_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix='reporting_ipc',
                initializer=lambda: os.sched_setaffinity(0, set(reporting_cores))
            )
            bt.logging.info(f"Query IPC executor assigned to cores: {query_cores}")
            bt.logging.info(f"Reporting IPC executor assigned to cores: {reporting_cores}")
            validator_cores = core_allocation['validator']
            os.sched_setaffinity(0, set(validator_cores))
            bt.logging.info(f"Validator assigned to cores: {validator_cores}")
            self.reward_cores = core_allocation['reward']
            self.reward_executor = ThreadPoolExecutor(max_workers=len(self.reward_cores),initializer=lambda: os.sched_setaffinity(0, set(self.reward_cores)))
            bt.logging.info(f"Reward executor assigned to cores: {self.reward_cores}")
            self.save_state_executor = ThreadPoolExecutor(max_workers=1)
            self.maintenance_executor = ThreadPoolExecutor(max_workers=1)
            self.maintenance_subtensor = bt.Subtensor(self.config.subtensor.chain_endpoint)
            # sync() (hence should_set_weights + set_weights) runs against
            # maintenance_subtensor, so its rate-limit read must be mechid-aware too.
            self._patch_blocks_since_last_update(self.maintenance_subtensor)

            # Metagraph sync worker: the resync's substrate scale-decode is a
            # 3-5s GIL burst that stalls whichever round it overlaps — do the
            # fetch+decode in a subprocess and ship back the pickled metagraph
            # (resync_metagraph swaps it in; falls back to in-process sync when
            # the worker is unavailable). Pinned to the unallocated OS-headroom
            # cores. METAGRAPH_WORKER=0 disables (legacy in-process sync).
            _allocated = {c for v in core_allocation.values() for c in v}
            _leftover = sorted(set(range(os.cpu_count() or 1)) - _allocated)
            self._mg_worker = None
            if os.environ.get("METAGRAPH_WORKER", "1") != "0" and not self.config.mock:
                try:
                    from taos.im.validator.metagraph_worker import MetagraphSyncWorker
                    self._mg_worker = MetagraphSyncWorker(
                        self.subtensor.chain_endpoint, self.config.netuid, cores=_leftover,
                        mechid=getattr(self.metagraph, "mechid", 0),
                    )
                    self._mg_worker.start()
                    bt.logging.info(
                        f"Metagraph sync worker started (cores: {_leftover or 'unpinned'})"
                    )
                except Exception as e:
                    self._mg_worker = None
                    bt.logging.warning(f"Metagraph sync worker unavailable, syncing in-process: {e}")

            # Shadow scoring service (Q3 Stage A): replicates the history
            # accounting in a subprocess from teed state bytes and reports
            # parity digests — groundwork (and proof) for moving reward/save/
            # report prep off the main process. Default OFF; observation only.
            self._scoring_shadow = None
            self._shadow_applied_ts = 0
            self._scoring_proc_cutover = False
            self._scoring_proc_n = 0
            if getattr(self.config, 'engine', 'simulation') != 'exchange' and not self.config.mock:
                try:
                    from taos.im.validator.scoring_shadow import (
                        ScoringShadow, compute_parity_components, cutover_enabled, shadow_enabled,
                    )
                    self._scoring_proc_cutover = cutover_enabled()
                    if self._scoring_proc_cutover or shadow_enabled():
                        self._shadow_digest_fn = compute_parity_components
                        from taos.im.validator.scoring_shadow import pnl_len_vector as _plv
                        self._shadow_pnl_vec_fn = _plv
                        # Cutover: the child scores authoritatively — give it the
                        # reward core slice (main's loky only runs on fallback/
                        # verify, which tolerates sharing). Shadow-only: leftover.
                        _shadow_cores = self.reward_cores if self._scoring_proc_cutover else _leftover
                        self._scoring_shadow = ScoringShadow(cores=_shadow_cores)
                        self._scoring_shadow.start()
                        bt.logging.info(
                            f"Scoring service started (mode: "
                            f"{'CUTOVER' if self._scoring_proc_cutover else 'shadow'}, "
                            f"cores: {_shadow_cores or 'unpinned'}, "
                            f"parity every {self._scoring_shadow.parity_ns/1e9:.0f} sim-s)"
                        )
                except Exception as e:
                    self._scoring_shadow = None
                    self._scoring_proc_cutover = False
                    bt.logging.warning(f"Scoring service unavailable: {e}")

            self.maintaining = False
            self.compressing = False
            self.querying = False
            # Mutual exclusion between the miner state-update query (forward)
            # and GTX assignment delivery (deliver_gentrx): they share the IPC
            # request queue/notify pipe and the miner network, so they must
            # never run concurrently. asyncio.Lock — both run on the Listen loop.
            self.miner_net_lock = asyncio.Lock()
            # Background GTX round-check/delivery task (single-flight).
            self._gentrx_task = None
            self._receiving_state = False
            self._rewarding = False
            self._pending_reward_tasks = 0
            self._saving = False
            self._reporting = False
            self._rewarding_lock = Lock()
            self._saving_lock = Lock()
            self._reporting_lock = Lock()
            self._reward_lock = asyncio.Lock()
            # Dedicated executor for MVTRX data-service push construction so it
            # doesn't contend with reporting on reporting_ipc_executor
            # (max_workers=4, already used by reporting's prep/serialize).
            # Small pool because payload build is CPU-bound and GIL-limited —
            # more workers don't help throughput, just headroom for backpressure.
            self._mvtrx_push_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix='mvtrx_push'
            )
            self._setup_signal_handlers()
            self._cleanup_done = False
            atexit.register(self.cleanup)


            self.router = APIRouter()
            self.router.add_api_route("/orderbook", self.orderbook, methods=["GET"])
            self.router.add_api_route("/account", self.account, methods=["GET"])
            self.router.add_api_route("/sltp", self.sltp_records, methods=["POST", "GET"])
            self.router.add_api_route("/sltp/levels", self.sltp_levels, methods=["GET"])

            # ---- GenTRX distributed training (optional) ----
            # Log via the `gtx_log` shim (imported at module top) — forwards
            # to bt.logging with the `[GTX]` prefix baked in, so every record
            # surfaces through the same stream the rest of the validator uses.
            self._gentrx = None
            _gentrx_enabled = getattr(getattr(self.config, "gentrx", None), "enabled", False)
            gtx_log.info(f"init: enabled={_gentrx_enabled}")
            if _gentrx_enabled:
                try:
                    from GenTRX.src.service import GenTRXService
                    # Miner UIDs are computed dynamically — metagraph changes
                    # over time as miners register/dereg, and miners may not be
                    # registered yet at validator init.
                    my_hotkey = self.wallet.hotkey.ss58_address

                    def _current_miner_uids():
                        uids = [
                            uid for uid in range(self.metagraph.n)
                            if self.metagraph.axons[uid].hotkey != my_hotkey
                            and self.metagraph.axons[uid].ip != "0.0.0.0"
                            and self.metagraph.axons[uid].port != 0
                        ]
                        # Include benchmark agents that have active axons (scored,
                        # not rewarded — same pattern as trading benchmarks).
                        for i, agent in enumerate(self.benchmark_agents):
                            axon = agent['axon']
                            if axon.ip != "0.0.0.0" and axon.port != 0:
                                uids.append(self.benchmark_start_uid + i)
                        return uids

                    gtx_log.info(f"init: server_url={self.config.gentrx.gradient_server_url}, current miner_uids={_current_miner_uids()}")

                    def _get_current_block():
                        try:
                            with self._subtensor_lock:
                                return self.subtensor.get_current_block()
                        except Exception:
                            return 0

                    self._gentrx = GenTRXService.from_config(
                        self.config,
                        deliver_fn=self._deliver_gentrx_assignments,
                        miner_uids_fn=_current_miner_uids,
                        get_block_fn=_get_current_block,
                        validator_uid=self.uid,
                    )
                    if self._gentrx:
                        if self._gentrx.is_healthy():
                            gtx_log.info(f"init: gradient server healthy at {self.config.gentrx.gradient_server_url}")
                        else:
                            gtx_log.warning(f"init: gradient server unreachable at {self.config.gentrx.gradient_server_url} — will retry on each tick")
                        # Commit validator bucket read credentials to chain.
                        # Miners and the aggregator use this to discover the bucket
                        # (data, scores, checkpoints) without pre-configured env vars.
                        try:
                            from GenTRX.src.chain import BucketInfo, GenTRXChain
                            val_bucket = BucketInfo.from_validator_env()
                            if val_bucket:
                                gtx_chain = GenTRXChain(self.subtensor, self.config.netuid, self.metagraph)
                                gtx_chain.commit_bucket(self.wallet, val_bucket)
                                gtx_log.info("validator bucket committed to chain")
                            else:
                                gtx_log.debug("GENTRX_VALIDATOR_S3_* not set — skipping chain commitment")
                        except Exception as exc:
                            import traceback
                            gtx_log.warning(f"validator chain commit failed: {exc}")
                            gtx_log.warning(traceback.format_exc())
                        # Register benchmark miner buckets with the gradient server.
                        # These miners can't commit to chain (not registered), so the
                        # validator injects their bucket info via the REST API.
                        for bm in self.benchmark_agents:
                            bkt = bm.get('gentrx_bucket')
                            if bkt:
                                ok = self._gentrx.register_benchmark_bucket(bm['uid'], bkt)
                                if ok:
                                    gtx_log.info(f"registered benchmark bucket: uid={bm['uid']} bucket={bkt.get('bucket_name')}")
                                else:
                                    gtx_log.warning(f"failed to register benchmark bucket: uid={bm['uid']}")
                    else:
                        gtx_log.warning("init: from_config returned None — check that --gentrx.gradient_server_url is set")
                except ImportError as exc:
                    gtx_log.warning(f"init: import failed — {exc}")
                except Exception as exc:
                    gtx_log.error(f"init: failed — {exc}")
                    import traceback
                    gtx_log.error(traceback.format_exc())
            gtx_log.info(f"init: {'ACTIVE' if self._gentrx else 'DISABLED'}")

            self.repo_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
            self.repo = Repo(self.repo_path)
            self.update_repo()

            self._reap_orphaned_services()
            self.query_process = None
            self.query_notify_read = -1
            self.query_notify_write = -1
            if not getattr(getattr(self.config, 'neuron', None), 'observe', False):
                self._start_query_service()
            else:
                bt.logging.info("Observe mode — skipping query service")
            self.report_process = None
            self._start_reporting_service()
            engine_mode = getattr(self.config, 'engine', 'simulation')
            if engine_mode == 'exchange':
                bt.logging.info("Starting validator in EXCHANGE engine mode")
                try:
                    from taos.im.validator.engines.exchange import ExchangeEngine
                except ImportError as _exc:
                    # Optional component; not part of this tree. Raised as a clear RuntimeError so
                    # --engine exchange does not fail on a raw ImportError with no hint that the mode
                    # itself is the problem.
                    raise RuntimeError(
                        "engine=exchange requires the exchange engine module, which this "
                        "distribution does not include; run with engine=simulation"
                    ) from _exc
                self.engine = ExchangeEngine(self.config, self)
                self.engine._push_fill_notify = _push_fill_notifications
            else:
                self.engine = SimulationEngine(self.config, self)
            self.engine.start()

        def _load_benchmark_agents(self):
            """Load benchmark agent configurations"""
            try:
                with open(self.config.benchmark.agents, 'r') as f:
                    benchmark_config = json.load(f)                
                for idx, agent in enumerate(benchmark_config['agents']):
                    uid = agent.get('uid', self.benchmark_start_uid + idx)
                    entry = {
                        'uid': uid,
                        'name': agent['name'],
                        'ip': agent['ip'],
                        'port': agent['port'],
                        'hotkey': agent['hotkey'],
                        'axon': bt.AxonInfo(
                            ip=agent['ip'],
                            port=agent['port'],
                            hotkey=agent['hotkey'],
                            coldkey='benchmark',
                            version=self.metagraph.axons[0].version,
                            ip_type=self.metagraph.axons[0].ip_type,
                            protocol=self.metagraph.axons[0].protocol,
                        )
                    }
                    if 'gentrx_bucket' in agent:
                        entry['gentrx_bucket'] = agent['gentrx_bucket']
                    self.benchmark_agents.append(entry)
                    bt.logging.info(f"Loaded benchmark agent: {agent['name']} (UID {uid}) at {agent['ip']}:{agent['port']}")            
            except Exception as ex:
                bt.logging.error(f"Failed to load benchmark agents: {ex}")
                bt.logging.error(traceback.format_exc())
                self.benchmark_agents = []
                
        def _initialize_all_structures(self):
            """
            Initialize all UID-indexed structures to effective_max_uids.
            """
            bt.logging.info(
                f"Initializing structures for {self.effective_max_uids} UIDs "
                f"(network: {self.subnet_info.max_uids}, benchmarks: {len(self.benchmark_agents)}) "
                f"with {self.simulation.book_count} books"
            )
            book_count = self.simulation.book_count
            # Engine's actual traded book-id set (root excluded on the mainnet-fork
            # -> [1..128]); fall back to 0-based range before the engine is assigned.
            # NOTE: this method is currently dead (structure init moved to the sim
            # engine) but kept book-id-correct to avoid a latent 0-based landmine.
            _bids = (self.engine.book_ids
                     if getattr(self, 'engine', None) is not None
                     else list(range(book_count)))
            if not hasattr(self, 'activity_factors') or len(self.activity_factors) < self.effective_max_uids:
                self.activity_factors = {
                    uid: {bookId: 0.0 for bookId in _bids}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'pnl_factors') or len(self.pnl_factors) < self.effective_max_uids:
                self.pnl_factors = {
                    uid: {bookId: 1.0 for bookId in _bids}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'kappa_values') or len(self.kappa_values) < self.effective_max_uids:
                self.kappa_values = {
                    uid: {
                        'books': {bookId: None for bookId in _bids},
                        'books_weighted': {bookId: 0.0 for bookId in _bids},
                        'total': None,
                        'average': None,
                        'median': None,
                        'normalized_average': 0.0,
                        'normalized_median': 0.0,
                        'normalized_total': 0.0,
                        'activity_weighted_normalized_median': 0.0,
                        'penalty': 0.0,
                        'score': 0.0,
                    }
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'unnormalized_scores') or len(self.unnormalized_scores) < self.effective_max_uids:
                self.unnormalized_scores = {uid: 0.0 for uid in range(self.effective_max_uids)}
            if not hasattr(self, 'trade_volumes') or len(self.trade_volumes) < self.effective_max_uids:
                self.trade_volumes = {
                    uid: {
                        bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                        for bookId in _bids
                    }
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'initial_balances') or len(self.initial_balances) < self.effective_max_uids:
                self.initial_balances = {
                    uid: {
                        bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                        for bookId in _bids
                    }
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'recent_miner_trades') or len(self.recent_miner_trades) < self.effective_max_uids:
                self.recent_miner_trades = {
                    uid: {bookId: [] for bookId in _bids}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'miner_stats') or len(self.miner_stats) < self.effective_max_uids:
                self.miner_stats = {
                    uid: {'requests': 0, 'timeouts': 0, 'failures': 0, 'rejections': 0, 'call_time': []}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'recent_trades'):
                self.recent_trades = {bookId: [] for bookId in _bids}
            if not hasattr(self, 'fundamental_price'):
                self.fundamental_price = {bookId: None for bookId in _bids}
            
            bt.logging.success(f"All structures initialized for {self.effective_max_uids} UIDs")

        def get_extended_metagraph(self):
            return self.engine.get_extended_metagraph()

        def __enter__(self):
            """
            Enables use of Validator as a context manager.

            Returns:
                Validator: The active validator instance.
            """
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Ensures cleanup is triggered when exiting a context manager block.

            Args:
                exc_type: Exception type, if any.
                exc_val: Exception instance, if any.
                exc_tb: Traceback, if any.

            Returns:
                bool: False to propagate exceptions.
            """
            self.cleanup()
            return False

        @property
        def shared_state_rewarding(self):
            with self._rewarding_lock:
                return self._rewarding

        @shared_state_rewarding.setter
        def shared_state_rewarding(self, value):
            with self._rewarding_lock:
                self._rewarding = value

        @property
        def shared_state_saving(self):
            with self._saving_lock:
                return self._saving

        @shared_state_saving.setter
        def shared_state_saving(self, value):
            with self._saving_lock:
                self._saving = value

        @property
        def shared_state_reporting(self):
            with self._reporting_lock:
                return self._reporting

        @shared_state_reporting.setter
        def shared_state_reporting(self, value):
            with self._reporting_lock:
                self._reporting = value

        async def wait_for(self, check_fn: callable, message: str, interval: float = 0.01):
            """
            Asynchronously waits for a condition to become False.

            Behavior:
                - Logs a message once per second while waiting.
                - Returns immediately if condition is already False.
                - Provides debug timing on completion.

            Args:
                check_fn (callable): Function returning a boolean condition.
                message (str): Log message describing the wait condition.
                interval (float): Interval between checks in seconds.

            Returns:
                None
            """
            if not check_fn():
                return

            start_time = time.time()
            last_log_time = start_time

            bt.logging.info(message)

            while check_fn():
                await asyncio.sleep(interval)

                current_time = time.time()
                elapsed = current_time - start_time

                if current_time - last_log_time >= 1.0:
                    bt.logging.info(f"{message} (waited {elapsed:.1f}s)")
                    last_log_time = current_time

            total_wait = time.time() - start_time
            bt.logging.debug(f"Wait completed after {total_wait:.1f}s")

        async def wait_for_event(self, event: asyncio.Event, wait_process: str, run_process: str):
            """
            Waits for an asyncio.Event to be set before continuing execution.

            Provides periodic logging while waiting, and measures the total wait
            duration for operational visibility.

            Args:
                event (asyncio.Event): The event that must be completed.
                wait_process (str): Name of the process being waited on.
                run_process (str): Name of the process to run afterward.

            Returns:
                None
            """
            if not event.is_set():
                bt.logging.debug(f"Waiting for {wait_process} to complete before {run_process}...")
                start_wait = time.time()
                while not event.is_set():
                    try:
                        await asyncio.wait_for(event.wait(), timeout=1.0)
                        break
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0)
                        elapsed = time.time() - start_wait
                        if int(elapsed) % 5 == 0 and elapsed > 0:
                            bt.logging.debug(f"Still waiting for {wait_process}... ({elapsed:.1f}s)")
                total_wait = time.time() - start_wait
                bt.logging.debug(f"Waited {total_wait:.1f}s for {wait_process}")

        async def _write_ipc_nonblocking(self, mem, queue, data_bytes, operation="IPC", executor=None):
            """
            Write to shared memory in executor thread.
            """
            if executor is None:
                executor = self.query_ipc_executor
            write_timeout = 10.0

            def write_worker():
                try:
                    mem.seek(0)
                    mem.write(struct.pack('Q', len(data_bytes)))
                    mem.write(data_bytes)
                    return True
                except Exception as e:
                    self.pagerduty_alert(f"{operation} write failed: {e}")
                    return False

            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        executor,
                        write_worker
                    ),
                    timeout=write_timeout
                )
                return result
            except asyncio.TimeoutError:
                self.pagerduty_alert(f"{operation} write timeout after {write_timeout}s")
                return False

        def onStart(self, timestamp, event : SimulationStartEvent) -> None:
            self.engine.on_start(timestamp, event)

        def onEnd(self) -> None:
            self.engine.on_end()


        def _build_validator_state(
            self,
            inventory_snapshot,
            realized_pnl_snapshot,
            volume_sums_snapshots,
            trade_volumes_snapshot,
            roundtrip_volumes_snapshot,
            open_positions_snapshot
        ):
            """Assemble validator state dict from all snapshots."""
            return build_validator_state(
                self, inventory_snapshot, realized_pnl_snapshot, volume_sums_snapshots,
                trade_volumes_snapshot, roundtrip_volumes_snapshot, open_positions_snapshot
            )


        def _snapshot_inventory_history(self):
            return snapshot_inventory_history(self)


        def _snapshot_realized_pnl_history(self):
            return snapshot_realized_pnl_history(self)


        def _snapshot_2_level_dict(self, source_dict):
            return snapshot_2_level_dict(self, source_dict)


        def _snapshot_volume_sums(self):
            return snapshot_volume_sums(self)


        def _snapshot_trade_volumes(self):
            return snapshot_trade_volumes(self)


        def _snapshot_roundtrip_volumes(self):
            return snapshot_roundtrip_volumes(self)


        def _snapshot_open_positions(self):
            return snapshot_open_positions(self)


        def _construct_save_data_sync(self):
            """Synchronously build all state data for saving."""
            return construct_save_data_sync(self)


        def _defragment_histories(self):
            """Rebuild history dicts to eliminate memory fragmentation."""
            defragment_histories(self)


        def save_state(self) -> None:
            """Schedules the asynchronous state-saving coroutine on the main event loop."""
            schedule_save(self)

        async def _save_state_sync(self):
            """Save validator state synchronously from an async context."""
            await save_state_sync(self)

        def migrate_sampling_interval(self, old_interval: int, new_interval: int):
            """Re-align trade volumes from old sampling interval to new interval.

            Args:
                old_interval: The interval the stored volumes were sampled at.
                new_interval: The configured interval to re-align onto.
            """
            migrate_sampling_interval(self, old_interval, new_interval)


        def load_state(self) -> None:
            """Loads validator and simulation state from msgpack or legacy PyTorch files."""
            load_state(self)


        def get_n_target_miners(self) -> int:
            """
            Count registered miner UIDs in the metagraph (validators excluded).

            Used by `prepare_weights` to size the GenTRX pool's participation
            denominator. A miner is identified by having a reachable axon
            endpoint and not being the running validator itself.
            """
            try:
                my_hotkey = self.wallet.hotkey.ss58_address
            except Exception:
                my_hotkey = None
            count = 0
            for uid in range(self.metagraph.n):
                axon = self.metagraph.axons[uid]
                if my_hotkey is not None and axon.hotkey == my_hotkey:
                    continue
                if axon.ip == "0.0.0.0" or axon.port == 0:
                    continue
                count += 1
            return count

        def handle_deregistration(self, uid, old_coldkey=None) -> None:
            """Engine handles primary deregistration; we also reset GenTRX state here.

            Args:
                uid: The deregistered uid.
                old_coldkey: The coldkey that held the slot.
            """
            self.engine.handle_deregistration(uid, old_coldkey)
            # Reset GenTRX EMA + service score cache so a new miner at the
            # same UID slot doesn't inherit the old miner's score history.
            if hasattr(self, 'gentrx_scores'):
                self.gentrx_scores[uid] = 0.0
            if hasattr(self, '_gentrx_ema') and self._gentrx_ema:
                self._gentrx_ema.pop(uid, None)
            # New occupant of the slot starts its track record fresh (seeds at its own
            # first score in get_rewards), not with the departed miner's standing.
            if hasattr(self, '_trading_score_ema') and self._trading_score_ema:
                self._trading_score_ema.pop(uid, None)
            if hasattr(self, '_trading_score_ema_n') and self._trading_score_ema_n:
                self._trading_score_ema_n.pop(uid, None)
            if getattr(self, '_gentrx', None) is not None:
                self._gentrx._scores.pop(uid, None)

        def process_resets(self, state: NormalizedState) -> None:
            """
            Collects reset notices delivered by the simulator and stashes them on
            the state. APPLICATION happens in _reward, immediately AFTER that
            round's update_trade_volumes — the same position the scoring service
            applies them (it derives the same set from the teed state), so the
            reset transition is deterministic on both sides. Applying here (before
            the queued volume update) left main one round ahead of the service on
            the reset uid's history — the recurring n_pnl parity mismatch.
            """
            pending = set()
            self.engine.collect_resets(state, pending)
            state._pending_resets = pending


        def _sync_metagraph_with_retry(self, attempts: int = 5):
            """In-process metagraph sync with bounded retry/backoff.

            The initial boot sync (super().__init__ -> sync) runs BEFORE the
            metagraph worker exists, so it always takes this in-process path. A
            transient chain-RPC hiccup here (e.g. a websocket keepalive-ping
            timeout) otherwise propagates straight out of __init__ and pm2's
            short kill-timeout SIGKILLs the half-booted validator, leaving it
            stopped. Retry with exponential backoff, refreshing the subtensor
            connection between tries so a dead websocket doesn't doom every
            attempt. Re-raises only if the chain is unreachable for the whole
            window (preserving the original fail-hard as a last resort)."""
            attempts = max(1, int(os.environ.get("METAGRAPH_SYNC_RETRIES", str(attempts))))
            delay = 2.0
            for attempt in range(1, attempts + 1):
                try:
                    self.metagraph.sync(subtensor=self.subtensor)
                    if attempt > 1:
                        bt.logging.success(f"Metagraph sync succeeded on attempt {attempt}/{attempts}")
                    return
                except Exception as ex:
                    if attempt >= attempts:
                        bt.logging.error(f"Metagraph sync failed after {attempts} attempts: {ex}")
                        raise
                    bt.logging.warning(
                        f"Metagraph sync attempt {attempt}/{attempts} failed ({ex}); "
                        f"refreshing subtensor and retrying in {delay:.0f}s"
                    )
                    try:
                        self.subtensor = bt.Subtensor(self.config.subtensor.chain_endpoint)
                    except Exception as re:
                        bt.logging.warning(f"Subtensor refresh failed (will retry with existing): {re}")
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)

        def resync_metagraph(self):
            """Resyncs the metagraph and updates hotkeys and scores."""
            bt.logging.trace("resync_metagraph()")
            # Preferred path: fetch+decode in the metagraph sync worker process
            # and swap the returned object in (replace-don't-mutate also makes
            # `previous_metagraph` a free reference instead of a deepcopy).
            # Any worker failure falls back to the legacy in-process sync.
            _new_mg = None
            _worker = getattr(self, '_mg_worker', None)
            if _worker is not None:
                _new_mg = _worker.sync()
                if _new_mg is None:
                    bt.logging.warning("Metagraph worker sync failed — falling back to in-process sync")
            if _new_mg is not None:
                previous_metagraph = self.metagraph
                # Worker ships the metagraph with .subtensor stripped (websocket
                # is unpicklable) — reattach ours so the object is functionally
                # identical to an in-process-synced one.
                try:
                    _new_mg.subtensor = self.subtensor
                except Exception:
                    pass
                self.metagraph = _new_mg
            else:
                previous_metagraph = copy.deepcopy(self.metagraph)
                bt.logging.debug("Syncing metagraph...")
                self._sync_metagraph_with_retry()
            # Compare the FULL hotkey list, not just its length. A burned_register
            # SWAP (a UID deregistered + a new miner taking its slot) keeps the count
            # at max_uids and, if the incoming miner hasn't served its axon yet, leaves
            # axons unchanged too — so a length+axons check early-returns and SKIPS the
            # deregistration path below (handle_deregistration → refund old proxy + pop;
            # ensure_wallets → create the new owner's proxy; register_wallet → link/load
            # signer). That left a reused-slot's new owner with no proxy (settlements
            # failed with "no loaded signer"). Detect any hotkey change so the proxy is
            # rotated on every dereg, incl. slot swaps.
            if previous_metagraph.axons == self.metagraph.axons and list(self.hotkeys) == list(self.metagraph.hotkeys):
                bt.logging.debug("No axon or hotkey changes!")
                # Re-register benchmark buckets even when the metagraph is unchanged,
                # so a gradient server restart self-heals within one resync cycle.
                _gtx = getattr(self, '_gentrx', None)
                if _gtx is not None:
                    for bm in self.benchmark_agents:
                        bkt = bm.get('gentrx_bucket')
                        if bkt:
                            ok = _gtx.register_benchmark_bucket(bm['uid'], bkt)
                            if ok:
                                gtx_log.debug("re-registered benchmark bucket: uid=%d", bm['uid'])
                            else:
                                gtx_log.warning("failed to re-register benchmark bucket: uid=%d", bm['uid'])
                return

            bt.logging.info(
                "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
            )
            for uid, hotkey in enumerate(self.hotkeys):
                if uid < len(self.metagraph.hotkeys) and hotkey != self.metagraph.hotkeys[uid]:
                    # Pass the departing miner's coldkey (pre-sync metagraph) so the
                    # proxy refund can find an on-disk proxy that isn't in the live
                    # runtime maps (not persisted across restarts).
                    _old_ck = (
                        previous_metagraph.coldkeys[uid]
                        if uid < len(previous_metagraph.coldkeys)
                        else None
                    )
                    self.handle_deregistration(uid, _old_ck)

            old_metagraph_size = len(self.hotkeys)
            new_metagraph_size = len(self.metagraph.hotkeys)
            if old_metagraph_size != new_metagraph_size:
                bt.logging.info(f"Metagraph size changed: {old_metagraph_size} -> {new_metagraph_size}")
                # `super().__init__` calls sync() → resync_metagraph() before `self.engine`
                # gets assigned in the subclass init body. During that first pass there is
                # nothing for the engine to resize yet, so skip the engine hook safely.
                if hasattr(self, 'engine'):
                    self.engine.on_resync_metagraph(old_metagraph_size, new_metagraph_size)

                # Expand per-UID dicts to cover any UIDs not yet present (new network
                # registrations, or holes left after benchmark shifting above).
                _bc = self.simulation.book_count
                # Engine's actual traded book-id set (root excluded -> [1..128]) so
                # new UIDs key identically to the engine's own structure init. The
                # first resync (super().__init__ -> sync() before self.engine exists)
                # falls back to 0-based range; those defaults are overwritten once the
                # engine initializes structures from its real book_ids.
                _bids = (self.engine.book_ids
                         if getattr(self, 'engine', None) is not None
                         else list(range(self.simulation.book_count)))
                _kappa_default = {
                    'books': {b: None for b in _bids},
                    'books_weighted': {b: 0.0 for b in _bids},
                    'total': None, 'average': None, 'median': None,
                    'normalized_average': 0.0, 'normalized_median': 0.0,
                    'normalized_total': 0.0,
                    'activity_weighted_normalized_median': 0.0,
                    'penalty': 0.0, 'score': 0.0,
                }
                for _uid in range(self.effective_max_uids):
                    if _uid not in self.miner_stats:
                        self.miner_stats[_uid] = {
                            'requests': 0, 'timeouts': 0, 'failures': 0,
                            'rejections': 0, 'call_time': [],
                        }
                    if _uid not in self.activity_factors:
                        self.activity_factors[_uid] = {b: 0.0 for b in _bids}
                    if _uid not in self.pnl_factors:
                        self.pnl_factors[_uid] = {b: 1.0 for b in _bids}
                    if _uid not in self.kappa_values:
                        self.kappa_values[_uid] = dict(_kappa_default)
                        self.kappa_values[_uid]['books'] = {b: None for b in _bids}
                        self.kappa_values[_uid]['books_weighted'] = {b: 0.0 for b in _bids}
                    if _uid not in self.unnormalized_scores:
                        self.unnormalized_scores[_uid] = 0.0
                    if _uid not in self.initial_balances:
                        self.initial_balances[_uid] = {
                            b: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                            for b in _bids
                        }
                    if _uid not in self.initial_balances_published:
                        self.initial_balances_published[_uid] = False
                    if _uid not in self.inventory_history:
                        self.inventory_history[_uid] = {}
                    if _uid not in self.recent_miner_trades:
                        self.recent_miner_trades[_uid] = {b: [] for b in _bids}

            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
            bt.logging.success(f"Metagraph resync complete: {len(self.hotkeys)} network hotkeys")
            # Create proxy wallets for any newly registered UIDs and refresh
            # executor uid→coldkey maps.  Must run after hotkeys is updated.
            if hasattr(self.engine, 'on_metagraph_synced'):
                self.engine.on_metagraph_synced(self.metagraph)

            # Re-register benchmark buckets after every resync so that a gradient
            # server restart self-heals without requiring a validator restart.
            _gtx = getattr(self, '_gentrx', None)
            if _gtx is not None:
                for bm in self.benchmark_agents:
                    bkt = bm.get('gentrx_bucket')
                    if bkt:
                        _gtx.register_benchmark_bucket(bm['uid'], bkt)

            # Re-register benchmark buckets after every resync so that a gradient
            # server restart self-heals without requiring a validator restart.
            _gtx = getattr(self, '_gentrx', None)
            if _gtx is not None:
                for bm in self.benchmark_agents:
                    bkt = bm.get('gentrx_bucket')
                    if bkt:
                        _gtx.register_benchmark_bucket(bm['uid'], bkt)

        async def _maintain(self) -> None:
            """
            Executes metagraph sync and maintenance operations asynchronously.

            Actions:
                - Marks the validator as in maintenance mode.
                - Runs synchronous maintenance work in an executor thread.
                - Logs timing and reports issues via PagerDuty.

            Returns:
                None
            """
            try:
                self.maintaining = True
                # Full chain sync (metagraph resync + set_weights check) only every
                # METAGRAPH_SYNC_EVERY-th firing: the metagraph scale-decode is a
                # 3-4s GIL burst that lands on some round's receive every interval,
                # and 26s freshness buys nothing (weights_rate_limit is ~72min,
                # dereg/axon changes tolerate ~2min). Simulator health check +
                # core re-pin still run EVERY firing (sim-death detection stays
                # at interval latency).
                self._maintain_count = getattr(self, '_maintain_count', 0) + 1
                _sync_every = max(1, int(os.environ.get("METAGRAPH_SYNC_EVERY", "4")))
                do_sync = (self._maintain_count % _sync_every) == 1 or _sync_every == 1
                if do_sync:
                    bt.logging.info(f"Synchronizing at Step {self.step}...")
                else:
                    bt.logging.debug(
                        f"Maintenance health check at Step {self.step} "
                        f"(sync {self._maintain_count % _sync_every}/{_sync_every})"
                    )
                start = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.maintenance_executor,
                    self._sync_and_check,
                    do_sync
                )
                if do_sync:
                    bt.logging.info(f"Synchronized ({time.time()-start:.4f}s)")

            except Exception as ex:
                self.pagerduty_alert(f"Failed to sync: {ex}", details={"trace": traceback.format_exc()})
            finally:
                self.maintaining = False

        async def _deliver_gentrx_assignments(self, assignments: dict) -> None:
            """Build delivery list from assignments and call deliver_gentrx."""
            try:
                from taos.im.protocol.gentrx import GenTRXAssignment

                my_hotkey = self.wallet.hotkey.ss58_address
                try:
                    my_uid = self.metagraph.hotkeys.index(my_hotkey)
                except ValueError:
                    my_uid = -1
                deliveries = []
                for uid, assignment in assignments.items():
                    if uid >= len(self.metagraph.axons):
                        bm_idx = uid - self.benchmark_start_uid
                        if bm_idx < 0 or bm_idx >= len(self.benchmark_agents):
                            continue
                        axon = self.benchmark_agents[bm_idx]['axon']
                    else:
                        axon = self.metagraph.axons[uid]
                    if axon.hotkey == my_hotkey:
                        continue
                    if axon.ip == "0.0.0.0" or axon.port == 0:
                        continue
                    try:
                        deliveries.append((
                            uid,
                            axon,
                            GenTRXAssignment(
                                round=assignment.get("round", 0),
                                model_version=assignment.get("model_version", 0),
                                books=assignment.get("books", []),
                                ts_start=assignment.get("ts_start", 0),
                                ts_end=assignment.get("ts_end", 0),
                                data=assignment.get("data", []),
                                data_source=assignment.get("data_source", "s3"),
                                data_endpoint=assignment.get("data_endpoint", ""),
                                data_bucket=assignment.get("data_bucket", ""),
                                data_access_key=assignment.get("data_access_key", ""),
                                data_secret_key=assignment.get("data_secret_key", ""),
                                validator_uid=my_uid,
                                advice=assignment.get("advice", {}),
                            ),
                        ))
                    except Exception as exc:
                        import traceback
                        gtx_log.warning(f"build assignment for uid {uid} failed: {exc}")
                        gtx_log.warning(traceback.format_exc())

                if not deliveries:
                    return

                round_id = next(iter(assignments.values())).get("round", "?")
                gtx_log.info(f"round={round_id}: delivering to uids={[u for u,_,_ in deliveries]}")
                # One atomic send to ALL miners (never chunked), but never
                # overlapping a state-update query: the lock is held by
                # forward() for its query window and by us for the delivery.
                async with self.miner_net_lock:
                    await deliver_gentrx(self, deliveries)
            except Exception as exc:
                gtx_log.error(f"delivery failed: {exc}")
                import traceback
                gtx_log.error(traceback.format_exc())

        def _read_block_count(self) -> int:
            """Parse the simulation blockCount from the XML config (root attribute).

            Read straight from the XML, not self.simulation, because core allocation
            runs before the engine loads the config. Sizing the sim core slice to
            blockCount means bumping blockCount in the XML scales the pin automatically.
            Returns 0 on any error or when SIM_CORE_PIN=0 (→ no pinning; sim floats).
            """
            if os.environ.get("SIM_CORE_PIN", "1") == "0":
                return 0
            # Observe / exchange mode has no simulation config → nothing to pin.
            # Bail silently instead of warning (config.simulation is None there).
            sim_cfg = getattr(self.config, "simulation", None)
            xml_path = getattr(sim_cfg, "xml_config", None) if sim_cfg else None
            if not xml_path:
                return 0
            try:
                import xml.etree.ElementTree as ET

                root = ET.parse(xml_path).getroot()
                return int(root.attrib["blockCount"])
            except Exception as ex:
                bt.logging.warning(f"_read_block_count: could not read blockCount ({ex}); sim will not be pinned")
                return 0

        def _pin_simulator(self) -> None:
            """Pin the C++ simulator process to its dedicated core slice.

            The simulator is a separate pm2 process launched with no affinity, so its
            worker threads otherwise float across the reward/save cores and starve the
            next-state handshake on scoring rounds. Bind it to self.sim_cores (disjoint
            from reward+save) to fix that. Best-effort + idempotent: re-run each
            maintenance cycle to catch simulator restarts.
            """
            sim_cores = getattr(self, "sim_cores", None)
            if not sim_cores:
                return
            try:
                import psutil

                target = set(sim_cores)
                for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                    try:
                        # Match the actual simulator binary only — NOT `htop -f taosim`
                        # (the run_validator monitor) or anything else that merely
                        # references taosim in its args.
                        name = proc.info.get("name") or ""
                        cmdline = proc.info.get("cmdline") or []
                        exe = cmdline[0] if cmdline else ""
                        if name != "taosim" and os.path.basename(exe) != "taosim":
                            continue
                        if set(proc.cpu_affinity()) == target:
                            continue
                        proc.cpu_affinity(sim_cores)
                        bt.logging.info(f"Pinned simulator (pid {proc.info['pid']}) to cores: {sim_cores}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as ex:
                bt.logging.warning(f"_pin_simulator: {ex}")

        def _sync_and_check(self, do_sync: bool = True):
            """
            Performs synchronous metagraph maintenance and simulator health checks.

            Steps:
                - Runs Bittensor sync (without saving state) when do_sync is True.
                - Verifies simulator health.
                - Restarts simulator if unhealthy.

            Returns:
                None
            """
            if do_sync:
                # Use a dedicated subtensor connection so this thread doesn't
                # share a websocket with the chain data provider running concurrently.
                primary, self.subtensor = self.subtensor, self.maintenance_subtensor
                try:
                    self.sync(save_state=False)
                finally:
                    self.subtensor = primary
            if not check_simulator(self):
                restart_simulator(self)
            # Re-pin the simulator each cycle (also re-applies after a restart above).
            self._pin_simulator()

        def maintain(self) -> None:
            """
            Schedules asynchronous maintenance work from the maintenance thread.

            Behavior:
                - Ensures maintenance is not already running.
                - Triggers only at specific simulation timestamps.
                - Sends a coroutine to the main event loop thread-safely.

            Returns:
                None
            """
            # Offset +3e9 (was +2e9): reward fires at +0 and completes ~2 rounds
            # later, releasing _reward_lock — reporting serialize, save queueing
            # and the mvtrx push all burst right then. +2 put the metagraph
            # scale-decode inside that same post-reward herd; +3 is the free slot
            # between the herd and the +4 save trigger. Same 1-interval cadence.
            if not self.maintaining and self.last_state and self.last_state.timestamp % self.config.scoring.interval == 3_000_000_000:
                bt.logging.debug(f"[MAINT] Scheduling from thread: {threading.current_thread().name}")
                bt.logging.debug(f"[MAINT] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
                self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._maintain()))

        def _prepare_reporting_data(self):
            bt.logging.debug("Retrieving fundamental prices...")
            start = time.time()
            if hasattr(self.engine, 'load_fundamental'):
                self.engine.load_fundamental()
            bt.logging.debug(f"Retrieved fundamental prices ({time.time()-start:.4f}s).")

            book_count = self.simulation.book_count
            bt.logging.debug("Computing realized P&L totals...")
            pnl_start = time.time()
            total_realized_pnl = {}
            realized_pnl_by_book = {}

            # Read from the incrementally-maintained running totals instead of
            # re-summing realized_pnl_history[uid][ts][book_id] every call. The
            # sum is O(N*T*B) (~66M ops on mainnet at N=259, T~2000, B=128) and
            # this function runs under _reward_lock, blocking every subsequent
            # _reward for its duration. The running totals are kept in sync by
            # trade.py:_apply_pnl_delta on write/delete/prune and rebuilt on
            # startup + sim-restart by persistence.bootstrap_pnl_totals — so
            # this is byte-identical to the old loop but constant-time per UID.
            _pnl_by_book = self.agent_pnl_by_book
            _pnl_total = self.agent_pnl_total
            for uid in self.realized_pnl_history.keys():
                book_totals_dict = _pnl_by_book.get(uid, {})
                realized_pnl_by_book[uid] = {
                    book_id: book_totals_dict.get(book_id, 0.0) for book_id in self.engine.book_ids
                }
                total_realized_pnl[uid] = _pnl_total.get(uid, 0.0)

            bt.logging.debug(f"Computed realized P&L totals ({time.time()-pnl_start:.4f}s)")

            bt.logging.debug("Building minimal inventory...")
            inv_start = time.time()
            minimal_inventory = {}
            for uid, hist in self.inventory_history.items():
                if not hist:
                    continue
                timestamps = sorted(hist.keys())
                n = len(timestamps)
                if n >= 3:
                    minimal_inventory[uid] = {
                        timestamps[0]: hist[timestamps[0]],
                        timestamps[-2]: hist[timestamps[-2]],
                        timestamps[-1]: hist[timestamps[-1]]
                    }
                elif n > 0:
                    minimal_inventory[uid] = {ts: hist[ts] for ts in timestamps}
            bt.logging.debug(f"Built minimal inventory ({time.time()-inv_start:.4f}s)")

            bt.logging.debug("Serializing volume sums...")
            serialize_start = time.time()

            # Send nested {uid: {book_id: vol}} directly. msgpack packs int keys
            # and the reporting side unpacks with strict_map_key=False, so the
            # previous flatten to "uid:book_id" strings here (6 dicts x ~N*B
            # f-string builds) and matching split() back in the report subprocess
            # was pure round-trip overhead on the reward-lock-held prep path.
            # The two-level copy snapshots the live defaultdicts for thread-safe
            # msgpack packing (same concurrency profile as the old flatten).
            def _nested_snapshot(d):
                return {uid: dict(books) for uid, books in d.items()}

            volume_sums_flat = _nested_snapshot(self.volume_sums)
            maker_volume_sums_flat = _nested_snapshot(self.maker_volume_sums)
            taker_volume_sums_flat = _nested_snapshot(self.taker_volume_sums)
            self_volume_sums_flat = _nested_snapshot(self.self_volume_sums)
            fee_sums_flat = _nested_snapshot(getattr(self, 'fee_sums', {}))
            roundtrip_volume_sums_flat = _nested_snapshot(self.roundtrip_volume_sums)

            # De-beta (P8) trading score for the reporting snapshot, finalized via the shared helper
            # (this process owns the fill-stream accumulators); passed as a compact {uid: score} map
            # rather than shipping the raw per-book sums over IPC. {} when disabled/warming.
            debeta_scores_flat = compute_debeta_scores(self)

            bt.logging.debug(f"Serialized volume sums ({time.time()-serialize_start:.4f}s)")

            bt.logging.debug("Building metagraph data...")
            meta_start = time.time()
            metagraph_data = {
                'hotkeys': [str(hk) for hk in self.metagraph.hotkeys],
                'coldkeys': [str(ck) for ck in self.metagraph.coldkeys] if hasattr(self.metagraph, 'coldkeys') else [],
                'stake': self.metagraph.stake.tolist(),
                'trust': self.metagraph.trust.tolist() if hasattr(self.metagraph, 'trust') else [],
                'consensus': self.metagraph.consensus.tolist(),
                'incentive': self.metagraph.incentive.tolist(),
                'emission': self.metagraph.emission.tolist(),
                'validator_trust': self.metagraph.validator_trust.tolist(),
                'validator_permit': self.metagraph.validator_permit.tolist() if hasattr(self.metagraph, 'validator_permit') else [],
                'dividends': self.metagraph.dividends.tolist(),
                'active': self.metagraph.active.tolist(),
                'last_update': self.metagraph.last_update.tolist(),
                'axon_ips': [str(getattr(ax, 'ip', '')) for ax in self.metagraph.axons] if hasattr(self.metagraph, 'axons') else [],
                'axon_ports': [int(getattr(ax, 'port', 0)) for ax in self.metagraph.axons] if hasattr(self.metagraph, 'axons') else [],
            }
            bt.logging.debug(f"Built metagraph data ({time.time()-meta_start:.4f}s)")

            bt.logging.debug("Building recent trades...")
            trades_start = time.time()
            recent_trades = {
                bookId: [t.model_dump() for t in trades[-25:]]
                for bookId, trades in self.recent_trades.items()
            }

            recent_miner_trades = {
                uid: {
                    bookId: [
                        {'trade': miner_trade.model_dump(), 'role': role}
                        for miner_trade, role in trades[-5:]
                    ]
                    for bookId, trades in book_trades.items()
                }
                for uid, book_trades in self.recent_miner_trades.items()
            }
            bt.logging.debug(f"Built minimal recent trades ({time.time()-trades_start:.4f}s)")

            def _cap_notices_per_book(notices, per_book=20):
                _NOTICE_TYPES = frozenset({
                    'EVENT_TRADE', 'ET',
                    'RDPOL', 'ERDPOL', 'RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT', 'ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT',
                    'RDPOM', 'ERDPOM', 'RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET', 'ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET',
                    'RDCO', 'ERDCO', 'RESPONSE_DISTRIBUTED_CANCEL_ORDERS', 'ERROR_RESPONSE_DISTRIBUTED_CANCEL_ORDERS',
                })
                by_book: dict = {}
                for n in notices:
                    if n.get('y') not in _NOTICE_TYPES:
                        continue
                    bk = n.get('b') or n.get('bookId') or 0
                    by_book.setdefault(bk, []).append(n)
                return [n for ns in by_book.values() for n in ns[-per_book:]]

            bt.logging.debug("Building minimal state...")
            state_start = time.time()
            minimal_state = {
                'accounts': self.last_state.accounts,
                'pools': self.last_state.pools,
                'books': {
                    bookId: {
                        'b': book['b'][:5] if book['b'] else [],
                        'a': book['a'][:5] if book['a'] else [],
                        # Include all event types (orders, fills, cancels, chain events)
                        # not just fills — required for the full L3 stream in the data service.
                        'e': (book.get('e') or [])[-50:],
                        'r': book.get('r', book.get('mtr', 0.0))
                    }
                    for bookId, book in self.last_state.books.items()
                },
                'notices': {
                    uid: _cap_notices_per_book(notices)
                    for uid, notices in self.last_state.notices.items()
                }
            }
            bt.logging.debug(f"Built minimal state ({time.time()-state_start:.4f}s)")

            bt.logging.debug("Building position summary...")
            pos_start = time.time()
            open_positions = {
                uid: {
                    book_id: {
                        'longs_count': len(pos['longs']),
                        'shorts_count': len(pos['shorts'])
                    }
                    for book_id, pos in books.items()
                }
                for uid, books in self.open_positions.items()
            }
            bt.logging.debug(f"Built position summary ({time.time()-pos_start:.4f}s)")

            bt.logging.debug("Assembling final data structure...")
            final_start = time.time()

            data = {
                'metagraph_data': metagraph_data,
                'simulation': self.simulation.model_dump(),
                'last_state': minimal_state,
                'simulation_timestamp': self.simulation_timestamp,
                'step': self.step,
                'step_rates': list(self.step_rates[-100:]),
                'volume_sums': volume_sums_flat,
                'maker_volume_sums': maker_volume_sums_flat,
                'taker_volume_sums': taker_volume_sums_flat,
                'self_volume_sums': self_volume_sums_flat,
                'fee_sums': fee_sums_flat,
                'roundtrip_volume_sums': roundtrip_volume_sums_flat,
                'inventory_history': minimal_inventory,
                'total_realized_pnl': total_realized_pnl,
                'realized_pnl_by_book': realized_pnl_by_book,
                'book_count': book_count,
                'activity_factors': self.activity_factors,
                'pnl_factors': self.pnl_factors,
                'kappa_values': self.kappa_values,
                'unnormalized_scores': self.unnormalized_scores,
                # Track-record EMA standing (post-smoothing trading score that actually drives
                # reward). Reported in place of raw kappa_score/combined_score so the dashboard
                # shows the smoothed standing, not the pre-EMA per-window value.
                'trading_score_ema': dict(getattr(self, '_trading_score_ema', {}) or {}),
                'debeta_scores': debeta_scores_flat,
                'scores': {i: score.item() for i, score in enumerate(self.scores)},
                'gentrx_scores': {i: score.item() for i, score in enumerate(self.gentrx_scores)},
                'gentrx_enabled': self._gentrx is not None,
                'gentrx_training': self._gentrx.get_training_stats() if self._gentrx is not None else {},
                'gentrx_scores_detailed': self._gentrx.get_scores() if self._gentrx is not None else {},
                'gentrx_config': self._gentrx.get_config() if self._gentrx is not None else {},
                'miner_stats': self.miner_stats,
                'initial_balances': self.initial_balances,
                'initial_balances_published': self.initial_balances_published,
                'recent_trades': recent_trades,
                'recent_miner_trades': recent_miner_trades,
                'open_positions': open_positions,
                'fundamental_price': self.fundamental_price,
                'shared_state_rewarding': self.shared_state_rewarding,
                'current_block': self.current_block,
                'uid': self.uid,
                'reconciliation': dict(getattr(getattr(self, 'engine', None), '_pending_reconciliation', None) or {}),
                # Exchange engine: include chain-level events and delegates for the data service.
                # block_events feeds the L3 stream (StakeAdded/StakeRemoved etc.).
                # delegates feeds the Redis cache used for order-placement delegate defaults.
                'block_events': list((getattr(getattr(self, 'engine', None), '_last_chain_state', None) or {}).get('block_events', [])),
                'delegates':    dict((getattr(getattr(self, 'engine', None), '_last_chain_state', None) or {}).get('delegates', {})),
                'validator_config' : {
                    'observe': bool(getattr(getattr(self.config, 'neuron', None), 'observe', False)),
                    'scoring': {
                        'interval': self.config.scoring.interval,
                        'max_instructions_per_book': self.config.scoring.max_instructions_per_book,
                        'min_delay': self.config.scoring.min_delay,
                        'max_delay': self.config.scoring.max_delay,
                        'min_instruction_delay': self.config.scoring.min_instruction_delay,
                        'max_instruction_delay': self.config.scoring.max_instruction_delay,
                        'kappa_weight': self.config.scoring.kappa.weight,
                        'kappa_lookback': self.config.scoring.kappa.lookback,
                        'kappa_min_lookback': self.config.scoring.kappa.min_lookback,
                        'kappa_tau': self.config.scoring.kappa.tau,
                        'kappa_min_realized_observations': self.config.scoring.kappa.min_realized_observations,
                        'kappa_normalization_min': self.config.scoring.kappa.normalization_min,
                        'kappa_normalization_max': self.config.scoring.kappa.normalization_max,
                        'kappa_pnl_impact': self.config.scoring.kappa.pnl.impact,
                        'pnl_weight': self.config.scoring.pnl.weight,
                        'pnl_lookback': getattr(self.config.scoring.pnl, 'lookback', self.config.scoring.kappa.lookback),
                        'gentrx_simulation_share': getattr(getattr(self.config.scoring, 'gentrx', None), 'simulation_share', 0.0) or 0.0,
                        'pnl_normalization_min_daily_return': self.config.scoring.pnl.min_daily_return,
                        'pnl_normalization_max_daily_return': self.config.scoring.pnl.max_daily_return,
                        'activity_impact': self.config.scoring.activity.impact,
                        'activity_trade_volume_sampling_interval': self.config.scoring.activity.trade_volume_sampling_interval,
                        'activity_trade_volume_assessment_period': self.config.scoring.activity.trade_volume_assessment_period,
                        'activity_decay_grace_period': self.config.scoring.activity.decay_grace_period,
                        'activity_decay_rate': self.config.scoring.activity.decay_rate,
                        'activity_capital_turnover_cap': self.config.scoring.activity.capital_turnover_cap,
                        'activity_max_volume': self.config.scoring.activity.capital_turnover_cap * self.simulation.miner_wealth,
                    }
                }
            }

            bt.logging.debug(f"Assembled final structure ({time.time()-final_start:.4f}s)")

            # One-line prep breakdown at INFO (sections run sequentially, so each
            # duration is next_start - this_start) so the dominant contributor to
            # prep is visible without enabling DEBUG. Cheap; ~1 line per report.
            bt.logging.info(
                "Reporting prep breakdown (s): "
                f"fund={pnl_start-start:.2f} "
                f"pnl={inv_start-pnl_start:.2f} inv={serialize_start-inv_start:.2f} "
                f"volser={meta_start-serialize_start:.2f} meta={trades_start-meta_start:.2f} "
                f"trades={state_start-trades_start:.2f} state={pos_start-state_start:.2f} "
                f"pos={final_start-pos_start:.2f} final={time.time()-final_start:.2f}"
            )

            return data

        async def _report(self):
            if self._reporting:
                bt.logging.warning(f"Previous reporting still in progress, skipping step {self.step}")
                return

            if self.reporting_process.poll() is not None:
                bt.logging.error(f"Reporting service died with exit code {self.reporting_process.returncode}")
                bt.logging.error("Attempting to restart reporting service...")
                self._start_reporting_service()
                if self.reporting_process.poll() is not None:
                    self.pagerduty_alert("Failed to restart reporting service")
                    return

            self._reporting = True
            reporting_step = self.step
            bt.logging.info(f"Starting Reporting at step {reporting_step}...")
            start = time.time()
            try:
                drain_start = time.time()
                drained = 0
                while True:
                    try:
                        await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                self.reporting_ipc_executor,
                                lambda: self.reporting_response_queue.receive(timeout=0.001)
                            ),
                            timeout=0.05
                        )
                        drained += 1
                    except (posix_ipc.BusyError, asyncio.TimeoutError):
                        break

                if drained > 0:
                    bt.logging.warning(f"Drained {drained} stale messages from reporting response queue ({time.time()-drain_start:.4f}s)")

                prep_start = time.time()
                async with self._reward_lock:
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(
                        self.reporting_ipc_executor,
                        self._prepare_reporting_data
                    )
                prep_time = time.time() - prep_start
                published_step = data.get('step') if isinstance(data, dict) else None
                bt.logging.info(f"Prepared reporting data ({prep_time:.4f}s)")

                serialize_start = time.time()

                def _pack_report_chunked():
                    # Byte-identical to msgpack.packb(data, use_bin_type=True)
                    # (see persistence._stream_pack) but sliced into per-subtree
                    # pack() calls so serializing the ~35MB report can't hold the
                    # GIL solid against the event loop in the post-reward window.
                    import io as _io
                    from taos.im.validator.persistence import _SAVE_STREAM_DEPTH, _stream_pack
                    _buf = _io.BytesIO()
                    _stream_pack(msgpack.Packer(use_bin_type=True), _buf.write, data, _SAVE_STREAM_DEPTH)
                    return _buf.getvalue()

                data_bytes = await asyncio.get_event_loop().run_in_executor(
                    self.reporting_ipc_executor,
                    _pack_report_chunked
                )
                serialize_time = time.time() - serialize_start
                data_mb = len(data_bytes) / 1024 / 1024
                bt.logging.info(f"Reporting data: {data_mb:.2f} MB (prep={prep_time:.4f}s, serialize={serialize_time:.4f}s)")

                write_start = time.time()
                write_success = await self._write_ipc_nonblocking(
                    self.reporting_request_mem,
                    self.reporting_request_queue,
                    data_bytes,
                    "Reporting",
                    self.reporting_ipc_executor
                )

                if not write_success:
                    self.pagerduty_alert("Failed to write reporting data")
                    return

                bt.logging.info(f"Wrote reporting data ({time.time()-write_start:.4f}s)")

                await asyncio.get_event_loop().run_in_executor(
                    self.reporting_ipc_executor,
                    self.reporting_request_queue.send,
                    b'publish'
                )

                receive_start = time.time()
                max_wait = 120.0
                elapsed = 0
                message = None
                poll_count = 0
                loop = asyncio.get_event_loop()
                while elapsed < max_wait:
                    poll_count += 1
                    try:
                        # Offload the blocking queue receive to the reporting IPC
                        # executor so the main event loop stays free while the
                        # (separate) reporting process applies its metric updates.
                        # A synchronous receive() on the loop blocks every other
                        # coroutine — including handle_state for incoming simulator
                        # ticks — for the full 30-60s a report takes, inflating
                        # "State update handled" time and causing miner-query
                        # timeouts. The prior form yielded only ~0.001s every 10
                        # polls, leaving the loop ~99% starved for the report's
                        # duration.
                        message, _ = await loop.run_in_executor(
                            self.reporting_ipc_executor,
                            lambda: self.reporting_response_queue.receive(timeout=0.5)
                        )
                        bt.logging.info(f"Received reporting response after {poll_count} polls ({time.time()-receive_start:.4f}s)")
                        break
                    except posix_ipc.BusyError:
                        elapsed = time.time() - receive_start
                        if poll_count % 20 == 0:
                            bt.logging.debug(f"Still polling for reporting response ({elapsed:.1f}s, {poll_count} polls)")
                        continue
                else:
                    self.pagerduty_alert(f"Reporting response timeout after {max_wait}s")
                    return
                if message is None:
                    self.pagerduty_alert(f"Reporting response timeout after {max_wait}s")
                    return

                read_start = time.time()
                self.reporting_response_mem.seek(0)
                size_bytes = self.reporting_response_mem.read(8)
                data_size = struct.unpack('Q', size_bytes)[0]
                result_bytes = self.reporting_response_mem.read(data_size)
                result_mb = len(result_bytes) / 1024 / 1024
                time.time()
                result = msgpack.unpackb(result_bytes, raw=False, strict_map_key=False)
                bt.logging.info(f"Read reporting response data ({time.time()-read_start:.4f}s | {result_mb:.2f}MB)")
                # Reject a stale/mismatched response: under backlog the validator can
                # pick up a prior cycle's signal and read the wrong cycle's result from
                # shared memory. The subprocess echoes the step it processed; if it does
                # not match the step we published, discard rather than ingest stale stats.
                _resp_step = result.get('step') if isinstance(result, dict) else None
                if _resp_step is not None and published_step is not None and _resp_step != published_step:
                    bt.logging.warning(
                        f"Discarding stale reporting response: got step {_resp_step}, "
                        f"expected {published_step} (desync — not ingesting stale stats)"
                    )
                    return
                self.initial_balances_published = result['initial_balances_published']
                self.miner_stats = result['miner_stats']

            except Exception as e:
                self.pagerduty_alert(f"Error sending to reporting service: {e}", details={"trace": traceback.format_exc()})
            finally:
                self._reporting = False
                bt.logging.info(f"Completed reporting for step {reporting_step} ({time.time() - start:.4f}s)")

        def report(self) -> None:
            if self.config.reporting.disabled or not self.last_state or self.last_state.timestamp % self.config.scoring.interval != 0:
                return
            if self._reporting:
                bt.logging.warning(f"Skipping reporting at step {self.step} — previous report still running.")
                return
            bt.logging.debug(f"[REPORT] Scheduling from thread: {threading.current_thread().name}")
            self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._report()))

        def _match_trade_fifo(self, uid: int, book_id: int, is_buy: bool, quantity: float,
                            price: float, fee: float, timestamp: int) -> tuple[float, float]:
            """FIFO matching including fee accounting."""
            return match_trade_fifo(self, uid, book_id, is_buy, quantity, price, fee, timestamp)


        def _update_trade_volumes(self, state: MarketSimulationStateUpdate):
            """Updates and maintains all trade volume tracking and position accounting structures."""
            update_trade_volumes(self, state)


        def should_block_queries(self) -> bool:
            """Block queries if reward is lagging."""
            if self._pending_reward_tasks >= 5:
                return True
            return False

        async def _reward(self, state: MarketSimulationStateUpdate):
            """
            Asynchronously perform the full reward computation pipeline.
            """
            with self._rewarding_lock:
                self._pending_reward_tasks += 1

            start_wait = time.time()
            rewarding_step = self.step
            try:
                async with self._reward_lock:
                    waited = time.time() - start_wait
                    if waited > 0:
                        bt.logging.debug(f"Acquired reward lock after waiting {waited:.3f}s")

                    self.shared_state_rewarding = True

                    timestamp = state.timestamp
                    duration = duration_from_timestamp(timestamp)
                    bt.logging.info(f"Starting reward calculation for step {rewarding_step}...")
                    start = time.time()

                    try:
                        # Sync mutation on _reward's coroutine — atomic against
                        # other coroutines (no awaits inside update_trade_volumes)
                        # and against the MVTRX push builder via the two-step
                        # atomic-snapshot pattern in _build_sim_push_payload.
                        self._update_trade_volumes(state)
                        _pending_resets = getattr(state, '_pending_resets', None)
                        if _pending_resets:
                            self.engine.apply_resets(_pending_resets)
                        _shadow = self._scoring_shadow
                        if _shadow is not None:
                            # Applied-ts + one-time INIT (under the held _reward_lock,
                            # structures frozen) + own parity digest at the same
                            # deterministic timestamps the shadow uses.
                            self._shadow_applied_ts = timestamp
                            if not _shadow.initialized:
                                await asyncio.get_event_loop().run_in_executor(
                                    None, lambda: _shadow.send_init(self).result()
                                )
                            elif timestamp % _shadow.parity_ns == 0:
                                _shadow.record_main_digest(timestamp, (self._shadow_digest_fn(self), self._shadow_pnl_vec_fn(self)))
                        if timestamp % self.config.scoring.interval != 0:
                            bt.logging.info(f"Agent Scores Data Updated for {duration} ({time.time()-start:.4f}s)")
                            return
                        bt.logging.info("Starting reward calculation...")
                        calc_start = time.time()
                        loop = asyncio.get_event_loop()

                        # ── Cutover: the scoring service computes; main adopts ──
                        # Falls back to the in-process get_rewards on ANY failure
                        # (dead child, timeout, uid-count drift). Every
                        # SCORING_PROC_VERIFY_EVERY-th interval main ALSO computes
                        # and cross-checks (prefer main + re-INIT child on mismatch).
                        adopted = None
                        if _shadow is not None and self._scoring_proc_cutover:
                            _eager = _shadow.eager_inputs_for(timestamp)
                            if _eager is not None:
                                # inputs were shipped at tee time — the child is
                                # already computing (or done); collect only.
                                _sim_ts = _eager['simulation_timestamp']
                                _deregs = _eager['deregistered_uids']
                                _gtx_scores = _eager['gentrx_scores']
                                _gtx_ema = _eager['gentrx_ema']
                                _t_ema = _eager.get('trading_ema', {})
                                _t_ema_n = _eager.get('trading_ema_n', {})
                                _t_ema_ts = _eager.get('trading_ema_ts')
                            else:
                                _sim_ts = self.simulation_timestamp
                                _deregs = list(self.deregistered_uids)
                                _gtx_scores = (
                                    self._gentrx.get_scores()
                                    if hasattr(self, '_gentrx') and self._gentrx is not None else {}
                                )
                                _gtx_ema = getattr(self, '_gentrx_ema', {})
                                _t_ema = dict(getattr(self, '_trading_score_ema', {}) or {})
                                _t_ema_n = dict(getattr(self, '_trading_score_ema_n', {}) or {})
                                _t_ema_ts = getattr(self, '_trading_score_ema_ts', None)
                            adopted = await loop.run_in_executor(
                                None,
                                lambda: _shadow.request_scores(
                                    timestamp, _sim_ts, _deregs, _gtx_scores, _gtx_ema,
                                    timeout=float(os.environ.get("SCORING_PROC_TIMEOUT", "45")),
                                    eager=_eager is not None,
                                    trading_ema=_t_ema, trading_ema_n=_t_ema_n,
                                    trading_ema_ts=_t_ema_ts,
                                ),
                            )
                            if adopted is not None and len(adopted['trading']) != self.effective_max_uids:
                                bt.logging.warning(
                                    f"[SCORING-PROC] uid-count drift (child {len(adopted['trading'])} "
                                    f"vs {self.effective_max_uids}) — falling back + re-INIT"
                                )
                                _shadow.request_reinit()
                                adopted = None
                            if adopted is None and _shadow.initialized:
                                bt.logging.warning("[SCORING-PROC] child scoring unavailable — in-process fallback")

                        self._scoring_proc_n += 1
                        _verify_every = max(1, int(os.environ.get("SCORING_PROC_VERIFY_EVERY", "10")))
                        _verify = adopted is not None and (self._scoring_proc_n % _verify_every == 0)

                        if adopted is None or _verify:
                            # Verify must compute with EXACTLY the inputs the child
                            # was given — simulation_timestamp advances during the
                            # adoption wait and shifts decay windows, producing
                            # false MISMATCHes (and spurious 15s re-INITs) against
                            # a live-read compute.
                            _pin = {
                                'simulation_timestamp': _sim_ts,
                                'deregistered_uids': _deregs,
                                'gentrx_scores': _gtx_scores,
                                'gentrx_ema': _gtx_ema,
                                'trading_ema': _t_ema,
                                'trading_ema_n': _t_ema_n,
                                'trading_ema_ts': _t_ema_ts,
                            } if _verify else None
                            trading_rewards, gentrx_rewards, updated_data, all_uids = await loop.run_in_executor(
                                self.reward_executor,
                                lambda: get_rewards(self, pinned_inputs=_pin)
                            )
                            if _verify:
                                _mine = [float(x) for x in trading_rewards.tolist()]
                                if _mine == adopted['trading']:
                                    bt.logging.info(
                                        f"[SCORING-PROC] VERIFY ts={timestamp} MATCH (n={len(_mine)})"
                                    )
                                else:
                                    _diffs = sum(1 for a, b in zip(_mine, adopted['trading']) if a != b)
                                    bt.logging.error(
                                        f"[SCORING-PROC] VERIFY ts={timestamp} MISMATCH "
                                        f"(diff_uids={_diffs}) — preferring main + re-INIT child"
                                    )
                                    _shadow.request_reinit()
                                # verify path keeps main's outputs (already bound)
                        else:
                            trading_rewards = torch.tensor(
                                adopted['trading'], dtype=torch.float32, device=self.device
                            )
                            gentrx_rewards = torch.tensor(
                                adopted['gentrx'], dtype=torch.float32, device=self.device
                            )
                            updated_data = adopted['factors']
                            all_uids = list(range(len(adopted['trading'])))
                            self._gentrx_ema = adopted['gentrx_ema']
                            if 'trading_ema' in adopted:
                                self._trading_score_ema = {int(k): float(v) for k, v in
                                                           (adopted['trading_ema'] or {}).items()}
                                self._trading_score_ema_n = {int(k): int(v) for k, v in
                                                             (adopted.get('trading_ema_n') or {}).items()}
                                self._trading_score_ema_ts = adopted.get('trading_ema_ts')

                        bt.logging.info(
                            f"Reward calculation completed ({time.time()-calc_start:.4f}s"
                            f"{', adopted from scoring service' if adopted is not None and not _verify else ''})"
                        )

                        if _shadow is not None and not self._scoring_proc_cutover and _shadow.initialized:
                            # Shadow-only mode: release the child's held boundary
                            # with the exact inputs this reward used + record
                            # main's trading scores for the [SHADOW-SCORES] compare.
                            _pre = getattr(self, '_trading_score_ema_pre', None) or ({}, {}, None)
                            _shadow.on_main_scored(
                                timestamp,
                                updated_data['sim_ts_used'],
                                updated_data['deregs_used'],
                                [float(x) for x in trading_rewards.tolist()],
                                trading_ema=_pre[0], trading_ema_n=_pre[1],
                                trading_ema_ts=_pre[2],
                            )

                        self.kappa_values = updated_data['kappa_values']
                        self.activity_factors = updated_data['activity_factors']
                        self.pnl_factors = updated_data['pnl_factors']

                        bt.logging.debug(
                            f"Agent Rewards Recalculated for {duration} ({time.time()-start:.4f}s):\n"
                            f"trading={trading_rewards}\ngentrx={gentrx_rewards}"
                        )
                        self.update_scores(trading_rewards, all_uids, gentrx_rewards=gentrx_rewards)
                        bt.logging.info(f"Agent Scores Updated for {duration} ({time.time()-start:.4f}s)")
                        self._last_rewarded_sim_timestamp = timestamp

                    except Exception as ex:
                        self.pagerduty_alert(f"Rewarding failed: {ex}", details={"trace": traceback.format_exc()})
                    finally:
                        self.shared_state_rewarding = False
                        bt.logging.debug(f"Completed rewarding (TOTAL {time.time()-start_wait:.4f}s).")
            finally:
                with self._rewarding_lock:
                    self._pending_reward_tasks -= 1

        def reward(self, state : MarketSimulationStateUpdate) -> None:
            """
            Schedule asynchronous reward calculation on the validator's main event loop.
            Offloads work to `_reward()` to ensure that:

            • Reward computation always occurs in the correct asyncio event loop
            • CPU-intensive work does not block the calling thread
            • Reward logic executes with proper async locking

            Args:
                state (MarketSimulationStateUpdate):
                    Simulation state for the current tick, forwarded to `_reward`.
            """
            if getattr(getattr(self.config, 'neuron', None), 'observe', False):
                return
            bt.logging.debug(f"[REWARD] Scheduling from thread: {threading.current_thread().name}")
            bt.logging.debug(f"[REWARD] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
            self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._reward(state)))

        def _build_agent_scoring_maps(self) -> dict:
            """Per-agent scoring maps (scores, kappa, volume, pnl, fees) for the
            data-service snapshot, computed from the incrementally-maintained
            accumulators.

            Both the simulation and exchange ingest payloads must carry these so the
            agent_snapshots row (top-level score/volume_24h/pnl_24h and per_book
            vol/fee/pnl/kappa) is populated identically in both modes — parity of
            process. The genuine difference is only in what feeds the accumulators:
            in exchange, update_trade_volumes(state) folds the RECONCILED
            NormalizedState (reward() → trade.py), so these reflect only on-chain-
            settled activity, satisfying "snapshots reflect reconciled data".

            Uses the same two-step atomic outer/inner snapshot as
            _build_sim_push_payload to stay safe against concurrent trade.py mutation
            from the reward thread (see the note there).

            NOTE: _build_sim_push_payload still inlines an equivalent computation; it
            should be consolidated onto this helper once the exchange path is verified
            in production, so the two cannot drift.
            """
            _vs_outer = dict(getattr(self, "volume_sums", {}))
            _mvs_outer = dict(getattr(self, "maker_volume_sums", {}))
            _tvs_outer = dict(getattr(self, "taker_volume_sums", {}))
            _fs_outer = dict(getattr(self, "fee_sums", {}))
            _rt_outer = dict(getattr(self, "roundtrip_volume_sums", {}))
            _af_outer = dict(getattr(self, "activity_factors", {}))
            _snap_kv = dict(getattr(self, "kappa_values", {}))
            _snap_scores = list(self.scores) if getattr(self, "scores", None) is not None else None
            _snap_pnl_book = {u: dict(b) for u, b in dict(getattr(self, "agent_pnl_by_book", {})).items()}
            _snap_pnl_total = dict(getattr(self, "agent_pnl_total", {}))
            _snap_vs = {u: dict(b) for u, b in _vs_outer.items()}
            _snap_mvs = {u: dict(b) for u, b in _mvs_outer.items()}
            _snap_tvs = {u: dict(b) for u, b in _tvs_outer.items()}
            _snap_fs = {u: dict(b) for u, b in _fs_outer.items()}
            _snap_rt = {u: dict(b) for u, b in _rt_outer.items()}
            _snap_af = {u: dict(b) for u, b in _af_outer.items()}

            def _sv(d):
                return {
                    str(uid): sum(float(v) for v in list(bks.values()))
                    for uid, bks in list(d.items())
                    if bks
                }

            _vs = _sv(_snap_vs)
            _mvs = _sv(_snap_mvs)
            _tvs = _sv(_snap_tvs)
            _pnl = {str(uid): round(float(v), 6) for uid, v in _snap_pnl_total.items() if v != 0.0}
            _sc = _snap_scores
            _sc_dict = {str(i): float(_sc[i]) for i in range(len(_sc))} if _sc is not None else {}
            _kappa_raw = {}
            _kappa_score = {}
            _kappa_penalty = {}
            _kappa_books = {}
            _kappa_books_w = {}
            for _kuid, _kv in _snap_kv.items():
                if _kv and isinstance(_kv, dict):
                    if _kv.get("total") is not None:
                        _kappa_raw[str(_kuid)] = float(_kv["total"])
                    if _kv.get("normalized_total") is not None:
                        _kappa_score[str(_kuid)] = float(_kv["normalized_total"])
                    if _kv.get("penalty") is not None:
                        _kappa_penalty[str(_kuid)] = float(_kv["penalty"])
                    _bks = {str(bid): float(v) for bid, v in (_kv.get("books") or {}).items() if v is not None}
                    if _bks:
                        _kappa_books[str(_kuid)] = _bks
                    _bw = {str(bid): float(v) for bid, v in (_kv.get("books_weighted") or {}).items() if v is not None}
                    if _bw:
                        _kappa_books_w[str(_kuid)] = _bw
            return {
                "agent_scores": _sc_dict,
                "agent_kappa": _kappa_raw,
                "agent_kappa_score": _kappa_score,
                "agent_kappa_penalty": _kappa_penalty,
                "agent_kappa_books": _kappa_books,
                "agent_kappa_books_w": _kappa_books_w,
                "agent_volume": _vs,
                "agent_maker_volume": _mvs,
                "agent_taker_volume": _tvs,
                "agent_pnl": _pnl,
                "agent_pnl_book": {
                    str(uid): {str(bid): round(float(v), 6) for bid, v in bks.items() if v != 0}
                    for uid, bks in _snap_pnl_book.items()
                    if bks
                },
                "agent_volume_book": {
                    str(uid): {str(bid): round(float(v), 4) for bid, v in bks.items() if v}
                    for uid, bks in _snap_vs.items()
                    if bks
                },
                "agent_fee_book": {
                    str(uid): {str(bid): round(float(v), 6) for bid, v in bks.items() if v != 0}
                    for uid, bks in _snap_fs.items()
                    if bks
                },
                "agent_roundtrip_volume": {
                    str(uid): round(float(sum(bks.values())), 4) for uid, bks in _snap_rt.items() if bks
                },
                "agent_activity_factor": {
                    str(uid): round(float(sum(bks.values()) / len(bks)), 4) for uid, bks in _snap_af.items() if bks
                },
                "agent_median_kappa": {
                    str(uid): round(float(kv.get("activity_weighted_normalized_median") or 0), 6)
                    for uid, kv in _snap_kv.items()
                    if kv
                },
            }

        def _build_sim_push_payload(self, state) -> dict:
            """Build the MVTRX data-service push payload for a simulation state update.

            Runs in _mvtrx_push_executor via _push_sim_state_bg. The former
            hot loop (agent_pnl_book re-summing realized_pnl_history O(N*T*B)
            per push) was the last remaining Python-CPU spike per state cycle
            in 0.5.x. It's been eliminated by maintaining agent_pnl_by_book /
            agent_pnl_total as incremental running totals in trade.py, which
            reduces the pnl aggregation portion of the push builder from
            ~5-7s to a direct dict read.

            The remaining volatile-dict reads (volume_sums, fee_sums, etc.)
            are already O(N*B) — tiny compared to what was replaced. They use
            a two-step atomic dict.copy() pattern to stay race-safe against
            trade.py's concurrent mutations (single C-level PyDict_Copy holds
            the GIL throughout, so no interleaving).
            """
            _sim_fills: list = []
            _sim_rejects: list = []
            _seen_trade_ids: set = set()
            # External-order UUID threading (exchange-API orders): RDPOL/RDPOM
            # placement events echo the clientOrderId minted in
            # _fetch_external_orders; capture engine-orderId → UUID (ET fills
            # reference engine ids) and stamp the UUID on the placement notice
            # ('xo') so the data service can flip its order store to OPEN.
            # NB 'o' on RDPOL/RDPOM is the ENGINE orderId (protocol alias) —
            # never overload it; the UUID always travels as 'xo'.
            _cloid_map = getattr(self.engine, '_ext_cloid_to_uuid', None) or {}
            _engoid_map = getattr(self.engine, '_ext_engineoid_to_uuid', None)
            if _cloid_map and _engoid_map is not None:
                for _uid_str, _evs in (getattr(state, 'notices', {}) or {}).items():
                    for _ev in (_evs or []):
                        if not isinstance(_ev, dict):
                            continue
                        if (_ev.get("y") or _ev.get("type")) not in ("RDPOL", "RDPOM"):
                            continue
                        try:
                            _uuid = _cloid_map.get(int(_ev.get("c") or 0))
                            if not _uuid:
                                continue
                            _ev["xo"] = _uuid
                            _eng_oid = _ev.get("o")
                            if _eng_oid is not None:
                                _engoid_map[int(_eng_oid)] = _uuid
                        except (TypeError, ValueError):
                            continue
            for _uid_str, _evs in (getattr(state, 'notices', {}) or {}).items():
                for _ev in (_evs or []):
                    if not isinstance(_ev, dict):
                        continue
                    _ev_type = _ev.get("y") or _ev.get("type")
                    if _ev_type in ("ET", "EVENT_TRADE"):
                        _tid = _ev.get("i") if _ev.get("i") is not None else _ev.get("tradeId")
                        if _tid is None:
                            continue
                        _book_id = _ev.get("b") if _ev.get("b") is not None else _ev.get("bookId")
                        _price   = float(_ev.get("p") or _ev.get("price") or 0)
                        _qty     = float(_ev.get("q") or _ev.get("quantity") or 0)
                        _side    = int(_ev.get("s") if _ev.get("s") is not None else _ev.get("side", 0))
                        _taker   = _ev.get("Ta") if _ev.get("Ta") is not None else _ev.get("takerAgentId")
                        _maker   = _ev.get("Ma") if _ev.get("Ma") is not None else _ev.get("makerAgentId")
                        _ti      = _ev.get("Ti")
                        _mi      = _ev.get("Mi")
                        _cr_raw  = _ev.get("cr", 0) or 0
                        _cr      = "SL" if _cr_raw == 1 else ("TP" if _cr_raw == 2 else None)
                        _toi     = _ev.get("Toi") or None
                        for _agent_uid, _is_taker in ((_taker, True), (_maker, False)):
                            if _agent_uid is None:
                                continue
                            _key = (_tid, int(_agent_uid))
                            if _key in _seen_trade_ids:
                                continue
                            _seen_trade_ids.add(_key)
                            _dir  = _side if _is_taker else (1 - _side)
                            _oid  = _ti if _is_taker else _mi
                            _fill_cr  = _cr if _is_taker else None
                            _fill_toi = _toi if (_is_taker and _cr) else None
                            # Exchange-API order UUID for this side's engine
                            # order id, when the order came in via the REST rail.
                            _xo = None
                            if _engoid_map and _oid is not None:
                                try:
                                    _xo = _engoid_map.get(int(_oid))
                                except (TypeError, ValueError):
                                    _xo = None
                            _sim_fills.append({
                                "uid":              int(_agent_uid),
                                "netuid":           int(_book_id) if _book_id is not None else 0,
                                "price":            _price,
                                "qty":              _qty,
                                "alpha_volume":     _qty,
                                "tao_volume":       round(_price * _qty, 9),
                                "direction":        _dir,
                                "side":             "buy" if _dir == 0 else "sell",
                                "role":             "taker" if _is_taker else "maker",
                                "trade_id":         _tid,
                                "order_id":         int(_oid) if _oid is not None else None,
                                "close_reason":     _fill_cr,
                                "linked_order_id":  int(_fill_toi) if _fill_toi is not None else None,
                                "maker_uid":        int(_maker) if _maker is not None else None,
                                "is_partial":       False,
                                "timestamp":        state.timestamp,
                                "xo":               _xo,
                            })
                    elif _ev_type in ("ERDPOL", "ERDPOM"):
                        _agent_uid = _ev.get("a") if _ev.get("a") is not None else _ev.get("agentId")
                        if _agent_uid is None:
                            continue
                        _seen_rej = (_ev_type, int(_agent_uid))
                        if _seen_rej in _seen_trade_ids:
                            continue
                        _seen_trade_ids.add(_seen_rej)
                        _sim_rejects.append({
                            "uid":    int(_agent_uid),
                            "netuid": int(_ev.get("b") if _ev.get("b") is not None else (_ev.get("bookId") or 0)),
                            "reason": _ev.get("m") or _ev.get("message") or "rejected",
                            "side":   int(_ev.get("s") if _ev.get("s") is not None else _ev.get("side", 0)),
                            "qty":    float(_ev.get("q") or _ev.get("quantity") or 0),
                            "price":  float(_ev.get("p") or _ev.get("price") or 0),
                        })
            _sim_open_orders: dict = {}
            _sim_orders_detail: dict = {}
            for _uid_str, _uid_data in (getattr(state, 'accounts', {}) or {}).items():
                try:
                    _uid = int(_uid_str)
                    if not isinstance(_uid_data, dict):
                        continue
                    for _nid_str, _acct in _uid_data.items():
                        if not isinstance(_acct, dict):
                            continue
                        for _o in (_acct.get('o') or []):
                            if not isinstance(_o, dict):
                                continue
                            _sim_open_orders[_uid] = _sim_open_orders.get(_uid, 0) + 1
                            _sim_orders_detail.setdefault(_uid, []).append({
                                "order_id":  _o.get('i') or _o.get('id'),
                                "netuid":    int(_nid_str),
                                "side":      int(_o.get('s', _o.get('side', 0))),
                                "price":     float(_o.get('p', _o.get('price', 0))),
                                "quantity":  float(_o.get('q', _o.get('quantity', 0))),
                                "placed_at": int(_o.get('t') or 0) // 1_000_000,
                            })
                except Exception:
                    continue
            # Two-step atomic snapshot — no lock. Each dict(some_dict) call is a
            # single C-level PyDict_Copy that holds the GIL for the whole copy,
            # so mutations from other Python threads (trade.py inside _reward,
            # etc.) cannot preempt during it. Step A snapshots the OUTER level
            # atomically. Step B iterates the STABLE snapshot to produce inner
            # copies — safe because the snapshot's key set is fixed, and each
            # dict(v) inside the comprehension is itself atomic.
            #
            # A single-step comprehension `{u: dict(h) for u, h in self.X.items()}`
            # is NOT safe: the outer for-loop is multiple bytecodes and a
            # concurrent add to self.X during iteration raises
            # `RuntimeError: dictionary changed size during iteration`.
            # Splitting into two atomic steps eliminates that window.
            _vs_outer  = dict(getattr(self, 'volume_sums', {}))
            _mvs_outer = dict(getattr(self, 'maker_volume_sums', {}))
            _tvs_outer = dict(getattr(self, 'taker_volume_sums', {}))
            _fs_outer  = dict(getattr(self, 'fee_sums', {}))
            _rt_outer  = dict(getattr(self, 'roundtrip_volume_sums', {}))
            _af_outer  = dict(getattr(self, 'activity_factors', {}))
            _snap_kv   = dict(getattr(self, 'kappa_values', {}))
            _snap_scores = list(self.scores) if getattr(self, 'scores', None) is not None else None
            # MVTRX push agent_pnl / agent_pnl_book: read pre-aggregated running
            # totals maintained incrementally by trade.py, NOT re-walk of
            # realized_pnl_history (which cost ~5-7s of Python CPU per push
            # cycle at N=259 UIDs × T~2000 timestamps × B=128 books).
            _snap_pnl_book  = {u: dict(b) for u, b in dict(getattr(self, 'agent_pnl_by_book', {})).items()}
            _snap_pnl_total = dict(getattr(self, 'agent_pnl_total', {}))
            _snap_vs   = {u: dict(b) for u, b in _vs_outer.items()}
            _snap_mvs  = {u: dict(b) for u, b in _mvs_outer.items()}
            _snap_tvs  = {u: dict(b) for u, b in _tvs_outer.items()}
            _snap_fs   = {u: dict(b) for u, b in _fs_outer.items()}
            _snap_rt   = {u: dict(b) for u, b in _rt_outer.items()}
            _snap_af   = {u: dict(b) for u, b in _af_outer.items()}
            def _sv(d):
                return {str(uid): sum(float(v) for v in list(bks.values()))
                        for uid, bks in list(d.items()) if bks}
            _vs  = _sv(_snap_vs)
            _mvs = _sv(_snap_mvs)
            _tvs = _sv(_snap_tvs)
            _pnl = {str(uid): round(float(v), 6) for uid, v in _snap_pnl_total.items() if v != 0.0}
            _sc  = _snap_scores
            _sc_dict = ({str(i): float(_sc[i]) for i in range(len(_sc))}
                        if _sc is not None else {})
            _kappa_raw   = {}
            _kappa_score = {}
            _kappa_penalty = {}
            _kappa_books = {}
            _kappa_books_w = {}
            for _kuid, _kv in _snap_kv.items():
                if _kv and isinstance(_kv, dict):
                    if _kv.get('total') is not None:
                        _kappa_raw[str(_kuid)] = float(_kv['total'])
                    if _kv.get('normalized_total') is not None:
                        _kappa_score[str(_kuid)] = float(_kv['normalized_total'])
                    if _kv.get('penalty') is not None:
                        _kappa_penalty[str(_kuid)] = float(_kv['penalty'])
                    _bks = {str(bid): float(v) for bid, v in (_kv.get('books') or {}).items() if v is not None}
                    if _bks:
                        _kappa_books[str(_kuid)] = _bks
                    _bw = {str(bid): float(v) for bid, v in (_kv.get('books_weighted') or {}).items() if v is not None}
                    if _bw:
                        _kappa_books_w[str(_kuid)] = _bw
            _book_vol: dict = {}
            for _uid_bk, _bk_dict in _snap_vs.items():
                for _bid, _bvol in _bk_dict.items():
                    _book_vol[int(_bid)] = _book_vol.get(int(_bid), 0.0) + float(_bvol)
            _sim_pools: dict = dict(state.pools or {})
            for _bid, _bvol in _book_vol.items():
                _bk = _sim_pools.get(_bid) or _sim_pools.get(str(_bid))
                if _bk is not None:
                    _bk['volume_24h'] = round(_bvol, 4)
            _sim_live_triggers = dict(getattr(getattr(self, 'engine', None), '_live_triggers', {}))
            if hasattr(self.engine, '_sltp_changed'):
                self.engine._sltp_changed = False
            return {
                "mode":                "simulation",
                "simulation_id":       getattr(self.simulation, 'simulation_id', None),
                # data-service uses this to attribute lifecycle events per
                # validator; NOT adding payload "network" here (the ingest
                # INGEST_NETWORK filter would reject a mismatch — the service
                # stamps its own network on events).
                "validator_hotkey":    self.wallet.hotkey.ss58_address if getattr(self, 'wallet', None) else None,
                "timestamp":           state.timestamp,
                "block":               state.block,
                "books":               state.books or {},
                "accounts":            state.accounts or {},
                "pools":               _sim_pools,
                "benchmark_agents":    [{"uid": _ba["uid"], "coldkey": _ba.get("coldkey", ""), "hotkey": _ba.get("hotkey", ""), "name": _ba.get("name", "")} for _ba in getattr(self, 'benchmark_agents', [])],
                "proxy_wallets":       getattr(self, 'proxy_wallets_published', {}) or {},
                # Separate key from proxy_wallets on purpose: the swap proxy is address-keyed and
                # permanent, the exchange proxy is uid-keyed and evicted on deregistration. One
                # key for both would let the service serve a swap wallet as an exchange proxy.
                "swap_wallets":        getattr(self, 'swap_wallets_published', {}) or {},
                "reconciliation":      {"fills": _sim_fills, "rejections": _sim_rejects},
                "notices":             {str(k): list(v) for k, v in (state.notices or {}).items()},
                "agent_open_orders":   _sim_open_orders,
                "agent_orders_detail": _sim_orders_detail,
                "validator_uid":       getattr(self, 'uid', None),
                "chain_block":         getattr(self, 'current_block', None),
                "metagraph":           (lambda _m: {
                    "hotkeys":         [str(hk) for hk in _m.hotkeys],
                    "coldkeys":        [str(ck) for ck in _m.coldkeys] if hasattr(_m, 'coldkeys') else [],
                    "stake":           _m.stake.tolist(),
                    "emission":        _m.emission.tolist(),
                    "incentive":       _m.incentive.tolist(),
                    "validator_trust": _m.validator_trust.tolist(),
                    "validator_permit":[bool(v) for v in _m.validator_permit] if hasattr(_m, 'validator_permit') else [],
                    "dividends":       _m.dividends.tolist() if hasattr(_m, 'dividends') else [],
                    "last_update":     _m.last_update.tolist() if hasattr(_m, 'last_update') else [],
                } if _m is not None else {})(getattr(self, 'metagraph', None)),
                "agent_scores":        _sc_dict,
                "agent_kappa":         _kappa_raw,
                "agent_kappa_score":   _kappa_score,
                "agent_kappa_penalty": _kappa_penalty,
                "agent_kappa_books":   _kappa_books,
                "agent_kappa_books_w": _kappa_books_w,
                "agent_volume":        _vs,
                "agent_maker_volume":  _mvs,
                "agent_taker_volume":  _tvs,
                "agent_pnl":           _pnl,
                "agent_pnl_book":      {str(uid): {str(bid): round(float(v), 6)
                                        for bid, v in bks.items() if v != 0}
                                       for uid, bks in _snap_pnl_book.items() if bks},
                "agent_volume_book":   {str(uid): {str(bid): round(float(v), 4)
                                        for bid, v in bks.items() if v}
                                       for uid, bks in _snap_vs.items() if bks},
                "agent_fee_book":      {str(uid): {str(bid): round(float(v), 6)
                                        for bid, v in bks.items() if v != 0}
                                       for uid, bks in _snap_fs.items() if bks},
                "agent_roundtrip_volume": {str(uid): round(float(sum(bks.values())), 4)
                                           for uid, bks in _snap_rt.items() if bks},
                "agent_activity_factor":   {str(uid): round(float(sum(bks.values()) / len(bks)), 4)
                                            for uid, bks in _snap_af.items() if bks},
                "agent_median_kappa":      {str(uid): round(float(kv.get('activity_weighted_normalized_median') or 0), 6)
                                           for uid, kv in _snap_kv.items() if kv},
                "fee_policy":              ({"fee_type": self.simulation.fee_policy.fee_type,
                                            **{k: float(v) for k, v in self.simulation.fee_policy.params.items()
                                               if k in ("targetMTR", "makerFee", "takerFee", "maxMakerRate", "maxTakerRate")}}
                                           if getattr(self.simulation, 'fee_policy', None) else None),
                "exchange_constraints":    ({"min_order_size":     float(getattr(self.simulation, 'min_order_size', 0.0)),
                                             "max_open_orders":    int(getattr(self.simulation, 'max_open_orders', 0) or 0),
                                             "max_leverage":       float(getattr(self.simulation, 'max_leverage', 0)),
                                             "max_loan":           float(getattr(self.simulation, 'max_loan', 0)),
                                             "maintenance_margin": float(getattr(self.simulation, 'maintenance_margin', 0)),
                                             "price_decimals":     int(getattr(self.simulation, 'priceDecimals', 4)),
                                             "volume_decimals":    int(getattr(self.simulation, 'volumeDecimals', 4)),
                                             "init_price":         float(getattr(self.simulation, 'init_price', 0)),
                                             "time_unit":          str(getattr(self.simulation, 'time_unit', 'ns')),
                                             "grace_period":       int(getattr(self.simulation, 'grace_period', 0)),
                                            }
                                           if getattr(self, 'simulation', None) else None),
                "sltp_triggers": _sim_live_triggers,
            }

        async def _push_sim_state_bg(self, state, url: str) -> None:
            """Async wrapper: build the MVTRX push payload in a dedicated
            executor pool (_mvtrx_push_executor), then POST it. Introduced to
            move ~7-11s of Python payload construction off the main event loop;
            the main loop was falling ~6x behind the 5s scoring interval
            because agent_pnl_book iterates realized_pnl_history for every UID
            (259 UIDs × ~2000 timestamps × 128 books).

            Race handling: no lock. _build_sim_push_payload uses a two-step
            atomic-snapshot pattern (outer dict.copy() first, then inner
            copies over the stable snapshot) — safe against trade.py mutations
            in _reward because each dict() call is a single C-level
            PyDict_Copy that holds the GIL. Removes lock contention that was
            blocking _reward and causing "waiting for rewarding to catch up"
            (5+ pending tasks blocking queries).
            """
            try:
                loop = asyncio.get_event_loop()
                payload = await loop.run_in_executor(
                    self._mvtrx_push_executor,
                    self._build_sim_push_payload,
                    state,
                )
                await _push_mvtrx(payload, url=url)
            except Exception as _exc:
                bt.logging.warning(f"_push_sim_state_bg failed: {_exc}")

        async def handle_state(self, state: NormalizedState, receive_start: float) -> dict:
            """
            Handle a full simulator state update, enrich it with validator data, compute responses,
            update internal validator state, and return instructions back to the simulator.


            This method is the central processing loop for each simulation step. It performs:
            - Periodic validator configuration reloads.
            - Per‑account volume injection.
            - Simulation metadata updates and logging.
            - State forwarding to miners and response aggregation.
            - Reward calculation, scoring, persistence, and metric publication.


            Args:
            state (NormalizedState): Canonical state envelope from the engine.
            receive_start (float): time.time() when the message arrived, used for latency metrics.


            Returns:
            dict: Serialized response batch to be returned to the simulator.
            """
            # [ROUND-PROFILE] phase timers — split the "State update handled"
            # time into recv_norm/pre/forward/post/sched to locate the tail.
            _rp_entry = time.time()

            # Per-tick simulation-specific updates (logDir, simulation_timestamp,
            # simulation_id, periodic compression/update_repo).
            self.engine.on_tick(state)

            start = time.time()
            for uid, accounts in state.accounts.items():
                for book_id in accounts:
                    # volume_sums IS NESTED BY UID, NOT KEYED BY A TUPLE.
                    #
                    # This read a 2-tuple key out of a {uid: {book_id: float}} defaultdict, so it
                    # missed on every call and wrote 0.0 for every miner. Silent twice over: .get()
                    # with a default raises nothing, and .get() on a defaultdict does not invoke
                    # default_factory, so no entry was created to hint at the miss. Every writer uses
                    # the nested shape (trade.py, persistence.py, both engines) and report.py consumes
                    # it nested. Measured working: volume_sums[171][5] = 1383.308 reaching the
                    # state dict and the wire.
                    state.accounts[uid][book_id]['v'] = self.volume_sums.get(uid, {}).get(book_id, 0.0)
            bt.logging.info(f"Volumes added to state ({time.time()-start:.4f}s).")

            # Update variables
            if not self.start_time:
                self.start_time = time.time()
                self.start_timestamp = state.timestamp

            self.step_rates.append((state.timestamp - (self.last_state.timestamp if self.last_state else self.start_timestamp)) / (time.time() - (self.last_state_time if self.last_state_time else self.start_time)))
            self.last_state = state
            self.step += 1

            # Log received state data
            # exchange/observe mode: state.timestamp is the engine BLOCK clock
            # (block x BLOCK_TIME_NS), not a real-world time. Report the engine's
            # batch-processing time instead: the chain state fetched for this batch
            # is stamped with a unix-SECONDS timestamp (chain_state['timestamp']),
            # which is the exchange engine's real-world batch clock. Format
            # dd/mm/yyyy HH:MM:SS.nnnnnnnnn (seconds source -> .000000000 nanos).
            # Simulation mode keeps the sim-elapsed duration format unchanged.
            if self.engine.mode == 'exchange':
                _cs = getattr(self.engine, '_last_chain_state', None) or {}
                _batch_ts = _cs.get('timestamp')
                if _batch_ts:
                    _t_disp = datetime.fromtimestamp(_batch_ts).strftime('%d/%m/%Y %H:%M:%S') + '.000000000'
                else:
                    # no chain state yet (pre-first-batch); fall back to wall clock
                    _now = datetime.now()
                    _t_disp = _now.strftime('%d/%m/%Y %H:%M:%S') + f".{_now.microsecond * 1000:09d}"
            else:
                _t_disp = duration_from_timestamp(state.timestamp)
            bt.logging.info(f"STATE UPDATE RECEIVED | VALIDATOR STEP : {self.step} | TIME : {_t_disp} (T={state.timestamp})")
            if self.config.logging.debug or self.config.logging.trace:
                debug_text = ''
                for bookId, book in state.books.items():
                    debug_text += '-' * 50 + "\n"
                    debug_text += f"BOOK {bookId}" + "\n"
                    if book['b'] and book['a']:
                        debug_text += ' | '.join([f"{level['q']:.4f}@{level['p']}" for level in reversed(book['b'][:5])]) + '||' + ' | '.join([f"{level['q']:.4f}@{level['p']}" for level in book['a'][:5]]) + "\n"
                    else:
                        debug_text += "EMPTY" + "\n"
                bt.logging.debug("\n" + debug_text.strip("\n"))

            # Process deregistration notices
            self.process_resets(state)

            # GenTRX: enqueue the raw state for the background packager.
            # As of PR-1 of the GenTRX isolation plan, push_state() does
            # NO extract_state / msgpack work synchronously — it just
            # appends a state-object reference to a bounded queue (drop-
            # oldest on overflow), which is sub-millisecond. The heavy
            # extract + pack runs on `GenTRX-pack` worker thread, the
            # HTTP POST on `GenTRX-tx` worker thread. Neither sits on
            # the validator's hot path.
            if self._gentrx is not None:
                try:
                    self._gentrx.push_state(state)
                except Exception as _gex:
                    bt.logging.warning(f"[GTX] handle_state push_state error: {_gex}")

            # Surface settled fills to miners before the synapse goes out.
            #
            # Miners received placement and cancel notices but never a fill notice in
            # only the ingest push consumed them, so a miner could not tell which of its
            # orders had traded. Merging here: the same consume-once point, just earlier
            #: puts them on the synapse as well. The id carried is the engine's own
            # integer, matching TradeEvent.i and what simulation miners already see; the
            # data service book-qualifies it later for its own surfaces.
            if self.engine.mode == 'exchange':
                # Settled fills, refusals and fill notices reach the miner from an optional companion
                # module that is not part of this tree; without it the validator does not surface them.
                # Order matters: fills, then refusals, then fill notices, as a miner reading them in
                # sequence expects.
                if _exchange_notices is not None:
                    # One call, one accounting line covering all four kinds. Four separate calls each
                    # logged only when they had something, so a kind that stopped being delivered was
                    # silent -- which is how acknowledgements went missing for every miner while fills
                    # kept flowing and every surface still looked healthy.
                    _exchange_notices.merge_all(self.engine, state, self.config)


            # Forward state synapse to miners and collect responses
            start = time.time()
            _rp_fwd0 = start
            miner_responses = await forward(self, state)
            _rp_fwd1 = time.time()
            bt.logging.debug(f"Gathered Response Batch ({time.time()-start}s)")

            # Exchange: send instructions to LOB, execute on-chain, inject trade events
            if self.engine.mode == 'exchange':
                start = time.time()
                _trade_events: list = []  # preserved for nexus book event injection
                try:
                    trade_events = await self.engine.execute(state, miner_responses)
                    if trade_events:
                        _trade_events = trade_events
                        for event in trade_events:
                            _notice = event.to_notice_dict() if hasattr(event, 'to_notice_dict') else event.__dict__
                            for _uid in filter(None, {
                                getattr(event, 'taker_uid', None),
                                getattr(event, 'maker_uid', None),
                            }):
                                state.notices.setdefault(_uid, []).append(_notice)
                        bt.logging.info(
                            f"ExchangeEngine.execute: {len(trade_events)} trade events "
                            f"injected into state ({time.time()-start:.4f}s)"
                        )
                        # NOTE: the pre-settlement fast-path fill notify is intentionally
                        # disabled. trade_events here are the engine's LOB match BEFORE
                        # on-chain settlement — they carry predicted quantities and no
                        # tao_volume, so reporting them announces a fill that has not yet
                        # (and may not exactly) settle. Fills are reported from a single
                        # authoritative source: the reconciled executed_fills carried by
                        # the per-block ingest push (neurons/validator.py exchange payload
                        # → ingest.py executed_fills → push_agent_fill), which reflect the
                        # actual on-chain-settled amounts. Restore a fast-notify only if
                        # it is driven from reconciled results, not from this pre-settle
                        # match.
                except Exception as _exc:
                    bt.logging.error(f"ExchangeEngine.execute failed: {_exc}")
                # Refresh state.books/accounts from post-execute LOB state so the
                # ingest payload carries fill events from this block (not just the
                # reconciliation snapshot from receive()).
                try:
                    _cs = getattr(self.engine, '_last_chain_state', None)
                    if _cs is not None and hasattr(self.engine, '_normalize'):
                        _post = self.engine._normalize(_cs)
                        state.books    = _post.books
                        state.accounts = _post.accounts
                except Exception:
                    pass
                # Inject the SINGLE authoritative fill event into state.books['e'] for
                # the nexus L3 stream. Post-reconcile, the surfaced fill is the reconciled
                # executed_fill: on-chain-SETTLED quantity + agent attribution + trade_id.
                # The predicted LOB match and the raw chain 't' are stripped in _normalize
                # (that pre-settlement view is deliberately surfaced only in the engine's
                # first-prediction-step output, never on this post-reconcile payload).
                # execute()'s trade_events are empty in the async-settlement flow, so the
                # event is built from the reconciliation (self._pending_ingest_reconciliation,
                # set post-settlement, consumed at the ingest push below).
                # Maker/taker attribution comes from a matching trade_event when one is
                # present (by trade_id); otherwise the reconciliation carries only the
                # settling agent, surfaced as the maker side (matches the LOB convention
                # where the miner's resting order is the maker, counterparty = pool).
                try:
                    _fill_ts_ns = int(time.time() * 1e9)
                    _te_by_tid = {}
                    for _te in (_trade_events or []):
                        _tid_k = getattr(_te, 'trade_id', None)
                        if _tid_k is not None:
                            _te_by_tid[str(_tid_k)] = _te
                    _recon_now = getattr(self.engine, '_pending_ingest_reconciliation', None) or {}
                    for _ef in _recon_now.get('executed_fills', []):
                        _nid = _ef.get('netuid')
                        if _nid is None:
                            continue
                        _tid = _ef.get('trade_id')
                        _agent = _ef.get('agent_id')
                        _te_m = _te_by_tid.get(str(_tid)) if _tid is not None else None
                        if _te_m is not None:
                            _ta = getattr(_te_m, 'taker_uid', None)
                            _ma = getattr(_te_m, 'maker_uid', None)
                        else:
                            _ta = None
                            _ma = _agent
                        _ev = {
                            'y':  't',
                            'nid': int(_nid),
                            'i':  _tid,
                            'p':  float(_ef.get('price', 0.0) or 0.0),
                            'q':  float(_ef.get('alpha_volume', 0.0) or 0.0),
                            's':  int(_ef.get('direction', 0) or 0),
                            'Ta': _ta,
                            'Ma': _ma,
                            't':  _fill_ts_ns,
                        }
                        _bk = state.books.get(_nid)
                        if _bk is None:
                            state.books[_nid] = {'b': [], 'a': [], 'e': [_ev]}
                        else:
                            _bk = dict(_bk)
                            _bk['e'] = list(_bk.get('e') or []) + [_ev]
                            state.books[_nid] = _bk
                        # Scoring parity: the reward path
                        # (trade.update_trade_volumes -> _process_uid_notices)
                        # accumulates per-UID volume/PnL/fees from state.notices, NOT
                        # from book['e']. In the async-settlement flow execute() returns
                        # no trade_events, so notices were empty and settled fills never
                        # reached scoring (snapshot score/volume = 0 despite rows landing
                        # in `trades`). Surface the SAME reconciled fill as a settled
                        # trade notice ('ET') keyed by the settling UID so snapshots
                        # reflect on-chain-settled activity. Pool fills: the settling
                        # miner is the aggressor (taker); a matched LOB trade_event
                        # reuses its maker/taker sides.
                        if _te_m is not None:
                            _n_ta = getattr(_te_m, 'taker_uid', None)
                            _n_ma = getattr(_te_m, 'maker_uid', None)
                        else:
                            _n_ta = _agent
                            _n_ma = None
                        # Built by the canonical serializer, which is the only producer of an ET notice.
                        # Consumers reach these keys by subscript, so a key absent from a notice is a
                        # KeyError that costs a uid its whole volume, PnL and roundtrip for the block.
                        _notice = NormalizedTradeEvent(
                            book_id=int(_nid),
                            quantity=float(_ef.get('alpha_volume', 0.0) or 0.0),
                            price=float(_ef.get('price', 0.0) or 0.0),
                            side=int(_ef.get('direction', 0) or 0),
                            maker_uid=_n_ma,
                            taker_uid=_n_ta,
                            maker_fee=float(_ef.get('maker_fee', 0.0) or 0.0),
                            taker_fee=float(_ef.get('taker_fee', 0.0) or 0.0),
                            trade_id=_tid,
                            taker_order_id=int(_ef.get('taker_order_id', 0) or 0),
                            maker_order_id=int(_ef.get('maker_order_id', 0) or 0),
                        ).to_notice_dict()
                        # One ET notice per fill per uid. Two sources describe the same
                        # fill: this reconciliation-derived one, and the trade-event one
                        # surfaced to miners before forward(). state.notices feeds BOTH
                        # the miner synapse and scoring (trade.update_trade_volumes reads
                        # it), so a second copy would double-count that fill's volume.
                        # They now share the engine's trade id, so it identifies the
                        # duplicate.
                        for _nuid in {_n_ta, _n_ma}:
                            if _nuid is None:
                                continue
                            _existing = state.notices.setdefault(_nuid, [])
                            if _tid is not None and any(
                                isinstance(_e, dict)
                                and (_e.get('y') or _e.get('type')) in ('ET', 'EVENT_TRADE')
                                and _e.get('i') == _tid
                                for _e in _existing
                            ):
                                continue
                            _existing.append(_notice)
                except Exception:
                    pass
                # Push post-execution state to MVTRX data service on every block
                # Build open-order detail from LOB account state (state.accounts[uid][nid]['o'])
                # which is populated by the C++ engine's packAccounts() on every block.
                _agent_open_orders: dict = {}
                _agent_orders_detail: dict = {}
                _ext_detail = getattr(getattr(self, 'engine', None), '_external_order_detail', {}) or {}
                for _uid_str, _uid_data in (getattr(state, 'accounts', {}) or {}).items():
                    try:
                        _uid = int(_uid_str)
                        if not isinstance(_uid_data, dict):
                            continue
                        # Seed every uid with an empty list. Without it a uid appears in the payload
                        # only when it HAS orders, so consumers cannot distinguish "no orders" from
                        # "no news" and a just-cancelled order lingers as open in stale downstream state.
                        _agent_orders_detail.setdefault(_uid, [])
                        for _nid_str, _acct in _uid_data.items():
                            if not isinstance(_acct, dict):
                                continue
                            for _o in (_acct.get('o') or []):
                                if not isinstance(_o, dict):
                                    continue
                                _o_side = int(_o.get('s', _o.get('side', 0)))
                                _o_placed_at = int(_o.get('t') or 0) // 1_000_000
                                if _o_placed_at == 0:
                                    _ext_info = _ext_detail.get((_uid, int(_nid_str), _o_side))
                                    if _ext_info:
                                        _o_placed_at = _ext_info.get("placed_at", 0)
                                _agent_open_orders[_uid] = _agent_open_orders.get(_uid, 0) + 1
                                _agent_orders_detail.setdefault(_uid, []).append({
                                    "order_id":  _o.get('i') or _o.get('id'),
                                    "netuid":    int(_nid_str),
                                    "side":      _o_side,
                                    "price":     float(_o.get('p', _o.get('price', 0))),
                                    "quantity":  float(_o.get('q', _o.get('quantity', 0))),
                                    "placed_at": _o_placed_at,
                                })
                    except Exception:
                        continue
                # Augment from engine's external-order registry (UI-submitted orders)
                # since the C++ LOB doesn't include open orders in account IPC responses.
                try:
                    for (_eod_uid, _eod_nid, _eod_dir), _eod_info in _ext_detail.items():
                        if _eod_uid not in _agent_orders_detail or not any(
                            o.get('netuid') == _eod_nid and o.get('side') == _eod_dir
                            for o in _agent_orders_detail.get(_eod_uid, [])
                        ):
                            _agent_open_orders[_eod_uid] = _agent_open_orders.get(_eod_uid, 0) + 1
                            _agent_orders_detail.setdefault(_eod_uid, []).append({
                                "order_id":  None,
                                "netuid":    _eod_nid,
                                "side":      _eod_dir,
                                "price":     _eod_info.get("price", 0.0),
                                "quantity":  _eod_info.get("quantity", 0.0),
                                "placed_at": _eod_info.get("placed_at", 0),
                            })
                except Exception:
                    pass
                _meta = {}
                try:
                    _meta = {
                        "hotkeys":         [str(hk) for hk in self.metagraph.hotkeys],
                        "coldkeys":        [str(ck) for ck in self.metagraph.coldkeys] if hasattr(self.metagraph, 'coldkeys') else [],
                        "stake":           self.metagraph.stake.tolist(),
                        "emission":        self.metagraph.emission.tolist(),
                        "incentive":       self.metagraph.incentive.tolist(),
                        "validator_trust": self.metagraph.validator_trust.tolist(),
                        "validator_permit": self.metagraph.validator_permit.tolist() if hasattr(self.metagraph, 'validator_permit') else [],
                        "dividends":       self.metagraph.dividends.tolist() if hasattr(self.metagraph, 'dividends') else [],
                        "last_update":     self.metagraph.last_update.tolist() if hasattr(self.metagraph, 'last_update') else [],
                    }
                except Exception:
                    pass
                _ingest_url = getattr(getattr(self.config, 'exchange', None), 'data_service_url', '') or ''
                _ingest_block = getattr(state, 'block', self.current_block)
                _ingest_books = len(getattr(state, 'books', {}) or {})
                bt.logging.info(f"Scheduling ingest push: block={_ingest_block} books={_ingest_books} url={_ingest_url}")
                # Exchange constraints for the data service (/exchangeInfo filters +
                # order-API LOT_SIZE/PRICE_FILTER enforcement). The C++ engine owns
                # these (exchange_0.xml); the env overrides exist for hosts whose
                # engine XML differs from the defaults. Cached after first build.
                _exch_constraints = getattr(self, '_exchange_constraints_cache', None)
                if _exch_constraints is None:
                    _exch_constraints = {
                        "min_order_size":  float(os.environ.get("EXCHANGE_MIN_ORDER_SIZE", "0.0001") or 0.0001),
                        "price_decimals":  int(os.environ.get("EXCHANGE_PRICE_DECIMALS", "4") or 4),
                        "volume_decimals": int(os.environ.get("EXCHANGE_VOLUME_DECIMALS", "4") or 4),
                        "max_open_orders": int(os.environ.get("EXCHANGE_MAX_OPEN_ORDERS", "100") or 100),
                    }
                    self._exchange_constraints_cache = _exch_constraints
                _sltp_changed = getattr(getattr(self, 'engine', None), '_sltp_changed', False)
                _live_triggers = dict(getattr(getattr(self, 'engine', None), '_live_triggers', {}))
                if hasattr(self.engine, '_sltp_changed'):
                    self.engine._sltp_changed = False
                # Merge ET fill notices from the previous block's settlement into the
                # current ingest push. The companion drains the same buffer the synapse merge used,
                # on-chain execution completes; by the time the next block arrives it is
                # ready.  Pattern mirrors _pending_reconciliation.
                _pending_et = (_exchange_notices.drain_pending_for_ingest(self.engine)
                               if _exchange_notices is not None else {})
                _state_notices = {str(k): list(v) for k, v in (getattr(state, 'notices', None) or {}).items()}
                for _et_uid, _et_evs in _pending_et.items():
                    _key = str(_et_uid)
                    if _key in _state_notices:
                        _state_notices[_key] = _state_notices[_key] + list(_et_evs)
                    else:
                        _state_notices[_key] = list(_et_evs)
                # Reconciliation for the DB trades write: consume the dedicated ingest
                # snapshot (_send_to_lob already cleared _pending_reconciliation before
                # this push). Same consume-at-push pattern.
                _pending_recon = dict(getattr(self.engine, '_pending_ingest_reconciliation', None) or {})
                if _pending_recon:
                    self.engine._pending_ingest_reconciliation = {}
                asyncio.create_task(_push_mvtrx({
                    "mode":                "exchange",
                    "network":             getattr(getattr(self.config, 'exchange', None), 'network', '') or '',
                    "timestamp":           int(time.time() * 1e9),
                    "block":               _ingest_block,
                    "books":               getattr(state, 'books', {}) or {},
                    "accounts":            getattr(state, 'accounts', {}) or {},
                    # Pools carry whether the subnet can settle at all. Without it the UI
                    # cannot mark a staking-disabled subnet or stop a miner trading one, and the
                    # only feedback is a failed settlement reporting SubtokenDisabled after the
                    # fact. A subnet the chain could not be asked about is left ABSENT rather
                    # than defaulted to enabled.
                    "pools":               _pools_with_subtoken(self, getattr(state, 'pools', None)),
                    "block_events":        list((getattr(self.engine, '_last_chain_state', None) or {}).get('block_events', [])),
                    "delegates":           dict((getattr(self.engine, '_last_chain_state', None) or {}).get('delegates', {})),
                    "chain_balances":      dict((getattr(self.engine, '_last_chain_state', None) or {}).get('balances', {})),
                    "offex_balances":      dict((getattr(self.engine, '_last_chain_state', None) or {}).get('offex_balances', {})),
                    "reconciliation":      _pending_recon,
                    "notices":             _state_notices,
                    "agent_open_orders":   _agent_open_orders,
                    "agent_orders_detail": _agent_orders_detail,
                    "metagraph":           _meta,
                    "validator_uid":       self.uid,
                    "benchmark_agents":    [{"uid": _ba["uid"], "coldkey": _ba.get("coldkey", ""), "hotkey": _ba.get("hotkey", ""), "name": _ba.get("name", "")} for _ba in getattr(self, 'benchmark_agents', [])],
                "proxy_wallets":       getattr(self, 'proxy_wallets_published', {}) or {},
                # Separate key from proxy_wallets on purpose: the swap proxy is address-keyed and
                # permanent, the exchange proxy is uid-keyed and evicted on deregistration. One
                # key for both would let the service serve a swap wallet as an exchange proxy.
                "swap_wallets":        getattr(self, 'swap_wallets_published', {}) or {},
                    "sltp_triggers":       _live_triggers,
                    "exchange_constraints": _exch_constraints,
                    # Per-agent scoring maps (score, kappa, volume, pnl, per-book
                    # vol/fee/pnl) so the exchange agent_snapshots row is populated
                    # like the sim one — from RECONCILED accumulators (see helper).
                    # Without these the exchange snapshot stores score/vol/pnl=0 and
                    # per_book carries only balances → portfolio charts read 0.
                    **self._build_agent_scoring_maps(),
                }, url=_ingest_url))
                # Yield once so the HTTP POST starts sending while the remaining
                # synchronous work (maintain/reward/save scheduling) runs.
                await asyncio.sleep(0)
                response = {"responses": []}
            else:
                start = time.time()
                response = SimulatorResponseBatch(miner_responses)
                response = response.serialize()
                bt.logging.debug(f"Serialized Response Batch ({time.time()-start}s)")
                # Inject UI-submitted external orders
                _ext = getattr(self.engine, '_external_instructions', [])
                if _ext:
                    response['responses'].extend(_ext)
                    self.engine._external_instructions = []
                    bt.logging.info(f"Injected {len(_ext)} external order(s) into simulation response")
                # Push simulation state to MVTRX data service each tick. Payload
                # construction is O(N * T * B) over realized_pnl_history and cost
                # ~7-11s per state on the main event loop; offloaded to
                # reporting_ipc_executor via _push_sim_state_bg. Skip when the URL
                # isn't configured so we don't spawn no-op tasks every tick.
                _push_url = (
                    getattr(getattr(self.config, 'simulation', None), 'data_service_url', '')
                    or getattr(getattr(self.config, 'exchange', None), 'data_service_url', '')
                )
                if _push_url:
                    asyncio.create_task(self._push_sim_state_bg(state, _push_url))

            # GenTRX: poll for round advance and deliver assignments after the
            # mining query completes. Round-check stays triggered per state
            # update (no wall-clock polling) but runs as a BACKGROUND task so
            # the grad-server HTTP (data-status / push_round / health check)
            # and the miner delivery never block the round. Exclusion with the
            # state-update query is enforced by miner_net_lock (held by
            # forward() for its query window, and by _deliver_gentrx_assignments
            # around the one atomic deliver to ALL miners) — so queries and
            # deliveries still never overlap, on the IPC channel or the wire.
            if self._gentrx is not None:
                if self._gentrx_task is not None and not self._gentrx_task.done():
                    # Single-flight: one deliver (~4-6s) per ~5-min round means a
                    # previous task still running at the NEXT state update is a
                    # sign something is genuinely stuck — surface it loudly.
                    gtx_log.warning(
                        "previous poll_and_deliver still running at next state "
                        "update — skipping this cycle (investigate if recurring)")
                else:
                    async def _run_gtx_background():
                        try:
                            # Generous hard bound purely as a stuck-task backstop;
                            # nothing in the round awaits this.
                            await asyncio.wait_for(
                                self._gentrx.poll_and_deliver(), timeout=120.0)
                        except asyncio.TimeoutError:
                            gtx_log.warning(
                                "background poll_and_deliver exceeded 120s — abandoned")
                        except Exception as _gex:
                            bt.logging.warning(
                                f"[GTX] background poll_and_deliver error: {_gex}")
                    self._gentrx_task = asyncio.create_task(_run_gtx_background())
            # Log response data, start state serialization and reporting threads, and return miner instructions to the simulator
            if len(response['responses']) > 0:
                bt.logging.trace(f"RESPONSE : {response}")
            bt.logging.info(f"RATE : {(self.step_rates[-1] if self.step_rates != [] else 0) / 1e9:.2f} STEPS/s | AVG : {(sum(self.step_rates) / len(self.step_rates) / 1e9 if self.step_rates != [] else 0):.2f}  STEPS/s")
            self.step_rates = self.step_rates[-10000:]
            self.last_state_time = time.time()

            # Calculate latest rewards, update miner scores, save state and publish metrics
            _rp_sched0 = time.time()
            self.maintain()
            self.reward(state)
            self.save_state()
            self.report()
            _rp_end = time.time()
            bt.logging.info(
                f"[ROUND-PROFILE] recv_norm={_rp_entry - receive_start:.3f}s "
                f"pre={_rp_fwd0 - _rp_entry:.3f}s forward={_rp_fwd1 - _rp_fwd0:.3f}s "
                f"post={_rp_sched0 - _rp_fwd1:.3f}s sched={_rp_end - _rp_sched0:.3f}s "
                f"total={_rp_end - receive_start:.3f}s"
            )
            bt.logging.info(f"State update handled ({time.time()-receive_start}s)")

            return response

        async def _listen(self):
            """
            Thin event loop: delegates IPC receive/respond to the engine.
            """
            # Startup push — fetch full exchange state (chain + LOB) and push to
            # the data service immediately, before waiting for the first block.
            if self.engine.mode == 'exchange':
                try:
                    bt.logging.info("MVTRX startup push: fetching chain + LOB state...")
                    _t0 = time.time()
                    _cs = await self.engine._fetch_chain_state()
                    if _cs:
                        # Mirror receive()'s auto-discovery so _normalize() has book_ids
                        if not self.engine._book_ids and _cs.get('pools'):
                            self.engine._book_ids = sorted(_cs['pools'].keys())
                            self.engine._update_exchange_config()
                        # Query LOB for current books/accounts/open orders (empty reconciliation
                        # payload triggers a state-only round-trip without on-chain execution).
                        loop = asyncio.get_event_loop()
                        _lob = await loop.run_in_executor(
                            None,
                            lambda: self.engine._send_to_lob(_cs, reconciliation={})
                        )
                        if _lob is not None:
                            _exc_state = _lob.get('state', _lob)
                            if _exc_state:
                                self.engine._last_exchange_state = {
                                    **_exc_state,
                                    'books':    _lob.get('books', {}),
                                    'accounts': _lob.get('accounts', {}),
                                }
                        _ss = self.engine._normalize(_cs)
                        _sm: dict = {}
                        try:
                            _sm = {
                                "hotkeys":         [str(hk) for hk in self.metagraph.hotkeys],
                                "coldkeys":        [str(ck) for ck in self.metagraph.coldkeys] if hasattr(self.metagraph, 'coldkeys') else [],
                                "stake":           self.metagraph.stake.tolist(),
                                "emission":        self.metagraph.emission.tolist(),
                                "incentive":       self.metagraph.incentive.tolist(),
                                "validator_trust": self.metagraph.validator_trust.tolist(),
                                "validator_permit": self.metagraph.validator_permit.tolist() if hasattr(self.metagraph, 'validator_permit') else [],
                            }
                        except Exception:
                            pass
                        # Build open-order detail from the LOB accounts in the startup state
                        _soo: dict = {}
                        _sod: dict = {}
                        for _uid_str, _uid_data in (getattr(_ss, 'accounts', {}) or {}).items():
                            try:
                                _uid = int(_uid_str)
                                if not isinstance(_uid_data, dict):
                                    continue
                                for _nid_str, _acct in _uid_data.items():
                                    if not isinstance(_acct, dict):
                                        continue
                                    for _o in (_acct.get('o') or []):
                                        if not isinstance(_o, dict):
                                            continue
                                        _soo[_uid] = _soo.get(_uid, 0) + 1
                                        _sod.setdefault(_uid, []).append({
                                            "order_id":  _o.get('i') or _o.get('id'),
                                            "netuid":    int(_nid_str),
                                            "side":      int(_o.get('s', _o.get('side', 0))),
                                            "price":     float(_o.get('p', _o.get('price', 0))),
                                            "quantity":  float(_o.get('q', _o.get('quantity', 0))),
                                            "placed_at": int(_o.get('t') or 0) // 1_000_000,
                                        })
                            except Exception:
                                continue
                        bt.logging.info(
                            f"MVTRX startup push: block={_cs.get('block')} "
                            f"books={len(getattr(_ss,'books',{}))} "
                            f"accounts={len(getattr(_ss,'accounts',{}))} "
                            f"pools={len(getattr(_ss,'pools',{}) or {})} "
                            f"open_orders={sum(_soo.values())} "
                            f"({time.time()-_t0:.2f}s)"
                        )
                        asyncio.create_task(_push_mvtrx({
                            "mode":                "exchange",
                            "network":             getattr(getattr(self.config, 'exchange', None), 'network', '') or '',
                            "timestamp":           int(time.time() * 1e9),
                            "block":               _cs.get('block', self.current_block),
                            "books":               getattr(_ss, 'books', {}) or {},
                            "accounts":            getattr(_ss, 'accounts', {}) or {},
                            "pools":               getattr(_ss, 'pools', None),
                            "reconciliation":      {},
                            "agent_open_orders":   _soo,
                            "agent_orders_detail": _sod,
                            "metagraph":           _sm,
                            "validator_uid":       self.uid,
                            "benchmark_agents":    [{"uid": _ba["uid"], "coldkey": _ba.get("coldkey", ""), "hotkey": _ba.get("hotkey", ""), "name": _ba.get("name", "")} for _ba in getattr(self, 'benchmark_agents', [])],
                "proxy_wallets":       getattr(self, 'proxy_wallets_published', {}) or {},
                # Separate key from proxy_wallets on purpose: the swap proxy is address-keyed and
                # permanent, the exchange proxy is uid-keyed and evicted on deregistration. One
                # key for both would let the service serve a swap wallet as an exchange proxy.
                "swap_wallets":        getattr(self, 'swap_wallets_published', {}) or {},
                        }, url=getattr(getattr(self.config, 'exchange', None), 'data_service_url', '')))
                except Exception as _exc:
                    bt.logging.warning(f"MVTRX startup push failed: {_exc}")
            bt.logging.info("Listener loop starting — awaiting engine.receive()...")

            # Liveness probe: if these stop, the event loop is blocked/starved
            # (sync native call on the loop, GIL monopolised, etc.) — if they
            # continue while receive() makes no progress, a specific await hangs.
            async def _listen_heartbeat():
                n = 0
                while True:
                    await asyncio.sleep(30)
                    n += 1
                    bt.logging.debug(f"[hb] listener event loop alive #{n}")
                    if n % 2 == 0:
                        # Dump every task's full await chain (get_stack only reports
                        # one frame for suspended coroutines) down to the awaited
                        # object — pinpoints exactly which await is parked.
                        for t in asyncio.all_tasks():
                            if t is asyncio.current_task():
                                continue
                            try:
                                coro = t.get_coro()
                                chain = []
                                while coro is not None and len(chain) < 12:
                                    fr = getattr(coro, 'cr_frame', None) or getattr(coro, 'gi_frame', None)
                                    if fr is not None:
                                        chain.append(
                                            f"{fr.f_code.co_name}@{fr.f_code.co_filename.rsplit('/', 1)[-1]}:{fr.f_lineno}"
                                        )
                                    nxt = getattr(coro, 'cr_await', None)
                                    if nxt is None:
                                        nxt = getattr(coro, 'gi_yieldfrom', None)
                                    if nxt is None or nxt is coro:
                                        break
                                    if not hasattr(nxt, 'cr_frame') and not hasattr(nxt, 'gi_frame'):
                                        chain.append(f"awaiting {repr(nxt)[:160]}")
                                        break
                                    coro = nxt
                                desc = ' <- '.join(chain)
                            except Exception as _e:
                                desc = f"<chain error: {_e}>"
                            bt.logging.debug(f"[hb] task {t.get_name()}: {desc or '<no frames>'}")
            _hb_task = asyncio.ensure_future(_listen_heartbeat())  # noqa: F841 — keep ref

            try:
                while True:
                    # Shutdown guard: cleanup() sets _cleanup_done and tears down
                    # the executors. Without this break the loop would keep
                    # calling engine.receive() -> run_in_executor on dead
                    # executors, catch "cannot schedule new futures after
                    # shutdown", and spin a tight PD-alert loop (the shutdown
                    # crash-loop). Exit cleanly instead.
                    if getattr(self, '_cleanup_done', False):
                        bt.logging.info("Shutdown in progress — exiting listener loop.")
                        break
                    response = {"responses": []}
                    raw_message = None
                    try:
                        raw_message, normalized_state, receive_start = await self.engine.receive()
                        if normalized_state is not None:
                            if self._scoring_shadow is not None and isinstance(raw_message, (bytes, bytearray)):
                                self._scoring_shadow.tee(raw_message, normalized_state.timestamp)
                                if self._scoring_proc_cutover and \
                                        normalized_state.timestamp % self.config.scoring.interval == 0:
                                    # Eager scoring: ship the boundary's inputs NOW so
                                    # the child computes during the lock-queue delay
                                    # instead of after _reward finally asks. sim_ts is
                                    # the boundary's own timestamp — deterministic and
                                    # main-defined (verify pins to the same values).
                                    self._scoring_shadow.tee_score_inputs(
                                        normalized_state.timestamp,
                                        normalized_state.timestamp,
                                        list(self.deregistered_uids),
                                        (self._gentrx.get_scores()
                                         if hasattr(self, '_gentrx') and self._gentrx is not None else {}),
                                        getattr(self, '_gentrx_ema', {}),
                                        trading_ema=dict(getattr(self, '_trading_score_ema', {}) or {}),
                                        trading_ema_n=dict(getattr(self, '_trading_score_ema_n', {}) or {}),
                                        trading_ema_ts=getattr(self, '_trading_score_ema_ts', None),
                                    )
                            response = await self.handle_state(normalized_state, receive_start)
                    except Exception as ex:
                        # Terminal teardown error: once the executors are gone,
                        # run_in_executor raises "cannot schedule new futures
                        # after shutdown". Retrying can never recover — break.
                        if getattr(self, '_cleanup_done', False) or \
                                "cannot schedule new futures after shutdown" in str(ex):
                            bt.logging.info("Executors shut down — exiting listener loop.")
                            break
                        traceback.print_exc()
                        self.pagerduty_alert(
                            f"Exception in listener loop: {ex}",
                            details={"trace": traceback.format_exc()}
                        )
                    finally:
                        # Don't attempt to respond during teardown — respond()
                        # also hits run_in_executor/IPC on torn-down resources.
                        if not getattr(self, '_cleanup_done', False):
                            self.engine.respond(raw_message, response)
            finally:
                pass

        def listen(self):
            """
            Synchronous wrapper for the asynchronous `_listen` method.
            """
            try:
                os.nice(-10)
            except PermissionError:
                bt.logging.warning("Cannot set process priority (need sudo for negative nice values)")
            try:
                asyncio.run(self._listen())
            except KeyboardInterrupt:
                print("Listening stopped by user.")

        async def orderbook(self, request : Request) -> dict:
            """
            HTTP route endpoint that receives a complete simulator state update over HTTP,
            parses it, and forwards it to `handle_state`.


            This is the HTTP equivalent of the IPC listener used when running a
            distributed or containerized simulator. It performs:
            - Streaming request‑body read.
            - Basic JSON‑structure validation.
            - Construction of a Ypy‑backed state object.
            - Conversion into a typed MarketSimulationStateUpdate model.
            - Delegation to the main validator processing pipeline.


            Args:
            request (Request): Incoming HTTP request containing a JSON‑encoded simulation state update.


            Returns:
            dict: Serialized simulation response batch.
            """
            bt.logging.info("Received state update from simulator.")
            global_start = time.time()
            start = time.time()
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
            bt.logging.info(f"Retrieved request body ({time.time()-start:.4f}s).")
            if not body:
                return {}
            if body[-3:].decode() != "]}}":
                raise Exception("Incomplete JSON!")
            message = YpyObject(body, 1)
            bt.logging.info(f"Constructed YpyObject ({time.time()-start:.4f}s).")
            state = MarketSimulationStateUpdate.from_ypy(message)
            bt.logging.info(f"Synapse populated ({time.time()-start:.4f}s).")
            del body

            normalized = self.engine._normalize(state)
            normalized.logDir = message['logDir']
            response = await self.handle_state(normalized, global_start)

            bt.logging.info(f"State update processed ({time.time()-global_start}s)")
            return response

        async def account(self, request : Request) -> None:
            """
            HTTP route endpoint for receiving event‑level notifications from the simulator
            (e.g., simulation start, simulation end, error reports, market notices).


            Responsibilities:
            - Immediately forward simulation‑start events to miners.
            - Handle simulation‑end markers.
            - Record and persist error‑report batches.
            - Forward all other event notifications to miners.
            - Trigger alerting for msgpack or simulation integrity errors.

            Args:
            request (Request): HTTP request containing a batch of simulator event messages.

            Returns:
            None | dict: `{"continue": True/False}` when error‑report limits are reached.
            Otherwise returns `None`.
            """
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
            batch = msgspec.json.decode(body)
            bt.logging.info(f"NOTICE : {batch}")
            notices = []
            ended = False
            for message in batch['messages']:
                if message['type'] == 'EVENT_SIMULATION_START':
                    self.engine.on_start(message['timestamp'], FinanceEventNotification.from_json(message).event)
                    continue
                elif message['type'] == 'EVENT_SIMULATION_END':
                    ended = True
                elif message['type'] == 'RESPONSES_ERROR_REPORT':
                    dump_file = self.config.neuron.full_path + f"/{self.last_state.config.simulation_id}.{message['timestamp']}.responses.json"
                    with open(dump_file, "w") as f:
                        json.dump(self.last_response, f, indent=4)
                    error_file = self.config.neuron.full_path + f"/{self.last_state.config.simulation_id}.{message['timestamp']}.error.json"
                    with open(error_file, "w") as f:
                        json.dump(message, f, indent=4)
                    self.msgpack_error_counter += len(message) - 3
                    if self.msgpack_error_counter < 10:
                        self.pagerduty_alert(f"{self.msgpack_error_counter} msgpack deserialization errors encountered in simulator - continuing.", details=message)
                        return { "continue": True }
                    else:
                        self.pagerduty_alert(f"{self.msgpack_error_counter} msgpack deserialization errors encountered in simulator - terminating simulation.", details=message)
                        return { "continue": False }
                notice = FinanceEventNotification.from_json(message)
                if not notice:
                    bt.logging.error(f"Unrecognized notification : {message}")
                else:
                    notices.append(notice)
            await notify(self, notices)
            if ended:
                self.engine.on_end()

        async def sltp_records(self, request: Request):
            """Delegate to the active engine; return [] when no extension installed."""
            if hasattr(self, 'engine') and hasattr(self.engine, 'sltp_records'):
                try:
                    return await self.engine.sltp_records(request)
                except Exception:
                    pass
            return []

        async def sltp_levels(self):
            """Return the engine's trigger snapshot if available, else {}."""
            if hasattr(self, 'engine') and hasattr(self.engine, '_live_triggers'):
                return self.engine._live_triggers
            return {}

        def cleanup_ipc(self):
            """
            Shuts down the query service and releases all POSIX IPC resources.
            """
            cleanup_ipc(self)

        def cleanup_executors(self):
            """
            Shuts down thread and process executors used by the validator.
            """
            cleanup_executors(self)

        def cleanup_event_loop(self):
            """
            Gracefully shuts down the main event loop and any pending tasks.
            """
            cleanup_event_loop(self)

        def cleanup(self):
            """
            Performs full resource cleanup for the validator during shutdown.
            """
            cleanup(self)

    from taos.im.validator.trade import match_trade_fifo, update_trade_volumes
    from taos.im.validator.cleanup import (
        cleanup_ipc, cleanup_executors, cleanup_event_loop, cleanup
    )
    from taos.im.validator.persistence import (
        load_state, build_validator_state,
        snapshot_inventory_history, snapshot_realized_pnl_history,
        snapshot_2_level_dict, snapshot_volume_sums, snapshot_trade_volumes,
        snapshot_roundtrip_volumes, snapshot_open_positions,
        construct_save_data_sync, defragment_histories, schedule_save,
        save_state_sync, migrate_sampling_interval,
    )

if __name__ == "__main__":
    # Make an unlocatable warning locatable.
    #
    # A Pydantic serializer warning fires ~299 times per day complaining that field `i` (the trade id,
    # declared int) received a string of the form x:0x<extrinsic_hash>:<uid>:<index>:<p|f>. That is a
    # real identity defect: no such id is ever recorded on the tape, so a consumer handed one cannot
    # join it to anything. But the warning carries no stack, static search across taos/ and
    # mvtrx/service did not find the assembly site, and the surrounding frames are stdlib
    # (the log-queue monitor) or a subprocess, so the warning may not even originate in this process.
    #
    # Rather than guess a fix, capture the stack the next time it fires. `warnings` gives the full
    # traceback when asked, so one run reproduces it (the shape includes :171: as well as :248:, so
    # the acceptance miner exercises it) and the fix lands with evidence instead of a hypothesis.
    import traceback
    import warnings as _warnings

    _seen_ser_warn = [False]
    _orig_showwarning = _warnings.showwarning

    def _showwarning_with_stack(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if "Pydantic serializer warnings" in text or "UnexpectedValue" in text:
            if not _seen_ser_warn[0]:
                _seen_ser_warn[0] = True
                try:
                    bt.logging.warning(
                        "PYDANTIC-SER-WARN (first occurrence, stack follows): "
                        + text[:300].replace("\n", " ")
                    )
                    for _ln in "".join(traceback.format_stack()).splitlines()[-24:]:
                        bt.logging.warning("PYDANTIC-SER-WARN | " + _ln.strip()[:200])
                except Exception:
                    pass
        return _orig_showwarning(message, category, filename, lineno, file, line)

    _warnings.showwarning = _showwarning_with_stack

    from taos.im.validator.update import check_repo, update_validator, check_simulator, rebuild_simulator, restart_simulator
    from taos.im.validator.forward import forward, notify, deliver_gentrx
    from taos.im.validator.reward import get_rewards, compute_debeta_scores

    if float(platform.freedesktop_os_release()['VERSION_ID']) < 22.04:
        raise Exception("taos validator requires Ubuntu >= 22.04!")

    def _pools_with_subtoken(validator, pools):
        """Attach subtoken_enabled to each pool, cheaply and without ever failing the payload.

        Cached for ten minutes in taos.im.validator.subtoken: a subnet owner toggling staking is
        a rare administrative act, and querying every subnet each block would cost far more than
        the flag is worth. Any failure returns the pools untouched, because losing the whole
        ingest payload over an advisory flag would be a poor trade.
        """
        try:
            from taos.im.validator.subtoken import merge_subtoken_flags, subtoken_flags
            _sub = getattr(getattr(validator, 'subtensor', None), 'substrate', None)
            # cached_only: this runs inside async handle_state, where a synchronous websocket
            # round-trip per subnet (~128 of them) stalls the event loop and everything queued
            # behind it. Serve the cached answer and let the refresh happen on its own thread.
            _flags = subtoken_flags(_sub, [int(k) for k in (pools or {})], cached_only=True)
            return merge_subtoken_flags(pools, _flags)
        except Exception:
            bt.logging.debug("subtoken flag merge skipped", exc_info=True)
            return pools or {}

    class _BittensorLogHandler(logging.Handler):
        """Forward stdlib logging records into bt.logging.

        The validator's engine, chain and reconcile modules log through
        logging.getLogger(__name__): 242 call sites across 8 modules. bittensor installs
        its own logging stack and those records reach none of it, so every one of those
        diagnostics was discarded. Measured on this box: build_reconciliation logs
        unconditionally at INFO on each reconciliation cycle, yet across three hours and
        ten settled trades it appeared zero times in the pm2 logs.

        That silence costs twice: it hides the state of a running validator, and it makes
        paths like the partial-fill remainder restore unobservable exactly when they need
        checking. Bridging once here fixes all 242 sites without touching a call site, and
        keeps the %s-style formatting they already use.
        """

        # bt.logging emits THROUGH a stdlib logger named "bittensor". Without this guard the
        # bridge is a feedback loop: bt.logging -> stdlib "bittensor" -> this handler ->
        # bt.logging -> ... Each pass prepended another "[bittensor]", and a single
        # "Sent query request" line reached 1776 bytes with ~40 nested prefixes before the
        # filter existed.
        _SKIP_PREFIXES = ("bittensor", "btlogging", "substrateinterface", "websockets")
        _local = threading.local()

        def emit(self, record) -> None:
            name = record.name or ""
            if name.split(".")[0] in self._SKIP_PREFIXES:
                return
            # Reentrancy: anything logged while we are inside bt.logging would come straight
            # back here. One flag per thread, because bt.logging is called from several.
            if getattr(self._local, "busy", False):
                return
            try:
                msg = f"[{record.name}] {record.getMessage()}"
                if record.exc_info:
                    msg += "\n" + logging.Formatter().formatException(record.exc_info)
            except Exception:
                return
            try:
                self._local.busy = True
                if record.levelno >= logging.ERROR:
                    bt.logging.error(msg)
                elif record.levelno >= logging.WARNING:
                    bt.logging.warning(msg)
                elif record.levelno >= logging.INFO:
                    bt.logging.info(msg)
                else:
                    bt.logging.debug(msg)
            except Exception:
                pass
            finally:
                self._local.busy = False

    def _bridge_stdlib_logging_to_bittensor() -> None:
        """Install the bridge on the root logger, once.

        Defaults to INFO rather than DEBUG: several bridged modules log per order at debug
        level, and this box has filled its disk with engine logs before. Raise it with
        TAOS_STDLIB_LOG_LEVEL=DEBUG when investigating.
        """
        root = logging.getLogger()
        if any(isinstance(h, _BittensorLogHandler) for h in root.handlers):
            return
        name = os.environ.get("TAOS_STDLIB_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, name, logging.INFO)
        handler = _BittensorLogHandler()
        handler.setLevel(level)
        root.addHandler(handler)
        # The root logger defaults to WARNING, which would drop INFO records before any
        # handler sees them, so the handler level alone is not enough.
        if root.level == logging.NOTSET or root.level > level:
            root.setLevel(level)
        bt.logging.info(
            f"stdlib logging bridged into bt.logging at {name} "
            f"({len(root.handlers)} root handler(s)); "
            f"set TAOS_STDLIB_LOG_LEVEL to change")

    # Apply logging config before any bt.logging calls — bt.logging starts in
    # Default/WARNING state at import time and never auto-applies CLI flags.
    # Use direct set_* calls (same pattern as BaseNeuron.__init__ and subprocesses)
    # because bt.logging(config=...) can silently fall through to enable_default()
    # if _extract_logging_config returns the wrong object.
    _cfg_lc = getattr(Validator.config(), 'logging', None)
    if getattr(_cfg_lc, 'trace', False):
        bt.logging.set_trace()
    elif getattr(_cfg_lc, 'debug', False):
        bt.logging.set_debug()
    else:
        bt.logging.set_info()
    _bridge_stdlib_logging_to_bittensor()
    bt.logging.info("Initializing validator...")
    app = FastAPI()
    validator = Validator()
    try:
        app.include_router(validator.router)

        bt.logging.info("Starting background threads...")
        threads = []
        engine_mode = getattr(validator.config, 'engine', 'simulation')

        # For exchange mode the block-monitoring loop drives everything;
        # for simulation mode uvicorn drives everything (simulator pushes state).
        # Build the background thread list accordingly.
        background_targets = [('Monitor', validator.monitor)]
        try:
            from taos.im.validator.exetrx import setup_attrs as _setup_sltp_attrs
            _setup_sltp_attrs(validator, engine_mode)
        except ImportError:
            validator.sltp_enabled = False
        # Observe-mode validators don't run agents and never launch the SL/TP
        # service, so don't monitor (and pointlessly try to PM2-restart) a
        # process that doesn't exist — it just spams ERROR logs.
        if bool(getattr(getattr(validator.config, 'neuron', None), 'observe', False)):
            validator.sltp_enabled = False
        if getattr(validator, 'sltp_enabled', False):
            background_targets.append(('SLTPMonitor', validator.monitor_sltp))
        if engine_mode == 'exchange':
            # Absorb POST /MarketSimulationStateUpdate from simulation validators that
            # query this axon. Return 404 (not 200) so the dendrite skips synapse
            # body-parsing and won't raise a ValidationError for missing fields.
            from fastapi.responses import JSONResponse as _JSONResponse
            @app.post("/MarketSimulationStateUpdate")
            async def _sim_stub():
                return _JSONResponse(status_code=404, content={"message": "exchange mode"})

            import logging as _logging
            class _SuppressSimStateLog(_logging.Filter):
                def filter(self, record):
                    return "MarketSimulationStateUpdate" not in record.getMessage()
            _logging.getLogger("uvicorn.access").addFilter(_SuppressSimStateLog())

            # Listen runs in the main thread below; uvicorn is a background thread.
            background_targets.append(('API', lambda: uvicorn.run(app, host="0.0.0.0", port=validator.config.port)))
        else:
            # Simulation mode: Listen is a background thread; uvicorn runs below.
            background_targets.append(('Listen', validator.listen))

        for name, target in background_targets:
            try:
                bt.logging.info(f"Starting {name} thread...")
                thread = Thread(target=target, daemon=True, name=name)
                thread.start()
                threads.append(thread)
            except Exception as ex:
                validator.pagerduty_alert(f"Exception starting {name} thread: {ex}")
                raise

        time.sleep(1)
        for thread in threads:
            if not thread.is_alive():
                validator.pagerduty_alert(f"Failed to start {thread.name} thread!")
                raise RuntimeError(f"Thread '{thread.name}' failed to start")

        bt.logging.info("All threads running. Starting FastAPI server and main event loop...")

        def run_main_loop():
            """Run the pre-created main event loop."""
            async def keep_alive():
                bt.logging.info("Main event loop started for background tasks")
                bt.logging.debug(f"[MAINLOOP] Thread: {threading.current_thread().name}")
                bt.logging.debug(f"[MAINLOOP] Loop: {id(validator.main_loop)}")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    bt.logging.info("Main event loop stopping...")
            loop = validator.main_loop
            asyncio.set_event_loop(loop)
            validator._main_loop_ready.set()
            bt.logging.debug(f"[MAINLOOP] Running loop: {id(loop)}")
            try:
                loop.run_until_complete(keep_alive())
            finally:
                loop.close()

        main_loop_thread = Thread(target=run_main_loop, daemon=True, name='main')
        main_loop_thread.start()
        threads.append(main_loop_thread)
        time.sleep(0.5)

        if engine_mode == 'exchange':
            bt.logging.info("Exchange mode: running block monitoring loop in main thread...")
            validator.listen()   # blocks; errors are visible and process dies on crash
        else:
            bt.logging.info(f"Starting FastAPI server on port {validator.config.port}...")
            uvicorn.run(app, host="0.0.0.0", port=validator.config.port)
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt received")
    except Exception as ex:
        bt.logging.error(f"Fatal error: {ex}")
        bt.logging.debug(traceback.format_exc())
        sys.exit(1)