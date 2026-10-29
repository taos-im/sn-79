# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""
Validator forward pass: IPC-based query dispatch and miner notification helpers.
"""

import os
import time
import bittensor as bt
import asyncio
import posix_ipc
import struct
import traceback
import pickle
from collections import deque
from types import SimpleNamespace
from typing import List

from taos.im.neurons.validator import Validator
from taos.im.protocol import FinanceAgentResponse, FinanceEventNotification, MarketSimulationStateUpdate
from taos.im.protocol.instructions import *
from taos.im.validator.delays import latency_profile, set_delays
from taos.im.validator.block_stats import block_row, bump
from taos.im.utils import duration_from_timestamp

_PLACEMENT_ANSWERS = frozenset({"RDPOL", "RDPOM", "ERDPOL", "ERDPOM"})
_CANCEL_ANSWERS = frozenset({"RDCO", "ERDCO"})
_FILL_EVENTS = frozenset({"ET", "EVENT_TRADE"})
_FILL_SEEN_CAP = 200000   # (uid, book, trade id) keys kept to count a redelivered exchange fill once; the oldest half goes when full
_CANCEL_PENDING_CAP = 200000   # (uid, book, order id) of cancels sent and not yet answered; the oldest half goes when full
_PLACEMENT_TYPES = frozenset({"PLACE_ORDER_LIMIT", "PLACE_ORDER_MARKET"})


def _field(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _remember(keys: dict, key, cap: int) -> None:
    if len(keys) >= cap:
        for old in list(keys)[:len(keys) // 2]:
            del keys[old]
    keys[key] = True


def _count_cancel_answer(self: Validator, st: dict, uid, notice, seen) -> None:
    """One answer's orders by outcome. An order a cancel instruction named is cancelled or not found; a success for an
    order no instruction named is a self-trade-prevention cancel. Until the first round is sent (after a restart) the
    answers belong to cancels sent before it, so an unmatched success is counted as a cancel."""
    pending = self.__dict__.get('_cancel_pending')
    y = _field(notice, 'y')
    book = _field(notice, 'b')
    for c in _field(notice, 'c') or ():
        oid = _field(c, 'o')
        if seen is not None:
            key = ('cancel', y, uid, book, oid)
            if key in seen:
                continue
            _remember(seen, key, _FILL_SEEN_CAP)
        asked = pending is not None and pending.pop((uid, book, oid), None) is not None
        if y.startswith('E'):
            name = 'orders_not_found'
        elif asked or pending is None:
            name = 'orders_cancelled'
        else:
            name = 'stp_cancels'
        st[name] = st.get(name, 0) + 1


def update_placement_stats(self: Validator, notices: dict | None) -> None:
    """Counts, into each miner's request stats so they publish and reset with them per block: its limit and market
    placements the engine answered in a state, accepted and refused (a refusal for CONTRACT_VIOLATION, post-only or
    time in force, also apart), the orders its cancels named by outcome, and the fills it took part in. The exchange
    redelivers a settled fill on every update for a window, so there a fill is counted once by (uid, book, trade id),
    and a cancel answer once per order; the simulation delivers each once."""
    dedup = getattr(getattr(self, 'engine', None), 'mode', 'simulation') == 'exchange'
    seen = self.__dict__.setdefault('_fill_notice_seen', {}) if dedup else None
    for uid, uid_notices in (notices or {}).items():
        st = block_row(self.miner_stats, uid)
        for n in uid_notices or ():
            y = n.get('y') if isinstance(n, dict) else getattr(n, 'y', None)
            if y in _FILL_EVENTS:
                if seen is not None:
                    tid = n.get('i') if isinstance(n, dict) else getattr(n, 'i', None)
                    book = n.get('b') if isinstance(n, dict) else getattr(n, 'b', None)
                    if tid is not None:
                        key = (uid, book, tid)
                        if key in seen:
                            continue
                        _remember(seen, key, _FILL_SEEN_CAP)
                st['fills'] = st.get('fills', 0) + 1
                continue
            if y in _CANCEL_ANSWERS:
                _count_cancel_answer(self, st, uid, n, seen)
                continue
            if y not in _PLACEMENT_ANSWERS:
                continue
            if y.startswith('E'):
                st['refused'] = st.get('refused', 0) + 1
                m = n.get('m') if isinstance(n, dict) else getattr(n, 'm', '')
                if 'CONTRACT_VIOLATION' in str(m or ''):
                    st['refused_contract'] = st.get('refused_contract', 0) + 1
            else:
                st['placed'] = st.get('placed', 0) + 1


def update_sent_stats(self: Validator, responses) -> None:
    """Counts, per miner, the placements, the cancel instructions and the orders they name that the validator sends to
    the engine this round, after validation and the delays, and keeps each named order until its answer, so the
    engine's answers can be read against what was asked."""
    pending = self.__dict__.setdefault('_cancel_pending', {})
    for response in responses or ():
        uid = getattr(response, 'agent_id', None)
        if uid is None:
            continue
        for instruction in getattr(response, 'instructions', None) or ():
            t = getattr(instruction, 'type', None)
            if t in _PLACEMENT_TYPES:
                bump(self.miner_stats, uid, 'placements_sent')
            elif t == 'CANCEL_ORDERS':
                bump(self.miner_stats, uid, 'cancels_sent')
                book = getattr(instruction, 'bookId', None)
                ids = [getattr(c, 'orderId', None) for c in getattr(instruction, 'cancellations', None) or ()]
                bump(self.miner_stats, uid, 'cancel_orders_sent', len(ids))
                for oid in ids:
                    _remember(pending, (uid, book, oid), _CANCEL_PENDING_CAP)


def update_dropped_stats(self: Validator, result: dict | None) -> None:
    """Adds the instructions the query service dropped this round under its caps (the volume cap, the per-book
    instruction cap) to each miner's block, from the service's validation stats."""
    dropped = ((result or {}).get('validation_stats') or {}).get('dropped') or {}
    for uid, n in dropped.items():
        if n:
            bump(self.miner_stats, uid, 'dropped', int(n))


def update_query_wait(self: Validator, result: dict | None) -> None:
    """Records the query service's wait for the round (dispatch to the last answer, or the global timeout), one of the
    round timing gauges the report publishes; the list keeps the last thousand rounds."""
    wait = ((result or {}).get('validation_stats') or {}).get('wait')
    if wait is None:
        return
    waits = getattr(self, 'query_waits', None)
    if waits is None:
        waits = self.query_waits = []
    waits.append(float(wait))
    del waits[:-1000]


def delay_settings(self: Validator) -> SimpleNamespace:
    """What set_delays reads from the validator, for the query process to apply it (neuron.orders_packed)."""
    scoring = self.config.scoring
    return SimpleNamespace(
        config=SimpleNamespace(neuron=SimpleNamespace(timeout=self.config.neuron.timeout), scoring=SimpleNamespace(
            min_delay=scoring.min_delay, max_delay=scoring.max_delay,
            min_instruction_delay=scoring.min_instruction_delay, max_instruction_delay=scoring.max_instruction_delay,
            delay_policy=getattr(scoring, 'delay_policy', None), delay_jitter=getattr(scoring, 'delay_jitter', None),
            delay_prompt_cutoff=getattr(scoring, 'delay_prompt_cutoff', None))),
        simulation=SimpleNamespace(time_unit=getattr(self.simulation, 'time_unit', 'ns')),
        # main's loaded profile, so the query process applies it without reading the file each round
        _latency_profile=latency_profile(self))


def update_stats(self : Validator, synapses : dict[int, MarketSimulationStateUpdate]) -> None:
    """
    Updates miner request statistics maintained and published by the validator.

    Args:
        self (Validator): The intelligent markets simulation validator.
        synapses (dict[int, MarketSimulationStateUpdate]): Mapping of miner UIDs to their corresponding synapse updates.

    Returns:
        None
    """
    for uid, synapse in synapses.items():
        block_row(self.miner_stats, uid)
        self.miner_stats[uid]['requests'] += 1
        if synapse.is_timeout:
            self.miner_stats[uid]['timeouts'] += 1
        elif synapse.is_failure or synapse.response is None:
            self.miner_stats[uid]['failures'] += 1
        elif synapse.is_blacklist:
            self.miner_stats[uid]['rejections'] += 1
        elif synapse.dendrite.process_time:
            self.miner_stats[uid]['call_time'].append(synapse.dendrite.process_time)
        # Presence: the last presence_window outcomes per uid, read by the de-beta presence gate
        # (debeta.absent_uids). Kept apart from miner_stats, whose window is the publish cadence.
        ok = not (synapse.is_timeout or synapse.is_failure or synapse.response is None or synapse.is_blacklist)
        presence = getattr(self, 'miner_presence', None)
        if presence is None:
            presence = self.miner_presence = {}
        outcomes = presence.get(uid)
        if outcomes is None:
            dcfg = getattr(getattr(getattr(self, 'config', None), 'scoring', None), 'debeta', None)
            window = int(getattr(dcfg, 'presence_window', None) or 50)
            outcomes = presence[uid] = deque(maxlen=max(window, 1))
        outcomes.append(ok)

def max_instruction_delay_ns(config) -> int:
    """The longest delay a miner's instruction may ask for, --neuron.max_instruction_delay_s in nanoseconds; the
    query service refuses a longer one at acceptance and the engine holds the same bound."""
    seconds = float(getattr(getattr(config, 'neuron', None), 'max_instruction_delay_s', 600.0) or 0.0)
    return int(seconds * 1_000_000_000)


async def forward(self, synapse: MarketSimulationStateUpdate) -> List[FinanceAgentResponse]:
    """
    Forwards a market simulation state update to miner agents via POSIX IPC.

    Packages request data, sends it to the external query service, receives agent responses,
    updates miner statistics, and applies delay scoring.

    Args:
        self (Validator): The intelligent markets simulation validator.
        synapse (MarketSimulationStateUpdate): The simulation state data to forward to miner agents.

    Returns:
        List[FinanceAgentResponse]: List of agent responses generated by miners.
    """
    responses = []

    if getattr(getattr(self.config, 'neuron', None), 'observe', False):
        return responses

    if self.deregistered_uids != []:
        response = FinanceAgentResponse(agent_id=self.uid)
        response.reset_agents(agent_ids=self.deregistered_uids)
        responses.append(response)

    bt.logging.info("Querying Miners...")

    if self.query_process.poll() is not None:
        bt.logging.error(f"Query service died with exit code {self.query_process.returncode}")
        bt.logging.error("Attempting to restart query service...")
        self._start_query_service()
        if self.query_process.poll() is not None:
            self.pagerduty_alert("Failed to restart query service")
            return responses
    # RELEASE ONLY WHAT THIS CALL ACQUIRED. asyncio.Lock has no owner, so release() from a coroutine
    # that never acquired it frees the lock a DIFFERENT in-flight forward() is holding. This try opens
    # ~60 lines before the acquire below, and forward() is re-entrant across that span because the
    # reward-catchup loop awaits inside it: an early return (observe) or any exception before the
    # acquire reached the unconditional release in the finally and stole a live holder's lock. That
    # surfaces as `RuntimeError: Lock is not acquired` raised by the VICTIM right after a successful
    # query, and the raising finally then replaces whatever exception actually occurred, destroying the
    # real fault.
    _lock_acquired = False
    try:
        session_start = time.time()
        bt.logging.debug(f"Session configured ({time.time() - session_start:.4f}s)")

        data_start = time.time()
        extended_metagraph = self.get_extended_metagraph()
        request_data = {
            'books': synapse.books,
            'accounts': synapse.accounts,
            'notices': synapse.notices,
            'config': synapse.config.model_dump(mode='json'),
            'version': synapse.version,
            'timestamp': synapse.timestamp,
            'block': synapse.block,
            'pools': synapse.pools or {},
            'book_ids': list(self.engine.book_ids),
            'engine_mode': self.engine.mode,
            'metagraph_axons': [
                {
                    'hotkey': axon.hotkey,
                    'coldkey': axon.coldkey,
                    'ip': axon.ip,
                    'port': axon.port,
                    'ip_type': axon.ip_type,
                    'protocol': axon.protocol,
                }
                for axon in extended_metagraph.axons
            ],
            'deregistered_uids': list(self.deregistered_uids),
            'uid': self.uid,
            'miner_wealth': self.simulation.miner_wealth,
            'volume_decimals': self.simulation.volumeDecimals,
            'book_count': self.simulation.book_count,
            'volume_sums': {uid: dict(books) for uid, books in self.volume_sums.items()},
            'capital_turnover_cap': self.config.scoring.activity.capital_turnover_cap,
            'max_instructions_per_book': self.config.scoring.max_instructions_per_book,
            'exchange_volume_cap': getattr(getattr(self.config, 'exchange', None), 'volume_cap', 50000.0),
            'max_instruction_delay_ns': max_instruction_delay_ns(self.config),
        }
        if getattr(self.config.neuron, 'orders_packed', False) and self.engine.mode != 'exchange':
            request_data['delays'] = delay_settings(self)  # the query process delays and encodes the orders
        bt.logging.info(f"Request Data Prepared ({time.time()-data_start:.4f}s).")

        if getattr(self.config.neuron, 'observe', False):
            bt.logging.info("Observe mode — skipping miner queries")
            return responses

        if self.should_block_queries():
            last_log_time = time.time()
            while self._pending_reward_tasks > 0:
                await asyncio.sleep(0.1)
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    bt.logging.warning(f"Waiting for rewarding to catch up before querying ({self._pending_reward_tasks} tasks pending)...")
                    last_log_time = current_time
        bt.logging.info("Rewarding up to date, proceeding with query!")

        # Mutual exclusion with GTX assignment delivery (deliver_gentrx): both
        # use the same IPC request queue / notify pipe AND the same miner
        # network — they must never overlap. Released in the finally below.
        await self.miner_net_lock.acquire()
        _lock_acquired = True
        self.querying = True
        query_start = time.time()
        try:
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self.query_ipc_executor,
                        lambda: self.response_queue.receive(timeout=0.001)
                    ),
                    timeout=0.1
                )
                bt.logging.warning("Drained stale message from query response queue")
            except (posix_ipc.BusyError, asyncio.TimeoutError):
                pass
            bt.logging.info(f"Drained Query Response Queue ({time.time()-query_start:.4f}s).")
            
            # Drain stale completion bytes from the notification pipe. The query service
            # can finish a cycle before the validator loops back around; the b'1' sits in
            # the pipe and would be read as an instant (false) completion on the next send.
            drained_pipe = 0
            try:
                while True:
                    data = os.read(self.query_notify_read, 1)
                    if data:
                        drained_pipe += 1
                    else:
                        break
            except (BlockingIOError, OSError):
                pass
            if drained_pipe > 0:
                bt.logging.warning(f"Drained {drained_pipe} stale pipe notification(s) before query")

            # Drain stale completion bytes from the notification pipe. The query service
            # can finish a cycle before the validator loops back around; the b'1' sits in
            # the pipe and would be read as an instant (false) completion on the next send.
            drained_pipe = 0
            try:
                while True:
                    data = os.read(self.query_notify_read, 1)
                    if data:
                        drained_pipe += 1
                    else:
                        break
            except (BlockingIOError, OSError):
                pass
            if drained_pipe > 0:
                bt.logging.warning(f"Drained {drained_pipe} stale pipe notification(s) before query")

            write_start = time.time()
            if getattr(self.config.neuron, 'query_prepayload', False) and request_data['engine_mode'] != 'exchange':
                # The miners' requests built here, so the query process unpickles strings, not the decoded state.
                from taos.im.validator.query import prebuild_request
                compression = self.config.compression
                request_data = prebuild_request(request_data, self.wallet.hotkey.ss58_address, compression.level,
                                                compression.engine, compression.parallel_workers)
                bt.logging.info(f"Prebuilt miner payloads ({time.time()-write_start:.4f}s).")
                write_start = time.time()
            data_bytes = pickle.dumps(request_data, protocol=5)
            bt.logging.info(f"Query request data size: {len(data_bytes) / 1024 / 1024:.2f} MB")
            
            self.request_mem.seek(0)
            self.request_mem.write(struct.pack('Q', len(data_bytes)))
            self.request_mem.write(data_bytes)
            bt.logging.info(f"Wrote query request data ({time.time()-write_start:.4f}s).")

            self.response_mem.seek(0)
            self.response_mem.write(struct.pack('Q', 0))

            receive_start = time.time()
            self.request_queue.send(b'query')
            bt.logging.info("Sent query request, waiting for notification...")

            max_wait = self.config.neuron.global_query_timeout + 10

            try:
                loop = asyncio.get_event_loop()
                notify_result = None
                deadline = time.time() + max_wait
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    future = asyncio.Future()

                    def pipe_readable():
                        loop.remove_reader(self.query_notify_read)
                        try:
                            data = os.read(self.query_notify_read, 1)
                            future.set_result(data)
                        except Exception as e:
                            future.set_exception(e)
                    loop.add_reader(self.query_notify_read, pipe_readable)
                    try:
                        notify_result = await asyncio.wait_for(future, timeout=remaining)
                    except asyncio.TimeoutError:
                        loop.remove_reader(self.query_notify_read)
                        raise
                    if notify_result == b'R':
                        bt.logging.debug("Ignoring stray ready signal, re-waiting")
                        continue
                    break
                pipe_wait_time = time.time() - receive_start
                if notify_result == b'1':
                    bt.logging.info(f"Received query completion notification ({pipe_wait_time:.4f}s)")
                else:
                    bt.logging.warning(f"Unexpected notification: {notify_result}")
                    return responses
            except asyncio.TimeoutError:
                self.pagerduty_alert(f"Query service notification timeout after {max_wait}s")
                return responses
            except Exception as e:
                self.pagerduty_alert(f"Error waiting for notification: {e}")
                return responses
            
            read_start = time.time()
            self.response_mem.seek(0)
            size_bytes = self.response_mem.read(8)
            data_size = struct.unpack('Q', size_bytes)[0]
            
            if data_size == 0:
                self.pagerduty_alert("Query service signaled ready but response size is 0")
                return responses
            
            result_bytes = self.response_mem.read(data_size)

            if len(result_bytes) != data_size:
                self.pagerduty_alert(f"Incomplete read: got {len(result_bytes)}, expected {data_size}")
                return responses

            _q2_shm_read_s = time.time() - read_start
            try:
                _q2_loads_start = time.time()
                result = pickle.loads(result_bytes)
                _q2_loads_s = time.time() - _q2_loads_start
                # [Q2-PROFILE] main-loop side of the forward IPC gap. notify_wait
                # = subprocess total (fanout + validate + dumps + write); shm_read
                # = reading the pickled blob; loads = pickle.loads ON THE EVENT
                # LOOP (holds the GIL, blocks the round). If loads is fat, the
                # win is a leaner response payload (send compact instructions).
                bt.logging.info(
                    f"[Q2-PROFILE main] notify_wait={pipe_wait_time:.3f}s "
                    f"shm_read={_q2_shm_read_s:.3f}s loads={_q2_loads_s:.3f}s "
                    f"resp={data_size/1048576:.1f}MB"
                )
                bt.logging.info(f"Read query response data ({time.time()-read_start:.4f}s)")
            except (pickle.UnpicklingError, UnicodeDecodeError) as e:
                self.pagerduty_alert(f"Failed to unpickle query response: {e}")
                return responses

        except posix_ipc.BusyError:
            self.pagerduty_alert("Query service response timeout")
            return responses
        except Exception as e:
            self.pagerduty_alert(f"Error communicating with query service: {e}", details={"traceback" : traceback.format_exc()})
            return responses

        if not result['success']:
            self.pagerduty_alert(f"Query service error: {result.get('error')}")
            if 'traceback' in result:
                bt.logging.error(f"Traceback: {result['traceback']}")
            return responses

        synapse_responses = result['responses']
        # the (uid, book) pairs at or over the volume cap at this round's snapshot, for the engine to refuse their
        # placements on arrival (row 67); None when the query service did not say, so the engine keeps its last set
        self._capped_pairs = result.get('capped')
        
        bt.logging.info(f"Query Completed ({time.time()-query_start:.4f}s).")
        update_query_wait(self, result)
        update_dropped_stats(self, result)
        
        start = time.time()
        update_stats(self, synapse_responses)
        bt.logging.info(f"Updated Stats ({time.time()-start:.4f}s).")

        start = time.time()
        if 'orders' in result:  # neuron.orders_packed: already delayed and encoded by the query process
            responses.append(result['orders'])
            self.engine.response_pack_drops = getattr(self.engine, 'response_pack_drops', 0) + result['pack_drops']
            for uid, row in result['sent'].items():  # update_sent_stats' counts, taken where the orders were packed
                for name, n in row.items():
                    if n:
                        bump(self.miner_stats, uid, name, n)
        else:
            delayed = set_delays(self, synapse_responses)
            responses.extend(delayed)
            update_sent_stats(self, delayed)
        bt.logging.info(f"Set Delays ({time.time()-start:.4f}s).")

        bt.logging.trace(f"Responses: {responses}")
        bt.logging.info(f"Received {result['validation_stats']['total_responses']} valid responses containing {result['validation_stats']['total_instructions']} instructions at {duration_from_timestamp(synapse.timestamp)} "
                    f"({result['validation_stats']['success']} SUCCESS | {result['validation_stats']['timeouts']} TIMEOUTS | {result['validation_stats']['failures']} FAILURES).")
        return responses
    finally:
        self.querying = False
        # NOT `if self.miner_net_lock.locked()`: that is true whenever ANYONE holds it, which
        # reproduces the cross-release this guard exists to prevent.
        if _lock_acquired:
            self.miner_net_lock.release()
        bt.logging.debug("Query flag cleared")


async def deliver_gentrx(self: Validator, deliveries: list) -> None:
    """Deliver GenTRX assignment synapses to miners via the query service IPC.

    Serializes delivery requests into shared memory, sends a 'deliver_gentrx'
    command to the query service, and waits for the b'G' pipe notification.
    Uses the same IPC channel as forward() so queries and deliveries are
    naturally serialized by the single request queue — they cannot overlap.

    Args:
        self (Validator): The validator instance.
        deliveries: List of (uid, axon, GenTRXAssignment) tuples.
    """
    from GenTRX.src.bt_log import gtx_log

    if not deliveries:
        return

    round_id = deliveries[0][2].round if deliveries else "?"

    request_data = {
        'round': round_id,
        'deliveries': [
            {
                'uid': uid,
                'axon_data': {
                    'hotkey': axon.hotkey,
                    'coldkey': axon.coldkey,
                    'ip': axon.ip,
                    'port': axon.port,
                    'ip_type': axon.ip_type,
                    'protocol': axon.protocol,
                },
                'assignment': {
                    'round': synapse.round,
                    'model_version': synapse.model_version,
                    'books': synapse.books,
                    'ts_start': synapse.ts_start,
                    'ts_end': synapse.ts_end,
                    'data': synapse.data,
                    'data_source': synapse.data_source,
                    # The aggregator's shard. This dict is a hand-listed flattening of the synapse for the
                    # query service, and query.py rebuilds a synapse FROM it -- so a field missing here is
                    # silently dropped between two objects that both declare it.
                    'bucket_prefix': synapse.bucket_prefix,
                    'data_endpoint': synapse.data_endpoint,
                    'data_bucket': synapse.data_bucket,
                    'data_access_key': synapse.data_access_key,
                    'data_secret_key': synapse.data_secret_key,
                    'validator_uid': synapse.validator_uid,
                    'advice': synapse.advice,
                },
            }
            for uid, axon, synapse in deliveries
        ],
    }

    try:
        data_bytes = pickle.dumps(request_data, protocol=5)
        gtx_log.info(
            f"deliver_gentrx IPC: round={round_id} "
            f"n={len(deliveries)} bytes={len(data_bytes)}"
        )

        self.response_mem.seek(0)
        self.response_mem.write(struct.pack('Q', 0))

        self.request_mem.seek(0)
        self.request_mem.write(struct.pack('Q', len(data_bytes)))
        self.request_mem.write(data_bytes)

        self.request_queue.send(b'deliver_gentrx')
        gtx_log.info("Sent deliver_gentrx command, waiting for notification...")

        max_wait = 60.0
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        def pipe_readable():
            loop.remove_reader(self.query_notify_read)
            try:
                data = os.read(self.query_notify_read, 1)
                future.set_result(data)
            except Exception as e:
                future.set_exception(e)

        loop.add_reader(self.query_notify_read, pipe_readable)
        try:
            notify_result = await asyncio.wait_for(future, timeout=max_wait)
        except asyncio.TimeoutError:
            loop.remove_reader(self.query_notify_read)
            gtx_log.error(f"deliver_gentrx notification timeout after {max_wait}s")
            return
        except Exception as e:
            gtx_log.error(f"deliver_gentrx notification error: {e}")
            return

        if notify_result != b'G':
            gtx_log.warning(f"Unexpected notification byte for deliver_gentrx: {notify_result}")
            return

        self.response_mem.seek(0)
        size_bytes = self.response_mem.read(8)
        data_size = struct.unpack('Q', size_bytes)[0]
        if data_size == 0:
            gtx_log.error("deliver_gentrx: response size is 0")
            return
        result_bytes = self.response_mem.read(data_size)
        result = pickle.loads(result_bytes)
        gtx_log.info(
            f"deliver_gentrx complete: ok={result.get('ok')} fail={result.get('fail')}"
        )

    except Exception as e:
        gtx_log.error(f"deliver_gentrx IPC error: {e}\n{traceback.format_exc()}")


async def notify(self : Validator, notices : List[FinanceEventNotification]) -> None:
    """
    Forwards finance event notifications to the appropriate miner agents.

    Uses the validator's dendrite to dispatch each event to the miner responsible
    for the associated agent, or to all miners if the event is broadcast.
    Handles both network UIDs and benchmark agent UIDs.

    Args:
        self (Validator): The intelligent markets simulation validator.
        notices (List[FinanceEventNotification]): List of event notifications to dispatch.

    Returns:
        None
    """
    responses = []
    extended_metagraph = self.get_extended_metagraph()
    # The validator's own uid is published at its external IP with nothing serving it, so a
    # notice sent there reaches whatever listens on that port. Same rule as query.py's
    # _queryable_axons.
    own_hotkey = self.wallet.hotkey.ss58_address

    def _not_own(uid, axon):
        return uid != self.uid and axon.hotkey != own_hotkey

    others = [axon for uid, axon in enumerate(extended_metagraph.axons) if _not_own(uid, axon)]

    for notice in notices:
        if notice.event.agentId is not None:
            agent_id = notice.event.agentId
            if agent_id < len(extended_metagraph.axons):
                target = extended_metagraph.axons[agent_id]
                axons = [target] if _not_own(agent_id, target) else []
            else:
                bt.logging.error(
                    f"Invalid agentId {agent_id} in notice (max: {len(extended_metagraph.axons)-1}), "
                    f"broadcasting to all agents instead"
                )
                axons = others
        else:
            axons = others
        if not axons:
            continue
        responses.extend(await self.dendrite(
            axons=axons,
            synapse=notice,
            timeout=1
        ))
    for response in responses:
        if response and response.acknowledged:
            bt.logging.info(f"{response.type} EventNotification Acknowledged by {response.axon.hotkey}")