# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import json
import logging
import uvicorn
import asyncio
import aiohttp
import argparse
import msgspec
import os
import time
import copy
import bittensor as bt
from pathlib import Path
from threading import Thread
import traceback

# Ensure GenTRX service logs are visible (uses standard Python logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
import posix_ipc
import mmap
import msgpack

from fastapi import FastAPI, APIRouter, Request

from taos import __spec_version__
from taos.common.neurons import BaseNeuron
from taos.im.neurons.validator import Validator
from taos.im.protocol import MarketSimulationStateUpdate, MarketSimulationConfig, FinanceAgentResponse, FinanceEventNotification
from taos.im.protocol.simulator import SimulatorResponseBatch
from taos.im.protocol.events import SimulationStartEvent
from taos.im.validator.reward import set_delays

from ypyjson import YpyObject
import xml.etree.ElementTree as ET

# Ensure GenTRX service logs are visible (uses standard Python logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_MVTRX_DATA_SERVICE_URL = os.environ.get("MVTRX_DATA_SERVICE_URL", "")

async def _push_proxy_state_to_mvtrx(state, block: int) -> None:
    """Delegate to the optional data-service push module when present."""
    try:
        from agents.proxy.mvtrx_push import push_proxy_state as _push
    except ImportError:
        return
    await _push(state, block, _MVTRX_DATA_SERVICE_URL)

# GenTRX integration — StatePackager + assignment fetch via GenTRXService
try:
    from GenTRX.src.service import GenTRXService

    _GENTRX_AVAILABLE = True
except ImportError as _e:
    bt.logging.warning(f"GenTRX not found — disabled ({_e})")
    _GENTRX_AVAILABLE = False

#--------------------------------------------------------------------------

class Proxy(Validator):
    def __init__(self, launcher_config):
        bt.logging.set_info()
        base_config = copy.deepcopy(BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self.config.neuron.timeout = launcher_config['proxy']['timeout']
        self.check_config(self.config)
        config_file = launcher_config['proxy']['simulation_xml']
        if not os.path.isabs(config_file):
            script_dir = Path(__file__).resolve().parent
            repo_root = Path(__file__).resolve().parents[2]
            anchor = script_dir if config_file.startswith("..") or config_file.startswith("./") else repo_root
            config_file = str((anchor / config_file).resolve())
        if not os.path.exists(config_file):
            raise Exception(f"Simulator config does not exist at {config_file}!")
        self.simulator_config_file = os.path.realpath(Path(config_file))
        self.simulation_config = self.load_simulation_config()

        self.agent_urls = {}
        port = launcher_config['agents']['start_port']
        for agent, agent_configs in launcher_config['agents'].items():
            if agent in ['start_port', 'path']:
                continue
            for agent_config in agent_configs:
                base_agent_name = f"{agent}_{'_'.join(list([str(a) for a in agent_config['params'].values()]))}"
                for i in range(agent_config['count']):
                    agent_name = f"{base_agent_name}_{i}"
                    self.agent_urls[agent_name] = f"http://127.0.0.1:{port}/handle"
                    port += 1

        self.compressing = False
        self.start_time = None
        self.start_timestamp = None
        self.step = 0

        self._gentrx = None
        grad_url = launcher_config.get("proxy", {}).get("gradient_server_url", "")
        if _GENTRX_AVAILABLE and grad_url:
            gs_interval = launcher_config.get("gradient_server", {}).get("interval", 60)
            n_agents = len(self.agent_urls)

            gs_log = launcher_config.get("gradient_server", {}).get("log", "")
            log_path = os.path.join(os.path.dirname(gs_log), "gentrx_service.log") if gs_log else None

            from GenTRX.src.state_packager import StatePackager
            agent_uids = list(range(n_agents))
            self._gentrx = GenTRXService(
                packager=StatePackager(),
                gradient_server_url=grad_url,
                poll_interval=gs_interval,
                deliver_fn=self._deliver_gentrx_assignments,
                miner_uids=agent_uids,
                log_path=log_path,
            )
            bt.logging.info(f"GenTRX service: server={grad_url}, poll={gs_interval}s, uids={agent_uids}")
        elif _GENTRX_AVAILABLE:
            bt.logging.info("GenTRX service: disabled (no proxy.gradient_server_url in config)")

        # Add routes for methods receiving input from simulator
        self.router = APIRouter()
        self.router.add_api_route("/orderbook", self.orderbook, methods=["GET"])
        self.router.add_api_route("/account", self.account, methods=["GET"])
        # Note: scores arrive via data bucket polling (poll_scores), no endpoint needed.

    def load_simulation_config(self):
        self.xml_config = ET.parse(self.simulator_config_file).getroot()
        self.simulation = MarketSimulationConfig.from_xml(self.xml_config)

    def compress_outputs(self, start: bool = False) -> None:
        """No-op for the local proxy harness. The production validator's
        log-archive + disk-cleanup loop lives on SimulationEngine; the proxy
        runs locally where disk hygiene isn't a concern, so skip it."""
        pass

    def seed(self) -> None:
        """
        Generates simulator seed data with improved error handling and non-blocking restarts.

        Returns:
            None
        """
        from taos.im.validator.seed import seed_thread
        
        # Wrap seed in try-except to prevent thread crashes
        while True:
            try:
                seed_thread(self)
            except Exception as ex:
                bt.logging.error(f"Seed process crashed: {ex}")
                bt.logging.error(traceback.format_exc())
                self.pagerduty_alert(
                    f"Seed process crashed, restarting in 5s: {ex}",
                    details={"trace": traceback.format_exc()}
                )
                time.sleep(5)

    def onStart(self, timestamp, event : SimulationStartEvent) -> None:
        """Triggered when start of simulation event is published by simulator.
        Sets the simulation output directory and compresses prior outputs."""
        bt.logging.info("-"*40)
        bt.logging.info("SIMULATION STARTED")
        self.start_time = time.time()
        self.simulation_timestamp = timestamp
        self.start_timestamp = self.simulation_timestamp
        self.simulation.logDir = event.logDir
        self.compress_outputs()
        bt.logging.info(f"START TIME: {self.start_time}")
        bt.logging.info(f"TIMESTAMP : {self.start_timestamp}")
        bt.logging.info(f"OUT DIR   : {self.simulation.logDir}")
        bt.logging.info("-"*40)

    async def handle_state(self, message : dict, state : MarketSimulationStateUpdate, receive_start : int) -> dict:
        time.time()
        state.version = __spec_version__
        state.dendrite.hotkey = 'proxy'
        if not self.start_time:
            self.start_time = time.time()
            self.start_timestamp = state.timestamp
        if self.simulation.logDir != message['logDir']:
            bt.logging.info(f"Simulation log directory changed : {self.simulation.logDir} -> {message['logDir']}")
            self.simulation.logDir = message['logDir']
        self.simulation_timestamp = state.timestamp
        if self.simulation:
            state.config = self.simulation.model_copy()
            state.config.simulation_id = os.path.basename(state.config.logDir)[:13]
            state.config.logDir = None
        self.step += 1
        bt.logging.debug(f"STATE : {state}")

        # GenTRX: save state to S3 + deliver assignments to agents
        if self._gentrx is not None:
            self._gentrx.push_state(state)
            await self._gentrx.poll_and_deliver()

        # Forward state to agents
        async def query_agent(uid, agent, url, session, json):
            response_time = None
            try:
                bt.logging.info(f"Querying {agent} at {url}...")
                start = time.time()
                async with session.post(url=url, json=json, timeout=self.config.neuron.timeout) as r:
                    response = await r.json()
                    response_time = time.time() - start
                    # Bounded: a rejected body comes back with the whole request echoed inside it,
                    # which at this book count is megabytes on one line. Report the status too, so a
                    # rejection reads as one rather than as a success.
                    body = str(response)
                    body = body if len(body) <= 512 else body[:512] + f"... ({len(body)} chars)"
                    if r.status == 200:
                        bt.logging.success(f"{agent} | Response : {body} ({response_time}s)")
                    else:
                        bt.logging.error(f"{agent} | HTTP {r.status} : {body} ({response_time}s)")
                return uid, agent, response, response_time
            except asyncio.exceptions.TimeoutError:
                bt.logging.error(f"{agent} | Timed out after {self.config.neuron.timeout}s while awaiting response from {url}.")
                return uid, agent, None, response_time
            except Exception as e:
                bt.logging.error(f"{agent} | Failed to query {url}: {e}")
                return uid, agent, None, response_time

        serialized_config = state.config.model_dump(mode='json')
        def create_agent_json(uid):
            return state.model_copy(update={
                "accounts": {uid: state.accounts[uid]},
                "notices": {uid: state.notices[uid]},
                "config" : serialized_config
            }).model_dump(mode='json', warnings=False)
        async with aiohttp.ClientSession() as session:
            responses = await asyncio.gather(*(query_agent(uid, agent, agent_url, session, create_agent_json(uid)) for uid, (agent, agent_url) in enumerate(self.agent_urls.items())))
        responses = {uid : (response, agent, response_time) for uid, agent, response, response_time in responses}
        synapse_responses = {}
        for uid, (response, agent, response_time) in responses.items():
            if response:
                try:
                    agent_response = FinanceAgentResponse.model_validate(response)
                    synapse = state.model_copy(update={"response":agent_response})
                    synapse.dendrite.process_time = response_time
                    synapse_responses[uid] = synapse
                except Exception as e:
                    bt.logging.error(f"{agent} | Failed to validate response : {e}")
        agent_responses = set_delays(self, synapse_responses)
        simulator_response = SimulatorResponseBatch(agent_responses).serialize()
        block_time_ns = getattr(getattr(self, 'simulation', None), 'block_time_ns', 12_000_000_000) or 12_000_000_000
        asyncio.create_task(_push_proxy_state_to_mvtrx(state, state.timestamp // block_time_ns))
        bt.logging.info(f"State update handled ({time.time()-receive_start}s)")
        return simulator_response

    async def _deliver_gentrx_assignments(self, assignments: dict) -> None:
        """Deliver GenTRX assignments to agents via HTTP POST."""
        async with aiohttp.ClientSession() as sess:
            for uid, (_agent, agent_url) in enumerate(self.agent_urls.items()):
                if uid in assignments:
                    assign_url = agent_url.replace("/handle", "/gentrx/assignment")
                    try:
                        await sess.post(
                            assign_url,
                            json=assignments[uid],
                            timeout=aiohttp.ClientTimeout(total=2),
                        )
                    except Exception:
                        pass
        bt.logging.info(f"GenTRX: delivered to {len(assignments)} agents via HTTP")

    async def _listen(self):
        def receive(mq_req) -> dict:
            msg, priority = mq_req.receive()
            receive_start = time.time()
            bt.logging.info("Received state update from simulator (msgpack)")
            byte_size_req = int.from_bytes(msg, byteorder="little")
            shm_req = posix_ipc.SharedMemory("/state")
            start = time.time()
            packed_data = None
            for attempt in range(1, 6):
                try:
                    with mmap.mmap(shm_req.fd, byte_size_req, mmap.MAP_SHARED, mmap.PROT_READ) as mm:
                        packed_data = mm.read(byte_size_req)
                    bt.logging.info(f"Unpacked state update ({time.time() - start:.4f}s)")
                    break
                except Exception as ex:
                    if attempt < 5:
                        bt.logging.error(f"mmap read failed (attempt {attempt}/5): {ex}")
                        time.sleep(0.005)
                    else:
                        bt.logging.error(f"mmap read failed on all 5 attempts: {ex}")
                        self.pagerduty_alert(f"Failed to mmap read after 5 attempts : {ex}", details={"trace" : traceback.format_exc()})
                        raise ex
                finally:
                    if packed_data is not None or attempt >= 5:
                        shm_req.close_fd()
            bt.logging.info(f"Retrieved State Update ({time.time() - receive_start}s)")
            start = time.time()
            result = None
            for attempt in range(1, 6):
                try:
                    result = msgpack.unpackb(packed_data, raw=False, use_list=True, strict_map_key=False)
                    bt.logging.info(f"Unpacked state update ({time.time() - start:.4f}s)")
                    break
                except Exception as ex:
                    if attempt < 5:
                        bt.logging.error(f"Msgpack unpack failed (attempt {attempt}/5): {ex}")
                        time.sleep(0.005)
                    else:
                        bt.logging.error(f"Msgpack unpack failed on all 5 attempts: {ex}")
                        self.pagerduty_alert(f"Failed to unpack simulator state after 5 attempts : {ex}", details={"trace" : traceback.format_exc()})
                        raise ex
            return result, receive_start

        def respond(response) -> dict:
            self.last_response = response
            packed_res = msgpack.packb(response, use_bin_type=True)
            byte_size_res = len(packed_res)
            mq_res = posix_ipc.MessageQueue("/taosim-res", flags=posix_ipc.O_CREAT, max_messages=1, max_message_size=8)
            shm_res = posix_ipc.SharedMemory("/responses", flags=posix_ipc.O_CREAT, size=byte_size_res)
            with mmap.mmap(shm_res.fd, byte_size_res, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ) as mm:
                shm_res.close_fd()
                mm.write(packed_res)
            mq_res.send(byte_size_res.to_bytes(8, byteorder="little"))
            mq_res.close()
        while True:
            response = {"responses" : []}
            try:
                mq_req = posix_ipc.MessageQueue("/taosim-req", flags=posix_ipc.O_CREAT, max_messages=1, max_message_size=8)
                # This blocks until the queue can provide a message
                message, receive_start = receive(mq_req)
                state = MarketSimulationStateUpdate.parse_dict(message)
                response = await self.handle_state(message, state, receive_start)
            except Exception:
                traceback.print_exc()
            finally:
                respond(response)
                mq_req.close()
            
    def listen(self):
        """Synchronous wrapper for the asynchronous _listen method."""
        try:
            asyncio.run(self._listen())
        except KeyboardInterrupt:
            bt.logging.info("Listening stopped by user.")

    async def orderbook(self, request : Request):
        time.time()
        body = await request.body()
        message = YpyObject(body, 1)
        state = MarketSimulationStateUpdate.from_ypy(message) # Populate synapse class from request data
        return self.handle_state(message, state)

    async def account(self, request : Request):
        body = await request.body()
        batch = msgspec.json.decode(body)
        bt.logging.info(f"NOTICE : {batch}")
        for message in batch['messages']:
            if message['type'] == 'EVENT_SIMULATION_START':
                self.onStart(message['timestamp'], FinanceEventNotification.from_json(message).event)

#--------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.json")
    parser.add_argument('--data-service-url', type=str, default=None,
                        help="Base URL of the MVTRX Data Service for ingest push.")
    args = parser.parse_args()

    if args.data_service_url is not None:
        _MVTRX_DATA_SERVICE_URL = args.data_service_url

    config = json.load(open(args.config))
    app = FastAPI()
    proxy = Proxy(config)
    app.include_router(proxy.router)
    # Start simulator price seeding data process in new thread
    Thread(target=proxy.seed, daemon=True, name='Seed').start()
    Thread(target=proxy.listen, daemon=True, name="Listen").start()
    # Run the proxy as a FastAPI client via uvicorn on the configured port
    uvicorn.run(app, port=config['proxy']['port'])