# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research

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
"""
Base miner neuron: registers the axon, handles forward/blacklist/priority
callbacks, and drives the async serving loop for subnet miners.
"""

import os
import time
import typing
import asyncio
import threading
import argparse
import traceback

import bittensor as bt

import importlib.util

from taos.common.neurons import BaseNeuron
from taos.common.config import add_miner_args
from taos.common.agents import SimulationAgent
from taos.common.protocol import SimulationStateUpdate, EventNotification


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for simulation subnet miners.    

    Handles basic initialization including loading of associated agent logic class defined at `{agent.path}/{agent.name}.py`, 
    as well as defining functions for validating, prioritizing and forwarding synapses to the agent for processing.
    """

    neuron_type: str = "MinerNeuron"
    agent: SimulationAgent

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """Add miner CLI arguments to the parser."""
        super().add_args(parser)
        add_miner_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Miners need a reasonably-recent metagraph (uid/hotkey/axon + stake for
        # blacklist/priority), but the resync's substrate scale-decode is a
        # multi-second GIL burst on a 129-subnet chain that would stall the axon
        # (query timeouts). Offload the fetch+decode to a subprocess — the same
        # MetagraphSyncWorker the validator uses — so it runs off the main GIL and
        # can never block the query/response path; resync_metagraph() consumes its
        # result. Falls back to inline sync if the worker can't start.
        self._mg_worker = None
        try:
            from taos.im.validator.metagraph_worker import MetagraphSyncWorker

            _mg_ep = getattr(getattr(self.config, "subtensor", None), "chain_endpoint", "") or ""
            self._mg_worker = MetagraphSyncWorker(_mg_ep, self.config.netuid)
            self._mg_worker.start()
        except Exception as e:
            bt.logging.warning(f"miner metagraph sync worker unavailable ({e}); using inline resync")
            self._mg_worker = None

        # Init to now so the FIRST run()-loop sync() is throttled (skips the heavy
        # metagraph resync) — __init__ already built a fresh metagraph. This lets
        # run() reach axon.start() immediately instead of blocking on a resync,
        # which otherwise leaves the axon not-listening and the miner un-queryable.
        self._last_mg_sync = time.time()

        # Warn if allowing incoming requests from anyone.
        if self.config.blacklist.allow_non_validators:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.Axon(wallet=self.wallet, config=self.config, ip=self.config.axon.ip, port=self.config.axon.port, external_ip=self.config.axon.external_ip, external_port=self.config.axon.external_port)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist_forward,
            priority_fn=self.priority_forward,
        ).attach(
            forward_fn=self.update,
            blacklist_fn=self.blacklist_update,
            priority_fn=self.priority_update,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()    
        
        module_spec = importlib.util.spec_from_file_location(self.config.agent.name, os.path.join(self.config.agent.path, self.config.agent.name + '.py'))
        agent_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(agent_module)
        agent_class = getattr(agent_module, self.config.agent.name)
        self.agent = agent_class(self.uid, self.config.agent.params, self.config.neuron.full_path)

    def should_sync_metagraph(self):
        """Throttle miner metagraph resync. Default (MINER_SYNC_INTERVAL unset/0) keeps
        base behaviour. On a mainnet-shaped chain the resync (neurons_lite fetch +
        256-neuron decode) is heavy enough to starve the axon → validator query
        timeouts; set MINER_SYNC_INTERVAL=<seconds> to resync sparsely so the miner
        stays responsive (the __init__ metagraph is sufficient for serving/uid)."""
        iv = float(os.environ.get("MINER_SYNC_INTERVAL", "0") or 0)
        if iv <= 0:
            return True
        now = time.time()
        if now - getattr(self, "_last_mg_sync", 0.0) < iv:
            return False
        self._last_mg_sync = now
        return True

    async def forward(
        self, synapse: SimulationStateUpdate
    ) -> SimulationStateUpdate:
        """
        Processes incoming simulation state synapse by forwarding to the associated agent class for handling.

        Args:
            synapse (protocol.common.SimulationStateUpdate): The synapse object containing the latest simulation state update.

        Returns:
            protocol.common.SimulationStateUpdate: The synapse object with the 'response' field updated with any instructions generated by the agent.
        """
        synapse.response = self.agent.handle(synapse)
        return synapse
    
    async def update(
        self, synapse: EventNotification
    ) -> EventNotification:
        """
        Processes incoming event notification synapse by forwarding to the associated agent class for handling.
        Validators do not require nor accept any response to event notification synapses, they are used only to provide information to the agent.

        Args:
            synapse (protocol.common.EventNotification): The synapse object containing the event data.

        Returns:
            protocol.common.EventNotification: The synapse object with the 'acknowledged' field updated to true.
        """
        synapse = self.agent.process(synapse)
        return synapse
    
    def blacklist_forward(
        self, synapse: SimulationStateUpdate
    ) -> typing.Tuple[bool, str]:
        """
        Apply default blacklisting to all received simulation state synapses.
        """
        return self.blacklist(synapse)
    
    def priority_forward(self, synapse: SimulationStateUpdate) -> float:
        """
        Apply default prioritization to all received simulation state synapses.
        """
        return self.priority(synapse)
    
    def blacklist_update(
        self, synapse: EventNotification
    ) -> typing.Tuple[bool, str]:
        """
        Apply default blacklisting to all received event notification synapses.
        """
        return self.blacklist(synapse)
    
    def priority_update(self, synapse: EventNotification) -> float:
        """Priority for an incoming notification synapse.

        Args:
            synapse (EventNotification): The incoming notification.

        Returns:
            float: Stake-derived priority.
        """
        return self.priority(synapse)

    def blacklist(
        self, synapse: bt.Synapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Unregistered hotkey: there is no uid / validator_permit to look up,
            # so the membership test MUST come before metagraph.hotkeys.index()
            # (which would otherwise raise ValueError on an unknown hotkey).
            if not self.config.blacklist.allow_non_registered:
                # Ignore requests from un-registered entities.
                bt.logging.trace(
                    f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey"
            bt.logging.trace(
                f"Not Blacklisting un-registered hotkey {synapse.dendrite.hotkey} (allow_non_registered)"
            )
            return False, "Hotkey recognized!"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if not self.config.blacklist.allow_non_validators:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    def priority(self, synapse: bt.Synapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # Retry axon serving if failed
        if self.subtensor is not None and hasattr(self.subtensor, "serve_axon"):
            served = False
            attempts = 0
            # Bound the serve retries: serve_axon can fail with ServingRateLimitExceeded
            # when the axon was published recently (common after restarts). The
            # previously-published on-chain axon info stays valid, so we must NOT block
            # here forever — start the axon regardless so it listens + is queryable.
            max_serve_attempts = int(os.environ.get("MINER_SERVE_ATTEMPTS", "3"))
            while not served and attempts < max_serve_attempts:
                try:
                    bt.logging.info(
                        f"Serving miner axon at {self.axon.external_ip}:{self.axon.external_port} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                    )
                    attempts += 1
                    served = self.subtensor.serve_axon(netuid=self.config.netuid, axon=self.axon)
                    if not served:
                        bt.logging.error(f"Failed to serve axon! Retrying (Attempt {attempts}/{max_serve_attempts})")
                        time.sleep(10)
                    else:
                        bt.logging.success("Published axon to chain.")
                except Exception as ex:
                    bt.logging.error(f"Exception when attempting to serve axon - Retrying in 5 secs (Attempt {attempts}/{max_serve_attempts}) : {ex}")
                    time.sleep(5)
            if not served:
                bt.logging.warning(
                    "Axon serve did not succeed within attempts (likely ServingRateLimitExceeded); "
                    "starting axon anyway — the previously-published on-chain axon info remains valid."
                )
        else:
            raise Exception("Cannot serve axon - invalid subtensor.  Check bittensor version and configuration and try again.")        

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()

        bt.logging.info(f"Miner starting at block {self.block} with UID {self.uid}")

        # This loop maintains the miner's operations until intentionally stopped. A dropped chain
        # connection (e.g. the local node restarting) must NOT terminate the loop: previously a single
        # sync() exception escaped to the outer handler, run() returned, and the miner wedged silently —
        # the process stayed alive so pm2 reported it 'online' while the axon served a stale/unreachable
        # state and nothing resynced. Now each iteration is guarded: on failure rebuild the subtensor +
        # metagraph in place, and only exit (for a clean pm2 restart) if the chain stays unreachable
        # across many consecutive attempts.
        consecutive_sync_failures = 0
        max_sync_failures = int(os.environ.get("MINER_MAX_SYNC_FAILURES", "12"))
        try:
            while not self.should_exit:
                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph
                try:
                    self.sync()
                    consecutive_sync_failures = 0
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    consecutive_sync_failures += 1
                    bt.logging.warning(
                        f"Miner sync failed ({consecutive_sync_failures}/{max_sync_failures}); chain "
                        f"connection likely dropped (node restart?). Reconnecting.\n{traceback.format_exc()}"
                    )
                    self._recover_chain_connection()
                    if consecutive_sync_failures >= max_sync_failures:
                        bt.logging.error(
                            f"Miner could not re-establish the chain connection after "
                            f"{consecutive_sync_failures} attempts; exiting for a clean pm2 restart."
                        )
                        try:
                            self.axon.stop()
                        except Exception:
                            pass
                        os._exit(1)
                self.step += 1
                time.sleep(bt.BLOCKTIME * 10)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception:
            bt.logging.error(traceback.format_exc())

    def _recover_chain_connection(self):
        """Rebuild the subtensor websocket + metagraph handle after a chain drop (e.g. the local node
        restarting) and re-publish the axon, so the miner self-heals instead of wedging with a dead
        connection while pm2 still reports it 'online'. The _mg_worker subprocess already self-respawns
        on its next sync(); this repairs the primary subtensor used by check_registered/update_block and
        the inline-resync fallback."""
        try:
            with self._subtensor_lock:
                _ep = getattr(getattr(self.config, "subtensor", None), "chain_endpoint", "") or ""
                self.subtensor = bt.Subtensor(network=_ep) if _ep else bt.Subtensor(config=self.config)
                _mechid = int(getattr(self, "_mechid", 0) or 0)
                self.metagraph = self.subtensor.metagraph(self.config.netuid, mechid=_mechid)
                bt.logging.info("Rebuilt subtensor + metagraph after chain drop.")
        except Exception:
            bt.logging.warning(f"Subtensor rebuild failed (node still unreachable?):\n{traceback.format_exc()}")
            return
        # Re-publish the axon so its on-chain serve record is refreshed against the new connection. A
        # ServingRateLimitExceeded here is harmless (the prior serve stays valid), so it's best-effort.
        try:
            if hasattr(self.subtensor, "serve_axon"):
                self.subtensor.serve_axon(netuid=self.config.netuid, axon=self.axon)
                bt.logging.info("Re-published axon after reconnect.")
        except Exception:
            bt.logging.debug(f"Axon re-serve after reconnect failed (rate limit ok):\n{traceback.format_exc()}")

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.trace("resync_metagraph()")

        # Sync the metagraph OFF the main GIL via the subprocess worker so the axon
        # keeps serving during the multi-second substrate scale-decode. Fall back to
        # inline sync only if the worker is unavailable or returns nothing.
        if getattr(self, "_mg_worker", None) is not None:
            _mg = self._mg_worker.sync()
            if _mg is not None:
                self.metagraph = _mg
                return
            bt.logging.warning("metagraph worker returned None; inline sync this cycle")
        self.metagraph.sync(subtensor=self.subtensor)

    def save_state(self):
        """Persist miner state; the base miner keeps none."""
        pass

    def load_state(self):
        """Restore miner state; the base miner keeps none."""
        pass
