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
Benchmark miner neuron: runs a deterministic reference agent with a fixed
validator-assigned UID (>= 256), bypassing on-chain registration.
"""

# Cap glibc malloc arenas to bound RSS growth. Each training-cycle thread
# otherwise gets its own arena (default 8×CPU) and arenas are never returned
# to the OS, so per-cycle thread creation leaks ~1.2 GB of anonymous CPU
# memory per round. Re-exec once with MALLOC_ARENA_MAX set so glibc sees it
# before allocating any arenas. Only triggers on direct script execution.
if __name__ == "__main__":
    import os
    import sys
    if os.environ.get("MALLOC_ARENA_MAX") is None:
        os.environ["MALLOC_ARENA_MAX"] = "2"
        os.execvp(sys.executable, [sys.executable] + sys.argv)

if __name__ != "__mp_main__":
    import time
    import copy
    import os
    import asyncio
    import threading
    import traceback
    import importlib.util
    import bittensor as bt

    from taos.common.config import loggable_config
    from taos.common.neurons import BaseNeuron
    from taos.im.neurons.miner import Miner, _HAS_PROTOCOL_EXCHANGE

    class BenchmarkMiner(Miner):
        """
        Benchmark Miner class implementation for intelligent market simulations.
        
        Unlike regular miners, benchmark miners:
        - Do not require registration on-chain
        - Are assigned a fixed UID (>= 256) by the validator
        - Skip metagraph lookups for UID assignment
        - Skip registration checks and related blockchain operations
        """
        
        @classmethod
        def add_args(cls, parser):
            """Add benchmark-specific arguments."""
            super().add_args(parser)
            parser.add_argument(
                '--benchmark.uid',
                type=int,
                default=None,
                help='Fixed UID for this benchmark agent (assigned by validator)'
            )
        
        def __init__(self, config=None, benchmark_uid: int = None):
            """
            Initialize benchmark miner with a fixed UID.
            
            Args:
                config: Configuration object
                benchmark_uid: Fixed UID assigned by validator (overrides config)
            """
            base_config = copy.deepcopy(config or BaseNeuron.config())
            self.config = self.config()
            self.config.merge(base_config)
            self.check_config(self.config)

            if benchmark_uid is not None:
                self.uid = benchmark_uid
            elif hasattr(self.config, 'benchmark') and hasattr(self.config.benchmark, 'uid') and self.config.benchmark.uid is not None:
                self.uid = self.config.benchmark.uid
            elif 'BENCHMARK_UID' in os.environ:
                self.uid = int(os.environ['BENCHMARK_UID'])
            else:
                raise ValueError(
                    "Benchmark miner requires a UID. Provide via:\n"
                    "  - Constructor: BenchmarkMiner(benchmark_uid=256)\n"
                    "  - CLI: --benchmark.uid 256\n"
                    "  - Environment: BENCHMARK_UID=256"
                )
            
            bt.logging.info(f"Benchmark miner assigned UID: {self.uid}")

            self.device = self.config.neuron.device
            bt.logging.info(f"Config:\n{loggable_config(self.config)}")

            bt.logging.info("Setting up bittensor objects:")

            self.wallet = bt.Wallet(
                path=self.config.wallet.path,
                name=self.config.wallet.name,
                hotkey=self.config.wallet.hotkey
            )
            # When chain_endpoint is set, pass it positionally — bt.Subtensor's
            # `network` arg accepts URLs and uses them directly. Using
            # `config=self.config` instead lets `config.subtensor.network` (often
            # default 'finney' or a non-matching alias) override chain_endpoint
            # with the alias's hardcoded URL, breaking custom-endpoint deploys.
            # GenTRX network detection has its own netuid-based fallback in
            # agents/gentrx.py for the "subtensor.network unknown" case.
            _chain_ep = (
                getattr(self.config.subtensor, "chain_endpoint", None) or ""
            )
            if _chain_ep:
                self.subtensor = bt.Subtensor(network=_chain_ep)
            else:
                self.subtensor = bt.Subtensor(config=self.config)

            self.metagraph = self.subtensor.metagraph(self.config.netuid)

            bt.logging.info(f"Wallet: {self.wallet}")
            bt.logging.info(f"Subtensor: {self.subtensor}")
            bt.logging.info(f"Metagraph: {self.metagraph}")
           
            # Retrieve the current block number
            self.update_block()
            # Retrieve the current subnet hyperparameters
            self.update_hyperparams()
            
            bt.logging.info(
                f"Benchmark Miner Running! Subnet: {self.config.netuid} | "
                f"Fixed UID: {self.uid} | "
                f"Endpoint: {self.subtensor.chain_endpoint}"
            )
            self.step = 0
            
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
            # Exchange-mode handler — gate on real taos.im.protocol.exchange
            # being importable. The public release ships a pydantic sentinel
            # ExchangeStateUpdate (miner.py:61) that does NOT inherit from
            # bittensor.Synapse; unconditionally attaching would trip
            # `assert issubclass(param_class, Synapse)` inside axon.attach.
            # Miner.__init__ already gates its own exchange attach the same way.
            if _HAS_PROTOCOL_EXCHANGE:
                self.axon.attach(
                    forward_fn=self.forward_exchange,
                    blacklist_fn=self.blacklist_forward_exchange,
                    priority_fn=self.priority_forward_exchange,
                )
            # Attach GenTRX assignment handler (inherited from Miner).
            # Benchmark miners participate in GenTRX training when
            # GENTRX_AGENT_S3_* env vars are set; handler is a no-op otherwise.
            try:
                from taos.im.protocol.gentrx import GenTRXAssignment  # noqa: F401  (import is the GenTRX-installed probe)
                self.axon.attach(
                    forward_fn=self.forward_gentrx_assignment,
                    blacklist_fn=self.blacklist_gentrx_assignment,
                    priority_fn=self.priority_gentrx_assignment,
                )
                bt.logging.info("GenTRX assignment handler attached to benchmark axon.")
            except ImportError:
                bt.logging.debug("GenTRX not installed — skipping assignment handler.")
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

            # Wire chain access onto the agent for GenTRX chain discovery.
            self.agent.subtensor = self.subtensor
            self.agent.metagraph = self.metagraph
            self.agent.config.netuid = self.config.netuid

            # Re-run model bootstrap now that chain is available.
            # initialize() already attempted this but subtensor wasn't wired yet.
            _gtx = getattr(self.agent, "_gtx", None)
            if _gtx is not None and _gtx.model is None:
                self.agent._ensure_model_version()

        def check_registered(self):
            """
            Override registration check - benchmark miners don't need registration.
            """
            bt.logging.debug("Skipping registration check for benchmark miner")
            return True

        def sync(self, save_state=True):
            """
            Wrapper for synchronizing the state of the network for the given miner or validator.
            """
            self.update_block()
            self.update_hyperparams()

            if self.should_sync_metagraph():
                self.resync_metagraph()
            if save_state:
                self.save_state()
        
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
            self.sync() 
            self.axon.start()
            bt.logging.info(f"Benchmark Miner starting at block {self.block} with UID {self.uid}")
            # This loop maintains the miner's operations until intentionally stopped.
            try:
                while not self.should_exit:
                    # Check if we should exit.
                    if self.should_exit:
                        break
                    # Sync metagraph
                    self.sync()
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


# This is the main function, which runs the benchmark miner.
if __name__ == "__main__":
    with BenchmarkMiner() as miner:
        while True:
            time.sleep(5)