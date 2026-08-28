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
Intelligent markets miner neuron: serves FinanceAgentResponse synapses in
response to MarketSimulationStateUpdate queries from the validator.
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
    import typing
    import traceback
    import bittensor as bt

    from taos.common.neurons.miner import BaseMinerNeuron
    from taos.im.protocol import MarketSimulationStateUpdate
    from taos.im.protocol.gentrx import GenTRXAssignment

    # taos.im.protocol.exchange ships in the public release from 0.6.0, but may be absent on older
    # checkouts or partial installs. Sentinel classes keep type hints and isinstance() checks valid
    # without forcing the module to be present; the exchange axon-attach below is
    # gated on the real import succeeding.
    try:
        from taos.im.protocol.exchange import ExchangeStateUpdate, ExchangeAgentResponse
        _HAS_PROTOCOL_EXCHANGE = True
    except ImportError:
        _HAS_PROTOCOL_EXCHANGE = False

        # Sentinels inherit from pydantic.BaseModel so FastAPI route registration
        # with Union[..., ExchangeStateUpdate] succeeds on the public surface.
        from pydantic import BaseModel as _SentinelBase

        class ExchangeStateUpdate(_SentinelBase):  # type: ignore[no-redef]
            pass

        class ExchangeAgentResponse(_SentinelBase):  # type: ignore[no-redef]
            pass

    class Miner(BaseMinerNeuron):
        """
        Intelligent markets miner neuron.

        Attaches synapse handlers for the simulation forward path, the exchange
        forward path, and (when GenTRX is enabled on the agent) the GenTRX
        training-assignment delivery path.
        """
        def __init__(self, config=None):
            super().__init__(config=config)
            # Exchange handler — required when running in exchange mode. Public
            # release omits taos.im.protocol.exchange, so this attach is gated
            # on the real ExchangeStateUpdate being importable; sim-only public
            # deployments simply never receive ExchangeStateUpdate synapses.
            if _HAS_PROTOCOL_EXCHANGE:
                self.axon.attach(
                    forward_fn=self.forward_exchange,
                    blacklist_fn=self.blacklist_forward_exchange,
                    priority_fn=self.priority_forward_exchange,
                )

            # GenTRX: wire chain access onto the agent so that
            # _get_aggregator_store_for_assignment can do on-chain bucket
            # discovery.  The base agent class receives no subtensor/metagraph
            # at construction time, so we set them here after super().__init__()
            # has established the chain connection.
            self.agent.subtensor = self.subtensor
            self.agent.metagraph = self.metagraph
            self.agent.config.netuid = self.config.netuid

            # Re-run model bootstrap now that chain is available.
            _gtx = getattr(self.agent, "_gtx", None)
            if _gtx is not None and _gtx.model is None:
                self.agent._ensure_model_version()

            # GenTRX: attach assignment handler so validators can push training
            # assignments via dendrite.  Ignored silently by non-GenTRX agents
            # (forward checks for _pending_assignment attribute before setting).
            self.axon.attach(
                forward_fn=self.forward_gentrx_assignment,
                blacklist_fn=self.blacklist_gentrx_assignment,
                priority_fn=self.priority_gentrx_assignment,
            )

            # GenTRX: commit S3 bucket credentials on-chain so validators can
            # discover where to fetch our gradients. Soft-fails when env vars
            # not set (non-GenTRX miner) or when the commit fails transiently.
            self._commit_gentrx_bucket()

            # Settlement readiness. Do this last so it is the final thing in the log.
            self._check_settlement_proxy()

        def _check_settlement_proxy(self) -> None:
            """Report settlement readiness: proxy authorisation AND proxy funding.

            Exchange-mode orders reach the matching engine and match there, but a trade
            only becomes real when the validator settles it on chain for the miner. That
            needs two things, and either missing one fails the same silent way:

              authorisation  the miner's COLDKEY must grant a Staking proxy to the
                             exchange's delegate. Only the miner can do this; add_proxy
                             must be signed by their own coldkey.
              funding        the delegate must hold TAO to pay extrinsic fees.

            Both are reported every time, not just whichever is broken, so a miner can see
            the whole picture rather than fixing one and rediscovering the other. Called at
            startup and periodically from the forward path: a proxy revoked or drained
            mid-run leaves a clean startup log behind it and fills simply stop.
            """
            import time as _t
            self._proxy_last_check = _t.time()
            try:
                ck = self.wallet.coldkeypub.ss58_address
            except Exception as exc:
                bt.logging.debug(f"settlement proxy check: no coldkey available ({exc})")
                return

            try:
                proxies = self.subtensor.query_module("Proxy", "Proxies", params=[ck])
                entries = (proxies.value[0] if proxies and proxies.value else []) or []
            except Exception as exc:
                bt.logging.warning(
                    f"Could not verify settlement readiness for coldkey {ck}: {exc}"
                )
                return

            delegates = []
            for e in entries:
                try:
                    if str(e.get("proxy_type")) == "Staking":
                        delegates.append(e.get("delegate"))
                except Exception:
                    continue

            authorised = bool(delegates)
            delegate = delegates[0] if delegates else None
            balance = None
            if delegate:
                try:
                    balance = float(self.subtensor.get_balance(delegate))
                except Exception:
                    balance = None
            funded = bool(balance is not None and balance > 0.0)

            wname = getattr(getattr(self.config, "wallet", None), "name", "<COLDKEY_WALLET>")
            wpath = getattr(getattr(self.config, "wallet", None), "path", "<WALLET_PATH>")
            endpoint = getattr(
                getattr(self.config, "subtensor", None), "chain_endpoint", "<CHAIN_ENDPOINT>"
            )

            ready = authorised and funded
            was_ready = getattr(self, "_proxy_ready", None)
            self._proxy_ready = ready

            if ready:
                if was_ready is False:
                    bt.logging.warning("Settlement is ready again: trades will now settle.")
                bt.logging.info(
                    f"Settlement ready: proxy AUTHORISED ({delegate}), "
                    f"FUNDED ({balance:.4f} TAO)"
                )
                return

            lines = ["", "=" * 78]
            if was_ready is True:
                lines.append("SETTLEMENT BROKE WHILE RUNNING: trades have STOPPED settling.")
            else:
                lines.append("SETTLEMENT NOT READY: this miner cannot settle trades.")
            lines += [
                "",
                "Your orders WILL be accepted and matched by the exchange, and will then",
                "silently fail to settle. No fills will appear against your uid.",
                "",
                f"  Coldkey        : {ck}",
                f"  Authorisation  : {'OK' if authorised else 'MISSING'}"
                + (f" (delegate {delegate})" if delegate else ""),
                "  Funding        : "
                + (f"OK ({balance:.4f} TAO)" if funded else
                   (f"EMPTY ({balance:.4f} TAO)" if balance is not None else "UNKNOWN")),
                "",
            ]
            if not authorised:
                lines += [
                    "FIX 1 of 2, authorisation. Grant a Staking proxy from YOUR coldkey to",
                    "the exchange's delegate. The validator has no authority to do this for",
                    "you. Get the delegate address from the exchange operator, or from",
                    "GET /api/v1/proxy/info (sr25519-signed; field individual_proxy.ss58).",
                    "",
                    "    btcli proxy add \\",
                    "      --delegate <DELEGATE_SS58> \\",
                    "      --proxy-type Staking --delay 0 \\",
                    f"      --wallet-name {wname} \\",
                    f"      --wallet-path {wpath} \\",
                    f"      --network {endpoint} --no-prompt",
                    "",
                ]
            if authorised and not funded:
                lines += [
                    "FIX, funding. The proxy is authorised but holds no TAO, so it cannot",
                    "pay the extrinsic fee that settles your trades. Around 0.5 TAO is ample.",
                    "",
                    "    btcli wallet transfer \\",
                    f"      --dest {delegate} \\",
                    "      --amount 0.5 \\",
                    f"      --wallet-name {wname} \\",
                    f"      --wallet-path {wpath} \\",
                    f"      --network {endpoint} --no-prompt",
                    "",
                ]
            if not authorised:
                lines += [
                    "FIX 2 of 2, funding. After granting the proxy, fund the delegate so it",
                    "can pay extrinsic fees:",
                    "",
                    "    btcli wallet transfer --dest <DELEGATE_SS58> --amount 0.5 \\",
                    f"      --wallet-name {wname} --wallet-path {wpath} \\",
                    f"      --network {endpoint} --no-prompt",
                    "",
                ]
            lines += [
                "Verify: this miner logs 'Settlement ready: proxy AUTHORISED ... FUNDED'",
                "once both are in place. It re-checks every 5 minutes while running.",
                "=" * 78,
            ]
            bt.logging.error("\n".join(lines))

        def _commit_gentrx_bucket(self) -> None:
            """Commit GenTRX S3 bucket credentials on-chain.

            Skips silently if GENTRX_AGENT_S3_BUCKET is not set (miner not
            participating in GenTRX). Warns on commit failure (retry next start).
            """
            try:
                from GenTRX.src.chain import BucketInfo, GenTRXChain
            except ImportError:
                bt.logging.debug("GenTRX not installed — skipping bucket commitment")
                return

            bucket_info = BucketInfo.from_env()
            if bucket_info is None:
                bt.logging.info(
                    "GenTRX env vars not set — skipping bucket commitment "
                    "(miner not participating in GenTRX)"
                )
                return

            try:
                chain = GenTRXChain(self.subtensor, self.config.netuid, self.metagraph)
                chain.commit_bucket(self.wallet, bucket_info)
                bt.logging.info(
                    f"GenTRX bucket committed on-chain: account={bucket_info.account_id}"
                )
            except Exception as exc:
                bt.logging.warning(
                    f"GenTRX bucket commitment failed (will retry on next start): {exc}"
                )

        async def forward_gentrx_assignment(
            self, synapse: GenTRXAssignment
        ) -> GenTRXAssignment:
            """Receive a GenTRX training assignment from the validator."""
            try:
                bt.logging.info(
                    f"[GTX] assignment received: round={synapse.round} "
                    f"model_version={synapse.model_version} "
                    f"books={synapse.books} "
                    f"data={len(synapse.data)} files "
                    f"ts={synapse.ts_start}..{synapse.ts_end} "
                    f"validator_uid={synapse.validator_uid} "
                    f"source={synapse.data_source}"
                )
                gtx = getattr(self.agent, "_gtx", None)
                if gtx is None:
                    bt.logging.debug("[GTX] agent has no _gtx — not a GenTRX agent, ignoring assignment")
                    return synapse
                gtx.pending_assignments.append({
                    "round":         synapse.round,
                    "model_version": synapse.model_version,
                    "books":         synapse.books,
                    "ts_start":      synapse.ts_start,
                    "ts_end":        synapse.ts_end,
                    "data":          synapse.data,
                    "data_source":   synapse.data_source,
                    "data_endpoint": synapse.data_endpoint,
                    "data_bucket":   synapse.data_bucket,
                    "data_access_key": synapse.data_access_key,
                    "data_secret_key": synapse.data_secret_key,
                    "validator_uid": synapse.validator_uid,
                    "advice":        synapse.advice,
                })
                bt.logging.info(
                    f"[GTX] assignment queued: round={synapse.round} "
                    f"pending={len(gtx.pending_assignments)}"
                )
            except Exception:
                bt.logging.error(f"[GTX] forward_gentrx_assignment failed:\n{traceback.format_exc()}")
                raise
            return synapse

        def blacklist_gentrx_assignment(
            self, synapse: GenTRXAssignment
        ) -> typing.Tuple[bool, str]:
            return self.blacklist(synapse)

        def priority_gentrx_assignment(self, synapse: GenTRXAssignment) -> float:
            return self.priority(synapse)

        async def forward(
            self, synapse: MarketSimulationStateUpdate
        ) -> MarketSimulationStateUpdate:
            """
            Processes incoming market simulation state synapse by forwarding to the associated agent class for handling.

            Args:
                synapse (taos.im.protocol.MarketSimulationStateUpdate): The synapse object containing the latest simulation state update.

            Returns:
                taos.im.protocol.MarketSimulationStateUpdate: The synapse object with the 'response' field updated with any instructions generated by the agent.
            """
            start = time.time()
            synapse.decompress(lazy=self.config.agent.params.lazy_load)
            bt.logging.info(f"Decompressed ({time.time() - start}s)")
            try:
                synapse.response = self.agent.handle(synapse)
            except Exception as e:
                bt.logging.error(f"Agent handle error: {e}\n{traceback.format_exc()}")
                raise
            start = time.time()
            compressed = synapse.clear_inputs().compress()
            bt.logging.debug(f"Compressed ({time.time() - start}s)")
            return compressed or synapse

        def blacklist_forward(
            self, synapse: MarketSimulationStateUpdate
        ) -> typing.Tuple[bool, str]:
            """
            Apply default blacklisting to all received market simulation state synapses.
            
            Args:
                synapse (taos.im.protocol.MarketSimulationStateUpdate): The synapse object containing the latest simulation state update.

            Returns:
                (bool, str): Tuple containing [1] boolean indicating if the request was blacklisted [2] string containing the message indicating reason for blacklisting.
            """
            return self.blacklist(synapse)
        
        def priority_forward(self, synapse: MarketSimulationStateUpdate) -> float:
            """
            Apply default prioritization to all received simulation state synapses.
            
            Args:
                synapse (taos.im.protocol.MarketSimulationStateUpdate): The synapse object containing the latest simulation state update.

            Returns:
                float: A priority score calculated using the standard priority function.
            """
            return self.priority(synapse)

        async def forward_exchange(self, synapse: ExchangeStateUpdate) -> ExchangeStateUpdate:
            # Re-verify settlement readiness periodically, not just at startup: a proxy
            # revoked mid-run leaves the miner trading into a void with a clean startup log
            # behind it. One chain read every 5 minutes is negligible against a per-block
            # synapse, and it is what turns a silent failure into a stated one.
            try:
                if time.time() - getattr(self, "_proxy_last_check", 0) > 300:
                    self._check_settlement_proxy()
            except Exception as exc:
                bt.logging.debug(f"periodic settlement proxy check failed: {exc}")
            start = time.time()
            synapse.decompress(lazy=self.config.agent.params.lazy_load)
            bt.logging.info(f"Decompressed ({time.time() - start}s)")
            try:
                synapse.response = self.agent.handle(synapse)
            except Exception as e:
                bt.logging.error(f"Agent handle error: {e}\n{traceback.format_exc()}")
                raise
            start = time.time()
            compressed = synapse.clear_inputs().compress()
            bt.logging.debug(f"Compressed ({time.time() - start}s)")
            return compressed or synapse

        def blacklist_forward_exchange(self, synapse: ExchangeStateUpdate) -> typing.Tuple[bool, str]:
            return self.blacklist(synapse)

        def priority_forward_exchange(self, synapse: ExchangeStateUpdate) -> float:
            return self.priority(synapse)

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            time.sleep(5)
