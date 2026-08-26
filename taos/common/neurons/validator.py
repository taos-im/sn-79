# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
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
Base validator neuron: metagraph sync, weight setting, and two-pool reward
allocation combining trading (Kappa-3) and GenTRX training scores.
"""

import copy
import torch
import asyncio
import argparse
import threading
import bittensor as bt

from typing import List, Optional, Tuple


def compute_two_pool_allocation(
    trading_scores: torch.Tensor,
    gentrx_scores: torch.Tensor,
    *,
    burn_ratio: float,
    gentrx_simulation_share: float,
    n_target: int,
    burn_uid: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute on-chain weights from per-UID trading and gentrx score vectors.

    Miner rewards split between two pools: trading (kappa+pnl) and training
    (gentrx). `gentrx_simulation_share` caps the training share at full
    participation; the actual training allocation scales by
    `n_active / n_target`, and any unused training share returns to trading.

    Returns:
        (raw_weights, summary) — the L1-normalized weight vector and a
        breakdown of the pool allocation suitable for logging or inspection.
    """
    trading_scores = torch.nan_to_num(trading_scores, 0.0)
    gentrx_scores = torch.nan_to_num(gentrx_scores, 0.0)

    trading_for_norm = trading_scores
    if trading_for_norm.numel() > 0 and float(torch.min(trading_for_norm)) < 0:
        trading_for_norm = trading_for_norm - torch.min(trading_for_norm)
    trading_weights = torch.nn.functional.normalize(trading_for_norm, p=1, dim=0)
    gentrx_weights = torch.nn.functional.normalize(gentrx_scores, p=1, dim=0)

    burn_ratio = max(0.0, min(1.0, float(burn_ratio or 0.0)))
    gentrx_sim_share = max(0.0, min(1.0, float(gentrx_simulation_share or 0.0)))
    n_active = int((gentrx_scores > 0).sum().item())
    shrink = (n_active / n_target) if n_target > 0 else 0.0
    shrink = max(0.0, min(1.0, shrink))

    simulation_pool = 1.0 - burn_ratio
    gentrx_alloc = simulation_pool * gentrx_sim_share * shrink
    trading_alloc = simulation_pool - gentrx_alloc
    burn_alloc = burn_ratio

    raw_weights = (
        trading_alloc * trading_weights
        + gentrx_alloc * gentrx_weights
    )

    if burn_uid is not None and 0 <= burn_uid < raw_weights.numel() and burn_alloc > 0:
        raw_weights[burn_uid] = raw_weights[burn_uid] + burn_alloc

    raw_weights = torch.nn.functional.normalize(raw_weights, p=1, dim=0)

    summary = {
        'simulation_pool': simulation_pool,
        'trading_alloc': trading_alloc,
        'gentrx_alloc': gentrx_alloc,
        'burn_alloc': burn_alloc,
        'n_active': n_active,
        'n_target': n_target,
        'shrink': shrink,
        'gentrx_simulation_share': gentrx_sim_share,
        'burn_ratio': burn_ratio,
    }
    return raw_weights, summary


def resync_inputs_unchanged(previous_axons, current_axons, remembered_hotkeys, current_hotkeys):
    """Whether resync_metagraph can return early without missing a uid slot changing hands.

    Covers the hotkeys as well as the axons: a uid changing occupant is detected by comparing the
    remembered hotkeys against the metagraph's, and an axon-only test misses the case where the new
    occupant publishes the axon info the departed miner had -- the same operator re-registering on the
    same box and port. handle_deregistration must still run there.

    Separate from resync_metagraph's body so it can be tested directly.

    Args:
        previous_axons: Axons as of the last resync.
        current_axons: Axons on the fresh metagraph.
        remembered_hotkeys: Hotkeys as of the last resync.
        current_hotkeys: Hotkeys on the fresh metagraph.

    Returns:
        bool: True when nothing changed hands and resync can return early.
    """
    return (
        previous_axons == current_axons
        and len(remembered_hotkeys) == len(current_hotkeys)
        and list(remembered_hotkeys) == list(current_hotkeys)
    )


from abc import abstractmethod

from taos.common.neurons import BaseNeuron
from taos.mock import MockDendrite
from taos.common.config import add_validator_args

import taos.common.utils.weights as weight_utils


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """Add validator CLI arguments to the parser."""
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.deregistered_uids = []

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.Dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # `self.scores` holds the slow-EMA of trading rewards (kappa+pnl
        # after Pareto). `self.gentrx_scores` holds the slow-EMA of the
        # GenTRX rank-normalized score (no Pareto). Combined in prepare_weights.
        self.scores = torch.zeros(
            self.metagraph.n, dtype=torch.float32, device=self.device
        )
        self.gentrx_scores = torch.zeros(
            self.metagraph.n, dtype=torch.float32, device=self.device
        )
        self.pending_weights = []
        self.last_commit = None

        # Init sync with the network. Updates the metagraph.
        self.sync(save_state=False)

        # Serve axon to enable external connections.
        _observe = getattr(getattr(self.config, 'neuron', None), 'observe', False)
        if not self.config.neuron.axon_off and not _observe:
            self.serve_axon()
        else:
            if _observe:
                bt.logging.info("Observe mode — skipping axon registration.")
            else:
                bt.logging.warning("`neuron.axon_off=True` - IP will not be served to chain.")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        bt.logging.info(f"Validator Started on net{self.config.netuid}! Address : {self.dendrite.keypair.ss58_address} | Stake : {self.metagraph.stake[self.uid]}τ | Alpha Stake : {self.subtensor.get_stake_for_hotkey(self.dendrite.keypair.ss58_address, self.config.netuid)}")

    def serve_axon(self):
        """Serve axon to enable external connections and advertise the IP of the validator to allow scraping of metrics."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.Axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass

    async def concurrent_forward(self):
        """Run the configured number of forward passes concurrently."""
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    # The `run` function is not used by this subnet since the validator is launched as a FastAPI client in order to receive communications from the simulator.
    def run(self):
        """The validator main loop: sync, forward, score and persist until stopped."""
        pass

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def prepare_weights(self):
        """Build the on-chain weight vector via the trading / training split.

        Delegates the math to `compute_two_pool_allocation` so the same
        function is exercised by the standalone scoring inspector.
        """
        network_trading_scores = self.scores[:self.metagraph.n]
        network_gentrx_scores = self.gentrx_scores[:self.metagraph.n]

        if torch.isnan(network_trading_scores).any():
            bt.logging.warning("Trading scores contain NaN values. Replacing with 0.")
        if torch.isnan(network_gentrx_scores).any():
            bt.logging.warning("GenTRX scores contain NaN values. Replacing with 0.")

        bt.logging.debug(f"Processing trading scores: {network_trading_scores}")
        bt.logging.debug(f"Processing gentrx scores:  {network_gentrx_scores}")

        burn_uid = getattr(self.config.neuron, 'burn_uid', None)
        burn_ratio = getattr(self.config.neuron, 'burn_ratio', 0.0) or 0.0

        scoring_cfg = getattr(self.config, 'scoring', None)
        gentrx_cfg = getattr(scoring_cfg, 'gentrx', None) if scoring_cfg is not None else None
        gentrx_sim_share = getattr(gentrx_cfg, 'simulation_share', 0.0) if gentrx_cfg is not None else 0.0

        get_n_target = getattr(self, 'get_n_target_miners', None)
        n_target = get_n_target() if callable(get_n_target) else self.metagraph.n

        if burn_uid is not None and burn_uid >= self.metagraph.n:
            bt.logging.warning(
                f"Burn UID {burn_uid} is out of range (metagraph size: {self.metagraph.n}). "
                f"Burn allocation will be lost."
            )
            burn_uid_arg = None
        else:
            burn_uid_arg = burn_uid

        raw_weights, summary = compute_two_pool_allocation(
            network_trading_scores,
            network_gentrx_scores,
            burn_ratio=burn_ratio,
            gentrx_simulation_share=gentrx_sim_share,
            n_target=n_target,
            burn_uid=burn_uid_arg,
        )

        bt.logging.info(
            f"Pool allocation: trading={summary['trading_alloc']*100:.4f}% "
            f"gentrx={summary['gentrx_alloc']*100:.4f}% "
            f"(sim_share={summary['gentrx_simulation_share']*100:.2f}%, "
            f"N={summary['n_active']}/{summary['n_target']}, "
            f"shrink={summary['shrink']:.4f}) "
            f"burn={summary['burn_alloc']*100:.4f}%"
        )
        if burn_uid is not None and 0 <= burn_uid < raw_weights.numel():
            bt.logging.debug(f"Post-burn weight for UID {burn_uid}: {raw_weights[burn_uid]:.6f}")

        bt.logging.debug(f"raw_weights={raw_weights}")
        bt.logging.debug(f"raw_weight_uids={self.metagraph.uids}")
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights.to("cpu").numpy(),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug(f"processed_weights={processed_weights}")
        bt.logging.debug(f"processed_weight_uids={processed_weight_uids}")

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug(f"uint_weights={uint_weights}")
        bt.logging.debug(f"uint_uids={uint_uids}")
        return uint_uids, uint_weights

    def set_weights(self):
        """
        Weight setting function
        """
        if getattr(self.config.neuron, 'observe', False):
            bt.logging.info("Observe mode — skipping weight submission")
            return

        # Get relevant hyperparameters
        commit_reveal_weights_enabled = bool(self.hyperparams.commit_reveal_weights_enabled)
        mechid = getattr(self.config.neuron, 'mechid', None)
        if mechid is None:
            mechid = 1 if getattr(self.config, 'engine', 'simulation') == 'exchange' else 0
        # Prepare weights for submission
        uint_uids, uint_weights = self.prepare_weights()
        bt.logging.info(f"`commit_reveal_weights_enabled` : {commit_reveal_weights_enabled}")
        bt.logging.info(f"`mechid` : {mechid}")
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            mechid=mechid,
            wait_for_inclusion=False,
            wait_for_finalization=False,
            version_key=self.spec_version,
        )
        return result

    @abstractmethod
    def handle_deregistration(self, uid):
        """
        Abstract method to enable specific handling of hotkey deregistration by the validator
        """
        ...

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.trace("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.       
        bt.logging.debug("Syncing metagraph...")
        self.metagraph.sync(subtensor=self.subtensor)

        if resync_inputs_unchanged(
            previous_metagraph.axons, self.metagraph.axons, self.hotkeys, self.metagraph.hotkeys
        ):
            bt.logging.debug("No axon changes!")
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.handle_deregistration(uid)

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages for both
        # the trading and gentrx score vectors.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            bt.logging.debug("Handling new hotkeys...")
            new_trading = torch.zeros((self.metagraph.n)).to(self.device)
            min_len = min(len(self.hotkeys), len(self.scores))
            new_trading[:min_len] = self.scores[:min_len]
            self.scores = new_trading

            new_gentrx = torch.zeros((self.metagraph.n)).to(self.device)
            min_len_g = min(len(self.hotkeys), len(self.gentrx_scores))
            new_gentrx[:min_len_g] = self.gentrx_scores[:min_len_g]
            self.gentrx_scores = new_gentrx

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(
        self,
        trading_rewards: torch.FloatTensor,
        uids: List[int],
        gentrx_rewards: torch.FloatTensor = None,
    ):
        """EMAs trading and gentrx rewards independently into self.scores and self.gentrx_scores.

        `trading_rewards` is the post-Pareto trading reward vector; `gentrx_rewards`
        is the rank-norm + per-UID EMA gentrx vector (no Pareto). Both apply the
        same slow `moving_average_alpha`. `gentrx_rewards=None` is treated as a
        zero vector (gentrx pool dormant).

        Args:
            trading_rewards: Post-Pareto trading reward vector.
            uids: Uids the vectors are aligned to.
            gentrx_rewards: Rank-normalised, per-uid-EMA GenTRX vector.
        """
        bt.logging.debug("Updating Scores...")
        if torch.isnan(trading_rewards).any():
            bt.logging.warning(f"NaN values detected in trading rewards: {trading_rewards}")
            trading_rewards = torch.nan_to_num(trading_rewards, 0)
        if gentrx_rewards is not None and torch.isnan(gentrx_rewards).any():
            bt.logging.warning(f"NaN values detected in gentrx rewards: {gentrx_rewards}")
            gentrx_rewards = torch.nan_to_num(gentrx_rewards, 0)

        bt.logging.debug("Cloning UIDs...")
        if isinstance(uids, torch.Tensor):
            uids_tensor = uids.clone().detach().to(self.device)
        else:
            uids_tensor = torch.tensor(uids).to(self.device)

        alpha: float = self.config.neuron.moving_average_alpha

        bt.logging.debug("Scattering trading rewards...")
        scattered_trading: torch.FloatTensor = self.scores.scatter(
            0, uids_tensor, trading_rewards
        ).to(self.device)
        self.scores: torch.FloatTensor = alpha * scattered_trading + (
            1 - alpha
        ) * self.scores.to(self.device)
        bt.logging.debug(f"Updated trading MA scores: {self.scores}")

        if gentrx_rewards is not None:
            bt.logging.debug("Scattering gentrx rewards...")
            scattered_gentrx: torch.FloatTensor = self.gentrx_scores.scatter(
                0, uids_tensor, gentrx_rewards
            ).to(self.device)
            self.gentrx_scores: torch.FloatTensor = alpha * scattered_gentrx + (
                1 - alpha
            ) * self.gentrx_scores.to(self.device)
            bt.logging.debug(f"Updated gentrx MA scores: {self.gentrx_scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.trace("Saving validator state.")

        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "gentrx_scores": self.gentrx_scores,
                "hotkeys": self.hotkeys,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        state = torch.load(self.config.neuron.full_path + "/state.pt", weights_only=False)
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
        self.gentrx_scores = state.get(
            "gentrx_scores", torch.zeros_like(self.scores)
        )
