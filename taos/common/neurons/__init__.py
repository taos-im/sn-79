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

import copy
import typing

import bittensor as bt

from abc import ABC, abstractmethod
from threading import Thread, Lock

# Sync calls set weights and also resyncs the metagraph.
from taos.common.config import check_config, add_args, config, prune_config
from taos.common.utils.misc import ttl_get_block
from taos.common.utils.pagerduty import triggerPagerDutyIncident, resolvePagerDutyIncident
from taos import __spec_version__ as spec_version
from taos.mock import MockSubtensor, MockMetagraph
from taos.common.utils.subnet_hyperparameters import get_subnet_hyperparameters


class BaseNeuron(ABC):
    """
    Base class for τaos subnet neurons. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    neuron_type: str = "BaseNeuron"

    @classmethod
    def check_config(cls, config: "bt.Config"):
        """Validate the merged config before the neuron starts."""
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        """Add this neuron's CLI arguments to the parser."""
        add_args(cls, parser)

    @classmethod
    def config(cls):
        """Build the neuron's config from its argument parser."""
        return config(cls)

    subtensor: "bt.Subtensor"
    wallet: "bt.Wallet"
    metagraph: "bt.Metagraph"
    spec_version: int = spec_version

    @property
    def block(self):
        """The current chain block number."""
        return ttl_get_block(self)

    def __init__(self, config=None):
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        # merge() rebuilds nested sections as fresh Config() instances, which
        # re-inject bittensor's default-config shadows — prune again before
        # the config is checked and logged.
        prune_config(self.config, type(self))
        self.check_config(self.config)

        # Apply parsed log level — bt.logging starts in Default/WARNING state
        # at import time; must be transitioned explicitly before any .info() calls.
        _lc = getattr(self.config, 'logging', None)
        if getattr(_lc, 'trace', False):
            bt.logging.set_trace()
        elif getattr(_lc, 'debug', False):
            bt.logging.set_debug()
        else:
            bt.logging.set_info()

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        bt.logging.info(f"Config:\n{self.config}")

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects:")

        # The wallet holds the cryptographic key pairs for the miner.
        if self.config.mock:
            self.wallet = bt.MockWallet(config=self.config)
            self.subtensor = MockSubtensor(
                self.config.netuid, wallet=self.wallet
            )
            self.metagraph = MockMetagraph(
                self.config.netuid, subtensor=self.subtensor
            )
        else:
            self.wallet = bt.Wallet(
                    path=self.config.wallet.path,
                    name=self.config.wallet.name,
                    hotkey=self.config.wallet.hotkey
                )
            # Pass chain_endpoint positionally (it's the `network` arg of
            # bt.Subtensor.__init__, which accepts URLs directly) when set.
            # Using `bt.Subtensor(config=self.config)` instead causes bittensor
            # 10.3.x to resolve via `config.subtensor.network` first — when
            # that's a known alias (default 'finney' or anything in the alias
            # table like 'local'), it overrides chain_endpoint with the alias's
            # hardcoded URL and yields ConnectionError on the wrong host.
            _chain_ep = (
                getattr(self.config.subtensor, "chain_endpoint", None) or ""
            )
            if _chain_ep:
                self.subtensor = bt.Subtensor(network=_chain_ep)
            else:
                self.subtensor = bt.Subtensor(config=self.config)
            # Metagraph must be mechid-aware: per-mechanism storages (Weights,
            # LastUpdate, Incentive, Bonds) live at storage index
            # netuid + mechid*GLOBAL_MAX_SUBNET_COUNT. Without mechid the
            # should_set_weights guard reads mechid-0 LastUpdate for every
            # engine, so an exchange validator (mechid 1) can never set weights.
            _mechid = getattr(self.config.neuron, "mechid", None)
            if _mechid is None:
                _mechid = 1 if getattr(self.config, "engine", "simulation") == "exchange" else 0
            self.metagraph = self.subtensor.metagraph(self.config.netuid, mechid=_mechid)
            self._mechid = int(_mechid)
            # Weight-setting reads must be mechid-aware. Applied to the primary
            # subtensor here; subclasses that set weights through additional
            # connections (e.g. the im validator's maintenance_subtensor) must
            # call this on those too.
            self._patch_blocks_since_last_update(self.subtensor)

        # bt 10.3.x's substrate websocket is not thread-safe — concurrent
        # ws.recv() from different threads raises ConcurrencyError. Serialize
        # all subtensor calls through this lock (taken inside sync(), set_weights,
        # and any callback that hits self.subtensor from a non-main thread).
        self._subtensor_lock = Lock()

        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Check if the hotkey is registered on the Bittensor network before proceeding further.
        self.check_registered()

        # Each registered hotkey gets a unique identity (UID) in the network for differentiation.
        _hk = self.wallet.hotkey.ss58_address
        self.uid = self.metagraph.hotkeys.index(_hk) if _hk in self.metagraph.hotkeys else -1
        # Retrieve the current block number
        self.update_block()
        # Retrieve the current subnet hyperparameters
        self.update_hyperparams()
        bt.logging.info(
            f"Neuron Running! Subnet : {self.config.netuid} | UID : {self.uid} | Endpoint : {self.subtensor.chain_endpoint}"
        )
        self.step = 0

    @abstractmethod
    def run(self):
        """
        Abstract method for running the neuron.
        """
        ...
    

    @abstractmethod
    def resync_metagraph(self):
        """
        Abstract method for resynchronizing metagraph.
        """
        ...

    def set_weights(self):
        """
        Default method for setting validator weights.
        """
        pass

    def sync(self, save_state=True):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        with self._subtensor_lock:
            # Ensure miner or validator hotkey is still registered on the network.
            self.check_registered()
            # Update block and hyperparameters
            self.update_block()
            self.update_hyperparams()

            if self.should_sync_metagraph():
                self.resync_metagraph()

            if self.should_set_weights():
                self.set_weights()

        if save_state:
            self.save_state()

    def check_registered(self):
        """
        Method to check if the hotkey configured to be used by the neuron is registered in the subnet.
        Skipped in observe mode since no weight submission or axon ops are performed.
        """
        if getattr(getattr(self.config, 'neuron', None), 'observe', False):
            bt.logging.info("Observe mode: skipping registration check.")
            return
        bt.logging.debug("Checking registration...")
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()
        bt.logging.debug(f"Key {self.config.wallet.name}.{self.config.wallet.hotkey} ({self.wallet.hotkey.ss58_address}) is registered.")

    def update_block(self, max_retries=5):
        """
        Method to update `self.current_block` with the latest block number from the chain.
        """
        bt.logging.debug("Getting latest block number...")
        retries = 0
        last_exception = None
        while retries < max_retries:
            try:
                self.current_block = self.subtensor.block                
                bt.logging.debug(f"Current Block : {self.current_block}")
                return True
            except Exception as e:
                last_exception = e
                bt.logging.error(f"Failed to retrieve current block : {e}")
                retries += 1
                continue
        raise(last_exception)

    def update_hyperparams(self):
        """
        Method to update `self.hyperparams` with the latest subnet hyperparameters from the chain.
        """
        bt.logging.debug("Updating Subnet Hyperparams...")
        self.hyperparams = get_subnet_hyperparameters(self.subtensor, self.config.netuid)
        bt.logging.debug(f"Subnet Hyperparams:\n{self.hyperparams}")

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return True

    def _patch_blocks_since_last_update(self, subtensor):
        """Make ``subtensor.blocks_since_last_update`` read this validator's
        mechanism storage index (netuid + mechid*GLOBAL_MAX_SUBNET_COUNT, 4096).

        set_weights writes LastUpdate at the mechanism index, but the SDK reads
        netuid-level LastUpdate (mechid 0). For mechid > 0 both the SDK's own
        set_weights rate-limit guard AND should_set_weights would otherwise gate
        on the wrong mechanism's LastUpdate. Must be applied to EVERY subtensor
        the validator sets weights through — the im validator swaps in
        maintenance_subtensor inside its sync path, so that one needs it too.
        No-op for mechid 0 / mock; delegates to the SDK for other netuids and on
        any query error.
        """
        mechid = int(getattr(self, "_mechid", 0) or 0)
        if not mechid or getattr(self.config, "mock", False) or subtensor is None:
            return
        nid = int(self.config.netuid)
        idx = nid + mechid * 4096
        orig = subtensor.blocks_since_last_update

        def _mechid_blocks_since_last_update(netuid, uid, block=None, _orig=orig, _idx=idx, _nid=nid, _st=subtensor):
            if int(netuid) != _nid:
                return _orig(netuid, uid, block)
            try:
                lu = _st.substrate.query("SubtensorModule", "LastUpdate", [_idx])
                vals = lu.value if lu is not None else None
                last = int(vals[uid]) if vals and uid < len(vals) else 0
                cur = block if block is not None else _st.get_current_block()
                return cur - last
            except Exception:
                return _orig(netuid, uid, block)

        subtensor.blocks_since_last_update = _mechid_blocks_since_last_update
        bt.logging.info(f"blocks_since_last_update → mechanism storage index {idx} (mechid {mechid})")

    def should_set_weights(self) -> bool:
        """
        Method to check whether weights should be set by the neuron at the current block.
        """
        # Don't set weights on initialization or if disabled in config.
        if self.step == 0 or self.config.neuron.disable_set_weights:
            return False

        # Miners never set weights.
        if self.neuron_type == "MinerNeuron":
            return False

        # Gate on the SAME primitive the SDK's set_weights uses
        # (blocks_since_last_update), so taos never says "set" while the SDK's
        # internal rate-limit guard silently skips (or vice-versa). That
        # primitive is made mechid-aware in __init__ for mechid > 0, so an
        # exchange validator checks LastUpdate at its mechanism storage index,
        # not the base netuid. Falls back to the metagraph on any query error.
        try:
            blocks_since = self.subtensor.blocks_since_last_update(self.config.netuid, self.uid)
        except Exception as e:
            bt.logging.warning(f"blocks_since_last_update failed, using metagraph fallback: {e}")
            blocks_since = self.current_block - int(self.metagraph.last_update[self.uid].item())
        bt.logging.trace(
            f"Blocks since last update : {blocks_since} | Rate Limit : {self.hyperparams.weights_rate_limit} | Current Block : {self.current_block}"
        )
        return blocks_since > self.hyperparams.weights_rate_limit

    def save_state(self):
        """
        Method to save (serialize) the state of the neuron (to be implemented by subclasses).
        """
        bt.logging.warning(
            "save_state() not implemented for this neuron."
        )

    def load_state(self):
        """
        Method to load (deserialize) the state of the neuron (to be implemented by subclasses).
        """
        bt.logging.warning(
            "load_state() not implemented for this neuron."
        )

    def pagerduty_alert(self, incident_text : str, method : str = "unknown", dedup_key : str | None = None, details : dict | None = None, event_class : str = "ERROR", severity : str = "error"):
        """Raise a PagerDuty incident.

        Args:
            incident_text (str): Incident summary.
            method (str): Caller context recorded on the incident.
            dedup_key (str | None): Stable key so repeats update one incident instead of opening more.
        """
        bt.logging.error("PD: " + incident_text + (f"\nDetails : {details}" if details else ''))
        if self.config.alerting.pagerduty.integration_key:
            triggerPagerDutyIncident(
                integration_keys=[self.config.alerting.pagerduty.integration_key], 
                source=f"{self.config.wallet.name}:{self.config.wallet.hotkey}:{method}", 
                group=f"{self.config.wallet.name}:{self.config.wallet.hotkey}",
                event_class=event_class, 
                msg=f"τaos {self.config.wallet.name}:{self.config.wallet.hotkey} - {method} : {incident_text}",
                custom_details=details, 
                severity=severity,
                dedup_key=dedup_key
            )
        else:
            bt.logging.debug("pagerduty_alert : PagerDuty is not configured.")
