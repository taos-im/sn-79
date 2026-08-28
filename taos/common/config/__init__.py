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

import os

os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "false")

import argparse
import bittensor as bt
from loguru import logger

from taos.common.utils.prometheus import prometheus


class ParseKwargs(argparse.Action):
    "Handles parsing of arbitrary agent parameters"
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, argparse.Namespace())
        for value in values:
            key, value = value.split('=')
            try:
                value = float(value)
            except (TypeError, ValueError):
                pass
            setattr(getattr(namespace, self.dest), key, value)

def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        logger.level("EVENTS", no=38, icon="📝")
        logger.add(
            os.path.join(config.neuron.full_path, "events.log"),
            rotation=config.neuron.events_retention_size,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            level="EVENTS",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default="cpu",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock neuron and all network components.",
        default=False,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default="2 GB",
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )
    
    parser.add_argument(
        "--alerting.pagerduty.integration_key",
        type=str,
        help="Integration key to enable triggering PagerDuty alerts for critical validation or mining errors.",
        default=None,
    )

def add_miner_args(cls, parser):
    """Add miner specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="miner",
    )

    parser.add_argument(
        "--blacklist.allow_non_validators",
        action="store_true",
        help="If set, we will force incoming requests to have a permit.",
        default=False,
    )

    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        help="If set, miners will accept queries from non registered entities. (Dangerous!)",
        default=False,
    )

    parser.add_argument(
        "--agent.path",
        type=str,
        help="Path where simulation agent logic files are located.",
        default="../../agents",
    )

    parser.add_argument(
        "--agent.name",
        type=str,
        help="Name of the agent (must correspond to file and class name in agent.path directory). ",
        default="RandomMakerAgent",
    )

    parser.add_argument(
        "--agent.params",
        nargs='*',
        action=ParseKwargs,
        default=argparse.Namespace(),
        help="Arbitrary user-defined parameters relevant to their specific agent implementation.  Pass in format `--agent.params p_0=x p_1=y ...`"
    )


def add_validator_args(cls, parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="validator",
    )

    parser.add_argument(
        "--neuron.reset",
        action="store_true",
        help="Skips state loading.",
        default=False,
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=3.0,
    )

    parser.add_argument(
        "--neuron.global_query_timeout",
        type=float,
        help="The hard wall-clock timeout for the entire dendrite query process to complete.",
        default=4.0,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="The number of concurrent forwards running at any time.",
        default=1,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.008298755,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument(
        "--neuron.burn_uid",
        type=int,
        help="Uid to assign weights to in order to burn emissions.",
        default=0,
    )

    parser.add_argument(
        "--neuron.burn_ratio",
        type=float,
        help="Ratio of miner emissions to burn.",
        default=0.0,
    )


def _prune_default_shadow_configs(cfg, parser):
    """bittensor >=10.3 builds every dotted-arg sub-namespace as a full
    Config() instance, which self-populates the standard default sections
    (axon/logging/priority/subtensor/wallet + config/strict/no_version_checking)
    at EVERY nesting level — so config.prometheus, config.neuron, etc. each
    shadow a complete default bittensor config with wrong values (wallet
    'default', finney endpoint) and the boot "Config:" dump explodes
    combinatorially. Prune every nested key the parser did not actually
    define, keeping intermediate groups on the path to real args."""
    legit = set()
    for action in parser._actions:
        if action.dest and action.dest != "help":
            parts = action.dest.split(".")
            for i in range(1, len(parts) + 1):
                legit.add(tuple(parts[:i]))
    for key in ("config", "strict", "no_version_checking"):
        legit.add((key,))

    def walk(node, path):
        """Walk a config tree depth-first, yielding each node with its path.

        Args:
            node: The current subtree.
            path: Dotted path to ``node``.
        """
        if not isinstance(node, bt.Config):
            return
        for key in list(node.keys()):
            if key.startswith("_"):
                continue
            sub_path = path + (key,)
            if sub_path not in legit:
                if path:
                    del node[key]
                continue
            walk(node[key], sub_path)

    walk(cfg, ())


def _build_parser(cls):
    parser = argparse.ArgumentParser()
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.Axon.add_args(parser)
    prometheus.add_args( parser )
    cls.add_args(parser)
    return parser


def prune_config(cfg, cls):
    """Re-prune the default-config shadows for a neuron class's config.
    bt.Config.merge() rebuilds nested sections as fresh Config() instances,
    re-injecting the default shadows — so this must run again after any
    merge (see BaseNeuron.__init__), not only at construction."""
    _prune_default_shadow_configs(cfg, _build_parser(cls))


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = _build_parser(cls)
    cfg = bt.Config(parser)
    _prune_default_shadow_configs(cfg, parser)
    return cfg
