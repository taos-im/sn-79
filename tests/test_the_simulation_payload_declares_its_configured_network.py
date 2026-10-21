# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A simulation validator's ingest payload names its network when the operator configures one.

--simulation.network takes the deployment's network as its launcher names it (local, testnet,
mainnet), so a receiving data service can tell a payload meant for another network. Unset, the
payload is exactly what it was before the option existed: no network key at all, so a validator
whose launcher does not pass the option is unaffected.
"""
import argparse
from types import SimpleNamespace as NS

import pytest

from taos.im.config.simulation import add_simulation_args
from taos.im.neurons.validator import Validator


class _Self:
    def __init__(self, sim_cfg):
        self.config = NS(simulation=sim_cfg)
        self.simulation = NS(simulation_id="s1")
        self.engine = NS()
        self.wallet = None
        self.benchmark_agents = []
        self.metagraph = None

    def _build_agent_scoring_maps(self):
        return {}


def _payload(sim_cfg):
    state = NS(notices={}, accounts={}, books={}, pools={}, timestamp=1, block=2)
    return Validator._build_sim_push_payload(_Self(sim_cfg), state)


@pytest.mark.parametrize("network", ["local", "testnet", "mainnet"])
def test_a_configured_network_is_declared(network):
    assert _payload(NS(network=network))["network"] == network


@pytest.mark.parametrize("sim_cfg", [NS(), NS(network=None), NS(network="")], ids=["absent", "none", "empty"])
def test_an_unset_network_leaves_the_payload_without_the_key(sim_cfg):
    assert "network" not in _payload(sim_cfg)


def test_the_option_exists_and_defaults_to_unset():
    parser = argparse.ArgumentParser()
    add_simulation_args(None, parser)
    assert getattr(parser.parse_args([]), "simulation.network") is None
    assert getattr(parser.parse_args(["--simulation.network", "testnet"]), "simulation.network") == "testnet"
