"""network_from_subtensor: bucket-prefix network resolution.

Regression for the aggregator/validator prefix split — a netuid-2 localnet
resolved to `localnet` for the validator (which passed netuid) but `mainnet`
for the gradient server (which didn't), so their S3 prefixes diverged and the
aggregator never saw miner gradients. The fix: always pass netuid.

Run: pytest GenTRX/tests/test_network_resolution.py -v
"""
import pytest

from GenTRX.src.gradient_store import (
    NETWORK_LOCALNET,
    NETWORK_MAINNET,
    NETWORK_TESTNET,
    network_from_subtensor,
)

_CUSTOM_EP = "ws://198.51.100.7:9945"  # a custom node: not finney, not loopback (RFC5737 doc range)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    # The env override wins over every signal; clear it so we test the
    # endpoint/netuid resolution paths in isolation.
    monkeypatch.delenv("GENTRX_NETWORK", raising=False)


def test_netuid_maps_custom_endpoint_to_localnet():
    """netuid 2 on a custom endpoint → localnet (the deterministic map),
    NOT the 'unknown wss → mainnet' fallback. This is the aggregator fix."""
    assert network_from_subtensor(_CUSTOM_EP, netuid=2) == NETWORK_LOCALNET


def test_netuid_maps_canonical_subnets():
    assert network_from_subtensor(_CUSTOM_EP, netuid=79) == NETWORK_MAINNET
    assert network_from_subtensor(_CUSTOM_EP, netuid=366) == NETWORK_TESTNET


def test_custom_endpoint_without_netuid_falls_back_to_mainnet():
    """Documents the fallback the aggregator hit before the fix: a custom
    wss/ws endpoint with no netuid hint resolves to mainnet."""
    assert network_from_subtensor(_CUSTOM_EP) == NETWORK_MAINNET


@pytest.mark.parametrize(
    "val,expected",
    [
        ("localnet", NETWORK_LOCALNET),
        ("local", NETWORK_LOCALNET),
        ("mainnet", NETWORK_MAINNET),
        ("testnet", NETWORK_TESTNET),
    ],
)
def test_env_override_wins(monkeypatch, val, expected):
    """GENTRX_NETWORK (and the equivalent --network arg) overrides everything,
    including a netuid that would map elsewhere. localnet must be accepted."""
    monkeypatch.setenv("GENTRX_NETWORK", val)
    assert network_from_subtensor(_CUSTOM_EP, netuid=79) == expected
