# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A local-node restart must not permanently wedge the miner.

THE DEFECT (found 2026-08-13). The miner run() loop wrapped its `while` in one try/except whose handler
sat OUTSIDE the loop. When the local subtensor node restarted, the next self.sync() raised (dead
websocket in check_registered/update_block); the exception escaped the loop, run() returned, and the run
thread died. The process stayed alive, so pm2 reported the miner 'online' while it had stopped syncing
and its metagraph-sync worker (killed by the same drop) was never respawned. Six Tier-3 miners wedged
this way on a shared-node bounce and produced zero valid responses until manually restarted.

THE FIX. Guard each iteration; on a sync failure rebuild the subtensor + metagraph in place
(_recover_chain_connection) and continue, exiting for a clean pm2 restart only after many consecutive
failures. These tests pin the reconnect mechanism: a dropped connection is rebuilt, not left dead, and a
still-down node or a rate-limited re-serve does not turn recovery into a fresh crash.
"""
import threading
import types

import taos.common.neurons.miner as miner_mod


def _stub_self(ep="ws://127.0.0.1:9944"):
    """A minimal stand-in carrying only the attributes _recover_chain_connection touches."""
    s = types.SimpleNamespace()
    s.config = types.SimpleNamespace(
        subtensor=types.SimpleNamespace(chain_endpoint=ep),
        netuid=79,
    )
    s._subtensor_lock = threading.Lock()
    s._mechid = 0
    s.axon = object()
    # the OLD (dead) connection we expect to be replaced
    s.subtensor = types.SimpleNamespace(name="dead")
    s.metagraph = types.SimpleNamespace(name="dead_mg")
    return s


def test_recover_rebuilds_subtensor_and_metagraph_and_reserves(monkeypatch):
    """The core of the fix: a dropped connection is rebuilt against the configured endpoint and the axon
    is re-published, rather than the dead handle being left in place."""
    built = {"subtensor": 0, "metagraph": 0, "serve": 0}

    class _FreshSubtensor:
        def __init__(self, network=None, config=None):
            built["subtensor"] += 1
            self._net = network

        def metagraph(self, netuid, mechid=0):
            built["metagraph"] += 1
            return types.SimpleNamespace(name="fresh_mg", netuid=netuid, mechid=mechid)

        def serve_axon(self, netuid=None, axon=None):
            built["serve"] += 1
            return True

    monkeypatch.setattr(miner_mod.bt, "Subtensor", _FreshSubtensor)
    s = _stub_self()
    miner_mod.BaseMinerNeuron._recover_chain_connection(s)

    assert built["subtensor"] == 1, "a fresh subtensor was not constructed"
    assert built["metagraph"] == 1 and s.metagraph.name == "fresh_mg", "metagraph handle not rebuilt"
    assert s.subtensor._net == "ws://127.0.0.1:9944", "reconnect ignored the configured chain endpoint"
    assert built["serve"] == 1, "the axon was not re-published after reconnect"


def test_recover_survives_a_still_down_node(monkeypatch):
    """If the node is still unreachable, recover must swallow and return (the loop retries next iteration),
    never raise -- a raise here would defeat the whole point and crash the loop again."""

    class _StillDown:
        def __init__(self, network=None, config=None):
            raise ConnectionError("node still down")

    monkeypatch.setattr(miner_mod.bt, "Subtensor", _StillDown)
    s = _stub_self()
    miner_mod.BaseMinerNeuron._recover_chain_connection(s)  # must not raise
    assert s.subtensor.name == "dead", "a failed rebuild must leave the old handle for the next retry"


def test_reserve_failure_is_nonfatal(monkeypatch):
    """A ServingRateLimitExceeded on re-serve must not abort the reconnect (the prior serve stays valid),
    so the rebuilt metagraph must persist."""

    class _FreshButRateLimited:
        def __init__(self, network=None, config=None):
            pass

        def metagraph(self, netuid, mechid=0):
            return types.SimpleNamespace(name="fresh_mg")

        def serve_axon(self, netuid=None, axon=None):
            raise Exception("ServingRateLimitExceeded")

    monkeypatch.setattr(miner_mod.bt, "Subtensor", _FreshButRateLimited)
    s = _stub_self()
    miner_mod.BaseMinerNeuron._recover_chain_connection(s)  # must not raise
    assert s.metagraph.name == "fresh_mg", "metagraph rebuild should persist despite a re-serve failure"
