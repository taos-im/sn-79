# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner leaves its loaded heap out of the garbage collector's scans before it starts serving.

With a large live heap (the agent, its model, its history) a full collection inside a request costs a miner 60 to 80 ms,
and the pause lands in the response time the validator records (the developers' measurement). Freezing the
heap once the agent has loaded removes those collections. It is on by default; --neuron.no_gc_freeze keeps the collector
as before.
"""
import argparse
import gc
import inspect

from taos.common.config import add_miner_args
from taos.common.neurons import miner


def _no_freeze(args):
    parser = argparse.ArgumentParser()
    add_miner_args(None, parser)
    known, _ = parser.parse_known_args(args)
    return getattr(known, "neuron.no_gc_freeze")


def test_the_miner_freezes_by_default_and_can_be_told_not_to():
    assert _no_freeze([]) is False
    assert _no_freeze(["--neuron.no_gc_freeze"]) is True


def test_freezing_moves_the_live_heap_out_of_the_scans():
    gc.unfreeze()
    try:
        assert gc.get_freeze_count() == 0
        miner.freeze_heap()
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()


def test_the_heap_is_frozen_after_the_agent_loads_and_before_the_axon_serves():
    src = inspect.getsource(miner.BaseMinerNeuron.run)
    assert "freeze_heap()" in src
    assert src.index("freeze_heap()") < src.index("self.axon.start()")
    assert "no_gc_freeze" in src
