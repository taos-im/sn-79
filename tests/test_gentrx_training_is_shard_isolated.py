"""A GenTRX training job must read and write the shard of ITS OWN assignment.

One axon serves both mechanisms, the agent's mechanism flag flips on every state update, and training
runs on a background thread -- so a job started from an exchange assignment can finish while the agent
is mid simulation tick. Before this was bound to the assignment, the upload took whichever prefix the
last tick left behind, and the download read the exchange aggregator's keys out of the simulation
prefix and 404'd every file.
"""
import types

import pytest

from taos.im.agents.gentrx import GenTRXAgent


class _Store:
    def __init__(self, prefix):
        self.prefix = prefix
        self.written = []

    def put_gradient(self, miner_uid, round_id, data):
        self.written.append((self.prefix, miner_uid, round_id))


def _agent(current_prefix):
    """A bare object carrying just the state these two helpers touch."""
    a = object.__new__(GenTRXAgent)
    a._gtx = types.SimpleNamespace(
        bucket_prefix=current_prefix,
        write_store=_Store(current_prefix),
        tlog=types.SimpleNamespace(info=lambda *_, **__: None,
                                   warning=lambda *_, **__: None),
    )
    return a


EXCH = "gentrx/localnet/exchange/"
SIM = "gentrx/localnet/simulation/"


def test_shard_comes_from_the_assignment_not_the_agent():
    a = _agent(SIM)                                  # agent currently on simulation
    assert a._shard_of({"bucket_prefix": EXCH}) == EXCH


def test_upload_goes_to_the_assignment_shard_while_agent_is_on_the_other():
    """The regression: agent flipped to simulation, job belongs to exchange."""
    a = _agent(SIM)
    store = a._write_store_for({"bucket_prefix": EXCH})
    store.put_gradient(miner_uid=8, round_id=5101, data=b"x")
    assert store.prefix == EXCH
    assert store.written[-1] == (EXCH, 8, 5101)
    # The agent's own store must not be REPOINTED. That is what keeps the other mechanism unaffected:
    # a job rebinding the shared store is exactly the oscillation this replaced.
    assert a._gtx.write_store.prefix == SIM
    assert store is not a._gtx.write_store


def test_same_shard_reuses_the_agent_store():
    a = _agent(SIM)
    assert a._write_store_for({"bucket_prefix": SIM}) is a._gtx.write_store


def test_missing_or_implausible_prefix_falls_back_to_the_agent():
    a = _agent(SIM)
    assert a._shard_of({}) == SIM
    assert a._shard_of({"bucket_prefix": "not/a/prefix"}) == SIM
    assert a._shard_of({"bucket_prefix": "gentrx/localnet/wat/"}) == SIM


def test_each_mechanism_keeps_its_own_store():
    a = _agent(SIM)
    e = a._write_store_for({"bucket_prefix": EXCH})
    s = a._write_store_for({"bucket_prefix": SIM})
    assert e.prefix == EXCH and s.prefix == SIM
    assert e is not s
