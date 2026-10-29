# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""With --neuron.reuse_connections the validator keeps one session to its miners across rounds.

Every round built a new Dendrite and closed its session after the round, and the miner's axon closed an idle
connection after 5 s, shorter than the time between rounds, so every round paid a TCP handshake and slow start on
the request body. The flag keeps the Dendrite and its session, clearing the synapse history it would otherwise grow
by every round, and the miner keeps an idle connection for 120 s. Kept connections are the default;
--neuron.no_reuse_connections turns them off.
"""
import argparse
import asyncio
import inspect
import types

import bittensor as bt

from taos.common.config import add_validator_args
from taos.im.validator import query


class _Session:
    def __init__(self, closed=False):
        self.closed = closed


class _Dendrite:
    def __init__(self, wallet=None, closed=False):
        self._session = _Session(closed)
        self.synapse_history = ["a", "b"]


def _service(reuse, closed=False):
    return types.SimpleNamespace(config=types.SimpleNamespace(neuron=types.SimpleNamespace(reuse_connections=reuse)),
                                 wallet=None, dendrite=_Dendrite(closed=closed))


def _round(svc, monkeypatch):
    closes = []

    async def close(d):
        closes.append(d)
    monkeypatch.setattr(query.DendriteManager, "close_session", staticmethod(close))
    monkeypatch.setattr(bt, "Dendrite", _Dendrite)
    before = svc.dendrite
    got = asyncio.run(query.QueryService._round_dendrite(svc))
    return before, got, closes


def test_by_default_every_round_gets_a_new_dendrite(monkeypatch):
    svc = _service(False)
    before, got, closes = _round(svc, monkeypatch)
    assert got is not before and svc.dendrite is got and closes == [before]


def test_reuse_keeps_the_dendrite_and_clears_its_history(monkeypatch):
    svc = _service(True)
    before, got, closes = _round(svc, monkeypatch)
    assert got is before and closes == [] and before.synapse_history == []


def test_reuse_rebuilds_a_closed_session(monkeypatch):
    svc = _service(True, closed=True)
    before, got, closes = _round(svc, monkeypatch)
    assert got is not before and closes == [before]


def test_the_cleanup_keeps_a_reused_session_open(monkeypatch):
    closes = []

    async def close(d):
        closes.append(d)
    monkeypatch.setattr(query.DendriteManager, "close_session", staticmethod(close))
    for reuse, want in ((True, 0), (False, 1)):
        closes.clear()
        svc = _service(reuse)
        asyncio.run(query.QueryService._post_query_cleanup(svc))
        assert len(closes) == want, (reuse, closes)


def _reuse(args):
    p = argparse.ArgumentParser()
    add_validator_args(None, p)
    strict = p.parse_args(args).__dict__["neuron.reuse_connections"]
    assert bt.Config(p, args=args).neuron.reuse_connections is strict
    return strict


def test_kept_connections_are_on_by_default():
    assert _reuse([]) is True


def test_a_validator_can_turn_them_off():
    assert _reuse(["--neuron.no_reuse_connections"]) is False


def test_the_old_switch_still_parses():
    assert _reuse(["--neuron.reuse_connections"]) is True


def test_the_query_round_takes_its_dendrite_from_the_one_place():
    src = inspect.getsource(query.QueryService.query_miners)
    assert "await self._round_dendrite()" in src
    assert "self.dendrite = bt.Dendrite(wallet=self.wallet)" not in src


def test_the_miner_keeps_an_idle_connection_for_two_minutes():
    from taos.common.neurons import miner
    src = inspect.getsource(miner)
    i, j = src.find("timeout_keep_alive"), src.find("self.axon.start()")
    assert 0 < i < j, "the keep-alive is set before the axon starts"
    assert miner.AXON_KEEP_ALIVE_S == 120
