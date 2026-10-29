# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""neuron.query_prepayload: the main process builds every queried miner's compressed request with the query
process's own function and hands those over instead of the decoded state.

The prebuilt requests are the ones the query process would build, for the same miners (its own uid and hotkey
skipped), and survive the pickled hand-off; the query process's checks still read the lighter request; query_miners
sends the prebuilt requests as they are, and builds its own when one it queries is missing.
"""
import asyncio
import pickle
from types import SimpleNamespace

import taos.im.validator.query as q
from taos.im.validator.query import QueryService

VALIDATOR_HOTKEY = "5Ebq3nnVj2YAyznQsFDJJSmyoYGV7sx5boXqXv9go4rNrT3G"


def _axon(hotkey, ip="203.0.113.1", port=8091):
    return {"hotkey": hotkey, "coldkey": "ck", "ip": ip, "port": port, "ip_type": 4, "protocol": 4}


def _request():
    book = {"i": 0, "b": [], "a": [], "e": []}
    return {
        "engine_mode": "simulation", "uid": 0, "version": 45, "timestamp": 10 ** 12, "config": None,
        "books": {0: book, 1: book}, "book_count": 2, "deregistered_uids": [],
        "accounts": {1: {0: {"marker": 1}}, 2: {0: {"marker": 2}}}, "notices": {1: [], 2: []},
        "metagraph_axons": [_axon(VALIDATOR_HOTKEY, ip="198.51.100.7"), _axon("5MinerA", port=8092),
                            _axon("5MinerB", port=8093), _axon("5Unserved", ip="0.0.0.0")],
    }


def test_the_main_process_builds_the_requests_the_query_process_would():
    books, built = q.build_axon_synapses(_request(), [1, 2], 1, "lz4", 0)
    handed = pickle.loads(pickle.dumps(q.prebuild_request(_request(), VALIDATOR_HOTKEY, 1, "lz4", 0), protocol=5))
    assert sorted(handed["axon_synapses"]) == [1, 2]
    assert handed["books_blob"] == books
    for uid in (1, 2):
        assert handed["axon_synapses"][uid].compressed == built[uid].compressed
    assert handed["accounts"] is None and handed["notices"] is None
    svc = SimpleNamespace(config=SimpleNamespace())
    assert QueryService.validate_responses(svc, {}, handed, set())[0] == 0


def _query(monkeypatch, request):
    """query_miners with stand-ins up to the send: returns the synapses it would send, or None if it built its own."""
    class _Stop(Exception):
        pass

    seen = {}

    def _build(*args):
        seen["built"] = True
        raise _Stop()

    async def _fanout(axon_synapses, *args):
        seen["sent"] = axon_synapses
        raise _Stop()

    monkeypatch.setattr(q, "build_axon_synapses", _build)
    monkeypatch.setattr(q.bt, "Dendrite", lambda wallet: SimpleNamespace(_session=None))
    monkeypatch.setattr(q.DendriteManager, "configure_session", staticmethod(lambda validator: None))
    monkeypatch.delenv("QUERY_FANOUT", raising=False)
    svc = SimpleNamespace(
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address=VALIDATOR_HOTKEY)),
        dendrite=SimpleNamespace(_session=None), _query_fanout=_fanout,
        config=SimpleNamespace(compression=SimpleNamespace(level=1, engine="lz4", parallel_workers=0),
                               neuron=SimpleNamespace(timeout=1.0, global_query_timeout=2.0, reuse_connections=False,
                                                      query_prebuilt=False)),
    )
    svc._round_dendrite = lambda: QueryService._round_dendrite(svc)  # as 0.6.5's own query_miners test binds it
    assert asyncio.run(QueryService.query_miners(svc, request))["success"] is False
    return seen.get("sent"), seen.get("built", False)


def test_query_miners_sends_the_prebuilt_requests_and_builds_its_own_when_one_is_missing(monkeypatch):
    handed = q.prebuild_request(_request(), VALIDATOR_HOTKEY, 1, "lz4", 0)
    sent, built = _query(monkeypatch, handed)
    assert sent is handed["axon_synapses"] and not built
    handed["axon_synapses"].pop(2)
    sent, built = _query(monkeypatch, handed)
    assert sent is None and built
