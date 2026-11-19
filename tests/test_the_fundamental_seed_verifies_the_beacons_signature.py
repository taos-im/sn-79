# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The fundamental seed's drand round was taken when two operators' endpoints served the same randomness, the
randomness checked to be the sha256 of the beacon's signature, the BLS signature itself unverified.
Where the chainlayer is installed its verify_quicknet_beacon (the drand-verify crate, scheme bls-unchained-g1-rfc9380
against quicknet's chain key) now checks each beacon's signature before the round is taken: a beacon that fails is
dropped and counted drand_bad_signature, as a disagreeing endpoint would be; a round taken with no verifier present
(the public tree ships without the chainlayer) is counted drand_unverified beside its source; the single-operator and
chain-pulse fallbacks stay. Both counts reach the validator gauges."""
import asyncio
import hashlib
import sys
import types

import pytest

from taos.im.validator import fundamental_seed as fs
from taos.im.validator import report

# quicknet round 1 as drand serves it
ROUND_1 = {"round": 1, "randomness": "1466a6cd24e327188770752f6134001c64d6efcc590ccc26b721611ad96f165a",
           "signature": "b55e7cb2d5c613ee0b2e28d6750aabbb78c39dcc96bd9d38c2c2e12198df95571de8e8e402a0cc48871c7089a2b3af4b"}


def _beacon(r=7, sig=b"\x11" * 48):
    return {"round": r, "signature": sig.hex(), "randomness": hashlib.sha256(sig).hexdigest()}


def test_a_verifier_judges_the_signature_and_none_means_unverified():
    b = _beacon()
    assert fs.signature_verified(b, 7, lambda r, s: True) is True
    assert fs.signature_verified(b, 7, lambda r, s: False) is False
    assert fs.signature_verified(b, 7, None) is None
    assert fs.signature_verified({"round": 7}, 7, lambda r, s: True) is False, "no signature to verify"

    def raising(r, s):
        raise ValueError("not a point")
    assert fs.signature_verified(b, 7, raising) is False


def test_a_beacon_is_accepted_only_when_its_randomness_and_its_signature_hold():
    b = _beacon()
    calls = []
    assert fs.accept_beacon(b, 7, lambda r, s: True, calls.append) == (bytes.fromhex(b["randomness"]), True)
    assert fs.accept_beacon(b, 7, None, calls.append) == (bytes.fromhex(b["randomness"]), False), "no verifier: taken, unverified"
    assert fs.accept_beacon(b, 7, lambda r, s: False, lambda: calls.append("bad")) == (None, False)
    assert calls == ["bad"]
    assert fs.accept_beacon(dict(b, randomness="00" * 32), 7, lambda r, s: True, lambda: calls.append("bad")) == (None, False)
    assert calls == ["bad"], "a randomness mismatch is not a signature failure"


class _Resp:
    def __init__(self, body):
        self.status = 200
        self._body = body

    async def json(self, content_type=None):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    bodies = {}

    def __init__(self, timeout=None):
        pass

    def get(self, url):
        base = url.split("/" + fs.QUICKNET_CHAIN)[0]
        return _Resp(self.bodies[base])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _fake_aiohttp(monkeypatch, bodies):
    mod = types.ModuleType("aiohttp")
    mod.ClientTimeout = lambda total=None: None
    mod.ClientError = type("ClientError", (Exception,), {})
    _Session.bodies = bodies
    mod.ClientSession = _Session
    monkeypatch.setitem(sys.modules, "aiohttp", mod)


def test_a_forged_pair_of_agreeing_endpoints_is_refused_and_counted(monkeypatch):
    b = _beacon()
    _fake_aiohttp(monkeypatch, {base: b for base in fs.ENDPOINTS})
    counts = []
    out = asyncio.run(fs.http_randomness(7, verifier=lambda r, s: False, on_bad_signature=lambda: counts.append("bad"),
                                         on_unverified=lambda: counts.append("unverified")))
    assert out is None and counts == ["bad"] * len(fs.ENDPOINTS)


def test_a_verified_pair_is_taken_and_an_unverified_one_is_counted_once(monkeypatch):
    b = _beacon()
    _fake_aiohttp(monkeypatch, {base: b for base in fs.ENDPOINTS})
    counts = []
    out = asyncio.run(fs.http_randomness(7, verifier=lambda r, s: r == 7 and s == b["signature"],
                                         on_bad_signature=lambda: counts.append("bad"), on_unverified=lambda: counts.append("unverified")))
    assert out == (bytes.fromhex(b["randomness"]), "drand") and counts == []
    out = asyncio.run(fs.http_randomness(7, verifier=None, on_bad_signature=lambda: counts.append("bad"),
                                         on_unverified=lambda: counts.append("unverified")))
    assert out == (bytes.fromhex(b["randomness"]), "drand") and counts == ["unverified"]


def test_the_default_verifier_is_the_chainlayers_and_the_two_counts_reach_the_gauges():
    import inspect
    src = inspect.getsource(fs.http_randomness)
    assert "verifier = beacon_verifier()" in src
    src = inspect.getsource(fs.attach)
    assert 'count("drand_bad_signature")' in src and 'count("drand_unverified")' in src
    src = inspect.getsource(report.fundamental_seed_gauges)
    assert '"drand_bad_signature"' in src and '"drand_unverified"' in src


def test_the_chainlayer_verifies_quicknets_published_round_1():
    verifier = fs.beacon_verifier()
    if verifier is None:
        pytest.skip("the chainlayer with verify_quicknet_beacon is not installed here")
    assert fs.signature_verified(ROUND_1, 1, verifier) is True
    assert fs.signature_verified(ROUND_1, 2, verifier) is False
    forged = dict(ROUND_1, signature=ROUND_1["signature"][:-2] + "00")
    assert fs.signature_verified(forged, 1, verifier) is False
    assert fs.accept_beacon(ROUND_1, 1, verifier) == (bytes.fromhex(ROUND_1["randomness"]), True)
