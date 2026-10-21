# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Requests from the query service pass a bittensor axon's verify.

The axon keeps the last nonce per f"{dendrite.hotkey}:{dendrite.uuid}" and rejects a
request whose nonce is not newer ("Nonce is too old, a newer one was last processed").
GenTRX delivery signed every miner with one nonce, so a second request to the same
axon under that uuid was rejected. query_miners also queried the validator's own
uid, whose published axon can land on a miner's port (Signature mismatch there).

Offline: a fake dendrite borrows bittensor's own preprocess_synapse_for_request, a
fake session records each request, and a fake axon applies verify's nonce and
signature rules to the recorded headers.
"""
import asyncio
import uuid as _uuid
from types import SimpleNamespace

import bittensor as bt
import pytest

import taos.im.validator.query as q
from taos.im.validator.query import QueryService

VALIDATOR_HOTKEY = "5Ebq3nnVj2YAyznQsFDJJSmyoYGV7sx5boXqXv9go4rNrT3G"
MINER_HOTKEY = "5GMyr8evxXH5qLxSEBfjyFeMBYK3kB9fXUs67NDFtt7A7fFX"


class _Keypair:
    """Records every signed message; the signature is the message's index."""

    def __init__(self):
        self.ss58_address = VALIDATOR_HOTKEY
        self.signed = []

    def sign(self, message):
        self.signed.append(message)
        return (len(self.signed) - 1).to_bytes(4, "big")

    def message_for(self, sig_hex):
        return self.signed[int(sig_hex[2:], 16)]


class _Resp:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def text(self):
        return ""


class _Session:
    closed = False

    def __init__(self):
        self.sent = []

    def post(self, url, headers, data, timeout):
        self.sent.append((url, dict(headers)))
        return _Resp()


class _Dendrite:
    preprocess_synapse_for_request = bt.Dendrite.preprocess_synapse_for_request

    def __init__(self, keypair, session, uuid=None):
        self.keypair = keypair
        self.external_ip = "127.0.0.1"
        self.uuid = uuid or str(_uuid.uuid1())
        self._session = session

    @property
    def session(self):
        async def _get():
            return self._session

        return _get()


class _Axon:
    """bittensor axon verify: nonce newer than the last per hotkey:uuid, then signature."""

    def __init__(self, keypair, hotkey):
        self.keypair = keypair
        self.hotkey = hotkey
        self.nonces = {}

    def verify(self, headers):
        nonce = int(headers["bt_header_dendrite_nonce"])
        dhk = headers["bt_header_dendrite_hotkey"]
        duuid = headers["bt_header_dendrite_uuid"]
        key = f"{dhk}:{duuid}"
        if key in self.nonces and nonce <= self.nonces[key]:
            return "Nonce is too old, a newer one was last processed"
        expected = f"{nonce}.{dhk}.{self.hotkey}.{duuid}.{headers['computed_body_hash']}"
        if self.keypair.message_for(headers["bt_header_dendrite_signature"]) != expected:
            return "Signature mismatch"
        self.nonces[key] = nonce
        return "ok"


def _service(keypair, factory):
    svc = SimpleNamespace(
        wallet=SimpleNamespace(hotkey=keypair),
        dendrite=None,
        config=SimpleNamespace(neuron=SimpleNamespace(timeout=1.0)),
    )
    return svc, factory


def _delivery(uid, hotkey=MINER_HOTKEY, port=8091):
    return {
        "uid": uid,
        "axon_data": {
            "hotkey": hotkey, "coldkey": "ck", "ip": "127.0.0.1", "port": port,
            "ip_type": 4, "protocol": 4,
        },
        "assignment": {"round": 7},
    }


def _deliver(monkeypatch, svc, factory, deliveries):
    monkeypatch.setattr(q.bt, "Dendrite", factory)
    return asyncio.run(
        QueryService.deliver_gentrx_miners(svc, {"round": 7, "deliveries": deliveries})
    )


def _assert_strictly_newer_per_key(sent):
    last = {}
    for _url, h in sent:
        key = (h["bt_header_dendrite_hotkey"], h["bt_header_dendrite_uuid"])
        nonce = int(h["bt_header_dendrite_nonce"])
        assert key not in last or nonce > last[key], f"nonce {nonce} not newer than {last[key]}"
        last[key] = nonce


def test_one_delivery_signs_each_request_with_a_newer_nonce(monkeypatch):
    kp, session = _Keypair(), _Session()
    svc, factory = _service(kp, lambda wallet: _Dendrite(kp, session))
    # Two uids published on one miner axon: both requests reach the same verify.
    res = _deliver(monkeypatch, svc, factory, [_delivery(4), _delivery(8)])
    assert res["success"] and res["ok"] == 2
    _assert_strictly_newer_per_key(session.sent)
    axon = _Axon(kp, MINER_HOTKEY)
    assert [axon.verify(h) for _u, h in session.sent] == ["ok", "ok"]


def test_two_deliveries_and_a_delivery_after_a_regular_query_stay_newer(monkeypatch):
    kp, session = _Keypair(), _Session()
    # Worst case: every dendrite in the process shares one uuid.
    shared = _Dendrite(kp, session, uuid="shared-uuid")
    svc, factory = _service(kp, lambda wallet: shared)
    axon = _Axon(kp, MINER_HOTKEY)

    # A regular query first: bittensor's dendrite stamps time.time_ns() and signs.
    regular = bt.Synapse()
    regular = shared.preprocess_synapse_for_request(
        bt.AxonInfo(version=1, ip="127.0.0.1", port=8091, ip_type=4,
                    hotkey=MINER_HOTKEY, coldkey="ck"),
        regular, 1.0,
    )
    regular_headers = regular.to_headers()
    session.sent.append(("regular", regular_headers))

    _deliver(monkeypatch, svc, factory, [_delivery(4), _delivery(8)])
    _deliver(monkeypatch, svc, factory, [_delivery(4), _delivery(8)])

    _assert_strictly_newer_per_key(session.sent)
    assert all(axon.verify(h) == "ok" for _u, h in session.sent)


def test_each_signature_covers_the_nonce_and_uuid_the_request_carries(monkeypatch):
    kp, session = _Keypair(), _Session()
    svc, factory = _service(kp, lambda wallet: _Dendrite(kp, session))
    _deliver(monkeypatch, svc, factory, [_delivery(1, port=8091), _delivery(2, port=8092)])
    for _url, h in session.sent:
        signed = kp.message_for(h["bt_header_dendrite_signature"])
        nonce, dhk, mhk, duuid, body_hash = signed.split(".")
        assert nonce == h["bt_header_dendrite_nonce"]
        assert duuid == h["bt_header_dendrite_uuid"]
        assert mhk == h["bt_header_axon_hotkey"]
        assert body_hash == h["computed_body_hash"]


# ── query_miners never queries the validator's own uid or hotkey ──────────────

def _axon(hotkey, ip="203.0.113.1", port=8091):
    return {"hotkey": hotkey, "coldkey": "ck", "ip": ip, "port": port, "ip_type": 4, "protocol": 4}


def test_queryable_axons_skip_own_uid_and_own_hotkey():
    axons = [
        _axon(VALIDATOR_HOTKEY, ip="156.155.10.180"),  # uid 0: this validator
        _axon("5Other", ip="0.0.0.0"),  # uid 1: not served
        _axon(MINER_HOTKEY),  # uid 2: a miner
        _axon(VALIDATOR_HOTKEY, ip="203.0.113.9"),  # uid 3: own hotkey again
    ]
    picked = q._queryable_axons(axons, own_uid=0, own_hotkey=VALIDATOR_HOTKEY)
    assert [uid for uid, _a in picked] == [2]


@pytest.mark.parametrize("engine_mode", ["simulation", "exchange"])
def test_query_miners_drops_own_uid_in_both_modes(monkeypatch, engine_mode):
    seen = {}

    class _Stop(Exception):
        pass

    real = q._queryable_axons

    def _capture(axons, own_uid, own_hotkey):
        seen["uids"] = [uid for uid, _a in real(axons, own_uid, own_hotkey)]
        raise _Stop()

    monkeypatch.setattr(q, "_queryable_axons", _capture)

    class _NoDendrite:
        def __init__(self, wallet):
            pass

    monkeypatch.setattr(q.bt, "Dendrite", _NoDendrite)
    svc = SimpleNamespace(
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address=VALIDATOR_HOTKEY)),
        dendrite=SimpleNamespace(_session=None),
        config=SimpleNamespace(neuron=SimpleNamespace(reuse_connections=False)),
    )
    svc._round_dendrite = lambda: QueryService._round_dendrite(svc)
    request = {
        "engine_mode": engine_mode,
        "uid": 0,
        "metagraph_axons": [_axon(VALIDATOR_HOTKEY, ip="156.155.10.180"), _axon(MINER_HOTKEY)],
        "deregistered_uids": [],
    }
    res = asyncio.run(QueryService.query_miners(svc, request))
    assert res["success"] is False, res  # the capture aborts the query after the filter
    assert seen.get("uids") == [1], res.get("error")

