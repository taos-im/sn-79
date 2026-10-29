# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""neuron.query_prebuilt in the query service.

A prebuilt request carries the headers and body Dendrite.call would send and passes an axon's
verify; every request is signed before the first is sent; a round sends in 0.6.5's round_order and records its
wait for the report as the fan-out does; a kept
dendrite's nonces keep rising after the clock steps back; with 0.6.5's neuron.reuse_connections a connection the miner closed is tried once
more. (Reusing the dendrite across rounds is tested with 0.6.5's option.)

Offline: a fake dendrite borrows bittensor's own request preparation and reply handling, a fake
session records each request and echoes its body, and a fake axon applies verify's nonce and
signature rules.
"""
import asyncio
import json
import time
import uuid as _uuid
from types import SimpleNamespace

import aiohttp
import bittensor as bt

import taos.im.validator.query as q
from taos.im.validator.query import QueryService

VALIDATOR_HOTKEY = "5Ebq3nnVj2YAyznQsFDJJSmyoYGV7sx5boXqXv9go4rNrT3G"
MINER_HOTKEYS = ("5GMyr8evxXH5qLxSEBfjyFeMBYK3kB9fXUs67NDFtt7A7fFX",
                 "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty")


class _State(bt.Synapse):
    compressed: dict | None = None


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
    headers = {}

    def __init__(self, body):
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        reply = json.loads(self.body)  # the request echoed, as an axon answers it
        reply["axon"]["status_code"] = 200
        return reply


class _Closed:
    """A kept connection the miner closed between rounds."""

    async def __aenter__(self):
        raise aiohttp.ServerDisconnectedError()

    async def __aexit__(self, *a):
        return False


class _Session:
    closed = False

    def __init__(self, keypair):
        self.keypair = keypair
        self.sent = []
        self.closed_ports = set()

    def post(self, url, headers, data, timeout):
        self.sent.append(SimpleNamespace(url=url, headers=dict(headers), body=data,
                                         signed_before=len(self.keypair.signed)))
        port = int(headers["bt_header_axon_port"])
        if port in self.closed_ports:
            self.closed_ports.discard(port)
            return _Closed()
        return _Resp(data)


class _Dendrite:
    preprocess_synapse_for_request = bt.Dendrite.preprocess_synapse_for_request
    process_server_response = bt.Dendrite.process_server_response
    process_error_message = bt.Dendrite.process_error_message
    _get_endpoint_url = bt.Dendrite._get_endpoint_url
    log_exception = bt.Dendrite.log_exception

    def __init__(self, keypair, session):
        self.keypair = keypair
        self.external_ip = "127.0.0.1"
        self.uuid = str(_uuid.uuid1())
        self._session = session
        self.synapse_history = []

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


def _service(keypair):
    session = _Session(keypair)
    axons = [bt.AxonInfo(version=1, ip="127.0.0.1", port=8091 + i, ip_type=4, hotkey=hk, coldkey="ck")
             for i, hk in enumerate(MINER_HOTKEYS)]
    return SimpleNamespace(
        dendrite=_Dendrite(keypair, session),
        metagraph=SimpleNamespace(axons=axons),
        config=SimpleNamespace(neuron=SimpleNamespace(timeout=1.0, global_query_timeout=2.0)),
    ), session


def _round(svc):
    books = "shared-books-blob"
    synapses = {uid: _State(compressed={"books": books, "payload": f"own-{uid}"}) for uid in range(len(MINER_HOTKEYS))}
    responses, n = asyncio.run(QueryService._query_prebuilt(svc, synapses, list(synapses), set(), time.time(), 1.5))
    return synapses, responses, n


def _uid(sent):
    return int(sent.headers["bt_header_axon_port"]) - 8091


def _unsigned(headers, body):
    headers = {k: v for k, v in headers.items() if k != "bt_header_dendrite_signature"}
    body = json.loads(body)
    body["dendrite"].pop("signature")
    return headers, body


def test_a_prebuilt_request_is_what_dendrite_call_sends_and_passes_verify(monkeypatch):
    monkeypatch.setattr(time, "time_ns", lambda: 1_800_000_000_000_000_000)
    kp = _Keypair()
    svc, session = _service(kp)
    _, responses, n = _round(svc)
    assert n == len(MINER_HOTKEYS)
    for sent in session.sent:
        uid = _uid(sent)
        # What Dendrite.call posts for this axon: the synapse after bittensor's own preparation, as JSON.
        ref = _State(compressed={"books": "shared-books-blob", "payload": f"own-{uid}"})
        ref = svc.dendrite.preprocess_synapse_for_request(svc.metagraph.axons[uid], ref, 1.0)
        assert _unsigned(sent.headers, sent.body) == _unsigned(
            {**ref.to_headers(), "Content-Type": "application/json"}, json.dumps(ref.model_dump()))
        assert sent.url == svc.dendrite._get_endpoint_url(svc.metagraph.axons[uid], request_name="_State")
        assert _Axon(kp, MINER_HOTKEYS[uid]).verify(sent.headers) == "ok"
        assert str(responses[uid].dendrite.status_code) == "200"
        assert float(responses[uid].dendrite.process_time) >= 0


def test_every_request_is_signed_before_the_first_is_sent():
    kp = _Keypair()
    svc, session = _service(kp)
    _round(svc)
    assert {sent.signed_before for sent in session.sent} == {len(kp.signed)}


def test_a_prebuilt_round_sends_in_round_order(monkeypatch):
    # 0.6.5 dispatches every round in a fresh order, so a miner's place in it does not follow its uid.
    monkeypatch.setattr(q, "round_order", lambda uid_list, rng=None: list(enumerate(uid_list))[::-1])
    svc, session = _service(_Keypair())
    _round(svc)
    assert [_uid(sent) for sent in session.sent] == [1, 0]


def test_a_prebuilt_round_records_its_wait_for_the_report():
    svc, _ = _service(_Keypair())
    _round(svc)
    assert 0 <= svc._last_wait_s < 2.0


def test_a_kept_dendrites_nonces_keep_rising_after_the_clock_steps_back(monkeypatch):
    kp = _Keypair()
    svc, session = _service(kp)
    axons = [_Axon(kp, hk) for hk in MINER_HOTKEYS]
    now = 1_800_000_000_000_000_000
    for clock in (now, now - 5_000_000_000):  # the second round's clock is 5 s behind the first's
        monkeypatch.setattr(time, "time_ns", lambda clock=clock: clock)
        session.sent.clear()
        _round(svc)
        assert [axons[_uid(sent)].verify(sent.headers) for sent in session.sent] == ["ok", "ok"]


def test_a_kept_connection_the_miner_closed_is_tried_once_more_on_a_new_one():
    for reuse, posts, status in ((True, 3, "200"), (False, 2, None)):
        svc, session = _service(_Keypair())
        svc.config.neuron.reuse_connections = reuse
        session.closed_ports = {8091}  # miner 0's connection, closed since the last round
        _, responses, _ = _round(svc)
        assert len(session.sent) == posts
        assert (str(responses[0].dendrite.status_code) == "200") == (status == "200")
        assert str(responses[1].dendrite.status_code) == "200"


def test_the_round_instruments_find_the_wait_in_a_prebuilt_round(caplog):
    import logging
    import re

    caplog.set_level(logging.INFO, logger="bittensor")
    svc, _ = _service(_Keypair())
    _round(svc)
    lines = [rec.getMessage() for rec in caplog.records]
    # the two marks the mainnet baseline's round_breakdown.py reads around the miners' wait
    assert any("fanout query tasks, starting wait" in line for line in lines)
    assert any(re.search(r"Wait completed \(fanout\): (\d+) done, (\d+) pending in ([\d.]+)s", line) for line in lines)
