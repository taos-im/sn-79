# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner's recorded response time runs from the validator's request written to the miner's answer read.

bittensor's Dendrite.call starts its clock before it signs and serialises the request, so the validator's own
preparation was in every miner's time; a trace on the pooled session supplies the wire clock, and the request on the
wire is unchanged. The prebuilt path (QUERY_PREBUILT) signs every request before the first is sent, but timed each
post from before the request went out to after the reply was decoded, so the validator's decode was in the miner's
time; it now reads the same trace stamps. Checked against a local server that answers after a known delay, on every
query path.
"""
import asyncio
import time
import types

import bittensor as bt
import pytest
from aiohttp import web
from bittensor.core.chain_data import AxonInfo

from taos.im.validator import query

ANSWER_S = 0.2
PREP_S = 0.3
DECODE_S = 0.3


class Synapse(bt.Synapse):
    """A synapse with the state update's compressed field, which the prebuilt path reads for its shared books blob."""
    compressed: dict | None = None


@pytest.fixture(autouse=True)
def _no_external_ip(monkeypatch):
    import bittensor.core.dendrite as dendrite_module
    monkeypatch.setattr(dendrite_module.networking, "get_external_ip", lambda: "127.0.0.1")


def _slow_decode(monkeypatch):
    original = bt.Dendrite.process_server_response

    def decode(self, *args, **kwargs):
        time.sleep(DECODE_S)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(bt.Dendrite, "process_server_response", decode)


def _slow_prep(monkeypatch):
    original = bt.Dendrite.preprocess_synapse_for_request

    def prep(self, *args, **kwargs):
        time.sleep(PREP_S)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(bt.Dendrite, "preprocess_synapse_for_request", prep)


async def _server(answer_s):
    async def handle(request):
        await request.read()
        await asyncio.sleep(answer_s)
        return web.json_response(bt.Synapse().model_dump(), headers={"bt_header_axon_status_code": "200",
                                                                      "bt_header_axon_status_message": "Success"})
    app = web.Application()
    app.router.add_post("/{name}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


def _service(port, keypair, timeout=3.0):
    svc = object.__new__(query.QueryService)
    svc.config = types.SimpleNamespace(neuron=types.SimpleNamespace(timeout=timeout, global_query_timeout=timeout + 2))
    svc.wallet = keypair
    svc.metagraph = types.SimpleNamespace(axons=[AxonInfo(version=1, ip="127.0.0.1", port=port, ip_type=4,
                                                          hotkey=keypair.ss58_address, coldkey=keypair.ss58_address)])
    return svc


async def _round(path, answer_s=ANSWER_S, timeout=3.0):
    keypair = bt.Keypair.create_from_uri("//Alice")
    runner, port = await _server(answer_s)
    try:
        svc = _service(port, keypair, timeout)
        pooled = path in ("fanout", "prebuilt")
        if pooled:
            svc.dendrite = bt.Dendrite(wallet=keypair)
            query.DendriteManager.configure_session(svc)
            run = svc._query_fanout if path == "fanout" else svc._query_prebuilt
        else:
            run = svc._query_threaded
        responses, _ = await run({0: Synapse()}, [0], set(), time.time(), timeout + 1)
        if pooled:
            await query.DendriteManager.close_session(svc.dendrite)
        return responses[0]
    finally:
        await runner.cleanup()


PATHS = ["fanout", "threaded", "prebuilt"]


@pytest.mark.parametrize("path", PATHS)
def test_the_validators_preparation_is_not_in_the_miners_time(monkeypatch, path):
    _slow_prep(monkeypatch)
    response = asyncio.run(_round(path))
    assert response.dendrite.status_code == 200
    assert ANSWER_S <= response.dendrite.process_time < ANSWER_S + PREP_S / 2, response.dendrite.process_time


@pytest.mark.parametrize("path", PATHS)
def test_the_validators_decode_is_not_in_the_miners_time(monkeypatch, path):
    _slow_decode(monkeypatch)
    response = asyncio.run(_round(path))
    assert response.dendrite.status_code == 200
    assert ANSWER_S <= float(response.dendrite.process_time) < ANSWER_S + DECODE_S / 2, response.dendrite.process_time


@pytest.mark.parametrize("path", PATHS)
def test_a_miner_that_does_not_answer_in_time_keeps_its_timeout(monkeypatch, path):
    response = asyncio.run(_round(path, answer_s=2.0, timeout=0.5))
    assert response.dendrite.status_code == 408
