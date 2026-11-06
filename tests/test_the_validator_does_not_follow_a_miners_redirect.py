# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner's answer is the answer to the request the validator sent it: a redirect is refused, not followed.

The validator never re-sends a request to an address a miner names, and the miner's recorded response time is that of
the one request it was sent. Checked on every query path against a local server that answers with a redirect to
itself, and against one that answers directly, which stays a success.
"""
import asyncio
import time
import types

import bittensor as bt
import pytest
from aiohttp import web
from bittensor.core.chain_data import AxonInfo

from taos.im.validator import query

WORK_S = 0.3


class Synapse(bt.Synapse):
    """A synapse with the state update's compressed field, which the prebuilt path reads for its shared books blob."""
    compressed: dict | None = None


@pytest.fixture(autouse=True)
def _no_external_ip(monkeypatch):
    import bittensor.core.dendrite as dendrite_module
    monkeypatch.setattr(dendrite_module.networking, "get_external_ip", lambda: "127.0.0.1")


async def _server(redirect):
    seen = []

    async def handle(request):
        await request.read()
        seen.append(str(request.rel_url))
        if redirect and request.query.get("again") != "1":
            await asyncio.sleep(WORK_S)
            raise redirect(location=str(request.rel_url.with_query({"again": "1"})))
        if not redirect:
            await asyncio.sleep(WORK_S)
        return web.json_response(bt.Synapse().model_dump(), headers={"bt_header_axon_status_code": "200",
                                                                      "bt_header_axon_status_message": "Success"})
    app = web.Application()
    app.router.add_post("/{name}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, runner.addresses[0][1], seen


def _service(port, keypair, timeout=3.0):
    svc = object.__new__(query.QueryService)
    svc.config = types.SimpleNamespace(neuron=types.SimpleNamespace(timeout=timeout, global_query_timeout=timeout + 2))
    svc.wallet = keypair
    svc.metagraph = types.SimpleNamespace(axons=[AxonInfo(version=1, ip="127.0.0.1", port=port, ip_type=4,
                                                          hotkey=keypair.ss58_address, coldkey=keypair.ss58_address)])
    return svc


async def _round(path, redirect, timeout=3.0):
    keypair = bt.Keypair.create_from_uri("//Alice")
    runner, port, seen = await _server(redirect)
    svc = _service(port, keypair, timeout)
    pooled = path in ("fanout", "prebuilt")
    try:
        if pooled:
            svc.dendrite = bt.Dendrite(wallet=keypair)
            query.DendriteManager.configure_session(svc)
            run = svc._query_fanout if path == "fanout" else svc._query_prebuilt
        else:
            run = svc._query_threaded
        responses, _ = await run({0: Synapse()}, [0], set(), time.time(), timeout + 1)
        return responses[0], seen
    finally:
        if pooled and getattr(svc, "dendrite", None) is not None:
            await query.DendriteManager.close_session(svc.dendrite)
        await runner.cleanup()


PATHS = ["fanout", "threaded", "prebuilt"]


@pytest.mark.parametrize("redirect", [web.HTTPTemporaryRedirect, web.HTTPPermanentRedirect])
@pytest.mark.parametrize("path", PATHS)
def test_a_redirect_is_refused_and_not_followed(path, redirect):
    response, seen = asyncio.run(_round(path, redirect=redirect))
    assert len(seen) == 1, seen
    assert response.dendrite.status_code != 200, (response.dendrite.status_code, response.dendrite.process_time)


@pytest.mark.parametrize("path", PATHS)
def test_a_direct_answer_is_still_a_success_timed_on_its_request(path):
    response, seen = asyncio.run(_round(path, redirect=None))
    assert len(seen) == 1, seen
    assert response.dendrite.status_code == 200
    assert WORK_S <= float(response.dendrite.process_time) < WORK_S + 2.0, response.dendrite.process_time
