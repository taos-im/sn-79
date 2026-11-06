# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A GenTRX delivery closes the session it opened and leaves the query's dendrite as it found it.

The delivery borrows a fresh dendrite for its posts, closes that dendrite's session itself, so no session is left for
garbage collection to close without an event loop, and puts the query's dendrite and its kept connections back.
Checked on a delivery that completes and on one that fails before it posts.
"""
import asyncio
import socket
import types

import bittensor as bt
import pytest

from taos.im.validator import query


@pytest.fixture(autouse=True)
def _no_external_ip(monkeypatch):
    import bittensor.core.dendrite as dendrite_module
    monkeypatch.setattr(dendrite_module.networking, "get_external_ip", lambda: "127.0.0.1")


def _closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _delivery(keypair, assignment):
    return {"uid": 0, "assignment": assignment,
            "axon_data": {"hotkey": keypair.ss58_address, "coldkey": keypair.ss58_address, "ip": "127.0.0.1",
                          "port": _closed_port(), "ip_type": 4, "protocol": 4}}


async def _deliver(monkeypatch, assignment):
    keypair = bt.Keypair.create_from_uri("//Alice")
    real_dendrite = bt.Dendrite
    monkeypatch.setattr(bt, "Dendrite", lambda wallet: real_dendrite(wallet=getattr(wallet, "hotkey", wallet)))
    svc = object.__new__(query.QueryService)
    svc.wallet = types.SimpleNamespace(hotkey=keypair)  # the delivery signs with the wallet's hotkey
    svc.config = types.SimpleNamespace(neuron=types.SimpleNamespace(timeout=1.0, global_query_timeout=2.0,
                                                                    reuse_connections=True))
    svc.dendrite = bt.Dendrite(wallet=keypair)
    query.DendriteManager.configure_session(svc)
    the_querys, its_session = svc.dendrite, svc.dendrite._session
    opened = []
    original = query.DendriteManager.configure_session

    def recording(validator):
        original(validator)
        opened.append(validator.dendrite._session)
    monkeypatch.setattr(query.DendriteManager, "configure_session", staticmethod(recording))
    result = await svc.deliver_gentrx_miners({"round": 1, "deliveries": [_delivery(keypair, assignment)]})
    kept = svc.dendrite is the_querys and not its_session.closed
    closed = [s.closed for s in opened]
    await query.DendriteManager.close_session(svc.dendrite)
    return result, kept, closed


@pytest.mark.parametrize("assignment,completes", [({}, True), ({"round": "not a round"}, False)])
def test_a_delivery_closes_its_session_and_keeps_the_querys_dendrite(monkeypatch, assignment, completes):
    result, kept, closed = asyncio.run(_deliver(monkeypatch, assignment))
    assert bool(result.get("success")) is completes, result
    assert kept, "the query's dendrite and its open session are back in place after the delivery"
    assert closed and all(closed), f"every session the delivery opened is closed: {closed}"
