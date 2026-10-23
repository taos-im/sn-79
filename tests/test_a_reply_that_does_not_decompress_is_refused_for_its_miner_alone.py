# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner's reply whose payload is not a frame of the compression engine is refused for that miner alone.

Text that decodes as base64 but is no lz4, zlib or zstd frame reaches the decompressor, whose own error
(RuntimeError, zlib.error, ZstdError) is not the ReplyRefused read_reply catches. It is refused as ReplyRefused, on the
round's other replies are read as usual.
"""
import base64
from types import SimpleNamespace

import pytest
from bittensor.core.synapse import TerminalInfo

from taos.im.protocol import MarketSimulationStateUpdate
from taos.im.protocol.models import OrderDirection
from taos.im.protocol.response import FinanceAgentResponse
from taos.im.utils import compress as cmod
from taos.im.validator.query import QueryService

GARBAGE = base64.standard_b64encode(b"valid base64, no compressed frame").decode("ascii")


@pytest.mark.parametrize("engine", ("lz4", "zlib", "zstd"))
def test_a_payload_that_is_no_frame_of_the_engine_is_refused(engine):
    with pytest.raises(cmod.ReplyRefused, match=f"does not decompress as {engine}"):
        cmod.decode_reply({"books": None, "payload": GARBAGE}, engine=engine, version=64, max_bytes=1 << 20,
                          max_instructions=100)


def _reply(uid, compressed=None):
    synapse = MarketSimulationStateUpdate(timestamp=1, version=64, compression_engine="lz4")
    if compressed is None:
        synapse.response = FinanceAgentResponse(agent_id=uid)
        synapse.response.limit_order(book_id=0, direction=OrderDirection.BUY, quantity=1.0, price=299.0)
        synapse = synapse.clear_inputs().compress()
    else:
        synapse.compressed = compressed
    synapse.dendrite = TerminalInfo(status_code=200)
    synapse.axon = TerminalInfo(status_code=200)
    return synapse


def test_the_other_miners_replies_are_still_read():
    synapses = {1: _reply(1, {"books": None, "payload": GARBAGE}), 2: _reply(2)}
    request = {"version": 64, "books": {}, "book_count": 1, "book_ids": [0], "volume_sums": {}}
    stats = QueryService.validate_responses(SimpleNamespace(), synapses, request, set())
    assert stats == (1, 1, 2, 0, 0)  # uid 2's one order read; both answered
    assert synapses[1].response is None and len(synapses[2].response.instructions) == 1
