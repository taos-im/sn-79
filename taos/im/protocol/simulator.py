# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Simulator response models: serialisable agent response and response-batch
types consumed by the C++ simulator via the validator protocol.
"""
from dataclasses import dataclass

import msgpack

from taos.common.protocol import BaseModel
from typing import Any

class SimulatorAgentResponse(BaseModel):
    """
    Represents a response from an agent.

    Attributes:
        agentId (int): Identifier for the agent sending the response.
        delay (int): Delay to be applied in processing the response.
        type (str): Type of the response.
        payload (dict[str, Any] | None): Additional data related to the response.
    """
    agentId: int
    delay: int
    type: str
    payload: dict[str, Any] | None  

    def serialize(self) -> dict:
        """
        Serializes the response into a dictionary format.
        """
        return {
            "agentId": self.agentId,
            "delay": self.delay,
            "type": self.type,
            "payload": self.payload,
        }

class SimulatorResponseBatch(BaseModel):
    """
    Represents a batch of responses from agents.

    Attributes:
        responses (list[SimulatorAgentResponse]): List of agent responses.
    """
    responses: list[SimulatorAgentResponse]

    def __init__(self, responses: list[SimulatorAgentResponse]):
        """
        Initialise the response batch.

        Args:
            responses (list[SimulatorAgentResponse]): Agent responses to include in the batch.
        """
        instructions = []
        for response in responses:
            if response:
                instructions.extend(response.serialize())
        super().__init__(responses=instructions)

    def serialize(self) -> dict:
        """
        Serialise the batch of responses into a dictionary format.

        Returns:
            dict: Dictionary with a 'responses' key mapping to a list of serialised responses.
        """
        return {
            "responses": [response.serialize() for response in self.responses]
        }


@dataclass
class PackedOrders:
    """Orders already delayed and encoded for the simulator (neuron.orders_packed): `count` msgpack items back to
    back. A batch carries them as one item and pack_batch writes them out in place."""
    count: int
    data: bytes

    def unpack(self) -> list:
        unpacker = msgpack.Unpacker(raw=False)
        unpacker.feed(self.data)
        return list(unpacker)



def with_capped(response: dict, pairs) -> dict:
    """The batch with the (uid, book) pairs at or over the volume cap beside its responses: the full current set,
    which replaces the engine's last one; an empty list clears it; None leaves the key out and the engine keeps it."""
    if pairs is None:
        return response
    return {**response, "capped": [[int(u), int(b)] for u, b in pairs]}


def response_batch(miner_responses: list) -> dict:
    """The simulator's batch for the miners' responses; PackedOrders (neuron.orders_packed) stay encoded, in their
    place."""
    if not any(isinstance(r, PackedOrders) for r in miner_responses):
        return SimulatorResponseBatch(miner_responses).serialize()
    return {"responses": [item for r in miner_responses for item in (
        [r] if isinstance(r, PackedOrders) else SimulatorResponseBatch([r]).serialize()["responses"])]}

def pack_batch(response: dict) -> tuple[bytes, list]:
    """msgpack.packb(response, use_bin_type=True) for a batch whose items may be PackedOrders, with the same bytes as
    packing their items one by one. Items the wire cannot encode are left out; returns the bytes and their agents."""
    pk = msgpack.Packer(use_bin_type=True)
    parts, count, dropped = [], 0, []
    for item in response.get("responses", ()):
        if isinstance(item, PackedOrders):
            parts.append(item.data)
            count += item.count
            continue
        try:
            parts.append(pk.pack(item))
            count += 1
        except (OverflowError, ValueError, TypeError):
            dropped.append(item.get("agentId") if isinstance(item, dict) else None)
    out = [pk.pack_map_header(len(response))]
    for key, value in response.items():
        out.append(pk.pack(key))
        out += [pk.pack_array_header(count), *parts] if key == "responses" else [pk.pack(value)]
    return b"".join(out), dropped
