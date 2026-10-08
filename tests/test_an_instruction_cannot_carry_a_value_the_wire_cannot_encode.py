# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Every miner's instructions for a step are packed into one message for the simulator. The integer fields a miner
fills (delay, expiryPeriod, settleFlag) had no upper bound, so a value the wire cannot encode made the whole pack
fail, and nothing reached the simulator that step for anyone (reported 7 October 2026). Three layers hold this
shut: the instruction models bound their integers, the validator's own delay addition stays within the bound, and
the engine's pack drops only the instructions it cannot encode, names their agents and counts them."""
import logging
from types import SimpleNamespace

import msgpack
import pytest
from pydantic import ValidationError

from taos.im.protocol import instructions as I
from taos.im.protocol.models import OrderDirection
from taos.im.protocol.response import FinanceAgentResponse
from taos.im.validator.engines.simulation import SimulationEngine
from taos.im.validator.reward import set_delays

HUGE = 2 ** 70


def _limit(**kw):
    base = dict(agentId=3, bookId=0, direction=OrderDirection.BUY, quantity=1.0, price=100.0, clientOrderId=None)
    base.update(kw)
    return I.PlaceLimitOrderInstruction(**base)


def test_the_integer_fields_a_miner_fills_are_bounded():
    for field in ("delay", "expiryPeriod", "settleFlag"):
        with pytest.raises(ValidationError):
            _limit(**{field: HUGE})
    ok = _limit(delay=I.MAX_INSTRUCTION_DELAY, expiryPeriod=I.WIRE_INT_MAX, settleFlag=I.WIRE_INT_MAX)
    msgpack.packb(ok.serialize(), use_bin_type=True)
    with pytest.raises(ValidationError):
        _limit(delay=I.MAX_INSTRUCTION_DELAY + 1)


def _validator_duck():
    return SimpleNamespace(
        config=SimpleNamespace(
            neuron=SimpleNamespace(timeout=12.0),
            scoring=SimpleNamespace(min_delay=10_000_000, max_delay=1_000_000_000,
                                    min_instruction_delay=5_000_000, max_instruction_delay=25_000_000),
        ),
    )


def test_the_validators_own_delay_addition_stays_within_the_bound():
    response = FinanceAgentResponse(agent_id=3, instructions=[_limit(delay=I.MAX_INSTRUCTION_DELAY), _limit(delay=I.MAX_INSTRUCTION_DELAY)])
    synapses = {3: SimpleNamespace(response=response, dendrite=SimpleNamespace(process_time=11.9))}
    out = set_delays(_validator_duck(), synapses)
    assert len(out) == 1
    for instruction in out[0].instructions:
        assert 0 <= instruction.delay <= I.MAX_INSTRUCTION_DELAY
    msgpack.packb({"responses": [i.serialize() for i in out[0].instructions]}, use_bin_type=True)


def _engine(sent):
    """A SimulationEngine with only what respond touches: the validator it writes last_response to and the send."""
    engine = SimulationEngine.__new__(SimulationEngine)
    engine.validator = SimpleNamespace()
    engine._send_bytes = lambda data: sent.append(data)
    return engine


def test_the_engine_drops_only_what_it_cannot_encode_and_names_the_agent(caplog):
    sent = []
    engine = _engine(sent)
    good_a = _limit(agentId=1).serialize()
    good_b = _limit(agentId=2).serialize()
    bad = dict(_limit(agentId=7).serialize(), delay=HUGE)   # past the model, as a hostile wire payload would be
    with caplog.at_level(logging.ERROR):
        engine.respond(None, {"responses": [good_a, bad, good_b]})
    assert len(sent) == 1, "the step's message still went out"
    unpacked = msgpack.unpackb(sent[0], raw=False, strict_map_key=False)
    assert [r["agentId"] for r in unpacked["responses"]] == [1, 2]
    assert engine.response_pack_drops == 1
    assert any("7" in rec.getMessage() and "drop" in rec.getMessage().lower() for rec in caplog.records), "the offending agent is named"


def test_an_encodable_batch_is_sent_untouched():
    sent = []
    engine = _engine(sent)
    batch = {"responses": [_limit(agentId=1).serialize()]}
    engine.respond(None, batch)
    assert msgpack.unpackb(sent[0], raw=False, strict_map_key=False) == batch
    assert getattr(engine, "response_pack_drops", 0) == 0
