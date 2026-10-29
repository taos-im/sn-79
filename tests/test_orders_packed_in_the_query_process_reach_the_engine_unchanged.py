# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""neuron.orders_packed: the query process delays the checked orders and encodes them for the simulator, and the main
process writes those bytes into the step's batch instead of rebuilding every order.

The engine receives the same bytes as when the main process delays and encodes the orders itself, beside the
validator's own reset instruction; an order the wire cannot encode is left out, counted and its agent named;
set_delays is the same function from its new module, and the settings the main process sends it give the same
delays as the validator.
"""
import random
from types import SimpleNamespace

import msgpack

from taos.im.protocol import instructions as I
from taos.im.protocol.models import OrderDirection
from taos.im.protocol.response import FinanceAgentResponse
from taos.im.protocol.simulator import PackedOrders, response_batch
from taos.im.validator import delays, reward
from taos.im.validator.engines.simulation import SimulationEngine
from taos.im.validator.forward import delay_settings, update_sent_stats
from taos.im.validator.query import QueryService


def _validator():
    return SimpleNamespace(
        config=SimpleNamespace(
            neuron=SimpleNamespace(timeout=3.0),
            scoring=SimpleNamespace(min_delay=10_000_000, max_delay=1_000_000_000,
                                    min_instruction_delay=5_000_000, max_instruction_delay=25_000_000),
        ),
        simulation=SimpleNamespace(time_unit="ns"),
    )


def _limit(agent, book, price):
    return I.PlaceLimitOrderInstruction(agentId=agent, bookId=book, direction=OrderDirection.BUY, quantity=1.0,
                                        price=price, clientOrderId=None)


def _replies():
    """Three miners' checked replies: several orders on one book (instruction delays drawn), a cancel, an empty one."""
    return {
        3: SimpleNamespace(response=FinanceAgentResponse(agent_id=3, instructions=[
            _limit(3, 0, 100.0), _limit(3, 0, 99.5), _limit(3, 1, 50.0)]), dendrite=SimpleNamespace(process_time=0.42)),
        5: SimpleNamespace(response=FinanceAgentResponse(agent_id=5, instructions=[
            I.CancelOrdersInstruction(agentId=5, bookId=1, cancellations=[I.CancelOrderInstruction(orderId=7, volume=None)])]),
            dendrite=SimpleNamespace(process_time=1.7)),
        8: SimpleNamespace(response=FinanceAgentResponse(agent_id=8), dendrite=SimpleNamespace(process_time=0.9)),
    }


def _reset():
    response = FinanceAgentResponse(agent_id=0)
    response.reset_agents(agent_ids=[11])
    return response


def _sent(miner_responses):
    sent = []
    engine = SimulationEngine.__new__(SimulationEngine)
    engine.validator = SimpleNamespace()
    engine._send_bytes = sent.append
    engine.respond(None, response_batch(miner_responses))
    return sent[0]


def test_the_engine_gets_the_same_bytes_whichever_process_encodes_the_orders():
    random.seed(7)
    today = [_reset()] + reward.set_delays(_validator(), _replies())
    random.seed(7)
    replies = _replies()
    packed = QueryService._pack_orders(None, replies, delay_settings(_validator()))
    assert packed["pack_drops"] == 0 and isinstance(packed["orders"], PackedOrders)
    assert all(not r.response.instructions for r in replies.values()), "the replies go back without their orders"
    sent = _sent([_reset(), packed["orders"]])
    assert sent == _sent(today)
    assert len(msgpack.unpackb(sent, raw=False)["responses"]) == 1 + 4


def test_an_order_the_wire_cannot_encode_is_left_out_counted_and_named(caplog):
    replies = _replies()
    bad = replies[3].response.instructions[2]
    object.__setattr__(bad, "expiryPeriod", 2 ** 70)  # past the model, as a hostile payload would be
    packed = QueryService._pack_orders(None, replies, delay_settings(_validator()))
    assert packed["pack_drops"] == 1 and packed["orders"].count == 3
    assert [item["agentId"] for item in packed["orders"].unpack()] == [3, 3, 5]
    assert any("[ORDERS-PACKED]" in rec.getMessage() and "3" in rec.getMessage() for rec in caplog.records)


def test_set_delays_moved_unchanged_and_the_sent_settings_give_the_validators_delays():
    assert reward.set_delays is delays.set_delays
    random.seed(11)
    own = [i.delay for r in delays.set_delays(_validator(), _replies()) for i in r.instructions]
    random.seed(11)
    sent = [i.delay for r in delays.set_delays(delay_settings(_validator()), _replies()) for i in r.instructions]
    assert own == sent and len(own) == 4


def test_each_miners_sent_counts_are_the_ones_main_would_count():
    """0.6.5's per-miner counters: under neuron.orders_packed main never sees the orders, so the query process counts
    each miner's placements and cancels by update_sent_stats' rule and main adds them to the miner's block."""
    random.seed(3)
    main = SimpleNamespace(miner_stats={})
    update_sent_stats(main, reward.set_delays(_validator(), _replies()))
    counted = {uid: {k: row[k] for k in ("placements_sent", "cancels_sent") if row.get(k)}
               for uid, row in main.miner_stats.items()}
    random.seed(3)
    packed = QueryService._pack_orders(None, _replies(), delay_settings(_validator()))
    assert {uid: {k: n for k, n in row.items() if n} for uid, row in packed["sent"].items() if any(row.values())} \
        == counted == {3: {"placements_sent": 3}, 5: {"cancels_sent": 1}}
