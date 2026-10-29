"""The volume cap enforced at execution and the instruction delay bound (the validator half).

The validator's acceptance check drops a placement from a (uid, book) already at or over the cap, but an instruction
accepted under the cap and delayed is released by the engine later with nothing checking the cap again. Each reply
batch now carries the set of (uid, book) at or over the cap ("capped", beside "responses"), the full current set every
round, so the engine refuses their non-cancel placements when they arrive at the exchange, delayed or not; an empty
list clears the engine's set, an absent key keeps it. The delay a miner may ask for is bounded by a dial,
--neuron.max_instruction_delay_s (600 simulated seconds), read at acceptance beside the protocol's ceiling.
"""
import msgpack
from types import SimpleNamespace


from taos.im.protocol.models import OrderDirection
from taos.im.protocol.response import FinanceAgentResponse
from taos.im.protocol.simulator import pack_batch, with_capped
from taos.im.validator.forward import max_instruction_delay_ns
from taos.im.validator.query import QueryService

SECOND = 1_000_000_000


def _reply(uid, book=0, delay=0):
    response = FinanceAgentResponse(agent_id=uid)
    response.limit_order(book_id=book, direction=OrderDirection.BUY, quantity=1.0, price=299.0)
    response.instructions[-1].delay = delay
    return SimpleNamespace(response=response, is_timeout=False, is_failure=False, is_success=True, compressed=False,
                           dendrite=SimpleNamespace(status_message="", hotkey=f"5Hotkey{uid:03d}"))


def _request(volume_sums, book_ids=(0, 1), **extra):
    req = {"version": 64, "books": {}, "book_count": max(book_ids) + 1, "book_ids": list(book_ids),
           "volume_sums": volume_sums, "capital_turnover_cap": 5.0, "miner_wealth": 100000.0, "volume_decimals": 2,
           "max_instructions_per_book": 10, "engine_mode": "simulation"}
    req.update(extra)
    return req


def test_the_capped_pairs_are_the_uids_and_books_at_or_over_the_cap():
    svc = SimpleNamespace()
    synapses = {1: _reply(1, book=0), 2: _reply(2, book=0), 3: _reply(3, book=1)}
    request = _request({1: {0: 500000.0, 1: 10.0}, 2: {0: 499999.99}, 3: {1: 600000.0}})
    QueryService.validate_responses(svc, synapses, request, set())
    assert svc._capped_pairs == [[1, 0], [3, 1]]
    assert synapses[1].response.instructions == [] and synapses[3].response.instructions == []
    assert len(synapses[2].response.instructions) == 1


def test_the_capped_pairs_name_the_books_by_their_global_id():
    svc = SimpleNamespace()
    synapses = {1: _reply(1, book=7)}
    request = _request({1: {5: 0.0, 7: 500000.0}}, book_ids=(5, 7))
    QueryService.validate_responses(svc, synapses, request, set())
    assert svc._capped_pairs == [[1, 7]]


def test_a_deregistered_uid_and_a_uid_under_the_cap_are_not_in_the_set():
    svc = SimpleNamespace()
    synapses = {1: _reply(1), 2: _reply(2)}
    request = _request({1: {0: 500000.0}, 2: {0: 1.0}})
    QueryService.validate_responses(svc, synapses, request, {1})
    assert svc._capped_pairs == []


def test_an_instruction_delayed_beyond_the_bound_is_refused_and_counted():
    svc = SimpleNamespace()
    synapses = {1: _reply(1, delay=601 * SECOND), 2: _reply(2, delay=600 * SECOND)}
    request = _request({}, max_instruction_delay_ns=600 * SECOND)
    QueryService.validate_responses(svc, synapses, request, set())
    assert synapses[1].response.instructions == []
    assert len(synapses[2].response.instructions) == 1
    assert svc._dropped_instructions == {1: 1}


def test_the_bound_defaults_to_600_simulated_seconds_when_the_request_names_none():
    svc = SimpleNamespace()
    synapses = {1: _reply(1, delay=600 * SECOND + 1)}
    QueryService.validate_responses(svc, synapses, _request({}), set())
    assert synapses[1].response.instructions == []


def test_the_dial_gives_the_request_its_bound_in_nanoseconds():
    assert max_instruction_delay_ns(SimpleNamespace(neuron=SimpleNamespace(max_instruction_delay_s=600.0))) == 600 * SECOND
    assert max_instruction_delay_ns(SimpleNamespace(neuron=SimpleNamespace(max_instruction_delay_s=0.5))) == SECOND // 2


def test_the_config_default_is_600_seconds():
    import argparse
    from taos.im.config import add_im_validator_args
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    assert parser.get_default("neuron.max_instruction_delay_s") == 600.0


def test_with_capped_replaces_clears_or_keeps_the_engines_set():
    assert with_capped({"responses": []}, [[1, 0], [3, 1]]) == {"responses": [], "capped": [[1, 0], [3, 1]]}
    assert with_capped({"responses": []}, []) == {"responses": [], "capped": []}
    assert with_capped({"responses": []}, None) == {"responses": []}


def test_the_wire_carries_the_capped_pairs_beside_the_responses():
    packed, dropped = pack_batch({"responses": [{"agentId": 2, "type": "CANCEL_ORDERS"}], "capped": [[1, 0]]})
    assert dropped == []
    assert msgpack.unpackb(packed, raw=False) == {"responses": [{"agentId": 2, "type": "CANCEL_ORDERS"}],
                                                  "capped": [[1, 0]]}
