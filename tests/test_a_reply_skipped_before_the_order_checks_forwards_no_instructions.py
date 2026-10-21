"""Instructions reach the engine only through the validator's order checks. A reply that is skipped before those checks,
whether its agent id does not match the uid it came from or its miner has deregistered, forwards no instructions; a
reply that passes keeps the instructions the checks accept."""
import types


def _instruction(agent_id, book_id=0):
    return types.SimpleNamespace(agentId=agent_id, bookId=book_id, type="PLACE_ORDER_MARKET")


def _synapse(agent_id, instructions, timeout=False):
    return types.SimpleNamespace(
        is_timeout=timeout, is_failure=False, is_success=not timeout, compressed=False,
        dendrite=types.SimpleNamespace(hotkey="5FakeHotkey", status_message=""),
        response=None if timeout else types.SimpleNamespace(agent_id=agent_id, instructions=instructions),
    )


REQUEST = {
    "book_count": 4,
    "book_ids": list(range(4)),
    "engine_mode": "simulation",
    "miner_wealth": 1_000_000,
    "capital_turnover_cap": 10,
    "volume_decimals": 4,
    "books": {},
    "volume_sums": {},
    "version": "0.6.4",
}


def _validate(synapses, deregistered=()):
    from taos.im.validator.query import QueryService
    QueryService.validate_responses(None, synapses, dict(REQUEST), set(deregistered))


def test_a_reply_whose_agent_id_does_not_match_its_uid_forwards_no_instructions():
    synapse = _synapse(agent_id=9, instructions=[_instruction(9), _instruction(9, 1)])
    _validate({7: synapse})
    assert synapse.response.instructions == []


def test_a_reply_from_a_deregistered_miner_forwards_no_instructions():
    synapse = _synapse(agent_id=8, instructions=[_instruction(8)])
    _validate({8: synapse}, deregistered={8})
    assert synapse.response.instructions == []


def test_a_reply_that_passes_keeps_its_checked_instructions_beside_skipped_ones():
    good = _synapse(agent_id=5, instructions=[_instruction(5)])
    skipped = _synapse(agent_id=9, instructions=[_instruction(9)])
    timed_out = _synapse(agent_id=6, instructions=[], timeout=True)
    _validate({5: good, 7: skipped, 6: timed_out})
    assert [i.agentId for i in good.response.instructions] == [5]
    assert skipped.response.instructions == []
    assert timed_out.response is None


def test_a_reply_refused_for_its_size_forwards_no_instructions():
    synapse = _synapse(agent_id=5, instructions=[_instruction(5)])
    synapse.compressed = True
    synapse.decompress_reply = lambda **_kw: "over the reply bound"
    _validate({5: synapse})
    assert synapse.response.instructions == []


def test_a_reply_in_an_invalid_state_forwards_no_instructions():
    synapse = _synapse(agent_id=5, instructions=[_instruction(5)])
    synapse.is_success = False
    _validate({5: synapse})
    assert synapse.response.instructions == []


def test_a_reset_instruction_in_a_skipped_reply_is_not_forwarded():
    reset = types.SimpleNamespace(agentId=9, bookId=0, type="RESET_AGENT", agentIds=[5, 6])
    synapse = _synapse(agent_id=9, instructions=[reset])
    _validate({7: synapse})
    assert synapse.response.instructions == []
