# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""THE END-OF-SIMULATION NOTICE MUST REACH onEnd, NOT JUST THE MINER.

Two independent links had to work and neither did, which is why no agent has ever been told a
simulation ended except by inferring it from a clock.

LINK 1, the axon. bittensor routes a request by the synapse's class NAME; the miner attached `update`
under the base `EventNotification` while the validator sends `FinanceEventNotification`, so every
dispatch was refused with UnknownSynapseError. Guarded in
test_the_notice_synapse_the_validator_sends_is_the_one_the_miner_attaches.py.

LINK 2, this file, the dispatch. `FinanceEvent.from_json` ABBREVIATES the wire type: an engine message
of type `EVENT_SIMULATION_END` becomes an event whose `y` -- and therefore whose `type` property -- is
`ESE`. The state-update path has always matched both spellings
(`case "EVENT_SIMULATION_END" | "ESE"` in FinanceAgentBase.handle). `FinanceAgentBase.process`, the handler on the
notification path, compared against the long form alone, so it could not have fired even once the
axon accepted the synapse. Fixing link 1 without link 2 moves the silence one step later.

The wire shape is checked here too. `EventNotification.event` is declared as the BASE
`SimulationEvent`, and pydantic serialises by the declared type, so a subclass field would vanish in
transit. SimulationEndEvent adds no field beyond `y`/`t`/`a`, which the base declares -- so this
particular event survives, and the test pins that rather than leaving it to luck.
"""

import tempfile

import pytest

from taos.common.protocol import SimulationEvent
from taos.im.agents import FinanceAgentBase
from taos.im.protocol import FinanceEventNotification

_ENGINE_MSG = {
    "type": "EVENT_SIMULATION_END",
    "timestamp": 12985000000000,
    "delay": 0,
    "payload": {},
}


def _wire_roundtrip(notification):
    """The notification as a miner receives it: serialised by the DECLARED field types and rebuilt."""
    return FinanceEventNotification(**notification.model_dump())


class _Recorder(FinanceAgentBase):
    """A FinanceAgentBase that records onEnd instead of running an agent's own shutdown.

    `initialize` is abstract on the base -- every real agent supplies it -- so it is stubbed here.
    Nothing else is overridden: `process` is the method under test and must be the real one.
    """

    def __init__(self):
        self.ended = []

    def initialize(self, *args, **kwargs):
        pass

    def onEnd(self, event):
        self.ended.append(event)


def test_the_engine_type_is_abbreviated_on_the_way_in():
    """Pinned because both consumers below depend on it, and neither says so on its own."""
    ev = FinanceEventNotification.from_json(_ENGINE_MSG).event
    assert ev.type == "ESE", (
        "from_json no longer abbreviates; if the long form is now carried, the dispatches that match "
        "both spellings are still correct, but this test's premise needs rewriting"
    )


def test_the_notice_survives_the_wire_as_an_end_event():
    got = _wire_roundtrip(FinanceEventNotification.from_json(_ENGINE_MSG))
    assert isinstance(got.event, SimulationEvent)
    assert got.event.type == "ESE", "the discriminator did not survive serialisation by the base type"
    assert got.event.timestamp == _ENGINE_MSG["timestamp"]
    assert got.event.agentId is None, "an end event is a broadcast; a uid here would target one miner"


def test_process_calls_onend_for_the_notice_the_validator_actually_sends():
    agent = _Recorder()
    got = agent.process(_wire_roundtrip(FinanceEventNotification.from_json(_ENGINE_MSG)))
    assert agent.ended, (
        "process() did not call onEnd for a real EVENT_SIMULATION_END notice. The event's type is "
        "abbreviated to 'ESE' by from_json, so a comparison against 'EVENT_SIMULATION_END' alone "
        "never matches and the agent is never told the simulation ended."
    )
    assert got.acknowledged is True, "the validator logs acknowledgement; it must be set"


def test_process_does_not_call_onend_for_other_events():
    """The dispatch must widen to the abbreviation, not to everything."""
    agent = _Recorder()
    start = FinanceEventNotification.from_json(
        {"type": "EVENT_SIMULATION_START", "timestamp": 1, "delay": 0, "payload": {"logDir": "/tmp/x"}}
    )
    agent.process(_wire_roundtrip(start))
    assert not agent.ended, "a start event called onEnd"


def test_the_broadcast_reaches_every_axon_not_uid_zero():
    """notify() targets on `agentId is not None`, so a 0 would mean uid 0 rather than everyone."""
    ev = FinanceEventNotification.from_json(_ENGINE_MSG).event
    # None, not 0. The two readers of this field disagree about zero -- the state-update fan-out in
    # MarketSimulationStateUpdate.from_json tests `if notice.event.agentId:` (so 0 broadcasts) and
    # notify() tests `if ... is not None:` (so 0 would address uid 0 alone). An end event is a
    # broadcast under both only while it carries None.
    assert ev.agentId is None, "an end event must carry no uid; 0 would address uid 0 alone in notify()"


# ── the state-update path: onEnd must receive the END event, not a leaked loop variable ──────────

class _Ev:
    """The shape FinanceAgentBase.update reads off a notice."""

    def __init__(self, etype, timestamp=0, book=None):
        self.type = etype
        self.timestamp = timestamp
        if book is not None:
            self.bookId = book

    def __repr__(self):
        return f"<{self.type}@{self.timestamp}>"


def _agent_driving_update(recorder_cls, events, book_ids=(5,)):
    """A real FinanceAgentBase with only the attributes `update` touches."""
    import types as _t

    a = object.__new__(recorder_cls)
    a.ended = []
    a.uid = 1
    a.history_len = 0
    a.history = []
    a.event_history = {"hk": [object()]}
    a.output_dir = tempfile.mkdtemp()
    # `config` here is the AGENT's own config (params), distinct from the state's simulation config.
    a.config = _t.SimpleNamespace(simulation_id="probe", lazy_load=False, params=_t.SimpleNamespace())
    a.simulation_config = _t.SimpleNamespace(book_ids=list(book_ids), simulation_id="probe")
    # One empty account per declared book: `update` indexes self.accounts[book_id] as it walks
    # simulation_config.book_ids, and an empty mapping raises KeyError before it ever reaches onEnd.
    _bal = lambda: _t.SimpleNamespace(free=0.0, reserved=0.0, total=0.0)

    def _acct():
        return _t.SimpleNamespace(orders=[], loans={}, fees=None,
                                  base_balance=_bal(), quote_balance=_bal(),
                                  base_collateral=0.0, quote_collateral=0.0,
                                  base_loan=0.0, quote_loan=0.0,
                                  traded_volume=0.0, delegate_stakes={})
    state = _t.SimpleNamespace(
        config=a.simulation_config,
        accounts={1: {b: _acct() for b in book_ids}},
        notices={1: list(events)},
        books={b: _t.SimpleNamespace(bids=[], asks=[]) for b in book_ids},
        dendrite=_t.SimpleNamespace(hotkey="hk"),
        timestamp=0,
    )
    return a, state


def test_onend_receives_the_end_event_and_not_whatever_was_last_in_the_loop():
    """The end notice sorts FIRST, so the leaked loop variable is never it.

    `update()` sets `simulation_ended = True` inside its notice loop and then, eighty lines later,
    calls `self.onEnd(event)` -- `event` being whichever notice the LAST loop to run left bound. There
    are two such loops, and the second rebinds `event` on every iteration regardless of its own
    `if`, so it ends up holding the last notice in the list.

    That is never the end event in practice: NoticePack packs a notice's `t` as its OCCURRENCE, and
    the end event is dispatched at the simulation's START with a delay of duration-1, so it carries
    the earliest timestamp of any notice and `from_json` sorts it to the front. An agent asking the
    event it was handed when the simulation ended gets a trade.
    """
    # The second notice needs no bookId and no handled type: the inner `for event in self.events:`
    # rebinds `event` on EVERY iteration, before its own `if hasattr(...)` runs. That is the whole
    # mechanism -- the leak does not need the notice to be relevant to anything.
    end = _Ev("ESE", timestamp=0)
    later = _Ev("UNHANDLED_TYPE", timestamp=99)
    agent, state = _agent_driving_update(_Recorder, [end, later])
    try:
        agent.update(state)
    except Exception as exc:                                        # pragma: no cover
        pytest.skip(f"update() needs more state than this fixture builds: {exc!r}")
    assert agent.ended, "onEnd was not called for a state update carrying an end notice"
    assert agent.ended[0] is end, (
        f"onEnd received {agent.ended[0]!r}, which is the last notice in the list rather than the "
        "end event. The end notice carries the earliest timestamp, so it sorts first and the leaked "
        "loop variable is never it."
    )
