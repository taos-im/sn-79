# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The documented notice handlers must fire in exchange mode, not only in simulation.

`agents/README.md` presents these as THE way a miner consumes notices, with no mention of a mode:

    onStart(event)                     onOrderCancelled(cancellation)
    onOrderAccepted(event)             onOrderCancellationFailed(cancellation)
    onOrderRejected(event)             onPositionClosed(close)
    onTrade(event, hotkey)             onPositionCloseFailed(close)
    onEnd(event)

They were dispatched only on the simulation path. The exchange branch of `update()` set `self.events`
and dispatched none of them, so an existing agent that handles `onTrade` ran on the exchange and was
simply never told about its fills -- no error, no warning, the handler just never called. That is the
opposite of the porting promise the README makes, and it is invisible to a miner reading it.

Dispatching them requires notices to BE event models, which on the exchange path they were not until
`parse_notices` landed. So this is the alignment's payoff rather than an independent change.

One handler per notice type is asserted, in exchange mode, off the real update() path.
"""

import pytest

from taos.im.protocol.exchange import ExchangeStateUpdate


class _Recorder:
    """Mixin recording which handlers fired, in order."""

    def _seen(self):
        if not hasattr(self, "_calls"):
            self._calls = []
        return self._calls

    def onStart(self, event):
        self._seen().append(("onStart", getattr(event, "type", None)))

    def onOrderAccepted(self, event):
        self._seen().append(("onOrderAccepted", getattr(event, "type", None)))

    def onOrderRejected(self, event):
        self._seen().append(("onOrderRejected", getattr(event, "type", None)))

    def onOrderCancelled(self, cancellation):
        self._seen().append(("onOrderCancelled", None))

    def onOrderCancellationFailed(self, cancellation):
        self._seen().append(("onOrderCancellationFailed", None))

    def onPositionClosed(self, close):
        self._seen().append(("onPositionClosed", None))

    def onPositionCloseFailed(self, close):
        self._seen().append(("onPositionCloseFailed", None))

    def onTrade(self, event, hotkey=None):
        self._seen().append(("onTrade", hotkey))

    def onEnd(self, event):
        self._seen().append(("onEnd", getattr(event, "type", None)))


def _agent(uid=4):
    from taos.im.agents import FinanceAgent

    class _A(_Recorder, FinanceAgent):
        def __init__(self):
            self.uid = uid
            self.accounts = {}
            self._pools = {}
            self._calls = []

        def initialize(self):
            return None

        def respond(self, state):
            return None

    return _A()


HOTKEY = "5HotKeyForTheValidator"


def _state(notices):
    st = ExchangeStateUpdate.model_construct()
    st.notices = notices
    st.accounts = {}
    st.config = {}
    st.pools = {1: {"price": 0.01}}
    st.dendrite = {"hotkey": HOTKEY}  # coerced to TerminalInfo by the field
    return st


def _wire(code, **extra):
    # `r` (currency) is required by the MARKET placement parse: from_json does OrderCurrency(json["r"]),
    # so a wire dict without it raises and the notice is dropped before any handler could see it.
    base = {"type": code, "timestamp": 5, "agentId": 4, "bookId": 1, "id": 9, "r": 0}
    base.update(extra)
    return base


def _parsed(uid, wires):
    from taos.im.protocol.exchange.events import parse_notices

    return parse_notices({uid: list(wires)})


@pytest.mark.parametrize(
    "code,handler",
    [
        ("RDPOL", "onOrderAccepted"),
        ("RDPOM", "onOrderAccepted"),
        ("ERDPOL", "onOrderRejected"),
        ("ERDPOM", "onOrderRejected"),
        ("ET", "onTrade"),
    ],
)
def test_each_notice_type_reaches_its_handler_in_exchange_mode(code, handler):
    agent = _agent()
    agent.update(_state(_parsed(4, [_wire(code)])))
    fired = [name for name, _ in agent._calls]
    assert handler in fired, (
        f"a {code} notice reached the agent in exchange mode and {handler}() was never called; "
        f"handlers that fired: {fired}"
    )


def test_onTrade_receives_the_validator_hotkey_like_the_simulation_path():
    """The simulation path calls onTrade(event, state.dendrite.hotkey).

    An exchange path calling it with one argument would break every ported agent on the argument count,
    which is a worse failure than not calling it at all: it raises inside the response builder.
    """
    agent = _agent()
    agent.update(_state(_parsed(4, [_wire("ET")])))
    trades = [(n, hk) for n, hk in agent._calls if n == "onTrade"]
    assert trades, "onTrade did not fire"
    assert trades[0][1] == HOTKEY, (
        f"onTrade was called without the validator hotkey the simulation path passes: {trades[0]}"
    )


def test_a_handler_that_raises_does_not_cost_the_miner_its_other_notices():
    """One bad handler must not swallow the rest.

    This is the SimpleRegressorAgent failure in a different guise: an exception raised while processing
    notices propagated out of the response builder and the agent lost its WHOLE response, not the one
    notice. A miner's own buggy handler should cost it that handler, nothing more.
    """
    agent = _agent()

    def _boom(event):
        raise RuntimeError("miner handler is buggy")

    agent.onOrderAccepted = _boom
    agent.update(_state(_parsed(4, [_wire("RDPOL"), _wire("ET")])))
    fired = [n for n, _ in agent._calls]
    assert "onTrade" in fired, (
        f"a raising onOrderAccepted stopped the later ET notice from reaching onTrade: {fired}"
    )


def test_simulation_only_events_are_not_synthesised_on_the_exchange():
    """An exchange does not start or end, so onStart/onEnd must not be invented for it.

    Firing them off every state update would be worse than silence: an agent using onStart to initialise
    would re-initialise on every tick.
    """
    agent = _agent()
    agent.update(_state(_parsed(4, [_wire("ET")])))
    fired = [n for n, _ in agent._calls]
    assert "onStart" not in fired and "onEnd" not in fired, (
        f"lifecycle handlers fired on an exchange state update that carried no such event: {fired}"
    )
