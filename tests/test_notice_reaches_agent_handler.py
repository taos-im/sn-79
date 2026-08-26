# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A wire notice must parse AND dispatch into the agent's handler.

THE TWO SILENT LINKS. Between the engine emitting a notice and a miner acting on it there are two steps
that fail without saying so:

* `parse_notices` never drops a notice -- it passes an unrecognised one through as a raw dict and only
  counts it. So delivery succeeds whatever the shape.
* `_dispatch_notice_handlers` reads `getattr(event, "type", None)`. A raw dict has no `.type`, so it
  matches `case _: pass`. No handler, no error, no log line.

Together they mean an exchange miner could receive every notice and run no code at all, which is what it
did until 2026-08-19. Every other test in this area asserts on a notice's fields; this one asserts on the
consequence, starting from the bytes the engine actually sends.
"""

import pytest

from taos.im.protocol.exchange.events import parse_notices


# Exactly what packNotice writes for these two types: abbreviated code in 'y', occurrence in 't',
# agentId in 'a', bookId in 'b', and for cancels a list under 'c' with one entry per order.
WIRE_RDCO = {"y": "RDCO", "t": 1_700_000_000, "a": 8, "b": 79,
             "c": [{"t": 1_700_000_000, "b": 79, "o": 778, "q": 1.5, "u": True, "m": ""}]}
WIRE_ERDCO = {"y": "ERDCO", "t": 1_700_000_001, "a": 8, "b": 79,
              "c": [{"t": 1_700_000_001, "b": 79, "o": 999_000_000, "q": 0.0, "u": False,
                     "m": "order not found"}]}


class _Recorder:
    """The dispatch half of FinanceAgent, bound to a bare object.

    Constructing a real agent needs a validator, a wallet and a metagraph. The dispatcher is a method on
    the class and touches only self.events and the handlers, so binding it directly tests the routing
    without any of that.
    """

    from taos.im.agents import FinanceAgent

    _dispatch_notice_handlers = FinanceAgent._dispatch_notice_handlers

    def __init__(self, events):
        self.events = events
        self.calls = []

    def onOrderAccepted(self, event):
        self.calls.append(("onOrderAccepted", getattr(event, "orderId", None)))

    def onOrderCancelled(self, event):
        self.calls.append(("onOrderCancelled", getattr(event, "orderId", None)))

    def onOrderCancellationFailed(self, event):
        self.calls.append(("onOrderCancellationFailed", getattr(event, "orderId", None)))


@pytest.mark.parametrize(
    "wire,handler,order_id",
    [(WIRE_RDCO, "onOrderCancelled", 778),
     (WIRE_ERDCO, "onOrderCancellationFailed", 999_000_000)],
    ids=["successful-cancel", "failed-cancel"],
)
def test_wire_notice_parses_and_reaches_its_handler(wire, handler, order_id):
    parsed = parse_notices({8: [wire]})[8]
    assert len(parsed) == 1
    event = parsed[0]
    assert not isinstance(event, dict), (
        "passed through as a raw dict, so no handler can ever dispatch on it"
    )
    assert getattr(event, "type", None) in (wire["y"], None) or event.type

    rec = _Recorder(parsed)
    rec._dispatch_notice_handlers(state=None)
    assert rec.calls == [(handler, order_id)], f"dispatched {rec.calls}"


def test_a_failed_cancel_never_dispatches_as_a_success():
    """The specific harm: a miner told its cancel worked stops tracking a resting order."""
    rec = _Recorder(parse_notices({8: [WIRE_ERDCO]})[8])
    rec._dispatch_notice_handlers(state=None)
    assert [c[0] for c in rec.calls] == ["onOrderCancellationFailed"]
