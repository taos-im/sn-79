# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A notice that contains a LIST must hand the agent events, not dicts.

Two notice types carry a list: cancellations (RDCO/ERDCO) and position closures (RDCP/ERDCP). The
dispatcher hands the ITEMS to the handler, one at a time --

    case "RESPONSE_DISTRIBUTED_CANCEL_ORDERS" | "RDCO":
        for cancellation in event.cancellations:
            self.onOrderCancelled(cancellation)

-- so what a miner's handler receives is an item, and the documented way to use it is
`event.orderId`. Both abbreviated forms were parsed with `model_construct`, which skips validation
entirely and therefore never coerces the nested list: every item stayed a raw dict. `cancellation.orderId`
raised AttributeError inside the miner's own handler, and on the simulation path that dispatch is not
wrapped, so the agent lost its whole update for the tick.

Abbreviated codes are the only form on the wire (packNotice writes them), so this was the live path in
both mechanisms, not an edge case.
"""

import pytest

from taos.im.protocol.events import parse_notices as sim_parse
from taos.im.protocol.exchange.events import parse_notices as xch_parse

WIRE_RDCO = {"y": "RDCO", "t": 1, "a": 8, "b": 79,
             "c": [{"t": 1, "b": 79, "o": 778, "q": 1.5, "u": True, "m": ""}]}
WIRE_ERDCO = {"y": "ERDCO", "t": 2, "a": 8, "b": 79,
              "c": [{"t": 2, "b": 79, "o": 999, "q": 0.0, "u": False, "m": "order not found"}]}
WIRE_RDCP = {"y": "RDCP", "t": 3, "a": 8, "b": 79,
             "o": [{"t": 3, "b": 79, "o": 778, "q": 1.5, "u": True, "m": ""}]}


@pytest.mark.parametrize("parse", [sim_parse, xch_parse], ids=["simulation", "exchange"])
@pytest.mark.parametrize("wire", [WIRE_RDCO, WIRE_ERDCO], ids=["RDCO", "ERDCO"])
def test_cancellation_items_expose_their_order_id(parse, wire):
    event = parse({8: [dict(wire)]})[8][0]
    items = event.cancellations
    assert items, "the notice arrived with no cancellations at all"
    item = items[0]
    assert not isinstance(item, dict), (
        "handed to onOrderCancelled as a dict: `cancellation.orderId` raises inside the miner's handler"
    )
    assert item.orderId == wire["c"][0]["o"]
    assert item.success is wire["c"][0]["u"]


def test_close_position_items_expose_their_order_id():
    """Simulation only: the exchange has no leverage, so no close-position notice reaches it."""
    event = sim_parse({8: [dict(WIRE_RDCP)]})[8][0]
    items = event.closes
    assert items, "the notice arrived with no closes at all"
    assert not isinstance(items[0], dict)
    assert items[0].orderId == 778
