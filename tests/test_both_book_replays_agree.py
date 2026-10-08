# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Two implementations rebuild a book from its events: the pydantic reference (Book.history in protocol/book.py) and
the dict replay the miner library runs in parallel (utils/history.py). They fixed the same cancellation defect on the
same day, so this holds them to the same result on the same events, with and without cancellation sides, so they
cannot drift apart again."""
from types import SimpleNamespace

from taos.im.protocol.book import Book, L2Snapshot, LevelInfo
from taos.im.utils import history as H

CFG = SimpleNamespace(volumeDecimals=4, publish_interval=1_000_000_000)


def _levels(d):
    return [LevelInfo.model_construct(price=p, quantity=q, orders=None) for p, q in d.items()]


def _events(with_sides):
    ev = [
        {"y": "o", "id": 1, "timestamp": 1, "quantity": 2.0, "side": 0, "price": 99.0},
        {"y": "o", "id": 2, "timestamp": 2, "quantity": 1.5, "side": 1, "price": 100.5},
        {"y": "t", "id": 3, "timestamp": 3, "quantity": 0.5, "side": 0, "price": 100.5, "taker_agent_id": 5, "maker_agent_id": 6},
        {"y": "c", "orderId": 2, "timestamp": 4, "price": 100.5, "quantity": 0.5},
        {"y": "c", "orderId": 1, "timestamp": 5, "price": 99.0, "quantity": 1.0},
    ]
    if with_sides:
        ev[3]["side"] = 1
        ev[4]["side"] = 0
    return ev


def _reference(with_sides):
    start = L2Snapshot(timestamp=0, bids={99.0: _levels({99.0: 5.0})[0]}, asks={100.5: _levels({100.5: 7.0})[0]})
    # the book's published levels are the target the reference compares against; make them the expected result
    book = Book.model_validate({"id": 0, "bids": [{"price": 99.0, "quantity": 6.0}], "asks": [{"price": 100.5, "quantity": 7.5}],
                                "events": _events(with_sides)})
    hist, matched, discrepancies = book.history(start, CFG, retention_mins=None, depth=None)
    last = list(hist.snapshots.values())[-1]
    return {p: l.quantity for p, l in last.bids.items()}, {p: l.quantity for p, l in last.asks.items()}, matched, discrepancies


def _dict_replay(with_sides):
    snap = {"timestamp": 0, "bids": {99.0: {"p": 99.0, "q": 5.0, "o": None}}, "asks": {100.5: {"p": 100.5, "q": 7.0, "o": None}}}
    wire = [{k: v for k, v in e.items()} for e in _events(with_sides)]
    # the dict replay reads the wire keys an event carries on the state update
    trans = {"id": "i", "orderId": "i", "timestamp": "t", "quantity": "q", "side": "s", "price": "p", "taker_agent_id": "Ta", "maker_agent_id": "Ma"}
    wire = [{trans.get(k, k): v for k, v in e.items()} for e in wire]
    hist, _ = H.history(snap, wire, 4)
    last = hist[max(hist)]
    return {p: l["q"] for p, l in last["bids"].items()}, {p: l["q"] for p, l in last["asks"].items()}


def test_the_two_replays_agree_with_sides_and_without():
    for with_sides in (True, False):
        rb, ra, matched, discrepancies = _reference(with_sides)
        db, da = _dict_replay(with_sides)
        assert rb == db and ra == da, f"sides={with_sides}: reference {rb, ra} dict {db, da}"
    rb, ra, matched, discrepancies = _reference(True)
    assert rb == {99.0: 6.0} and ra == {100.5: 7.5}, (rb, ra)
    assert matched, discrepancies
