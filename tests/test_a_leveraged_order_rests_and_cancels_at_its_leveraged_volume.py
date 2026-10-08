# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The engine rests a leveraged order at volume x (1 + leverage) and removes the same when it is cancelled
(simulate/trading/src/cpp/book/Book.cpp, the placement and the cancel path dividing by 1 + leverage). Both replays
in the miner library added and removed the order's own quantity, so a leveraged order left q x leverage on the
level after its cancellation: on the 7 October capture every remaining mismatch inside the published window was one
(book 21, order 23637, q 992.5902 at leverage 0.0113 published 1003.8065 and left 11.2163 behind). Both replays
rest q x (1 + l) when the order carries leverage and remove the same on its cancellation, keyed by order id.

The reference replay reconciles toward the levels the book publishes, so each case gives it the levels the engine
would publish and asks whether the replay matches them without a discrepancy."""
from types import SimpleNamespace

from taos.im.protocol.book import Book, L2Snapshot, LevelInfo
from taos.im.utils import history as H

CFG = SimpleNamespace(volumeDecimals=4, publish_interval=1_000_000_000)
Q, L, P = 992.5902, 0.0113, 298.40
RESTED = round(Q * (1 + L), 4)


def _ref(events, start, published):
    s = L2Snapshot(timestamp=0, bids={p: LevelInfo.model_construct(price=p, quantity=q, orders=None) for p, q in start.items()}, asks={})
    book = Book.model_validate({"id": 21, "bids": [{"price": p, "quantity": q} for p, q in published.items()], "asks": [], "events": events})
    hist, matched, discrepancies = book.history(s, CFG, retention_mins=None, depth=None)
    last = list(hist.snapshots.values())[-1]
    return matched, discrepancies, {p: lv.quantity for p, lv in last.bids.items()}


def _dict(events, start):
    snap = {"timestamp": 0, "bids": {p: {"p": p, "q": q, "o": None} for p, q in start.items()}, "asks": {}}
    hist, _ = H.history(snap, events, 4)
    last = hist[max(hist)]
    return {p: lv["q"] for p, lv in last["bids"].items()}


def test_the_reference_replay_rests_a_leveraged_order_at_its_leveraged_volume():
    matched, discrepancies, last = _ref([{"y": "o", "id": 23637, "timestamp": 1, "quantity": Q, "side": 0, "price": P, "leverage": L}],
                                        start={}, published={P: RESTED})
    assert matched, discrepancies
    assert last == {P: RESTED}


def test_the_reference_replay_removes_the_leveraged_volume_on_the_cancellation():
    matched, discrepancies, last = _ref([{"y": "o", "id": 23637, "timestamp": 1, "quantity": Q, "side": 0, "price": P, "leverage": L},
                                         {"y": "c", "orderId": 23637, "timestamp": 2, "price": P, "quantity": Q, "side": 0}],
                                        start={}, published={})
    assert matched, discrepancies
    assert last == {}


def test_the_dict_replay_does_the_same():
    o = {"y": "o", "i": 23637, "t": 1, "q": Q, "s": 0, "p": P, "l": L}
    c = {"y": "c", "i": 23637, "t": 2, "p": P, "q": Q, "s": 0}
    assert _dict([o], {}) == {P: RESTED}
    assert _dict([o, c], {}) == {}


def test_an_unleveraged_order_behaves_as_before_in_both_replays():
    matched, discrepancies, last = _ref([{"y": "o", "id": 1, "timestamp": 1, "quantity": 2.0, "side": 0, "price": 100.0}],
                                        start={100.0: 1.0}, published={100.0: 3.0})
    assert matched and last == {100.0: 3.0}, discrepancies
    assert _dict([{"y": "o", "i": 1, "t": 1, "q": 2.0, "s": 0, "p": 100.0}], {100.0: 1.0}) == {100.0: 3.0}


def test_a_cancellation_carrying_its_leverage_removes_the_leveraged_volume_without_the_placement():
    # the placement sat in an earlier state update; the engine publishes the order's leverage on the cancellation
    matched, discrepancies, last = _ref([{"y": "c", "orderId": 23637, "timestamp": 2, "price": P, "quantity": Q, "side": 0, "leverage": L}],
                                        start={P: RESTED}, published={})
    assert matched and last == {}, discrepancies
    assert _dict([{"y": "c", "i": 23637, "t": 2, "p": P, "q": Q, "s": 0, "l": L}], {P: RESTED}) == {}


def test_a_cancellation_of_an_order_the_replay_never_saw_removes_its_own_quantity():
    # the order rested before the snapshot; without a leverage to look up, the cancellation's quantity is taken as is
    matched, discrepancies, last = _ref([{"y": "c", "orderId": 5, "timestamp": 2, "price": 100.0, "quantity": 0.5, "side": 0}],
                                        start={100.0: 1.0}, published={100.0: 0.5})
    assert matched and last == {100.0: 0.5}, discrepancies
    assert _dict([{"y": "c", "i": 5, "t": 2, "p": 100.0, "q": 0.5, "s": 0}], {100.0: 1.0}) == {100.0: 0.5}
