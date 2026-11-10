# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Each miner's resting orders and open loans, summarised for the report.

The state carries every account's open orders ('o') and loans ('l') per book. The dashboards show a summary: per uid
and book the orders by side and the open loans; per uid the same counts, the resting notional by side and the books it
rests on. A book's own best bid and ask and its resting volume per side stay out of the published series: together they
are a maker's quoting. Books with no order and no loan are left out, so the per-book series follow the books a miner
uses.
"""
from typing import NamedTuple

BUY = 0


class BookOrders(NamedTuple):
    bids: int
    asks: int
    bid_volume: float
    ask_volume: float
    best_bid: float | None
    best_ask: float | None
    bid_notional: float
    ask_notional: float
    loans: int


# gauge name -> BookOrders field, in the order the per-book gauges are published
BOOK_GAUGES = {
    'open_bids': 'bids',
    'open_asks': 'asks',
    'open_loans': 'loans',
}


def _book(orders, loans) -> BookOrders:
    bids = asks = 0
    bid_volume = ask_volume = bid_notional = ask_notional = 0.0
    best_bid = best_ask = None
    for o in orders:
        q = float(o.get('q') or 0.0)
        p = o.get('p')
        if o.get('s') == BUY:
            bids += 1
            bid_volume += q
            if p is not None:
                p = float(p)
                bid_notional += p * q
                if best_bid is None or p > best_bid:
                    best_bid = p
        else:
            asks += 1
            ask_volume += q
            if p is not None:
                p = float(p)
                ask_notional += p * q
                if best_ask is None or p < best_ask:
                    best_ask = p
    return BookOrders(bids, asks, bid_volume, ask_volume, best_bid, best_ask, bid_notional, ask_notional,
                      len(loans or ()))


def book_stats(accounts) -> dict:
    """``{uid: {book: BookOrders}}`` over ``{uid: {book: account}}``, for the books with an order or a loan."""
    out = {}
    for uid, books in (accounts or {}).items():
        row = {}
        for book, account in (books or {}).items():
            if not isinstance(account, dict):
                continue
            orders = account.get('o') or ()
            loans = account.get('l') or ()
            if orders or loans:
                row[book] = _book(orders, loans)
        if row:
            out[uid] = row
    return out


def uid_totals(stats) -> dict:
    """Per uid: the orders by side, the resting notional by side, the books it rests orders on and its open loans."""
    out = {}
    for uid, books in stats.items():
        out[uid] = {
            'open_bids': sum(b.bids for b in books.values()),
            'open_asks': sum(b.asks for b in books.values()),
            'resting_bid_notional': sum(b.bid_notional for b in books.values()),
            'resting_ask_notional': sum(b.ask_notional for b in books.values()),
            'order_books': sum(1 for b in books.values() if b.bids or b.asks),
            'open_loans': sum(b.loans for b in books.values()),
        }
    return out


NO_ORDERS = {'open_bids': 0, 'open_asks': 0, 'resting_bid_notional': 0.0, 'resting_ask_notional': 0.0,
             'order_books': 0, 'open_loans': 0}


def book_gauges(row: BookOrders):
    """``(gauge name, value)`` for one book."""
    for name, field in BOOK_GAUGES.items():
        value = getattr(row, field)
        if value is not None:
            yield name, value
