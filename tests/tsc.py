# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Shared shapes for the scoring-shadow tests.

A state built with notices but no books made the volume-slot loop (trade._process_uid_trade_volumes, which creates each
uid's slot per book the state carries) create nothing, so every uid with a fill raised at the fill write, the per-uid
guard swallowed it on main and on the child alike, and the parity tests passed while no trading uid's FIFO, realized
P&L, round trips or inventory ever ran. fill_state gives each fill's book a touch, so the accounting runs."""
from collections import defaultdict
from types import SimpleNamespace

from taos.im.validator.engines import NormalizedTradeEvent


def fill_books(trades, book_count=0):
    """{book: a touch} for every book in range(book_count) and every book the trades name, the touch at the fill's
    price where there is one. Every book, as a live state carries every book: the per-uid slot cache
    (trade._process_uid_trade_volumes) takes the first book's slot for the sampled interval as proof that every
    book's slot exists, which holds only when every state carries every book; a state naming a subset leaves a
    later book without its slot, the fill write raises into the guard, and main (its cache warm) and a freshly
    rebuilt child (no cache) account differently."""
    books = {b: {"b": [{"p": 100.0, "q": 1.0}], "a": [{"p": 100.0, "q": 1.0}], "e": []} for b in range(book_count)}
    for _taker, _maker, book, _qty, price, _side in trades:
        books[book] = {"b": [{"p": price, "q": 1.0}], "a": [{"p": price, "q": 1.0}], "e": []}
    return books


def fill_state(ts, trades, book_count=0, maker_fee_rate=0.001, taker_fee_rate=0.002):
    """Duck-typed state as trade.py reads it (.timestamp/.notices/.books/.accounts), with a book for every id in
    range(book_count) and for every fill. trades: list of (taker_uid, maker_uid, book, qty, price, side)."""
    notices = defaultdict(list)
    for taker, maker, book, qty, price, side in trades:
        ev = NormalizedTradeEvent(
            book_id=book, quantity=qty, price=price, side=side,
            maker_uid=maker, taker_uid=taker, maker_fee=qty * price * maker_fee_rate,
            taker_fee=qty * price * taker_fee_rate, timestamp=ts,
        ).to_notice_dict()
        notices[taker].append(ev)
        if maker != taker:
            notices[maker].append(ev)
    return SimpleNamespace(timestamp=ts, notices=dict(notices), books=fill_books(trades, book_count), accounts={})
