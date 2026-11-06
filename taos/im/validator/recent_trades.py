# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The recent trades the report carries: each miner's newest trades across its books, and every book's newest fee
rates.

The validator keeps each miner's last trades per book (recent_miner_trades, ``{uid: {book: [(trade, role), ...]}}``,
oldest first). The report's table of a miner's recent trades shows the newest of them, so the payload carries the
RECENT_TRADES_PER_UID newest of each miner, each book's in the order kept; the per-book fee-rate gauges get the newest
maker and taker rate of every book separately.
"""
import heapq

RECENT_TRADES_PER_UID = 50


def _rate(trade, role):
    # a trade restored from a state saved before the fee fields existed has neither
    try:
        return trade.makerFeeRate if role == 'maker' else trade.takerFeeRate
    except AttributeError:
        return None


def book_fee_rates(trades) -> list:
    """``[maker rate, taker rate]`` of the newest trade of each role that carries its fee, None for a role with none."""
    maker = taker = None
    for trade, role in reversed(trades):
        if role == 'maker' and maker is None:
            maker = _rate(trade, role)
        elif role == 'taker' and taker is None:
            taker = _rate(trade, role)
        if maker is not None and taker is not None:
            break
    return [maker, taker]


def newest_per_uid(recent_miner_trades, keep=RECENT_TRADES_PER_UID):
    """``({uid: {book: [{'trade': dump, 'role': role}, ...]}}, {uid: {book: [maker rate, taker rate]}})``: each uid's
    ``keep`` newest trades across its books, each book's oldest first as kept, and the newest fee rates of every book
    the uid traded on."""
    kept, rates = {}, {}
    for uid, books in recent_miner_trades.items():
        candidates = []
        uid_rates = {}
        for book, trades in books.items():
            if not trades:
                continue
            uid_rates[book] = book_fee_rates(trades)
            for idx, (trade, _role) in enumerate(trades):
                t = getattr(trade, 't', None)
                candidates.append((t if t is not None else -1, book, idx))
        if uid_rates:
            rates[uid] = uid_rates
        chosen = {}
        for _t, book, idx in heapq.nlargest(keep, candidates):
            chosen.setdefault(book, []).append(idx)
        kept[uid] = {
            book: [{'trade': books[book][i][0].model_dump(), 'role': books[book][i][1]} for i in sorted(idxs)]
            for book, idxs in chosen.items()
        }
    return kept, rates
