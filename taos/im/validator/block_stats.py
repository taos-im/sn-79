# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Each miner's per-block counters: its request stats and the engine's view of its instructions, kept in one row per
uid, published and reset together per block of rounds (report.round_block_gauges)."""

BLOCK_COUNTERS = (
    "requests", "timeouts", "failures", "rejections",
    "placed", "refused", "refused_contract",
    "placements_sent", "cancels_sent", "cancel_orders_sent", "orders_cancelled", "orders_not_found", "stp_cancels",
    "fills", "dropped",
)


def empty_block_stats() -> dict:
    row = {name: 0 for name in BLOCK_COUNTERS}
    row["call_time"] = []
    return row


def block_row(stats: dict, uid) -> dict:
    """The uid's row, created empty when absent. A row restored from an older state may lack the newer counters, so
    every reader takes them with a default of zero."""
    return stats.setdefault(uid, empty_block_stats())


def bump(stats: dict, uid, name: str, by: int = 1) -> None:
    row = block_row(stats, uid)
    row[name] = row.get(name, 0) + by


PLACEMENT_TYPES = frozenset({"PLACE_ORDER_LIMIT", "PLACE_ORDER_MARKET"})


def sent_counts(responses) -> tuple[dict, list]:
    """What the validator sends the engine in a round, after validation and the delays: per miner, its placements, its
    cancel instructions and the orders they name, and each named order as (uid, book, order id). Main counts it this way,
    and the query process where it packs the orders (neuron.orders_packed), so both paths count alike."""
    rows, named = {}, []
    for response in responses or ():
        uid = getattr(response, 'agent_id', None)
        if uid is None:
            continue
        for instruction in getattr(response, 'instructions', None) or ():
            t = getattr(instruction, 'type', None)
            if t in PLACEMENT_TYPES:
                row = rows.setdefault(uid, {})
                row['placements_sent'] = row.get('placements_sent', 0) + 1
            elif t == 'CANCEL_ORDERS':
                row = rows.setdefault(uid, {})
                book = getattr(instruction, 'bookId', None)
                ids = [getattr(c, 'orderId', None) for c in getattr(instruction, 'cancellations', None) or ()]
                row['cancels_sent'] = row.get('cancels_sent', 0) + 1
                row['cancel_orders_sent'] = row.get('cancel_orders_sent', 0) + len(ids)
                named.extend((uid, book, oid) for oid in ids)
    return rows, named
