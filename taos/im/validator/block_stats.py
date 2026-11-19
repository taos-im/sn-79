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


# kind -> (timestamped history, running sum): messages and fills for the order-to-trade reading; fills taken and made
# under the book's dust floor for the dust-fill reading (measured, not paid)
_OTR_KINDS = {
    "msg": ("debeta_msg_hist", "debeta_msgs"), "fill": ("debeta_fill_hist", "debeta_fills"),
    "dust_take": ("debeta_dusttake_hist", "debeta_dusttakes"),
    "dust_make": ("debeta_dustmake_hist", "debeta_dustmakes"),
}


def note_otr(self, uid, kind: str, ts, n: int = 1) -> None:
    """Counts a message (a placement or a cancel the engine applied) or a fill, `n` of them at once when a replay has
    a state's count ready, into the uid's timestamped history for the order-to-trade reading (debeta_msg_hist /
    debeta_fill_hist, {uid: {ts: count}}) and its running sum (debeta_msgs / debeta_fills), kept over the making window
    like the de-beta histories. Nothing is counted without a state timestamp."""
    if ts is None or n <= 0:
        return
    hist_name, run_name = _OTR_KINDS.get(kind, ("debeta_msg_hist", "debeta_msgs"))
    hist = self.__dict__.setdefault(hist_name, {})
    running = self.__dict__.setdefault(run_name, {})
    row = hist.setdefault(uid, {})
    row[ts] = row.get(ts, 0) + int(n)
    running[uid] = running.get(uid, 0) + int(n)


def dust_fill_shares(self) -> dict:
    """{uid: {takes, makes, take_share}} over the making window: the fills the uid took and made under their print's
    dust floor (the capture's per-print floor, debeta._push_floor_reference) and the takes' share of the uid's fills;
    a uid with neither has no reading. Published per uid so the board names who prints the dust the floor excludes;
    nothing is paid or discounted on it."""
    takes = getattr(self, "debeta_dusttakes", None) or {}
    makes = getattr(self, "debeta_dustmakes", None) or {}
    fills = getattr(self, "debeta_fills", None) or {}
    out = {}
    for uid in set(takes) | set(makes):
        t, k = int(takes.get(uid, 0) or 0), int(makes.get(uid, 0) or 0)
        if t <= 0 and k <= 0:
            continue
        f = int(fills.get(uid, 0) or 0)
        out[uid] = {"takes": t, "makes": k, "take_share": (t / f) if f > 0 else 0.0}
    return out


def messages_per_fill(self) -> dict:
    """{uid: messages per fill over the making window}: the running message and fill sums' ratio, inf for a uid with
    messages and no fills; a uid with neither has no reading."""
    msgs = getattr(self, "debeta_msgs", None) or {}
    fills = getattr(self, "debeta_fills", None) or {}
    out = {}
    for uid in set(msgs) | set(fills):
        m, f = int(msgs.get(uid, 0) or 0), int(fills.get(uid, 0) or 0)
        if m <= 0 and f <= 0:
            continue
        out[uid] = (m / f) if f > 0 else float("inf")
    return out


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
