# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The queued (interface or API) orders a validator drains, as totals for its gauges.

After each drain the engine carries that drain's counts, queued_order_counts (MarketEngine._drain_external_orders):
the orders drained, those without the service's receipt stamp (each goes in at delay 0, ahead of every miner), and the
largest delay given. A report samples every minute or so while a drain runs every round, so the engines call
tally() right after each drain and the counts are added up here: every count since the process started, the largest
delay over the last WINDOW_S of drains and the latest drain's delay (each from drains that carried orders). A reading
resets nothing: the validator prepares a report more often than it publishes metrics, so a figure reset by a report
that was never published would be lost. An engine whose drains carry no counts has no totals and publishes no gauges,
so a board reads 'not measured' rather than a zero."""

import time
from collections import deque

DELAY_KEY = "delay_max_ns"
LAST_KEY = "delay_last_ns"
RECEIPT_KEY = "receivedNs"
WINDOW_S = 3600.0
_DELAYS_KEPT = 4096


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _prune(recent, now: float) -> None:
    while recent and recent[0][0] < now - WINDOW_S:
        recent.popleft()


def tally(engine, now: float | None = None) -> None:
    """Add the drain's counts to the engine's totals; nothing when the drain carried none."""
    counts = getattr(engine, "queued_order_counts", None)
    if not isinstance(counts, dict):
        return
    totals = getattr(engine, "_queued_order_totals", None)
    if totals is None:
        totals = {"drained": 0, "unstamped": 0, DELAY_KEY: 0}
        engine._queued_order_totals = totals
    recent = getattr(engine, "_queued_order_delays", None)
    if recent is None:
        recent = engine._queued_order_delays = deque(maxlen=_DELAYS_KEPT)
    t = time.monotonic() if now is None else now
    for key, value in counts.items():
        if key != DELAY_KEY:
            totals[key] = totals.get(key, 0) + _int(value)
    if _int(counts.get("drained")) > 0:
        delay = _int(counts.get(DELAY_KEY))
        recent.append((t, delay))
        totals[LAST_KEY] = delay
    _prune(recent, t)


def health(engine, now: float | None = None) -> dict | None:
    """The totals for the report, None before the first counted drain: the counts since the process started, the
    largest delay over the last WINDOW_S and the latest drain's delay. Reading it changes nothing."""
    totals = getattr(engine, "_queued_order_totals", None)
    if totals is None:
        return None
    recent = getattr(engine, "_queued_order_delays", None)
    if recent is None:
        recent = engine._queued_order_delays = deque(maxlen=_DELAYS_KEPT)
    _prune(recent, time.monotonic() if now is None else now)
    out = dict(totals)
    out[DELAY_KEY] = max((d for _, d in recent), default=0)
    return out


def gauge_name(key: str) -> str:
    """queued_orders for the orders drained, queued_order_delay_max_ns and queued_order_delay_last_ns for the delays,
    queued_orders_<key> otherwise."""
    if key == "drained":
        return "queued_orders"
    if key == DELAY_KEY:
        return "queued_order_delay_max_ns"
    if key == LAST_KEY:
        return "queued_order_delay_last_ns"
    return f"queued_orders_{key}"


def carry_receipt(raw, translated):
    """Put the data service's receipt stamp (receivedNs) from a fetched order onto the instruction it was translated
    into, where the drain reads it. The translation builds a new dict, and without this every queued order would go in
    unstamped at delay 0. Returns translated."""
    received = raw.get(RECEIPT_KEY) if isinstance(raw, dict) else None
    if received is not None and isinstance(translated, dict):
        translated[RECEIPT_KEY] = received
    return translated
