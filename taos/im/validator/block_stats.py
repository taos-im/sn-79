# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Each miner's per-block counters: its request stats and the engine's view of its instructions, kept in one row per
uid, published and reset together per block of rounds (report.round_block_gauges)."""

BLOCK_COUNTERS = (
    "requests", "timeouts", "failures", "rejections",
    "placed", "refused", "refused_contract",
    "placements_sent", "cancels_sent", "cancels_ok", "cancels_failed", "fills", "dropped",
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
