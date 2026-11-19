# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The exchange engine's round timed by phase, per chain block, for the validator's gauges.

Every exchange engine response carries 'phases': the chain block of its batch ('block'), the instructions the batch
carried into the engine's queue ('instructions') and the steady-clock nanoseconds handleBatch spent in each phase:
reconcile (balances and the reconciliation's corrections), reseed (band samples, armed closes, GTT expiry, restored
remainders), match (the instruction queue) and respond (building the response). The engine sends it once a round and a
chain block holds several rounds, so each phase is summed over the rounds that share a block, as the block's figure; the
round trip the validator timed, less the engine's own time, is the part outside the engine (packing and the IPC). A
block is complete when the first round of the next block arrives; a jump of more than one block number in between
counts the blocks that held no round at all (blocks_skipped). A no-op answer (a duplicate or an unprocessable batch)
carries zeros with block 0 and is skipped.

health() hands the report the last completed block's figures and, as <name>_max_ms, the largest a block took over the
last WINDOW_BLOCKS completed blocks (an hour of 12 s blocks), so a reader of any cadence up to an hour sees an
overrun, and instructions_total, every instruction the engine reported since the process started: the last block's
count holds a round's instructions for one block only, about 12 s, which a report or a reader can miss, while the
total's growth cannot be missed. A monitor reads them against one bound per phase. An engine without the key publishes
nothing, so a reader sees 'not measured'.

The first SETTLE_BLOCKS blocks after the process's first round (about five minutes) stay out of the window, so they set
no maximum and count no skipped block: a start's first rounds wait on the engine's own start, which can take seconds
and skip blocks, and the restart itself already tells that. The last block's figures are published from the first
block on, and a slow block or a gap after the settle counts."""
from collections import deque

PHASES = ("reconcile", "reseed", "match", "respond")
WINDOW_BLOCKS = 300
SETTLE_BLOCKS = 25
_NS_PER_MS = 1_000_000


def _round_ms(phases):
    """The round's figures in ms, None when the key is malformed."""
    try:
        out = {p: int(phases.get(f"{p}_ns") or 0) / _NS_PER_MS for p in PHASES}
        out["block"] = int(phases["block"])
        out["instructions"] = int(phases.get("instructions") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    out["engine_round"] = sum(out[p] for p in PHASES)
    return out


def _block_figures(acc):
    """A completed block's figures, keyed as the gauges are named after 'exchange_'."""
    out = {f"phase_{p}_ms": acc[p] for p in PHASES}
    out["engine_round_ms"] = acc["engine_round"]
    if acc["ipc"] is not None:
        out["ipc_ms"] = acc["ipc"]
    out["rounds_per_block"] = acc["rounds"]
    out["block_instructions"] = acc["instructions"]
    out["blocks_skipped"] = acc.get("skipped_after", 0)
    return out


def note(engine, response, elapsed_ns) -> None:
    """Add one response's phases to its block's sums; closes the previous block when a new one starts. `elapsed_ns`
    is the validator's request-to-response time for this round, None when it was not timed."""
    if engine is None or not isinstance(response, dict) or not isinstance(response.get("phases"), dict):
        return
    r = _round_ms(response["phases"])
    if r is None or r["block"] == 0:
        return
    st = engine.__dict__.setdefault("_exchange_phases", {"acc": None, "recent": deque(maxlen=WINDOW_BLOCKS),
                                                         "instructions_total": 0, "first": r["block"]})
    st["instructions_total"] = st.get("instructions_total", 0) + r["instructions"]
    acc = st["acc"]
    if acc is not None and acc["block"] != r["block"]:
        acc["skipped_after"] = max(0, r["block"] - acc["block"] - 1)
        st["last"] = _block_figures(acc)
        if acc["block"] - st.get("first", acc["block"]) >= SETTLE_BLOCKS:
            st["recent"].append(st["last"])
        acc = None
    if acc is None:
        acc = {p: 0.0 for p in PHASES}
        acc.update(block=r["block"], engine_round=0.0, ipc=0.0, rounds=0, instructions=0)
        st["acc"] = acc
    for p in PHASES:
        acc[p] += r[p]
    acc["engine_round"] += r["engine_round"]
    acc["rounds"] += 1
    acc["instructions"] += r["instructions"]
    if elapsed_ns is None or acc["ipc"] is None:
        acc["ipc"] = None
    else:
        acc["ipc"] += max(0.0, int(elapsed_ns) / _NS_PER_MS - r["engine_round"])


def health(engine) -> dict | None:
    """The last completed block's figures, for each _ms figure the largest over the recent blocks as <name>_max_ms,
    and blocks_skipped as the blocks over the same window that held no round, for the report. None before the first
    block completes; while the start settles the maxima are the last block's own and no block counts as skipped."""
    st = getattr(engine, "_exchange_phases", None) if engine is not None else None
    if not st or not st.get("last"):
        return None
    recent = st["recent"]
    out = dict(st["last"])
    for k in out.copy():
        if k.endswith("_ms"):
            out[k[:-3] + "_max_ms"] = max([b[k] for b in recent if k in b] or [out[k]])
    out["blocks_skipped"] = sum(b.get("blocks_skipped", 0) for b in recent)
    out["instructions_total"] = st.get("instructions_total", 0)
    return out


def gauge_name(key: str) -> str:
    """exchange_phase_<phase>_ms, exchange_engine_round_ms, exchange_ipc_ms, exchange_rounds_per_block, ..."""
    return f"exchange_{key}"
