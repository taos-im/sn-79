# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Whether a subnet can settle a trade at all.

A subnet with ``SubtokenEnabled = False`` rejects every stake extrinsic, so an order there can
never settle however healthy its pool looks. On the seeded localnet netuid 90 holds 81k alpha in
its pool with staking switched off, and netuid 86 likewise. Measured consequence: a stake returns

    Subtensor returned `SubtokenDisabled(Module)` error. This means: `SubToken disabled now`

after the fact. Carrying the flag on the pool lets the UI mark the subnet and stop a miner
trading it, instead of the only feedback being a failed settlement.

Read rarely: this changes when a subnet owner toggles it, not per block, so it is cached.
"""
from __future__ import annotations

import threading
import time

# Long by design. A subnet owner toggling staking is a rare administrative act, and querying
# every subnet each block would cost far more than the flag is worth.
_TTL_SECONDS = 600

_cache: dict[int, bool] = {}
_cache_at: float = 0.0
# The netuids the last sweep ASKED about, not the ones it managed to read. A subnet whose query
# fails is deliberately omitted from _cache, so testing `wanted <= set(_cache)` re-queried EVERY
# subnet on EVERY call for as long as any single one kept failing -- the cache existed but could
# never be hit. Recording the attempt makes the TTL bound the work, not the success rate.
_attempted: set[int] = set()
_refreshing = threading.Lock()


def merge_subtoken_flags(pools, flags) -> dict:
    """Attach ``subtoken_enabled`` to each pool the chain was asked about.

    A subnet with no answer is left ABSENT rather than defaulted to True: a stale or failed
    lookup must not be able to present a closed subnet as tradable. Never drops a pool, never
    mutates the input, and never raises on a malformed entry, because this runs inside the
    ingest payload builder where an exception would cost the whole payload.
    """
    if not pools:
        return {}
    flags = flags or {}
    out: dict = {}
    for key, pool in pools.items():
        if not isinstance(pool, dict):
            out[key] = pool
            continue
        try:
            nid = int(key)
        except (TypeError, ValueError):
            out[key] = dict(pool)
            continue
        entry = dict(pool)
        if nid in flags:
            entry["subtoken_enabled"] = bool(flags[nid])
        out[key] = entry
    return out


def subtoken_flags(substrate, netuids, *, now=None, cached_only: bool = False) -> dict[int, bool]:
    """``{netuid: enabled}`` for the given subnets, cached for _TTL_SECONDS.

    Returns whatever it managed to read. A subnet whose query fails is omitted rather than
    guessed, so merge_subtoken_flags leaves it absent and no caller can mistake a failed read
    for permission to trade.
    """
    global _cache, _cache_at, _attempted
    t = time.time() if now is None else now
    wanted = {int(n) for n in (netuids or [])}
    fresh = _cache_at > 0.0 and (t - _cache_at) < _TTL_SECONDS and wanted <= _attempted
    if fresh or substrate is None:
        return dict(_cache)
    if cached_only:
        # NEVER BLOCK A CALLER THAT CANNOT AFFORD TO. substrate.query is a synchronous websocket
        # round-trip; doing ~128 of them inline in the validator's async handle_state froze the
        # event loop for seconds at a time and delayed everything behind it, including the block
        # push. The stale value is served immediately and the sweep runs on its own thread.
        _refresh_async(substrate, wanted, t)
        return dict(_cache)
    _sweep(substrate, wanted, t)
    return dict(_cache)


def _sweep(substrate, wanted: set[int], t: float) -> None:
    """Query every wanted subnet and replace the cache. Blocking; call off the event loop."""
    global _cache, _cache_at, _attempted
    found: dict[int, bool] = {}
    for nid in sorted(wanted):
        try:
            v = substrate.query("SubtensorModule", "SubtokenEnabled", [nid])
            found[nid] = bool(getattr(v, "value", v))
        except Exception:
            continue
    # Stamp the attempt even when nothing was read, so a subnet that always fails cannot pin the
    # cache open and turn every call back into a full sweep.
    _cache = found or _cache
    _attempted = set(wanted)
    _cache_at = t


def _refresh_async(substrate, wanted: set[int], t: float) -> None:
    """Run one sweep on a daemon thread, at most one at a time."""
    if not _refreshing.acquire(blocking=False):
        return

    def _run():
        try:
            _sweep(substrate, wanted, t)
        finally:
            _refreshing.release()

    threading.Thread(target=_run, name="subtoken-refresh", daemon=True).start()


def reset_cache() -> None:
    """For tests."""
    global _cache, _cache_at
    _cache, _cache_at = {}, 0.0
