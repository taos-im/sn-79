# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The fundamental's step seed: a drand quicknet round published after the miners' replies are in.

Every SEED_INTERVAL_NS of simulated time the engine draws each book's fundamental step from a seed. The validator sends
it with the batch answering the state whose reply window crosses that boundary: the first quicknet round scheduled
strictly after the validator closed the reply window, so no reply can have been decided on it. The round comes from
drand's HTTP endpoints when two of them serve the same randomness, the randomness checked to be the sha256 of the
beacon's signature (the BLS signature itself is not verified here), else from the chain's Drand pallet, whose pulses the
chain verifies; after RANDOM_AFTER_S without either, 32 random bytes go as round 0, a round no beacon issues, so a
fallback stands out in the record. Each seed sent is appended to the run's fundamental_seed.csv as
round,randomness_hex,sim_time_ns for a recorded replay."""
import asyncio
import hashlib
import os
import time

import bittensor as bt

QUICKNET_CHAIN = "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
GENESIS_S = 1692803367
PERIOD_S = 3
ENDPOINTS = ("https://api.drand.sh", "https://drand.cloudflare.com", "https://api2.drand.sh", "https://api3.drand.sh")
SEED_INTERVAL_NS = 30 * 10 ** 9
RANDOM_AFTER_S = 6.0
HTTP_TIMEOUT_S = 1.0
POLL_S = 0.25
SEED_FILE = "fundamental_seed.csv"


def crosses_seed_boundary(ts_ns: int, interval_ns: int) -> bool:
    """Whether a seed boundary lies in (ts_ns, ts_ns + interval_ns], the reply window of the state at ts_ns."""
    interval_ns = int(interval_ns or 0)
    return interval_ns > 0 and int(ts_ns) // SEED_INTERVAL_NS < (int(ts_ns) + interval_ns) // SEED_INTERVAL_NS


def round_time(r: int) -> float:
    return GENESIS_S + (int(r) - 1) * PERIOD_S


def round_after(t: float) -> int:
    """The first quicknet round scheduled strictly after wall time t."""
    return int((t - GENESIS_S) // PERIOD_S) + 2


def randomness_of(beacon: dict, r: int) -> bytes | None:
    """A beacon's randomness when it is round r's and is the sha256 of its signature, else None."""
    try:
        if int(beacon["round"]) != int(r):
            return None
        sig = bytes.fromhex(beacon["signature"])
        rnd = bytes.fromhex(beacon["randomness"])
    except (KeyError, TypeError, ValueError):
        return None
    return rnd if len(rnd) == 32 and hashlib.sha256(sig).digest() == rnd else None


def _hex(v) -> str:
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).hex()
    if isinstance(v, (list, tuple)):
        return bytes(v).hex()
    s = str(v)
    return s[2:] if s.startswith("0x") else s


def pulse_randomness(pulse: dict, r: int) -> bytes | None:
    """A chain Drand pulse ({round, randomness, signature}) read as a beacon."""
    try:
        return randomness_of({"round": pulse["round"], "randomness": _hex(pulse["randomness"]),
                              "signature": _hex(pulse["signature"])}, r)
    except (KeyError, TypeError, ValueError):
        return None


def agreed(values) -> tuple:
    """(the randomness at least two endpoints served, or None; whether two served different ones)."""
    got = [v for v in values if v is not None]
    if not got:
        return None, False
    best = max(set(got), key=got.count)
    return (best if got.count(best) >= 2 else None), len(set(got)) > 1


async def http_randomness(r: int, on_disagree=None) -> bytes | None:
    """Round r's randomness when two of drand's endpoints serve the same checked value, else None; on_disagree() is
    called when two serve different ones."""
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)

    async def one(session, base):
        try:
            async with session.get(f"{base}/{QUICKNET_CHAIN}/public/{int(r)}") as resp:
                if resp.status != 200:
                    return None
                return randomness_of(await resp.json(content_type=None), r)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return None

    values = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.ensure_future(one(session, base)) for base in ENDPOINTS]
        try:
            for done in asyncio.as_completed(tasks):
                values.append(await done)
                if agreed(values)[0] is not None:
                    break
        finally:
            for t in tasks:
                t.cancel()
    got, disagreed = agreed(values)
    if disagreed and on_disagree is not None:
        on_disagree()
    return got


def chain_randomness_getter(subtensor):
    async def get(r: int) -> bytes | None:
        def query():
            try:
                pulse = subtensor.substrate.query("Drand", "Pulses", [int(r)])
                value = getattr(pulse, "value", pulse)
                return pulse_randomness(value, r) if value else None
            except Exception:
                bt.logging.debug(f"[SEED] chain Drand pulse {r} unavailable", exc_info=True)
                return None
        return await asyncio.to_thread(query)
    return get


async def seed_for(closed_at: float, http_get, chain_get, now=time.time, sleep=asyncio.sleep):
    """(round, randomness, source) for a reply window closed at wall time closed_at: the first round after it, from
    drand or the chain once it is scheduled, else 32 random bytes as round 0 after RANDOM_AFTER_S."""
    r = round_after(closed_at)
    while True:
        t = now()
        if t - closed_at >= RANDOM_AFTER_S:
            return 0, os.urandom(32), "random"
        if t >= round_time(r):
            got = await http_get(r)
            if got is not None:
                return r, got, "drand"
            got = await chain_get(r)
            if got is not None:
                return r, got, "chain"
            await sleep(POLL_S)
        else:
            await sleep(min(POLL_S, round_time(r) - t))


def _record(v, r: int, randomness: bytes, ts_ns: int) -> None:
    log_dir = getattr(getattr(v, "simulation", None), "logDir", None)
    if not log_dir:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, SEED_FILE), "a") as fh:
            fh.write(f"{int(r)},{randomness.hex()},{int(ts_ns)}\n")
    except OSError:
        bt.logging.warning(f"[SEED] could not record round {r} in {log_dir}", exc_info=True)


async def attach(v, response: dict, ts_ns: int, closed_at: float, seed_fn=seed_for) -> dict:
    """The batch with the step seed beside its responses when the state at ts_ns ends a seed interval; the batch
    unchanged otherwise. The seed is recorded and its source counted (v.fundamental_seed_sources)."""
    if not crosses_seed_boundary(ts_ns, getattr(v.simulation, "publish_interval", 0)):
        return response
    subtensor = getattr(v, "subtensor", None)
    sources = getattr(v, "fundamental_seed_sources", None)
    if not isinstance(sources, dict):
        sources = v.fundamental_seed_sources = {}

    def count(key):
        sources[key] = sources.get(key, 0) + 1

    async def http_get(r):
        return await http_randomness(r, on_disagree=lambda: count("drand_disagree"))

    async def no_chain(r):
        return None
    r, randomness, source = await seed_fn(closed_at, http_get,
                                          chain_randomness_getter(subtensor) if subtensor is not None else no_chain)
    count(source)
    if source != "drand":
        bt.logging.warning(f"[SEED] step seed at {ts_ns} from {source} (round {r})")
    _record(v, r, randomness, ts_ns)
    return {**response, "seed": [int(r), randomness]}
