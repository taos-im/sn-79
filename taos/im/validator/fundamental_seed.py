# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The fundamental's step seed: a validator secret over a drand quicknet round published after the miners' replies.

Every SEED_INTERVAL_NS of simulated time the engine draws each book's fundamental step from a seed. The validator sends
it with the batch answering the state whose reply window crosses that boundary. The round is the first quicknet round
scheduled MARGIN_S (one period) after the validator closed the reply window, and never below the first round after
drand's own latest at the close, so a host clock slow by a period, or by more, still picks a round no reply can have been
decided on; the close against the latest round's schedule is kept as a clock-skew sample (clock_skew_gauge: negative
when the host clock runs slow). The round comes from drand's HTTP endpoints when endpoints of two operators serve the
same randomness, the randomness checked to be the sha256 of the beacon's signature (the BLS signature itself is not
verified here); when only one operator's endpoints agree, the chain's Drand pallet, whose pulses the chain verifies, is
asked first and the single operator's value stands only in its absence, counted apart; after RANDOM_AFTER_S without
any, 32 random bytes go as round 0, a round no beacon issues, so a fallback stands out in the record.

The value sent is HMAC-SHA256(secret, randomness): the validator draws a 32-byte secret when a run starts and commits
to it (its sha256 in the run's fundamental_seed_commit.json and on the report's info metric), and reveals it when the
next run starts (fundamental_seed_secret.json in the ended run's directory and the report's reveal metric), so anyone
can check afterwards that every seed came from the public round and none was chosen, by the validator or anyone, while
in-run nobody rebuilds the fundamental's innovations from the public rounds. The engine and a recorded replay read the
first 8 bytes of the value sent, appended to the run's fundamental_seed.csv as round,value_hex,sim_time_ns; the public
input goes beside it in fundamental_seed_drand.csv as round,randomness_hex,sim_time_ns,source."""
import asyncio
import hashlib
import hmac
import json
import os
import time

import bittensor as bt

QUICKNET_CHAIN = "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
GENESIS_S = 1692803367
PERIOD_S = 3
MARGIN_S = PERIOD_S
ENDPOINTS = ("https://api.drand.sh", "https://drand.cloudflare.com", "https://api2.drand.sh", "https://api3.drand.sh")
OPERATOR = {"https://api.drand.sh": "drand.sh", "https://api2.drand.sh": "drand.sh", "https://api3.drand.sh": "drand.sh",
            "https://drand.cloudflare.com": "cloudflare"}
LATEST_ENDPOINTS = ("https://drand.cloudflare.com", "https://api.drand.sh")
SEED_INTERVAL_NS = 30 * 10 ** 9
RANDOM_AFTER_S = 6.0
HTTP_TIMEOUT_S = 1.0
POLL_S = 0.25
CLOCK_SAMPLES = 20
SEED_FILE = "fundamental_seed.csv"
DRAND_FILE = "fundamental_seed_drand.csv"
COMMIT_FILE = "fundamental_seed_commit.json"
REVEAL_FILE = "fundamental_seed_secret.json"


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
    """values: (endpoint, randomness or None) pairs. (the randomness at least two endpoints served, or None; "drand"
    when endpoints of two operators served it, "drand_one_operator" when one operator's did, else None; whether two
    endpoints served different values)."""
    got = [(b, v) for b, v in values if v is not None]
    if not got:
        return None, None, False
    operators, endpoints = {}, {}
    for b, v in got:
        operators.setdefault(v, set()).add(OPERATOR.get(b, b))
        endpoints[v] = endpoints.get(v, 0) + 1
    disagreed = len(endpoints) > 1
    by_two = [v for v, ops in operators.items() if len(ops) >= 2]
    if by_two:
        return by_two[0], "drand", disagreed
    best = max(endpoints, key=endpoints.get)
    if endpoints[best] >= 2:
        return best, "drand_one_operator", disagreed
    return None, None, disagreed


async def http_randomness(r: int, on_disagree=None) -> tuple | None:
    """(round r's randomness, "drand" or "drand_one_operator") when two endpoints serve the same checked value, else
    None; the first kind needs two operators. on_disagree() is called when two serve different values."""
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)

    async def one(session, base):
        try:
            async with session.get(f"{base}/{QUICKNET_CHAIN}/public/{int(r)}") as resp:
                if resp.status != 200:
                    return base, None
                return base, randomness_of(await resp.json(content_type=None), r)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return base, None

    values = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.ensure_future(one(session, base)) for base in ENDPOINTS]
        try:
            for done in asyncio.as_completed(tasks):
                values.append(await done)
                if agreed(values)[1] == "drand":
                    break
        finally:
            for t in tasks:
                t.cancel()
    got, kind, disagreed = agreed(values)
    if disagreed and on_disagree is not None:
        on_disagree()
    return (got, kind) if got is not None else None


async def http_latest() -> int | None:
    """The highest round drand's endpoints report as their latest, or None when none answers in time."""
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)

    async def one(session, base):
        try:
            async with session.get(f"{base}/{QUICKNET_CHAIN}/public/latest") as resp:
                if resp.status != 200:
                    return None
                return int((await resp.json(content_type=None))["round"])
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, TypeError):
            return None

    async with aiohttp.ClientSession(timeout=timeout) as session:
        got = [x for x in await asyncio.gather(*(one(session, b) for b in LATEST_ENDPOINTS)) if x is not None]
    return max(got) if got else None


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


async def seed_for(closed_at: float, http_get, chain_get, now=time.time, sleep=asyncio.sleep, min_round: int = 0):
    """(round, randomness, source) for a reply window closed at wall time closed_at: the first round MARGIN_S after it
    and not below min_round (the first after drand's latest at the close), from drand or the chain once it is
    scheduled, else 32 random bytes as round 0 after RANDOM_AFTER_S past the margin."""
    scheduled = round_after(closed_at + MARGIN_S)
    r = max(scheduled, int(min_round or 0))
    # drand's latest ahead of the local schedule means the host clock runs slow by more than the margin: the round is
    # then asked for at once, as drand publishes it, instead of waiting for the local clock to reach its schedule
    anchored = r > scheduled
    while True:
        t = now()
        if t - closed_at >= RANDOM_AFTER_S + MARGIN_S:
            return 0, os.urandom(32), "random"
        if anchored or t >= round_time(r):
            got = await http_get(r)
            if got is not None and got[1] == "drand":
                return r, got[0], "drand"
            from_chain = await chain_get(r)
            if from_chain is not None:
                return r, from_chain, "chain"
            if got is not None:
                return r, got[0], got[1]
            await sleep(POLL_S)
        else:
            await sleep(min(POLL_S, round_time(r) - t))


# --- the clock against drand ---------------------------------------------------------------------------------------------
def clock_skew_sample(closed_at: float, latest) -> float | None:
    """The close on the host clock against the latest published round's schedule; None without a latest. A right
    clock reads between 0 and a period plus the endpoint's lag; a negative sample means the host clock runs slow."""
    return None if latest is None else float(closed_at) - round_time(int(latest))


def note_clock_skew(v, sample) -> None:
    if sample is None:
        return
    samples = getattr(v, "fundamental_seed_clock", None)
    if not isinstance(samples, list):
        samples = v.fundamental_seed_clock = []
    samples.append(float(sample))
    del samples[:-CLOCK_SAMPLES]


def clock_skew_gauge(v) -> float | None:
    samples = getattr(v, "fundamental_seed_clock", None) or []
    return min(samples) if samples else None


# --- the secret ----------------------------------------------------------------------------------------------------------
def step_value(secret: bytes, randomness: bytes) -> bytes:
    return hmac.new(secret, randomness, hashlib.sha256).digest()


def _write_json(path, payload) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh)
    except OSError:
        bt.logging.warning(f"[SEED] could not write {path}", exc_info=True)


def _reveal(v, old: dict, log_root, count) -> None:
    v.fundamental_seed_reveal = {"run": old["run"], "commit": old["commit"], "secret": old["secret"]}
    if log_root:
        _write_json(os.path.join(log_root, old["run"], REVEAL_FILE),
                    {**v.fundamental_seed_reveal, "revealed_at": time.time()})
    count("reveal")
    bt.logging.info(f"[SEED] run {old['run']}'s seed secret revealed, commitment {old['commit'][:16]}")


def secret_for(v, log_dir, count) -> bytes:
    """The run's secret, drawn and committed at its first seed boundary; a run change reveals the previous one."""
    run = os.path.basename(os.path.normpath(log_dir)) if log_dir else ""
    log_root = os.path.dirname(os.path.normpath(log_dir)) if log_dir else None
    current = getattr(v, "fundamental_seed_secret", None)
    if isinstance(current, dict) and current.get("run") == run and current.get("secret"):
        return bytes.fromhex(current["secret"])
    if isinstance(current, dict) and current.get("secret"):
        _reveal(v, current, log_root, count)
    else:
        v.fundamental_seed_reveal = getattr(v, "fundamental_seed_reveal", None)
    secret = os.urandom(32)
    commit = hashlib.sha256(secret).hexdigest()
    v.fundamental_seed_secret = {"run": run, "secret": secret.hex(), "commit": commit}
    if log_dir:
        _write_json(os.path.join(log_dir, COMMIT_FILE), {"run": run, "commit": commit, "committed_at": time.time()})
    count("commit")
    bt.logging.info(f"[SEED] run {run}: seed secret committed, sha256 {commit[:16]}")
    return secret


def state_to_save(v) -> dict:
    """What the validator's state file keeps so a restart mid-run continues the same secret and clock samples."""
    return {"fundamental_seed_secret": getattr(v, "fundamental_seed_secret", None),
            "fundamental_seed_reveal": getattr(v, "fundamental_seed_reveal", None),
            "fundamental_seed_clock": list(getattr(v, "fundamental_seed_clock", None) or [])}


def restore_state(v, data) -> None:
    data = data or {}
    for key in ("fundamental_seed_secret", "fundamental_seed_reveal"):
        value = data.get(key)
        setattr(v, key, dict(value) if isinstance(value, dict) else None)
    v.fundamental_seed_clock = [float(x) for x in (data.get("fundamental_seed_clock") or [])]


def report_fields(v) -> dict:
    """What the report publishes: the current run's commitment, the ended run's reveal, the clock samples. Never the
    current secret."""
    current = getattr(v, "fundamental_seed_secret", None) or {}
    commit = {"run": current["run"], "commit": current["commit"]} if current.get("commit") else None
    reveal = getattr(v, "fundamental_seed_reveal", None)
    return {"fundamental_seed_commit": commit,
            "fundamental_seed_reveal": dict(reveal) if isinstance(reveal, dict) else None,
            "fundamental_seed_clock": list(getattr(v, "fundamental_seed_clock", None) or [])}


def _record(v, r: int, value: bytes, randomness: bytes, source: str, ts_ns: int) -> None:
    log_dir = getattr(getattr(v, "simulation", None), "logDir", None)
    if not log_dir:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, SEED_FILE), "a") as fh:
            fh.write(f"{int(r)},{value.hex()},{int(ts_ns)}\n")
        with open(os.path.join(log_dir, DRAND_FILE), "a") as fh:
            fh.write(f"{int(r)},{randomness.hex()},{int(ts_ns)},{source}\n")
    except OSError:
        bt.logging.warning(f"[SEED] could not record round {r} in {log_dir}", exc_info=True)


async def attach(v, response: dict, ts_ns: int, closed_at: float, seed_fn=seed_for, latest_fn=http_latest) -> dict:
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
    secret = secret_for(v, getattr(v.simulation, "logDir", None), count)
    latest = await latest_fn()
    note_clock_skew(v, clock_skew_sample(closed_at, latest))
    r, randomness, source = await seed_fn(closed_at, http_get,
                                          chain_randomness_getter(subtensor) if subtensor is not None else no_chain,
                                          min_round=(int(latest) + 1) if latest else 0)
    count(source)
    if source != "drand":
        bt.logging.warning(f"[SEED] step seed at {ts_ns} from {source} (round {r})")
    value = step_value(secret, randomness)
    _record(v, r, value, randomness, source, ts_ns)
    return {**response, "seed": [int(r), value]}
