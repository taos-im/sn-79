# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The fundamental's step seed: a validator secret over a drand quicknet round published after the miners' replies.

At every seedInterval of simulated time (each class file's FundamentalPrice seedInterval; SEED_INTERVAL_NS where a
layout states none) the engine draws each book's fundamental step from a seed. The validator sends it with the batch
answering the state whose reply window crosses that boundary. The round is the first quicknet round
scheduled MARGIN_S (one period) after the validator closed the reply window, and never below the first round after
drand's own latest at the close, so a host clock slow by a period, or by more, still picks a round no reply can have been
decided on; the close against the latest round's schedule is kept as a clock-skew sample (clock_skew_gauge: negative
when the host clock runs slow). The round comes from drand's HTTP endpoints when endpoints of two operators serve the
same randomness, the randomness checked to be the sha256 of the beacon's signature and, where the chainlayer's
verifier is present (mvtrx_chainlayer.verify_quicknet_beacon), the BLS signature itself checked against
quicknet's chain key (a beacon that fails is dropped and counted drand_bad_signature; a round taken without the verifier
is counted drand_unverified); when only one operator's endpoints agree, the chain's Drand pallet, whose pulses the chain verifies, is
asked first and the single operator's value stands only in its absence, counted apart; after RANDOM_AFTER_S without
any, 32 random bytes go as round 0, a round no beacon issues, so a fallback stands out in the record. drand's endpoints
are asked through one HTTP session per event loop, and the chain's pulses over a subtensor connection of the seed's
own, never the maintenance subtensor's websocket from the seed's thread. --simulation.seeding.drand off skips drand: no
endpoint is asked, the chain's pulse is taken where there is a chain, and round 0 goes at once where there is none.

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
SEED_INTERVAL_NS = 30 * 10 ** 9   # the interval where a layout's classes state no seedInterval
RANDOM_AFTER_S = 6.0
HTTP_TIMEOUT_S = 1.0
POLL_S = 0.25
CLOCK_SAMPLES = 20
SEED_FILE = "fundamental_seed.csv"
DRAND_FILE = "fundamental_seed_drand.csv"
COMMIT_FILE = "fundamental_seed_commit.json"
REVEAL_FILE = "fundamental_seed_secret.json"


def seed_intervals(sim) -> tuple:
    """The seed intervals of a layout, ascending: each class file's FundamentalPrice seedInterval, the engine's own
    reseed interval, so the validator reads it where the engine does; (SEED_INTERVAL_NS,) where no
    class states one."""
    try:
        classes = sim.asset_classes()
    except (AttributeError, TypeError, KeyError):
        classes = []
    got = {int(i) for c in classes if (i := getattr(getattr(c, "config", None), "fp_seed_interval", None))}
    return tuple(sorted(got)) or (SEED_INTERVAL_NS,)


def crosses_seed_boundary(ts_ns: int, interval_ns: int, seed_intervals_ns=(SEED_INTERVAL_NS,)) -> bool:
    """Whether a seed boundary of any of the seed intervals lies in (ts_ns, ts_ns + interval_ns], the reply window of
    the state at ts_ns."""
    interval_ns = int(interval_ns or 0)
    return interval_ns > 0 and any(
        int(ts_ns) // int(k) < (int(ts_ns) + interval_ns) // int(k) for k in seed_intervals_ns if int(k) > 0)


def drand_enabled(v) -> bool:
    """--simulation.seeding.drand: on unless the validator's configuration turns it off."""
    on = getattr(getattr(getattr(getattr(v, "config", None), "simulation", None), "seeding", None), "drand", True)
    return on is not False and on is not None


def round_time(r: int) -> float:
    return GENESIS_S + (int(r) - 1) * PERIOD_S


def round_after(t: float) -> int:
    """The first quicknet round scheduled strictly after wall time t."""
    return int((t - GENESIS_S) // PERIOD_S) + 2


_SESSIONS: dict = {}   # event loop -> the one aiohttp session drand's endpoints are asked through on it


def _http_session():
    """The HTTP session for drand's endpoints: one per event loop rather than one per poll. A session belongs
    to the loop it was opened on, so a loop of its own (a test's asyncio.run) opens its own; those of closed loops are
    let go."""
    import aiohttp
    loop = asyncio.get_running_loop()
    for old in [lp for lp in _SESSIONS if lp.is_closed()]:
        _SESSIONS.pop(old, None)
    s = _SESSIONS.get(loop)
    if s is None or getattr(s, "closed", False):
        s = _SESSIONS[loop] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S))
    return s


_VERIFIER = None   # the chainlayer's verify_quicknet_beacon once imported; False when the module or the function is absent


def beacon_verifier():
    """The chainlayer's quicknet signature verifier (round, signature_hex) -> bool, or None where it is not installed
    (the public tree ships without the chainlayer; the seed then counts its rounds as drand_unverified)."""
    global _VERIFIER
    if _VERIFIER is None:
        try:
            from mvtrx_chainlayer import verify_quicknet_beacon
            _VERIFIER = verify_quicknet_beacon
        except ImportError:
            _VERIFIER = False
    return _VERIFIER or None


def signature_verified(beacon: dict, r: int, verifier) -> bool | None:
    """True or False when `verifier` checks the beacon's BLS signature on round r; None without a verifier. A beacon
    without a readable signature, or one the verifier cannot read as a point, is False."""
    if verifier is None:
        return None
    try:
        return bool(verifier(int(r), str(beacon["signature"])))
    except (KeyError, TypeError, ValueError):
        return False


def accept_beacon(beacon: dict, r: int, verifier, on_bad_signature=None) -> tuple:
    """(the beacon's randomness or None, whether its signature was verified): the randomness check (randomness_of)
    then the signature check; a beacon that fails the signature check gives None and on_bad_signature() is called."""
    rnd = randomness_of(beacon, r)
    if rnd is None:
        return None, False
    verified = signature_verified(beacon, r, verifier)
    if verified is False:
        if on_bad_signature is not None:
            on_bad_signature()
        return None, False
    return rnd, bool(verified)


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


_DEFAULT = object()


async def http_randomness(r: int, on_disagree=None, verifier=_DEFAULT, on_bad_signature=None, on_unverified=None) -> tuple | None:
    """(round r's randomness, "drand" or "drand_one_operator") when two endpoints serve the same checked value, else
    None; the first kind needs two operators. on_disagree() is called when two serve different values. Each beacon's
    BLS signature is checked with `verifier` (the chainlayer's by default; None checks nothing): a beacon that fails
    is dropped and on_bad_signature() called; a value taken with no verifier calls on_unverified() once."""
    import aiohttp
    if verifier is _DEFAULT:
        verifier = beacon_verifier()

    async def one(session, base):
        try:
            async with session.get(f"{base}/{QUICKNET_CHAIN}/public/{int(r)}") as resp:
                if resp.status != 200:
                    return base, None
                rnd, _verified = accept_beacon(await resp.json(content_type=None), r, verifier, on_bad_signature)
                return base, rnd
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return base, None

    values = []
    session = _http_session()
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
    if got is not None and verifier is None and on_unverified is not None:
        on_unverified()
    return (got, kind) if got is not None else None


async def http_latest() -> int | None:
    """The highest round drand's endpoints report as their latest, or None when none answers in time."""
    import aiohttp

    async def one(session, base):
        try:
            async with session.get(f"{base}/{QUICKNET_CHAIN}/public/latest") as resp:
                if resp.status != 200:
                    return None
                return int((await resp.json(content_type=None))["round"])
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, TypeError):
            return None

    session = _http_session()
    got = [x for x in await asyncio.gather(*(one(session, b) for b in LATEST_ENDPOINTS)) if x is not None]
    return max(got) if got else None


def _pulse_of(subtensor, r: int) -> bytes | None:
    pulse = subtensor.substrate.query("Drand", "Pulses", [int(r)])
    value = getattr(pulse, "value", pulse)
    return pulse_randomness(value, r) if value else None


def chain_randomness_getter(subtensor):
    async def get(r: int) -> bytes | None:
        def query():
            try:
                return _pulse_of(subtensor, r)
            except Exception:
                bt.logging.debug(f"[SEED] chain Drand pulse {r} unavailable", exc_info=True)
                return None
        return await asyncio.to_thread(query)
    return get


def _seed_subtensor(v):
    """The seed's own subtensor connection for the chain's Drand pulses: the maintenance subtensor's
    websocket is not asked from the seed's thread. Opened on first use on the validator's chain endpoint, as the
    validator opens its own; None when it cannot be opened, and opened again at the next ask."""
    st = getattr(v, "_seed_subtensor", None)
    if st is not None:
        return st
    cfg = getattr(v, "config", None)
    endpoint = getattr(getattr(cfg, "subtensor", None), "chain_endpoint", None) or ""
    try:
        st = bt.Subtensor(network=endpoint) if endpoint else bt.Subtensor(config=cfg)
    except Exception:
        bt.logging.warning("[SEED] the chain fallback could not open its connection")
        bt.logging.debug("[SEED] chain fallback connection", exc_info=True)
        return None
    v._seed_subtensor = st
    return st


def seed_chain_getter(v):
    """round -> the chain's Drand pulse randomness for round r, asked over the seed's own connection; a failed ask
    drops the connection so the next one opens it again."""
    async def get(r: int) -> bytes | None:
        def query():
            st = _seed_subtensor(v)
            if st is None:
                return None
            try:
                return _pulse_of(st, r)
            except Exception:
                bt.logging.debug(f"[SEED] chain Drand pulse {r} unavailable", exc_info=True)
                v._seed_subtensor = None
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
    if not crosses_seed_boundary(ts_ns, getattr(v.simulation, "publish_interval", 0), seed_intervals(v.simulation)):
        return response
    subtensor = getattr(v, "subtensor", None)
    use_drand = drand_enabled(v)
    sources = getattr(v, "fundamental_seed_sources", None)
    if not isinstance(sources, dict):
        sources = v.fundamental_seed_sources = {}

    def count(key):
        sources[key] = sources.get(key, 0) + 1

    async def http_get(r):
        return await http_randomness(r, on_disagree=lambda: count("drand_disagree"),
                                     on_bad_signature=lambda: count("drand_bad_signature"),
                                     on_unverified=lambda: count("drand_unverified"))

    async def no_http(r):
        return None

    async def no_chain(r):
        return None
    secret = secret_for(v, getattr(v.simulation, "logDir", None), count)
    if not use_drand and subtensor is None:
        r, randomness, source = 0, os.urandom(32), "random"
    else:
        latest = await latest_fn() if use_drand else None
        if use_drand:
            note_clock_skew(v, clock_skew_sample(closed_at, latest))
        r, randomness, source = await seed_fn(closed_at, http_get if use_drand else no_http,
                                              seed_chain_getter(v) if subtensor is not None else no_chain,
                                              min_round=(int(latest) + 1) if latest else 0)
    count(source)
    if source != "drand":
        bt.logging.warning(f"[SEED] step seed at {ts_ns} from {source} (round {r})")
    value = step_value(secret, randomness)
    _record(v, r, value, randomness, source, ts_ns)
    return {**response, "seed": [int(r), value]}
