# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Runtime switches for the round-time levers, so any of them can be turned off on a live validator in seconds, with
no restart, if it misbehaves at scale.

A lever is on unless it is named in TAOS_LEVERS_OFF (comma-separated, read once at start) or in the levers file
(TAOS_LEVERS_FILE, default ~/.taos/levers_off: one name a line, '#' starts a comment). The file is re-read when its
modification time changes, looked at no more than every CHECK_S seconds, so a lever goes off with
`echo push_in_wait >> ~/.taos/levers_off` and back on when the line is removed. A file that cannot be read, or is
not UTF-8, turns no lever off. Each change is logged once, as are the levers TAOS_LEVERS_OFF turns off; the report
publishes each lever's state as validator_gauges lever_<name>. A lever turned off takes the path it replaced.

- push_in_wait: the data-service push starts once the query is sent (validator.on_query_sent); off, it starts after
  the forward.
- pack_shared: the report's pack copies only the payload's live parts (persistence.pack_report); off, it copies all
  of it.
- account_deltas: an engine under dirtyAccountsOnly leaves unchanged account entries out and the validator fills them
  in (kept_accounts); off, every answer asks the engine for every account in full (SimulationEngine.respond), so each
  state lists them all, as from an engine without it, with no engine restart.
- prereq_in_wait: the pre-request work that feeds nothing the request carries (the reset notices, the placement
  counts and the GenTRX push) runs once the query is sent, in the miners' wait (validator._pre_request_work); off, it
  runs after the forward.
- reward_in_wait: a state's accounting starts once the query is sent, only after its pre-request work ran there, and
  its boundary scoring waits for the forward's end; off, the reward starts after the forward.

An opt-in lever (OPT_IN) is off until it is named on: in TAOS_LEVERS_ON (comma-separated, read once at start) or in the
levers-on file (TAOS_LEVERS_ON_FILE, default ~/.taos/levers_on), read like the off file, so it goes on with
`echo offmain_build >> ~/.taos/levers_on` and off when the line is removed. Named off as well, it is off. The report
publishes its state beside the others.

- offmain_build: the query process builds the miners' requests from the state it reads itself, and main hands over
  only what it adds (offmain_build.py); off, main builds them, as before."""
import os
import time

LEVERS = ("push_in_wait", "pack_shared", "account_deltas", "prereq_in_wait", "reward_in_wait")
OPT_IN = ("offmain_build",)
CHECK_S = 2.0
FILE = os.path.expanduser(os.environ.get("TAOS_LEVERS_FILE") or "~/.taos/levers_off")
ON_FILE = os.path.expanduser(os.environ.get("TAOS_LEVERS_ON_FILE") or "~/.taos/levers_on")
_ENV_OFF = frozenset(n.strip() for n in (os.environ.get("TAOS_LEVERS_OFF") or "").split(",") if n.strip())
_ENV_ON = frozenset(n.strip() for n in (os.environ.get("TAOS_LEVERS_ON") or "").split(",") if n.strip())

_seen = {"checked": float("-inf"), "mtime": None, "off": frozenset(), "env_said": False}
_seen_on = {"checked": float("-inf"), "mtime": None, "on": frozenset(), "env_said": False}


def _say(msg: str) -> None:
    import bittensor as bt
    bt.logging.warning(msg)


def _read(path: str) -> frozenset:
    try:
        with open(path, encoding="utf-8") as f:
            return frozenset(n for n in (line.split("#", 1)[0].strip() for line in f) if n)
    except (OSError, ValueError):
        return frozenset()


def _names(off_now: frozenset) -> str:
    unknown = sorted(off_now - set(LEVERS))
    return (", ".join(f"{n} {'off' if n in off_now else 'on'}" for n in LEVERS)
            + (f"; not a lever, ignored: {', '.join(unknown)}" if unknown else ""))


def off() -> frozenset:
    """The levers turned off now: the environment's from the start, and the file's as last read."""
    if _ENV_OFF and not _seen["env_said"]:
        _seen["env_said"] = True
        _say(f"levers (TAOS_LEVERS_OFF): {_names(_ENV_OFF)}")
    now = time.monotonic()
    if now - _seen["checked"] >= CHECK_S:
        _seen["checked"] = now
        try:
            mtime = os.stat(FILE).st_mtime_ns
        except OSError:
            mtime = None
        if mtime != _seen["mtime"]:
            _seen["mtime"] = mtime
            new = _read(FILE) if mtime is not None else frozenset()
            if new != _seen["off"]:
                _say(f"levers ({FILE}): {_names(new)}")
                _seen["off"] = new
    return _ENV_OFF | _seen["off"]


def _on_names(on_now: frozenset) -> str:
    unknown = sorted(on_now - set(OPT_IN) - set(LEVERS))
    return (", ".join(f"{n} {'on' if n in on_now else 'off'}" for n in OPT_IN)
            + (f"; not an opt-in lever, ignored: {', '.join(unknown)}" if unknown else ""))


def on() -> frozenset:
    """The opt-in levers named on now: the environment's from the start, and the levers-on file's as last read."""
    if _ENV_ON and not _seen_on["env_said"]:
        _seen_on["env_said"] = True
        _say(f"opt-in levers (TAOS_LEVERS_ON): {_on_names(_ENV_ON)}")
    now = time.monotonic()
    if now - _seen_on["checked"] >= CHECK_S:
        _seen_on["checked"] = now
        try:
            mtime = os.stat(ON_FILE).st_mtime_ns
        except OSError:
            mtime = None
        if mtime != _seen_on["mtime"]:
            _seen_on["mtime"] = mtime
            new = _read(ON_FILE) if mtime is not None else frozenset()
            if new != _seen_on["on"]:
                _say(f"opt-in levers ({ON_FILE}): {_on_names(new)}")
                _seen_on["on"] = new
    return _ENV_ON | _seen_on["on"]


def enabled(name: str) -> bool:
    if name in OPT_IN:
        return name in on() and name not in off()
    return name not in off()


def states() -> dict:
    """{lever: on} for every round-time lever, on unless turned off."""
    gone = off()
    return {n: n not in gone for n in LEVERS}


def opt_in_states() -> dict:
    """{lever: on} for every opt-in lever, off unless named on."""
    return {n: enabled(n) for n in OPT_IN}
