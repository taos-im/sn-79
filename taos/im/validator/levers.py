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
  of it."""
import os
import time

LEVERS = ("push_in_wait", "pack_shared")
CHECK_S = 2.0
FILE = os.path.expanduser(os.environ.get("TAOS_LEVERS_FILE") or "~/.taos/levers_off")
_ENV_OFF = frozenset(n.strip() for n in (os.environ.get("TAOS_LEVERS_OFF") or "").split(",") if n.strip())

_seen = {"checked": float("-inf"), "mtime": None, "off": frozenset(), "env_said": False}


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


def enabled(name: str) -> bool:
    return name not in off()


def states() -> dict:
    """{lever: on} for every lever, as the report publishes them."""
    gone = off()
    return {n: n not in gone for n in LEVERS}
