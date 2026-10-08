# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Opt-in capture of the state stream a validator receives, for the scoring acceptance's replay harness.

The validator scores what it receives, not what a tape shows, so the harness that certifies its pay replays the
same bytes: the raw msgpack state the engine hands the listener loop, one file per update, named by the update's
timestamp. Off unless STATE_CAPTURE_DIR is set; bounded by STATE_CAPTURE_MAX_BYTES (oldest files go first); never
raises into the validator. A mainnet update is about 12 MB raw, so files are lz4 frames when the lz4 package is
present, which the engine already requires.
"""
import logging
import os

try:
    import lz4.frame as _lz4
except ImportError:  # pragma: no cover
    _lz4 = None

ENV_DIR = "STATE_CAPTURE_DIR"
ENV_MAX_BYTES = "STATE_CAPTURE_MAX_BYTES"
DEFAULT_MAX_BYTES = 20 * 1024 ** 3
SUFFIX_RAW = ".mp"
SUFFIX_LZ4 = ".mp.lz4"
_log = logging.getLogger(__name__)


def capture_state(raw, timestamp: int, env=None):
    """Write one received state (its raw bytes) under the capture directory; returns the path, or None when capture
    is off or the write failed. The write goes to a temporary name and is renamed, so a reader never sees a partial
    file, and the directory is trimmed to the byte budget after each write."""
    env = os.environ if env is None else env
    directory = env.get(ENV_DIR)
    if not directory:
        return None
    try:
        os.makedirs(directory, exist_ok=True)
        data = bytes(raw)
        suffix = SUFFIX_RAW
        if _lz4 is not None:
            data, suffix = _lz4.compress(data), SUFFIX_LZ4
        final = os.path.join(directory, f"{int(timestamp)}{suffix}")
        tmp = os.path.join(directory, f".{int(timestamp)}{suffix}.tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, final)
        try:
            budget = int(env.get(ENV_MAX_BYTES, DEFAULT_MAX_BYTES))
        except (TypeError, ValueError):
            budget = DEFAULT_MAX_BYTES
        enforce_budget(directory, budget)
        return final
    except Exception:
        _log.debug("state capture failed", exc_info=True)
        return None


def captured_files(directory):
    """(timestamp, path) for every captured update in the directory, oldest first."""
    out = []
    for name in os.listdir(directory):
        if name.startswith(".") or not (name.endswith(SUFFIX_RAW) or name.endswith(SUFFIX_LZ4)):
            continue
        stem = name[: -len(SUFFIX_LZ4)] if name.endswith(SUFFIX_LZ4) else name[: -len(SUFFIX_RAW)]
        try:
            out.append((int(stem), os.path.join(directory, name)))
        except ValueError:
            continue
    return sorted(out)


def enforce_budget(directory, max_bytes: int) -> int:
    """Remove the oldest captured updates until the directory is within max_bytes, always keeping the newest one;
    returns how many were removed."""
    files = captured_files(directory)
    sizes = {p: os.path.getsize(p) for _, p in files}
    total = sum(sizes.values())
    removed = 0
    for _, p in files[:-1]:
        if total <= max_bytes:
            break
        os.remove(p)
        total -= sizes[p]
        removed += 1
    return removed


def read_captured(path) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    if path.endswith(SUFFIX_LZ4):
        if _lz4 is None:
            raise RuntimeError("an lz4 capture needs the lz4 package to read")
        return _lz4.decompress(data)
    return data


def iter_captured(directory):
    """(timestamp, raw bytes) for every captured update, oldest first: the harness's recorded-stream input."""
    for ts, path in captured_files(directory):
        yield ts, read_captured(path)
