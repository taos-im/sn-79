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
import queue
import threading
import time

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


def _budget(env):
    try:
        return int(env.get(ENV_MAX_BYTES, DEFAULT_MAX_BYTES))
    except (TypeError, ValueError):
        return DEFAULT_MAX_BYTES


def capture_state(raw, timestamp: int, env=None):
    """Write one received state (its raw bytes) under the capture directory; returns the path, or None when capture
    is off or the write failed. The write goes to a temporary name and is renamed, so a reader never sees a partial
    file, and the directory is trimmed to the byte budget after each write. Synchronous: the listener loop uses
    submit_capture instead."""
    return _write(raw, timestamp, os.environ if env is None else env, trim=True)


def _write(raw, timestamp: int, env, trim=True):
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
        if trim:
            enforce_budget(directory, _budget(env))
        return final
    except Exception:
        _log.debug("state capture failed", exc_info=True)
        return None


class CaptureWriter:
    """Captures states off the caller's thread. submit() never blocks: it puts the state on a bounded queue, and when
    the queue is full it drops the oldest pending state and counts it. A daemon thread compresses, writes and renames
    each state, and trims the directory to its budget at most every trim_every_s seconds."""

    def __init__(self, env=None, maxsize=4, trim_every_s=60.0):
        self.env = os.environ if env is None else env
        self.queue = queue.Queue(maxsize=max(1, int(maxsize)))
        self.trim_every_s = float(trim_every_s)
        self.dropped = 0
        self.written = 0
        self.failed = 0
        self._last_trim = None
        self._thread = None
        self._start_lock = threading.Lock()

    def submit(self, raw, timestamp: int) -> bool:
        """Hand one state to the writer; False when capture is off."""
        if not self.env.get(ENV_DIR):
            return False
        item = (bytes(raw), int(timestamp))
        while True:
            try:
                self.queue.put_nowait(item)
                break
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                    self.dropped += 1
                    if self.dropped in (1, 10, 100) or self.dropped % 1000 == 0:
                        _log.warning("state capture behind the stream: %d state(s) dropped so far", self.dropped)
                except queue.Empty:
                    pass
        self._ensure_thread()
        return True

    def flush(self, timeout=None):
        """Wait until every queued state is written (tests and shutdown)."""
        end = None if timeout is None else time.monotonic() + timeout
        while self.queue.unfinished_tasks:
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(0.01)
        return True

    def _ensure_thread(self):
        with self._start_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="state_capture", daemon=True)
                self._thread.start()

    def _run(self):
        while True:
            raw, ts = self.queue.get()
            try:
                if _write(raw, ts, self.env, trim=False):
                    self.written += 1
                else:
                    self.failed += 1
                now = time.monotonic()
                if self._last_trim is None or now - self._last_trim >= self.trim_every_s:
                    self._last_trim = now
                    directory = self.env.get(ENV_DIR)
                    if directory and os.path.isdir(directory):
                        enforce_budget(directory, _budget(self.env))
            except Exception:
                self.failed += 1
                _log.debug("state capture writer failed", exc_info=True)
            finally:
                self.queue.task_done()


_writer = None


def submit_capture(raw, timestamp: int) -> bool:
    """The listener loop's capture: returns at once, the module's CaptureWriter does the work."""
    global _writer
    if not os.environ.get(ENV_DIR):
        return False
    if _writer is None:
        _writer = CaptureWriter()
    return _writer.submit(raw, timestamp)


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
