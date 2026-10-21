# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Descriptor watch: who closes a descriptor an event loop still watches, and what the process looked like when a loop
error named one.

A watched descriptor closed elsewhere and reused by another file makes the loop's next re-arm fail; under uvloop that
aborted the process without a message, on the standard loop it is a logged error. The tracer records every close made
through os.close or socket.close with its thread and caller, and raises an alarm at once when the descriptor is
registered with a watched loop, which is the closer the fault needs. The loop handler, on an error about a descriptor,
logs the open descriptors with their targets, each watched loop's registrations, the threads alive and the last closes.
"""
import asyncio
import collections
import errno
import os
import socket
import threading
import time
import traceback
import weakref

_RECENT = collections.deque(maxlen=256)
_LOOPS = {}
_LOCK = threading.Lock()
_INSTALLED = False
_ALARM = None
_LAST_SNAPSHOT = 0.0
_SNAPSHOT_EVERY_S = 60.0
_DESCRIPTOR_ERRNOS = {errno.EBADF, errno.EPERM, errno.ENOENT, errno.EEXIST}


def _log_default(text):
    try:
        import bittensor as bt
        bt.logging.error(text)
    except Exception:
        print(text)


def set_alarm(fn):
    """The call made with the alarm text when a watched descriptor is closed; None restores the default logger."""
    global _ALARM
    _ALARM = fn


def watch_loop(loop, name):
    """Register a loop whose selector registrations the tracer checks every close against."""
    with _LOCK:
        _LOOPS[name] = weakref.ref(loop)


def _registered(fd):
    out = []
    with _LOCK:
        items = list(_LOOPS.items())
    for name, ref in items:
        loop = ref()
        sel = getattr(loop, "_selector", None) if loop is not None else None
        if sel is None:
            continue
        try:
            if fd in sel.get_map():
                out.append(name)
        except Exception:
            continue
    return out


def _where(skip=2, depth=3):
    frames = traceback.extract_stack(limit=skip + depth)[:-skip]
    return " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(frames))


def _record(fd):
    rec = {"fd": int(fd), "thread": threading.current_thread().name, "t": time.time(), "where": _where(skip=3)}
    _RECENT.append(rec)
    loops = _registered(rec["fd"])
    if loops:
        text = (f"fd {rec['fd']} closed by thread {rec['thread']} while registered with loop {', '.join(loops)}: "
                f"{rec['where']}")
        (_ALARM or _log_default)(text)


def install_close_tracer():
    """Wrap os.close and socket.socket.close once; both record the close before it happens."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    _os_close = os.close

    def close(fd):
        try:
            _record(fd)
        except Exception:
            pass
        return _os_close(fd)

    os.close = close
    _sock_close = socket.socket.close

    def sock_close(self):
        try:
            fd = self.fileno()
            if fd >= 0:
                _record(fd)
        except Exception:
            pass
        return _sock_close(self)

    socket.socket.close = sock_close


def recent_closes():
    return list(_RECENT)


def _open_descriptors():
    out = []
    try:
        for name in sorted(os.listdir("/proc/self/fd"), key=int):
            try:
                out.append(f"{name} -> {os.readlink(f'/proc/self/fd/{name}')}")
            except OSError:
                continue
    except OSError:
        return ["(no /proc/self/fd)"]
    return out


def snapshot(reason=""):
    """The process at this moment, as a text block: the reason, open descriptors, each watched loop's registrations,
    the threads alive and the recent closes."""
    lines = [f"descriptor watch: {reason}", "open descriptors:"]
    lines += [f"  {x}" for x in _open_descriptors()]
    with _LOCK:
        items = list(_LOOPS.items())
    for name, ref in items:
        loop = ref()
        sel = getattr(loop, "_selector", None) if loop is not None else None
        lines.append(f"loop {name} registered:")
        if sel is None:
            lines.append("  (no selector)")
            continue
        try:
            for fd, key in sorted(sel.get_map().items()):
                lines.append(f"  fd {fd} events {key.events} data {key.data!r}"[:200])
        except Exception as ex:
            lines.append(f"  (unreadable: {ex})")
    lines.append("threads: " + ", ".join(t.name for t in threading.enumerate()))
    lines.append("recent closes:")
    now = time.time()
    for rec in list(_RECENT)[-30:]:
        lines.append(f"  fd {rec['fd']} by {rec['thread']} {now - rec['t']:.1f}s ago at {rec['where']}")
    return "\n".join(lines)


def _about_descriptor(exc):
    if exc is None:
        return False
    if isinstance(exc, OSError) and exc.errno in _DESCRIPTOR_ERRNOS:
        return True
    text = str(exc).lower()
    return isinstance(exc, (OSError, ValueError)) and ("descriptor" in text or "fd " in text or "epoll" in text)


def install_loop_handler(loop, name, log=None):
    """Keep the loop's default exception handling and, on an error about a descriptor, log the snapshot (at most one
    a minute)."""
    log = log or _log_default
    default = loop.get_exception_handler() or loop.default_exception_handler

    def handler(lp, context):
        try:
            default(context)
        except Exception:
            pass
        exc = context.get("exception")
        if not _about_descriptor(exc):
            return
        global _LAST_SNAPSHOT
        now = time.time()
        if now - _LAST_SNAPSHOT < _SNAPSHOT_EVERY_S:
            return
        _LAST_SNAPSHOT = now
        try:
            log(snapshot(f"loop {name}: {context.get('message', '')} {exc!r}"))
        except Exception as ex:
            log(f"descriptor watch: snapshot failed: {ex!r}")

    loop.set_exception_handler(handler)
