# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A descriptor closed while an event loop still watches it is named with the thread and the call that closed it, and a
loop error about a descriptor is logged with the process's open descriptors, the loop's registrations, the threads
alive and the last closes, so the cause of such a fault is read from the log rather than reconstructed.

The validator's loops abort, under uvloop, when a watched descriptor is closed elsewhere and its number reused; on the
standard loop the same fault is a logged error. This gives that error everything a diagnosis needs.
"""
import asyncio
import inspect
import os
import socket
import threading

from taos.im.validator import fdwatch


def test_a_close_is_recorded_with_its_thread_and_its_caller():
    fdwatch.install_close_tracer()
    r, w = os.pipe()
    done = threading.Event()

    def closer():
        os.close(w)
        done.set()

    threading.Thread(target=closer, name="the-closer").start()
    done.wait(2)
    os.close(r)
    recent = fdwatch.recent_closes()
    rec = next(x for x in recent if x["fd"] == w)
    assert rec["thread"] == "the-closer" and "closer" in rec["where"]
    assert any(x["fd"] == r and x["thread"] == threading.current_thread().name for x in recent)


def test_a_close_of_a_descriptor_a_watched_loop_registers_is_flagged():
    fdwatch.install_close_tracer()
    flagged = []
    fdwatch.set_alarm(lambda text: flagged.append(text))
    loop = asyncio.new_event_loop()
    a, b = socket.socketpair()
    try:
        fdwatch.watch_loop(loop, "listener")
        loop.add_reader(a.fileno(), lambda: None)
        fd = a.fileno()
        t = threading.Thread(target=a.close, name="other-thread")
        t.start(); t.join(2)
        assert flagged and f"fd {fd}" in flagged[0] and "other-thread" in flagged[0] and "listener" in flagged[0]
    finally:
        fdwatch.set_alarm(None)
        try:
            loop.remove_reader(fd)
        except Exception:
            pass
        b.close()
        loop.close()


def test_a_loop_error_about_a_descriptor_logs_the_snapshot():
    logged = []
    loop = asyncio.new_event_loop()
    fdwatch.install_loop_handler(loop, "main", log=logged.append)
    r, w = os.pipe()

    def bad():
        raise OSError(9, "Bad file descriptor", f"fd {w}")

    loop.call_soon(bad)
    loop.call_soon(loop.stop)
    loop.run_forever()
    loop.close()
    text = "\n".join(logged)
    assert "Bad file descriptor" in text and "open descriptors" in text and "registered" in text
    assert "threads" in text and "recent closes" in text and f"{r} ->" in text, "the open descriptor listing names each fd and its target"
    os.close(r); os.close(w)


def test_the_validator_installs_the_tracer_and_watches_both_loops():
    src = open(os.path.join(os.path.dirname(__file__), "..", "taos", "im", "neurons", "validator.py")).read()
    assert "fdwatch.install_close_tracer()" in src
    assert 'fdwatch.watch_loop(self.main_loop, "main")' in src
    assert 'fdwatch.watch_loop(asyncio.get_running_loop(), "listener")' in src
    assert src.count("fdwatch.install_loop_handler(") >= 2, "both loops get the descriptor-error handler"
    assert "def snapshot" in inspect.getsource(fdwatch)
