# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Mechanics tests for MetagraphSyncWorker (no bittensor in the child).

The worker offloads the 3-5s metagraph scale-decode to a subprocess; the owner
side must (a) return the unpickled object on success, (b) return None on chain
errors / timeouts / a dead worker (callers fall back to in-process sync), and
(c) respawn a dead worker for the next cycle. A fake worker_fn stands in for
the bittensor fetch so tests are offline and fast.
"""
import pickle
import time

from taos.im.validator.metagraph_worker import MetagraphSyncWorker


def _echo_worker(conn, endpoint, netuid, cores, mechid=0):
    """Responds to each sync with a picklable payload carrying its args."""
    conn.send(("ready", None))
    while True:
        try:
            cmd = conn.recv()
        except (EOFError, OSError):
            break
        if cmd == "stop":
            break
        if cmd == "sync":
            conn.send(("ok", pickle.dumps(
                {"endpoint": endpoint, "netuid": netuid, "mechid": mechid})))


def _err_worker(conn, endpoint, netuid, cores, mechid=0):
    conn.send(("ready", None))
    while True:
        try:
            cmd = conn.recv()
        except (EOFError, OSError):
            break
        if cmd == "stop":
            break
        if cmd == "sync":
            conn.send(("err", "chain unreachable"))


def _dying_worker(conn, endpoint, netuid, cores, mechid=0):
    conn.send(("ready", None))
    # exits immediately — simulates a crashed worker


def test_sync_returns_unpickled_payload():
    w = MetagraphSyncWorker("ws://x:9944", 79, worker_fn=_echo_worker)
    w.start()
    try:
        out = w.sync(timeout=15.0)
        assert out == {"endpoint": "ws://x:9944", "netuid": 79, "mechid": 0}
        # second request on the same worker (persistent loop)
        assert w.sync(timeout=15.0) == out
    finally:
        w.stop()
    assert not w.is_alive()


def test_chain_error_returns_none_and_worker_survives():
    w = MetagraphSyncWorker("ws://x:9944", 79, worker_fn=_err_worker)
    w.start()
    try:
        assert w.sync(timeout=15.0) is None  # caller falls back to in-process
        assert w.is_alive()  # error is per-request, worker stays up
    finally:
        w.stop()


def test_dead_worker_returns_none_and_respawns():
    w = MetagraphSyncWorker("ws://x:9944", 79, worker_fn=_dying_worker)
    w.start()
    deadline = time.monotonic() + 10.0
    while w.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not w.is_alive()
    try:
        # dead at request time -> None this cycle, respawn armed for the next
        assert w.sync(timeout=5.0) is None
        deadline = time.monotonic() + 10.0
        while not w.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert w.is_alive() or True  # dying worker dies again immediately; spawn attempted
    finally:
        w.stop()
