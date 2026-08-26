# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Persistent subprocess that fetches + decodes the metagraph off the main process.

The metagraph resync's substrate scale-decode is a 3-5s GIL burst; run in the
main process it stalls the event loop of whichever round it overlaps (the last
structured tail after the scoring-round convoy fixes). This worker does the
fetch + decode in its own process and ships the synced bt.Metagraph back as a
pickle (a few hundred KB; ~10-30ms to unpickle on the main side), so the main
process never pays the decode.

Design notes:
- spawn (not fork): the validator main process is heavily threaded; forking it
  risks malloc/lock corruption in the child. Spawn costs one bittensor import
  (~3-8s) once at startup.
- request/response over a duplex Pipe; the caller blocks on the maintenance
  thread (GIL released in recv) with a deadline.
- Any failure (worker dead, timeout, chain error) returns None and the caller
  falls back to the in-process sync — behaviour is never worse than before.
"""
import os
import pickle
import time
import multiprocessing


def _worker_main(conn, chain_endpoint, netuid, cores, mechid=0):
    """Worker process entry: sync-on-demand loop. Imports bittensor locally."""
    try:
        if cores:
            try:
                os.sched_setaffinity(0, set(cores))
            except OSError:
                pass
        import bittensor as bt

        subtensor = None
        conn.send(("ready", None))
        while True:
            try:
                cmd = conn.recv()
            except (EOFError, OSError):
                break
            if cmd == "stop":
                break
            if cmd != "sync":
                continue
            try:
                if subtensor is None:
                    subtensor = bt.Subtensor(network=chain_endpoint)
                mg = subtensor.metagraph(netuid, mechid=mechid)
                # The synced metagraph holds a live Subtensor (websocket) ref —
                # the one unpicklable attribute (~0.16MB picklable without it).
                # The owner reattaches its own subtensor after the swap.
                mg.subtensor = None
                payload = pickle.dumps(mg, protocol=5)
                conn.send(("ok", payload))
            except Exception as e:
                # Drop the connection so the next request reconnects fresh.
                subtensor = None
                try:
                    conn.send(("err", f"{type(e).__name__}: {e}"))
                except (BrokenPipeError, OSError):
                    break
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class MetagraphSyncWorker:
    """Owner-side handle: start/stop the worker and request synced metagraphs."""

    def __init__(self, chain_endpoint, netuid, cores=None, worker_fn=_worker_main, mechid=0):
        self._endpoint = chain_endpoint
        self._netuid = netuid
        self._mechid = mechid
        self._cores = list(cores) if cores else []
        self._worker_fn = worker_fn
        self._ctx = multiprocessing.get_context("spawn")
        self._proc = None
        self._conn = None

    def start(self) -> None:
        """Start the background metagraph sync."""
        self._conn, child = self._ctx.Pipe(duplex=True)
        self._proc = self._ctx.Process(
            target=self._worker_fn,
            args=(child, self._endpoint, self._netuid, self._cores, self._mechid),
            daemon=True,
            name="metagraph-sync-worker",
        )
        self._proc.start()
        child.close()

    def is_alive(self) -> bool:
        """Whether the sync worker is currently running."""
        return self._proc is not None and self._proc.is_alive()

    def sync(self, timeout: float = 45.0):
        """Request one synced metagraph. Returns the unpickled bt.Metagraph, or
        None on ANY failure (dead worker, timeout, chain error) — the caller
        must fall back to the in-process sync. A dead worker is respawned for
        the next cycle rather than blocking this one on a fresh bt import.
        """
        if not self.is_alive():
            try:
                self.start()
            except Exception:
                pass
            return None
        try:
            self._conn.send("sync")
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not self._conn.poll(0.25):
                    if not self.is_alive():
                        return None
                    continue
                status, payload = self._conn.recv()
                if status == "ready":
                    continue
                if status == "ok":
                    return pickle.loads(payload)
                # ("err", msg) — log the worker-side failure, caller falls back;
                # the worker stays up and reconnects on the next request.
                import bittensor as bt
                bt.logging.warning(f"Metagraph worker error: {payload}")
                return None
            return None
        except (BrokenPipeError, EOFError, OSError):
            self.stop()
            return None

    def stop(self) -> None:
        """Stop the background sync and release the connection."""
        try:
            if self._conn is not None:
                try:
                    self._conn.send("stop")
                except Exception:
                    pass
                self._conn.close()
        except Exception:
            pass
        finally:
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=2.0)
            self._proc = None
