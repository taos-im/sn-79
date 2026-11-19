# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The miners' requests built in the query process instead of main (lever offmain_build, off unless named on).

With neuron.query_prepayload, main builds and compresses every queried miner's request on the round's path
(query.prebuild_request) and hands those over. With this lever on:

- as main receives a state it tells the query process the state's size, in one small message stamped with a
  sequence number (on_state_bytes, called from the simulation engine's receive);
- the query process reads the state from the engine's shared memory itself, completes the accounts the engine left
  out from a full copy of its own exactly as main does (kept_accounts.fill_accounts), parses it while main is still
  parsing, and writes a status naming the sequence, the state's timestamp and whether the accounts were complete
  (Builder.on_state). A gap in the sequence clears its copy, since a state it did not see may have changed an entry;
- at the forward main reads that status without waiting (hand_over). When it names this state and the accounts were
  complete, main hands over the request without the books, accounts and notices, with each account's traded volume
  'v', the one thing main writes into them, and the query process builds the requests with main's own function
  (Builder.complete); otherwise main builds them itself as before, and when the builder's copy could not complete the
  accounts, main's next answer asks the engine for every account in full so that it can.

The requests are byte for byte main's own (the guard test compares them over a full state and states that leave
entries out). With the lever off nothing here runs: main sends no message and reads no status, and the query process
creates no builder."""
import json
import os
import time

import msgpack

from taos.im.validator import levers

LEVER = "offmain_build"
COMMAND = "offmain_state"
KEY = "offmain"
READ_RETRIES = 8


class NotParsed(Exception):
    """A hand-over for a state the query process has not parsed, or not completed."""


STATUS_BYTES = 256


def status_path(queue_name: str) -> str:
    """The shared-memory segment the query process writes its status to, named after its query queue."""
    return f"{queue_name}_offmain"


def message(seq: int, size: int) -> bytes:
    return f"{COMMAND}:{int(seq)}:{int(size)}".encode()


def parse_message(raw) -> tuple[int, int]:
    text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    _, seq, size = text.split(":")
    return int(seq), int(size)


def write_status(name: str, seq: int, timestamp, complete: bool) -> None:
    """One write of the whole record, so a reader sees this record or the last; a torn read does not parse."""
    import posix_ipc

    body = json.dumps({"seq": seq, "timestamp": timestamp, "complete": bool(complete)}).encode()
    shm = posix_ipc.SharedMemory(name, flags=posix_ipc.O_CREAT, size=STATUS_BYTES)
    try:
        os.pwrite(shm.fd, body.ljust(STATUS_BYTES, b"\0"), 0)
    finally:
        shm.close_fd()


def read_status(name: str) -> dict | None:
    """The builder's last status, or None when there is none or it cannot be read whole."""
    import posix_ipc

    try:
        shm = posix_ipc.SharedMemory(name)
    except (posix_ipc.Error, ValueError):
        return None
    try:
        return json.loads(os.pread(shm.fd, STATUS_BYTES, 0).rstrip(b"\0"))
    except (OSError, ValueError):
        return None
    finally:
        shm.close_fd()


# ── main's side ──────────────────────────────────────────────────────────────────────────────────────────────────────


def _active(validator) -> bool:
    if not levers.enabled(LEVER):
        return False
    if not getattr(getattr(validator.config, "neuron", None), "query_prepayload", False):
        return False
    return getattr(validator.engine, "mode", "simulation") == "simulation"


def on_state_bytes(validator, size: int) -> None:
    """The simulation engine's receive has read a state of `size` bytes: tell the query process, stamped with the
    next sequence number. A message the queue would not take leaves the state with main; the builder sees the gap."""
    if not _active(validator):
        return
    import posix_ipc

    validator._offmain_seq = getattr(validator, "_offmain_seq", 0) + 1
    validator._offmain_sent = None
    try:
        validator.request_queue.send(message(validator._offmain_seq, size), timeout=0)
    except (posix_ipc.BusyError, posix_ipc.SignalError, OSError, AttributeError):
        return
    validator._offmain_sent = validator._offmain_seq


def hand_over(validator, request_data: dict) -> dict | None:
    """The request to hand the query process in place of main's own build, or None when main builds it: the lever
    off, no state announced for this round, the builder not yet on this state, or its accounts incomplete (then the
    next answer asks the engine for every account in full)."""
    if not _active(validator):
        return None
    seq = getattr(validator, "_offmain_sent", None)
    if seq is None:
        return None
    status = read_status(validator._offmain_status)
    if not status or status.get("seq") != seq or status.get("timestamp") != request_data.get("timestamp"):
        return None
    if not status.get("complete"):
        if getattr(validator.engine, "dirty_accounts_only", False):
            validator.engine._ask_full_accounts = True
        return None
    accounts = request_data.get("accounts") or {}
    volumes = {uid: {book: entry.get("v") for book, entry in books.items()} for uid, books in accounts.items()}
    return {**request_data, "books": {}, "accounts": None, "notices": None,
            KEY: {"seq": seq, "timestamp": request_data.get("timestamp"), "v": volumes}}


# ── the query process's side ─────────────────────────────────────────────────────────────────────────────────────────


def _read_state(size: int) -> bytes:
    """The state's first `size` bytes from the engine's shared memory, as main reads them."""
    import posix_ipc

    from taos.im.validator.engines.simulation import _ipc_name, _read_segment

    shm = posix_ipc.SharedMemory(_ipc_name("state"))
    try:
        return _read_segment(shm.fd, size)
    finally:
        shm.close_fd()


class Builder:
    """The query process's copy of each state, completed and parsed as main does, ready for main's hand-over."""

    def __init__(self, status_file: str):
        self.status_file = status_file
        self.kept: dict = {}
        self.last_seq: int | None = None
        self.pending: dict | None = None

    def on_state(self, seq: int, size: int, read=_read_state) -> bool:
        """Read, complete and parse the state main announced; write the status. Returns whether it is ready."""
        from taos.im.protocol import MarketSimulationStateUpdate
        from taos.im.validator.kept_accounts import fill_accounts
        from taos.im.validator.multiasset_state import flatten_multiasset_state

        if self.last_seq is not None and seq != self.last_seq + 1:
            self.kept = {}
        self.last_seq, self.pending = seq, None
        flat = None
        for attempt in range(READ_RETRIES):
            try:
                data = read(size)
            except Exception:
                break
            try:
                raw = msgpack.unpackb(data, raw=False, use_list=True, strict_map_key=False)
                flat, _ = flatten_multiasset_state(raw)
                break
            except (ValueError, TypeError, msgpack.UnpackValueError):
                # the engine can announce a state a moment before its write lands, as main's own read allows for
                time.sleep(0.005 * (2 ** attempt))
        if flat is None:
            # main's own copy now has this state and this one does not: start again from a full state
            self.kept = {}
            write_status(self.status_file, seq, None, False)
            return False
        complete = True
        if flat.get("dirtyAccountsOnly"):
            complete = fill_accounts(self.kept, flat.get("accounts") or {}, flat.get("books"),
                                     flat.get("bookFees")) == 0
        timestamp = flat.get("timestamp")
        if complete:
            state = MarketSimulationStateUpdate.parse_dict(flat)
            timestamp = state.timestamp
            self.pending = {"seq": seq, "timestamp": timestamp, "books": state.books, "accounts": state.accounts,
                            "notices": state.notices}
        write_status(self.status_file, seq, timestamp, complete)
        return complete

    def complete(self, request_data: dict, own_hotkey: str, level, engine, parallel_workers) -> dict:
        """Main's hand-over made whole with this process's parse of the same state, then built with main's own
        function, so the requests are the ones main would have built."""
        from taos.im.validator.query import prebuild_request

        handed = request_data.get(KEY) or {}
        pending = self.pending
        if pending is None or pending["seq"] != handed.get("seq") or pending["timestamp"] != handed.get("timestamp"):
            raise NotParsed(f"hand-over for state {handed.get('seq')} at {handed.get('timestamp')}, parsed "
                            f"{None if pending is None else (pending['seq'], pending['timestamp'])}")
        self.pending = None
        accounts = pending["accounts"]
        try:
            for uid, books in (handed.get("v") or {}).items():
                for book, volume in books.items():
                    accounts[uid][book]["v"] = volume
        except (KeyError, TypeError) as exc:
            raise NotParsed(f"main wrote a volume for an account this parse does not hold: {exc!r}") from exc
        whole = {k: v for k, v in request_data.items() if k != KEY}
        whole.update(books=pending["books"], accounts=accounts, notices=pending["notices"])
        return prebuild_request(whole, own_hotkey, level, engine, parallel_workers)
