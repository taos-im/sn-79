# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Lever offmain_build (off unless named on): the query process builds the miners' requests main would have built.

With neuron.query_prepayload on, main builds and compresses every queried miner's request on the round's path. With
the lever on, main tells the query process each state's size as it receives it (one small message, stamped with a
sequence number); the query process reads the state from the engine's shared memory, completes its accounts from a
full copy of its own exactly as main does, and parses it while main is still parsing. At the forward main reads the
builder's status: when it names this state's sequence and timestamp and the accounts were complete, main hands over
only the request's other fields and each queried account's traded volume, and the query process builds the requests
with main's own function; otherwise main builds them itself as before, and asks the engine for every account in full
when the builder's copy could not complete them. A gap in the sequence clears the builder's copy.

The guard: over a full state then states that leave entries out (entirely, and as nil fields), every queried miner's
request and the shared books blob are byte for byte main's, and a planted difference in the builder's copy is
caught. With the lever off, main sends no message and reads no status, and the query process creates no builder."""
import inspect
import os
import pickle
from types import SimpleNamespace

import msgpack
import posix_ipc
import pytest

import taos.im.validator.offmain_build as ob
import taos.im.validator.query as q
from taos.im.validator.engines.simulation import SimulationEngine
from taos.im.validator.multiasset_state import flatten_multiasset_state

HOTKEY = "5Ebq3nnVj2YAyznQsFDJJSmyoYGV7sx5boXqXv9go4rNrT3G"
FEES = {"v": None, "m": 0.001, "t": 0.002}
NIL = {k: None for k in ("bb", "qb", "bl", "ql", "bc", "qc", "o", "l")}
UIDS, BOOKS = (1, 2, 3), (0, 1)


def _entry(uid, book, total):
    bal = {"c": "BASE", "t": total, "f": total, "r": 0.0, "i": total}
    return {"i": uid, "b": book, "bb": bal, "qb": dict(bal, c="QUOTE", t=total * 300), "bl": 0.0, "ql": 0.0,
            "bc": 0.0, "qc": 0.0, "o": [{"i": uid * 10 + book, "p": 300.0 + total}], "l": {}, "f": FEES}


def _raw(ts, entries):
    """A state's bytes as an engine under dirtyAccountsOnly sends them. `entries` {(uid, book): total | None | 'out'}:
    a total is a full entry, None one sent with nil fields, 'out' one left out entirely."""
    accounts = {}
    for uid in UIDS:
        accounts[uid] = {}
        for book in BOOKS:
            v = entries.get((uid, book), "out")
            if v == "out":
                continue
            accounts[uid][book] = {"i": uid, "b": book, **NIL, "f": FEES} if v is None else _entry(uid, book, v)
    books = {b: {"i": b, "b": [{"p": 299.0 + ts % 7, "q": 1.0}], "a": [{"p": 301.0, "q": 2.0}], "e": []}
             for b in BOOKS}
    return msgpack.packb({"timestamp": ts, "books": books, "accounts": accounts, "bookFees": {b: FEES for b in BOOKS},
                          "notices": {uid: [{"y": "RDPOL", "s": True, "o": ts + uid}] for uid in UIDS},
                          "dirtyAccountsOnly": True}, use_bin_type=True)


FULL = {(u, b): 5.0 + u + b for u in UIDS for b in BOOKS}
SEQUENCE = [
    (10 ** 12, FULL),
    (10 ** 12 + 12, {(1, 0): 9.5}),                                  # one change, the rest left out entirely
    (10 ** 12 + 24, {(2, 1): 4.25, (3, 0): None, (3, 1): None}),     # a change and two sent as nil fields
]


def _axon(hotkey, port):
    return {"hotkey": hotkey, "coldkey": "ck", "ip": "203.0.113.1", "port": port, "ip_type": 4, "protocol": 4}


def _main_request(raw_bytes, engine, volumes):
    """Main's side: the engine's own unpack, flatten, completion and parse, the volumes handle_state writes, and the
    request the forward assembles."""
    flat, _ = flatten_multiasset_state(msgpack.unpackb(raw_bytes, raw=False, use_list=True, strict_map_key=False))
    assert engine._complete_accounts(flat)
    state = engine._parse_state(flat)
    for uid, books in state.accounts.items():
        for book in books:
            state.accounts[uid][book]["v"] = volumes.get(uid, {}).get(book, 0.0)
    return {"books": state.books, "accounts": state.accounts, "notices": state.notices, "config": None,
            "version": 68, "timestamp": state.timestamp, "block": state.timestamp // 12, "pools": {},
            "engine_mode": "simulation", "uid": 0, "deregistered_uids": [],
            "metagraph_axons": [_axon(HOTKEY, 8091)] + [_axon(f"5Miner{u}", 8091 + u) for u in UIDS]}


def _engine():
    engine = SimulationEngine.__new__(SimulationEngine)
    engine.validator = SimpleNamespace()
    return engine


def _validator(engine, status):
    return SimpleNamespace(engine=engine, _offmain_status=str(status), _offmain_sent=None, _offmain_seq=0,
                           config=SimpleNamespace(neuron=SimpleNamespace(query_prepayload=True)))


def _built(request):
    return pickle.loads(pickle.dumps(request, protocol=5))


def _same(a, b):
    """Every queried miner's compressed request and the shared books blob, byte for byte."""
    return a["books_blob"] == b["books_blob"] and sorted(a["axon_synapses"]) == sorted(b["axon_synapses"]) and all(
        a["axon_synapses"][u].compressed == b["axon_synapses"][u].compressed for u in a["axon_synapses"])


@pytest.fixture
def tmp_path(request):
    """A status segment of the test's own (the builder's status lives in shared memory), unlinked after."""
    name = f"/offmain_test_{os.getpid()}_{request.node.name[:40]}"
    yield SimpleNamespace(__truediv__=None, name=name)
    try:
        posix_ipc.unlink_shared_memory(name)
    except posix_ipc.ExistentialError:
        pass


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(ob.levers, "enabled", lambda name: name == ob.LEVER or ob.levers.LEVERS.count(name) > 0)


def _round(seq, raw_bytes, engine, builder, validator, volumes):
    builder.on_state(seq, len(raw_bytes), read=lambda size: raw_bytes)
    validator._offmain_sent = seq
    request = _main_request(raw_bytes, engine, volumes)
    mains = _built(q.prebuild_request(dict(request), HOTKEY, 1, "lz4", 0))
    handed = ob.hand_over(validator, request)
    return mains, handed


def test_over_a_full_then_delta_sequence_every_request_is_mains_byte_for_byte(tmp_path, on):
    engine, status = _engine(), tmp_path.name
    builder, validator = ob.Builder(str(status)), _validator(engine, status)
    previous = None
    for seq, (ts, entries) in enumerate(SEQUENCE, start=1):
        volumes = {u: {b: round(ts % 1000 + u * 1.5 + b, 6) for b in BOOKS} for u in UIDS}
        mains, handed = _round(seq, _raw(ts, entries), engine, builder, validator, volumes)
        assert handed is not None, f"state {seq}: the builder was ready and complete, so main hands the build over"
        assert handed["accounts"] is None and handed["books"] == {} and handed["notices"] is None
        theirs = _built(builder.complete(_built(handed), HOTKEY, 1, "lz4", 0))
        assert _same(mains, theirs), f"state {seq}: the query process's requests are not main's"
        if previous is not None:
            assert not _same(previous, mains), "the states differ, so the comparison is not of identical requests"
        previous = mains


def test_a_planted_difference_in_the_builders_copy_is_caught(tmp_path, on):
    engine, status = _engine(), tmp_path.name
    builder, validator = ob.Builder(str(status)), _validator(engine, status)
    volumes = {u: {b: 1.0 for b in BOOKS} for u in UIDS}
    (ts0, full), (ts1, delta) = SEQUENCE[0], SEQUENCE[1]
    _round(1, _raw(ts0, full), engine, builder, validator, volumes)
    builder.kept[2][1]["bb"]["t"] = 999.0          # uid 2's book 1 is left out of the next state
    mains, handed = _round(2, _raw(ts1, delta), engine, builder, validator, volumes)
    theirs = _built(builder.complete(_built(handed), HOTKEY, 1, "lz4", 0))
    assert not _same(mains, theirs)
    assert mains["axon_synapses"][2].compressed != theirs["axon_synapses"][2].compressed
    assert mains["axon_synapses"][1].compressed == theirs["axon_synapses"][1].compressed


def test_a_gap_in_the_sequence_clears_the_copy_and_main_builds_and_asks_for_every_account(tmp_path, on):
    engine, status = _engine(), tmp_path.name
    builder, validator = ob.Builder(str(status)), _validator(engine, status)
    volumes = {}
    (ts0, full), (ts1, delta) = SEQUENCE[0], SEQUENCE[1]
    _round(1, _raw(ts0, full), engine, builder, validator, volumes)
    engine.dirty_accounts_only = True
    mains, handed = _round(3, _raw(ts1, delta), engine, builder, validator, volumes)
    held = {(uid, book) for uid, books in builder.kept.items() for book in books}
    assert held == {(1, 0)}, "the copy restarted from this state's one full entry"
    assert handed is None, "the builder could not complete the accounts, so main builds"
    assert engine._ask_full_accounts, "and the next answer asks the engine for every account, to complete the copy"


def test_a_builder_still_on_an_older_state_leaves_the_build_to_main_and_asks_for_nothing(tmp_path, on):
    engine, status = _engine(), tmp_path.name
    builder, validator = ob.Builder(str(status)), _validator(engine, status)
    ts0, full = SEQUENCE[0]
    _round(1, _raw(ts0, full), engine, builder, validator, {})
    validator._offmain_sent = 2                     # main has the next state, the builder has not written it yet
    request = _main_request(_raw(SEQUENCE[1][0], SEQUENCE[1][1]), engine, {})
    assert ob.hand_over(validator, request) is None
    assert not engine._ask_full_accounts


def test_the_query_process_refuses_a_hand_over_for_a_state_it_has_not_parsed(tmp_path, on):
    engine, status = _engine(), tmp_path.name
    builder, validator = ob.Builder(str(status)), _validator(engine, status)
    ts0, full = SEQUENCE[0]
    _, handed = _round(1, _raw(ts0, full), engine, builder, validator, {})
    handed[ob.KEY]["seq"] = 7
    with pytest.raises(ob.NotParsed):
        builder.complete(handed, HOTKEY, 1, "lz4", 0)


def test_with_the_lever_off_main_sends_no_message_and_reads_no_status(tmp_path, monkeypatch):
    monkeypatch.setattr(ob.levers, "enabled", lambda name: name != ob.LEVER)
    sent = []
    validator = _validator(_engine(), tmp_path.name)
    validator.request_queue = SimpleNamespace(send=lambda *a, **k: sent.append(a))
    monkeypatch.setattr(ob, "read_status", lambda path: pytest.fail("the status was read with the lever off"))
    ob.on_state_bytes(validator, 12345)
    assert sent == [] and validator._offmain_seq == 0 and validator._offmain_sent is None
    assert ob.hand_over(validator, {"timestamp": 1}) is None


def test_with_the_lever_on_each_state_is_announced_with_the_next_sequence(tmp_path, on):
    sent = []
    validator = _validator(_engine(), tmp_path.name)
    validator.request_queue = SimpleNamespace(send=lambda msg, timeout=None: sent.append((msg, timeout)))
    ob.on_state_bytes(validator, 100)
    ob.on_state_bytes(validator, 200)
    assert [ob.parse_message(m) for m, _ in sent] == [(1, 100), (2, 200)] and all(t == 0 for _, t in sent)
    assert validator._offmain_sent == 2

    def _full(msg, timeout=None):
        raise posix_ipc.BusyError("full")
    validator.request_queue = SimpleNamespace(send=_full)
    ob.on_state_bytes(validator, 300)
    assert validator._offmain_seq == 3 and validator._offmain_sent is None, \
        "a state the queue would not take is not handed over, and the builder sees the gap at the next one"


def test_the_off_path_runs_nothing_new_in_main_or_the_query_process():
    from taos.im.validator import forward as fwd
    receive = inspect.getsource(SimulationEngine.receive)
    hook = receive.index("self._on_state_bytes")
    assert "if _hook is not None" in receive[hook - 200:hook + 200]
    src = inspect.getsource(fwd.forward)
    handed = src.index("offmain_build.hand_over(self, request_data)")
    assert src.index("prebuild_request(request_data") > handed, "main's own build stays the fallback"
    run = inspect.getsource(q.QueryService.run)
    branch = run.index("command.startswith(offmain_build.COMMAND)")
    assert run.index("offmain_build.Builder(") > branch, "the builder is created only on the first state message"
    assert "self._offmain = None" in inspect.getsource(q.QueryService.__init__)
