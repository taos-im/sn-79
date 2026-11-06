# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The report payload is packed without copying again the parts the preparation built for it.

The report's pack copies each subtree before packing it (persistence._snapshot), so a worker thread packs a
consistent document while the event loop may write the live one. Most of the payload is not live: the preparation
built it for this report (the volume sums, the recent trades, the metagraph lists, the positions), and the state's
accounts are written once, by the volume pass at the state's start, before the report reads them. Those parts are
packed as they are; the live ones keep the copy. The bytes are the same either way."""
import inspect
from collections import defaultdict

import msgpack

from taos.im.validator import persistence


def _doc():
    nested = defaultdict(dict, {1: {0: (1.5, 2), 3: [4, {"a": (5,)}]}, 2: {}})
    return {"last_state": {"accounts": {7: {0: {"bb": {"t": 1.0}, "o": [{"p": 2.5, "q": (1, 2)}]}}}},
            "volume_sums": nested, "kappa_values": {3: {"books": {0: None, 1: 0.25}}}, "step": 12,
            "recent_trades": {0: [{"p": 1.0}]}}


def test_a_subtree_packed_without_its_copy_is_the_same_bytes():
    for depth in (0, 1, 2, 3):
        for snapshot in (True, False):
            out = []
            persistence._stream_pack(msgpack.Packer(use_bin_type=True), out.append, _doc(), depth, snapshot=snapshot)
            assert b"".join(out) == msgpack.packb(_doc(), use_bin_type=True), (depth, snapshot)


def test_only_the_shared_parts_are_copied(monkeypatch):
    copied = []
    real = persistence._snapshot

    def counting(obj):
        copied.append(obj)
        return real(obj)
    monkeypatch.setattr(persistence, "_snapshot", counting)
    doc = _doc()
    raw = persistence.pack_report(doc, unshared={"last_state", "volume_sums", "recent_trades"})
    assert raw == msgpack.packb(_doc(), use_bin_type=True)
    unshared_ids = set()

    def ids(o):
        unshared_ids.add(id(o))
        for v in (o.values() if isinstance(o, dict) else o if isinstance(o, (list, tuple)) else ()):
            ids(v)
    for k in ("last_state", "volume_sums", "recent_trades"):
        ids(doc[k])
    assert copied and not any(id(o) in unshared_ids for o in copied)


def test_the_validator_packs_its_own_parts_uncopied_and_the_state_only_in_simulation_mode():
    from taos.im.neurons import validator
    src = inspect.getsource(validator)
    at = src.index("(REPORT_UNSHARED | _state) if levers.enabled('pack_shared')")
    assert "pack_report(data, unshared)" in src[at:at + 300]
    assert "'last_state'" in src[at - 400:at + 200]
    assert "last_state" not in persistence.REPORT_UNSHARED, "the state's accounts go uncopied in simulation mode only"
