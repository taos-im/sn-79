# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The template miner's book history sees every gap in the states it receives, not only the first.

A state update the miner skips under load leaves a gap: the next state's events cover only its own interval, so the
book can be rebuilt only from a snapshot taken exactly one interval earlier. After a first gap the history manager
marks the book as in a gap and keeps the history across it; a second state skipped before the book has recovered is a
second gap, and rebuilding from the older snapshot would replay one interval's events onto a book two intervals old.
"""
from types import SimpleNamespace

from taos.im.agents import StateHistoryManager

I = 1_000_000_000
V = "validator"


def _manager(history_end):
    m = StateHistoryManager.__new__(StateHistoryManager)
    m.history_retention_mins = 600
    m.gap = {V: {}}
    m.last_snapshot = {V: {0: SimpleNamespace(timestamp=history_end)}}
    m.history = {V: {0: SimpleNamespace(end=history_end, start=0, snapshots={history_end: "at the end"})}}
    return m


def _state(ts):
    return SimpleNamespace(dendrite=SimpleNamespace(hotkey=V), timestamp=ts,
                           config=SimpleNamespace(publish_interval=I))


def _book():
    return SimpleNamespace(id=0, snapshot=lambda ts: SimpleNamespace(timestamp=ts))


def test_a_first_gap_gives_no_snapshot_and_the_next_state_rebuilds_from_the_one_before_it():
    m = _manager(10 * I)
    assert m._prepare_snapshot(_state(13 * I), _book()) is None
    assert m.gap[V][0] is True
    snap = m._prepare_snapshot(_state(14 * I), _book())
    assert snap is not None and snap.timestamp == 13 * I


def test_a_second_gap_in_the_same_load_spike_is_seen():
    m = _manager(10 * I)
    assert m._prepare_snapshot(_state(13 * I), _book()) is None
    # 14 is skipped too, before the book has recovered: rebuilding 15 from 13 would miss 13 to 14's events.
    assert m._prepare_snapshot(_state(15 * I), _book()) is None
    snap = m._prepare_snapshot(_state(16 * I), _book())
    assert snap is not None and snap.timestamp == 15 * I
    assert 0 in m.history[V], "a small gap keeps the history"


def test_consecutive_states_never_read_as_a_gap():
    m = _manager(10 * I)
    snap = m._prepare_snapshot(_state(11 * I), _book())
    assert snap == "at the end"
    assert not m.gap[V].get(0)
