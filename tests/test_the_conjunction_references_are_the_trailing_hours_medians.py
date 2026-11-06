# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The conjunction's book references are the medians of the trailing hour.

The band is the class's bound times the book's quoted spread and the gate a fraction of its fill size, each read as the
median over one simulated hour, so a few large fills do not move the gate. The validator keeps each book's samples of
the trailing hour, the quoted spread at every state where the book has a bid and an ask that do not cross and every
fill size, and takes their medians every five simulated minutes, each book on its own phase of that period so that a
state refreshes a few books, not all of them. A book without references takes them at once; after a restart, whose
samples are not saved, the saved references hold until the book's next refresh; a state earlier than the newest
sample, a new run's clock, drops the samples and keeps the references. The samples are flat arrays per book, follow
their book at a layout change and reach the scoring child with the rest of the state.
"""
import inspect
from types import SimpleNamespace

from taos.im.validator import book_layout, scoring_shadow, trade

S = 10 ** 9
M = 60 * S
H = 3600 * S
R = trade.CONJ_REF_REFRESH_NS


def _book(bid, ask):
    return {"b": [{"p": bid, "q": 1.0}], "a": [{"p": ask, "q": 1.0}]}


def _v():
    v = SimpleNamespace()
    trade.init_conjunction_state(v)
    return v


def _fills(*qs):
    return [{"y": "t", "q": q} for q in qs]


def test_the_refresh_is_five_simulated_minutes_and_the_window_one_hour():
    assert R == 300 * S and trade.CONJ_REF_WINDOW_NS == H


def test_the_spread_reference_is_the_median_of_the_trailing_hours_quoted_spreads():
    v = _v()
    for i, (bid, ask) in enumerate([(99.9, 100.1)] * 3 + [(99.0, 101.0)]):
        trade.conjunction_book_refs(v, 0, _book(bid, ask), _fills(1.0), ts=i * M)
    trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), _fills(1.0), ts=R)
    assert abs(v.debeta_conj_ref[0][0] - 20.0) < 1e-9, "one wide state among five does not move the median"


def test_the_size_reference_is_the_median_fill_size_of_the_trailing_hour():
    v = _v()
    trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), _fills(1.0, 1.0, 1.0), ts=0)
    trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), _fills(1.0, 100.0), ts=R)
    assert v.debeta_conj_ref[0][1] == 1.0, "a mean would read 20.8; the median of the five fills is 1"


def test_samples_older_than_the_hour_leave_the_references():
    v = _v()
    for i in range(3):
        trade.conjunction_book_refs(v, 0, _book(99.8, 100.2), _fills(4.0), ts=i * M)
    assert abs(v.debeta_conj_ref[0][0] - 40.0) < 1e-9 and v.debeta_conj_ref[0][1] == 4.0
    for i in range(2):
        trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), _fills(1.0), ts=H + 5 * M + i * M)
    spread, size = v.debeta_conj_ref[0]
    assert abs(spread - 20.0) < 1e-9 and size == 1.0, "the first hour's samples are out of the window"
    assert len(v.debeta_conj_samples[0]["sv"]) == 2 and len(v.debeta_conj_samples[0]["qv"]) == 2


def test_the_references_hold_between_refreshes():
    v = _v()
    trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), _fills(1.0), ts=0)
    first = v.debeta_conj_ref[0]
    assert abs(first[0] - 20.0) < 1e-9 and first[1] == 1.0, "a book without references takes them at once"
    for k in range(1, 6):
        trade.conjunction_book_refs(v, 0, _book(99.0, 101.0), _fills(9.0), ts=k * 50 * S)
    assert v.debeta_conj_ref[0] == first, "inside the period the references hold"
    trade.conjunction_book_refs(v, 0, _book(99.0, 101.0), _fills(9.0), ts=R)
    spread, size = v.debeta_conj_ref[0]
    assert abs(spread - 200.0) < 1e-9 and size == 9.0, "at the refresh they take the window's medians"


def test_the_books_refresh_on_their_own_phases_a_few_a_state():
    v = _v()
    books = range(128)
    for b in books:
        trade.conjunction_book_refs(v, b, _book(99.9, 100.1), _fills(1.0), ts=0)
    per_state, seen = [], {b: 0 for b in books}
    for k in range(1, 2 * (R // (5 * S)) + 1):
        before = {b: v.debeta_conj_samples[b]["at"] for b in books}
        for b in books:
            trade.conjunction_book_refs(v, b, _book(99.9, 100.1), _fills(1.0), ts=k * 5 * S)
        moved = [b for b in books if v.debeta_conj_samples[b]["at"] != before[b]]
        per_state.append(len(moved))
        for b in moved:
            seen[b] += 1
    assert max(per_state) <= 4, per_state
    assert set(seen.values()) == {2}, "every book refreshes once a period"


def test_a_crossed_book_adds_no_spread_sample_and_a_book_without_fills_takes_no_size():
    v = _v()
    trade.conjunction_book_refs(v, 1, _book(100.2, 100.1), [], ts=0)
    assert 1 not in v.debeta_conj_ref, "a crossed book sets no spread reference"
    trade.conjunction_book_refs(v, 2, _book(99.9, 100.1), [], ts=0)
    spread, size = v.debeta_conj_ref[2]
    assert abs(spread - 20.0) < 1e-9 and size is None, "no fill, no size reference: the band waits for one"
    trade.conjunction_book_refs(v, 2, _book(99.9, 100.1), _fills(3.0), ts=S)
    assert v.debeta_conj_ref[2][1] == 3.0, "a missing reference is taken at its first sample, not a period later"


def test_after_a_restart_the_saved_references_hold_until_the_books_next_refresh():
    v = _v()
    v.debeta_conj_ref[0] = (50.0, 7.0)
    trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), _fills(1.0), ts=10 * H)
    assert v.debeta_conj_ref[0] == (50.0, 7.0)
    trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), _fills(1.0), ts=10 * H + R)
    spread, size = v.debeta_conj_ref[0]
    assert abs(spread - 20.0) < 1e-9 and size == 1.0


def test_a_new_runs_clock_drops_the_samples_and_keeps_the_references():
    v = _v()
    for i in range(3):
        trade.conjunction_book_refs(v, 0, _book(99.8, 100.2), _fills(4.0), ts=20 * H + i * M)
    ref = v.debeta_conj_ref[0]
    trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), _fills(1.0), ts=5 * S)
    assert v.debeta_conj_ref[0] == ref, "the references carry into the new run"
    assert list(v.debeta_conj_samples[0]["st"]) == [5 * S], "the old run's samples are gone"
    trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), _fills(1.0), ts=5 * S + R)
    assert v.debeta_conj_ref[0][1] == 1.0


def test_an_hour_of_five_second_states_keeps_flat_arrays_of_a_few_tens_of_kilobytes_a_book():
    v = _v()
    for k in range(720):
        trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), _fills(*([1.0] * 5)), ts=k * 5 * S)
    s = v.debeta_conj_samples[0]
    assert len(s["sv"]) == 720 and len(s["qv"]) == 3600
    nbytes = sum(len(s[k]) * s[k].itemsize for k in ("st", "sv", "qt", "qv"))
    assert nbytes <= 70_000, nbytes


def test_the_state_update_passes_the_states_time():
    src = inspect.getsource(trade.update_trade_volumes)
    assert "conjunction_book_refs(self, bookId, book, trades, ts=timestamp)" in src


def test_the_samples_follow_their_book_and_reach_the_scoring_child():
    assert "debeta_conj_samples" in book_layout.BOOK_KEYED["book"]
    assert "debeta_conj_samples" in scoring_shadow._DEBETA_PLAIN
    v = _v()
    trade.conjunction_book_refs(v, 3, _book(99.9, 100.1), _fills(2.0), ts=0)
    book_layout.rekey_books(v, {3: 7}, keep_unmapped=False)
    assert 7 in v.debeta_conj_samples and 3 not in v.debeta_conj_samples
    assert list(v.debeta_conj_samples[7]["qv"]) == [2.0]
