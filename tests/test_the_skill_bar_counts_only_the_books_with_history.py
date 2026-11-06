# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A class's skill bar counts the books with history, not every book of the layout.

The skill bar is a share of a class's books. When a run boundary adds books to a class, the added books start empty:
no miner has traded them inside the lookback, so a bar counted on the whole layout asks every miner at once for history
that cannot exist yet. A miner over the old bar and under the new one would lose its skill share in the first scoring
round, and the rest would take it. So each added book counts towards the bar in proportion to how much of the lookback
it has existed for: at the boundary the bar is the old layout's, one lookback later the new layout's, and in between
it rises a book at a time as the added books' history fills. The kept books' records, the making half and the class
weights are untouched by it.
"""
import inspect
from types import SimpleNamespace

from taos.im.validator import persistence, reward, trade

_H = 3 * 3600 * 10 ** 9  # one lookback, simulated nanoseconds
SHARE = 0.15625


def _young(since, span=_H):
    return {"span": span, "since": dict(since)}


def _bar(young, books, now):
    return reward.resolve_skill_min_books(0, SHARE, reward.books_with_history(young, books, now))


def test_a_book_counts_by_how_much_of_the_lookback_it_has_existed_for():
    books = range(128)
    young = _young({b: 1000 for b in range(64, 128)})
    assert reward.books_with_history(None, books, 0) == 128
    assert reward.books_with_history({}, books, 0) == 128
    assert reward.books_with_history(young, books, 1000) == 64
    assert reward.books_with_history(young, books, 1000 + _H // 2) == 96
    assert reward.books_with_history(young, books, 1000 + _H) == 128
    assert reward.books_with_history(young, books, 1000 + 5 * _H) == 128
    assert reward.books_with_history(young, books, 0) == 64, "a clock before the boundary counts the added books as 0"
    assert reward.books_with_history(young, range(64), 1000) == 64, "a class none of whose books were added"


def test_the_bar_rises_from_the_old_layouts_to_the_new_one_a_book_at_a_time():
    young = _young({b: 0 for b in range(64, 128)})
    assert _bar(young, range(128), 0) == 10 and _bar(young, range(128), _H) == 20
    assert _bar(young, range(128), _H // 2) == 15
    path = [_bar(young, range(128), k * _H // 200) for k in range(201)]
    assert path == sorted(path) and all(b - a in (0, 1) for a, b in zip(path, path[1:])), "never a jump"
    small = _young({b: 0 for b in range(32, 64)})
    assert _bar(small, range(64), 0) == 5 and _bar(small, range(64), _H) == 10


def test_the_boundary_records_when_each_added_book_started():
    v = SimpleNamespace()
    ch = {"books": {**{b: "keep" for b in range(64)}, **{b: "new" for b in range(64, 128)}},
          "dest": {b: b for b in range(64)}}
    trade.note_young_books(v, ch, start_ts=500, span=_H)
    assert v._debeta_young_books == {"span": _H, "since": {b: 500 for b in range(64, 128)}}


def test_a_book_whose_market_changed_starts_young_too():
    v = SimpleNamespace()
    trade.note_young_books(v, {"books": {0: "clear", 1: "keep"}, "dest": {1: 1}}, start_ts=7, span=_H)
    assert v._debeta_young_books["since"] == {0: 7}


def test_a_later_change_moves_the_young_books_with_their_start_and_forgets_those_a_lookback_old():
    v = SimpleNamespace(_debeta_young_books=_young({4: 100, 5: 100, 6: 100 - _H, 7: 100}))
    ch = {"books": {14: "move", 15: "move", 16: "move", 0: "clear", 20: "new", 7: "drop"},
          "dest": {4: 14, 5: 15, 6: 16, 7: None}}
    trade.note_young_books(v, ch, start_ts=200, span=_H)
    assert v._debeta_young_books["since"] == {14: 100, 15: 100, 0: 200, 20: 200}


def test_a_change_that_adds_no_book_leaves_nothing_young():
    v = SimpleNamespace()
    trade.note_young_books(v, {"books": {b: "keep" for b in range(8)}, "dest": {b: b for b in range(8)}}, 0, _H)
    assert not v._debeta_young_books


def test_the_boundary_notes_the_added_books_and_the_class_bars_count_them():
    boundary = inspect.getsource(trade.shift_simulation_histories)
    assert "note_young_books(self, book_changes, new_ts, lookback)" in boundary
    scores = inspect.getsource(reward.compute_debeta_scores)
    i = scores.index("_bars[_c] = resolve_skill_min_books(")
    assert "books_with_history(getattr(self, '_debeta_young_books', None), _part['books']" in scores[i:i + 400]


def test_a_save_and_a_restart_keep_the_added_books_start():
    young = _young({64: 500, 65: 500})
    saved = persistence.build_debeta_state(SimpleNamespace(_debeta_young_books=young))
    assert saved["debeta_young_books"] == young
    assert persistence.load_young_books(saved["debeta_young_books"]) == young
    assert persistence.load_young_books({"span": _H, "since": {"64": "500"}}) == _young({64: 500})
    assert persistence.load_young_books(None) == {}
