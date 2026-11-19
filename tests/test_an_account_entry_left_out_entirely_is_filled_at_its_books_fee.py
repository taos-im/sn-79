# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""An engine under dirtyAccountsOnly leaves an unchanged entry out of the state entirely, not as a skeleton of its ids,
fees and eight nils: at 256 books those skeletons were 4.6 MB of a 6.3 MB state in which no account had changed. The
fees an untiered policy charges move with the book (the dynamic fee follows its maker/taker ratio), so the state
carries each book's shared rates once ("bookFees") and an entry left out is filled from the last full copy at its
book's current rates. Only the books the state has are filled, so a copy kept across a layout change never puts an
old book back."""
from test_account_entries_left_out_are_filled_in import OMITTED, entry, full_accounts

from taos.im.validator.kept_accounts import fill_accounts
from taos.im.validator.multiasset_state import flatten_multiasset_state

BOOKS = {0: {}, 1: {}}
RATES = {0: {"m": 0.0, "t": 0.0003}, 1: {"m": -0.0001, "t": 0.0004}}


def test_an_entry_left_out_is_filled_from_the_copy_at_its_books_current_rates():
    kept = {}
    assert fill_accounts(kept, full_accounts(5.0), BOOKS, RATES) == 0
    later = {0: {0: entry(0, 0, 7.0)}, 1: {}}
    assert fill_accounts(kept, later, BOOKS, RATES) == 0
    assert later[0][0]["bb"]["t"] == 7.0, "a full entry is taken as sent"
    for a, b in ((0, 1), (1, 0), (1, 1)):
        filled = later[a][b]
        assert filled["bb"]["t"] == 5.0 and filled["i"] == a and filled["b"] == b
        assert all(filled[k] is not None for k in OMITTED)
        assert filled["f"] == {"v": None, **RATES[b]}, "the book's rates now, not the copy's"


def test_what_main_adds_to_an_entry_does_not_travel_into_a_filled_one():
    kept = {}
    fill_accounts(kept, full_accounts(5.0), BOOKS, RATES)
    kept[1][1]["v"] = 99.0  # main's traded volume, written at the entry's own level of the state it applied
    later = {0: {}, 1: {}}
    fill_accounts(kept, later, BOOKS, RATES)
    assert "v" not in later[1][1]
    assert later[1][1] is not kept[1][1], "a filled entry is its own dict"


def test_a_book_the_state_no_longer_has_is_not_filled():
    kept = {}
    old = {a: {b: entry(a, b, 5.0) for b in (0, 1, 2)} for a in (0, 1)}
    fill_accounts(kept, old, {0: {}, 1: {}, 2: {}}, RATES)
    now = {0: {}, 1: {}}
    assert fill_accounts(kept, now, BOOKS, RATES) == 0
    assert set(now[0]) == set(now[1]) == {0, 1}


def test_an_entry_left_out_with_no_copy_counts_as_missing():
    assert fill_accounts({}, {0: {0: entry(0, 0, 1.0)}, 1: {}}, BOOKS, RATES) == 3


def test_the_flattened_state_carries_every_realizations_book_rates():
    def instance(book, rates):
        return {"books": {book: {}}, "accounts": {}, "notices": {}, "bookFees": {book: rates}}
    raw = {"timestamp": 1, "backgrounds": [{"bg": 0, "name": "simulation_0", "instances": [instance(0, RATES[0])]},
                                           {"bg": 1, "name": "simulation_1", "instances": [instance(1, RATES[1])]}]}
    flat, _ = flatten_multiasset_state(raw)
    assert flat["bookFees"] == RATES
