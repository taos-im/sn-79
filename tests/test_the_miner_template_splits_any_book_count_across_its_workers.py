# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The miner template's history manager splits any number of books across its parallel workers. It refused a worker
count that did not divide the book count, so a layout change (a class shrinking, or a book count the miner's setting
does not divide) stopped a miner that worked the run before. The books now go to the workers in contiguous batches
whose sizes differ by at most one; every book is reconstructed once."""
import inspect

from taos.im.agents import StateHistoryManager, book_batches


def test_three_workers_take_every_one_of_256_books_once():
    batches = book_batches(list(range(256)), 3)
    assert len(batches) == 3 and [len(b) for b in batches] == [86, 85, 85]
    assert [b for batch in batches for b in batch] == list(range(256))


def test_a_divisible_count_keeps_todays_even_batches():
    assert book_batches(list(range(128)), 4) == [list(range(i, i + 32)) for i in range(0, 128, 32)]


def test_more_workers_than_books_gives_one_book_each():
    assert book_batches([7, 9], 5) == [[7], [9]]


def test_the_manager_batches_with_it_and_no_longer_refuses_a_count():
    src = inspect.getsource(StateHistoryManager)
    assert "book_batches(list(state.books.keys()), self.parallel_workers)" in src
    assert "must divide number of books" not in src
