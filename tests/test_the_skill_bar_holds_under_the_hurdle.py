# SPDX-License-Identifier: MIT
"""Under the size-neutral basis shipped on 28 September 2026 (pool-relative floor off, a hurdle of 2.3 bps of the
uid's own notional per book), skill must still need 20 qualifying books, and a qualifying book is one that clears
the hurdle. Two things broke that on the day it shipped:

- the hurdle branch computed skill with kappa_floored's default four-book minimum, not skill_min_books, so an
  account with five to twelve hurdle-passing books published positive skill (four uids on the first board);
- the book count the pool gates on (skill_books) counted books above the floor, and with the floor at 0 that is
  every filled book, including those the hurdle rejected, so an account with 20 filled books and fewer than 20
  clearing the hurdle was paid on the skill half.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taos.im.validator.debeta import hurdle_skill  # noqa: E402
import taos.im.validator.reward as reward  # noqa: E402


def _alphas(n, start=1.0):
    return {b: start + 0.1 * b for b in range(n)}


def test_skill_needs_the_bar_in_hurdle_passing_books():
    pool = {1: _alphas(25), 2: _alphas(12), 3: _alphas(5)}
    values, books = hurdle_skill(pool, floor=0.0, min_books=20)
    assert values[1] > 0 and values[2] == 0.0 and values[3] == 0.0
    assert books == {1: 25, 2: 12, 3: 5}


def test_the_count_is_of_kept_books_not_filled_ones():
    kept = {1: _alphas(8)}            # 25 filled, 8 cleared the hurdle
    values, books = hurdle_skill(kept, floor=0.0, min_books=20)
    assert books[1] == 8 and values[1] == 0.0


def test_the_reward_path_uses_it_with_the_configured_bar():
    src = inspect.getsource(reward.compute_debeta_scores)
    assert "hurdle_skill(" in src
    assert "kappa_floored(list(b.values()), floor) for u, b in skill_pool.items()" not in src


def test_the_pool_call_is_handed_the_configured_bar():
    # build_scoring_config did not carry skill_min_books, so the pool read its fallback of 4.
    assert "'skill_min_books'" in inspect.getsource(reward.build_scoring_config)
