# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The sub-window skill path applies the same skill bar as the single-window path.

With scoring.debeta.skill_subwindows above 1, skill came from subwindow_skill, which called kappa_floored without the
bar, so a uid with fewer qualifying books than scoring.debeta.skill_min_books could still score skill there, though
the single-window path gives it none. Latent at the shipped default of 1 sub-window.
"""
import inspect

from taos.im.validator import debeta, reward


def _pool(n_books):
    """One uid with n_books positive alphas, each the same across three sub-windows."""
    by_book = {b: 1.0 + 0.1 * b for b in range(n_books)}
    subs = {b: [a, a, a] for b, a in by_book.items()}
    return {7: by_book}, {7: subs}


def test_weakest_form_scores_nothing_under_the_bar():
    pool, subs = _pool(6)
    under, _ = debeta.subwindow_skill(pool, subs, "weakest", 2, 0.0, min_books=8)
    over, _ = debeta.subwindow_skill(pool, subs, "weakest", 2, 0.0, min_books=5)
    assert under[7] == 0.0 and over[7] > 0.0


def test_sign_gated_form_scores_nothing_under_the_bar():
    pool, subs = _pool(6)
    under, _ = debeta.subwindow_skill(pool, subs, "sign_gated", 2, 0.0, min_books=8)
    over, _ = debeta.subwindow_skill(pool, subs, "sign_gated", 2, 0.0, min_books=5)
    assert under[7] == 0.0 and over[7] > 0.0


def test_the_default_bar_is_the_single_window_default():
    pool, subs = _pool(4)
    default, _ = debeta.subwindow_skill(pool, subs, "weakest", 2, 0.0)
    assert default[7] == debeta.subwindow_skill(pool, subs, "weakest", 2, 0.0,
                                                min_books=debeta.MIN_SKILL_BOOKS_DEFAULT)[0][7]


def test_the_scorer_passes_its_bar_on_the_sub_window_path():
    src = inspect.getsource(reward)
    assert "subwindow_skill(skill_pool, sub_k, _form, _agree, floor, min_books=_bar)" in src
