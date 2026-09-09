# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""debeta_scores must expose its legs, or a miner cannot be told why they scored what they did.

The score is `w_make*rank01(making) + (1-w_make)*rank01(skill)`. Until now the function returned
only the combined value and discarded making, skill, both ranks and the P11 factor. Nothing
downstream could publish them, so every scoring panel on the dashboards showed kappa -- which
de-beta full-replaces -- and none showed either leg that actually drives emissions.

The detail must be the SAME numbers the score was built from, not a recomputation, or the dashboard
and the emission can disagree.
"""
import pytest

from taos.im.validator.debeta import debeta_scores, median_abs_floor

# uid 1 two-sided on book 0; uid 2 one-sided per book but globally balanced; uid 3 smaller two-sided.
CB = {1: {0: 5.0}, 2: {0: 5.0, 1: 0.0}, 3: {0: 1.0, 1: 1.5}}
CS = {1: {0: 5.0}, 2: {0: 0.0, 1: 5.0}, 3: {0: 1.0, 1: 1.5}}
AL = {1: [1.0, 1.1, 0.9, 1.2], 2: [0.4, 0.5, 0.35, 0.45], 3: [2.0, 2.2, 1.8, 2.1]}


def _call(**kw):
    return debeta_scores(CB, CS, AL, floor=median_abs_floor(AL), w_make=0.30, **kw)


def test_scores_unchanged_when_detail_not_requested():
    """Backward compatible: the plain call still returns just {uid: score}."""
    out = _call()
    assert isinstance(out, dict)
    assert set(out) == {1, 2, 3}
    assert all(isinstance(v, float) for v in out.values())


def test_detail_carries_both_legs_and_their_ranks():
    detail = {}
    scores = _call(detail=detail)
    assert set(detail) == set(scores)
    for uid, d in detail.items():
        for k in ("making_raw", "making_rank", "skill_raw", "skill_rank", "p11_factor"):
            assert k in d, f"uid {uid} detail missing {k}"
        assert 0.0 <= d["making_rank"] <= 1.0
        assert 0.0 <= d["skill_rank"] <= 1.0


def test_detail_reproduces_the_score_it_was_built_from():
    """The published legs must recombine to the emitted score, or the dashboard lies."""
    detail = {}
    scores = _call(detail=detail)
    w = 0.30
    for uid, s in scores.items():
        d = detail[uid]
        assert d["making_rank"] * w + d["skill_rank"] * (1 - w) == pytest.approx(s, abs=1e-12)


def test_making_raw_reflects_the_per_book_rule():
    """uid 2 is one-sided on every book: its making magnitude must be 0, not its global pair."""
    detail = {}
    _call(detail=detail)
    assert detail[2]["making_raw"] == pytest.approx(0.0)
    assert detail[1]["making_raw"] > 0.0


def test_p11_factor_is_one_when_the_discount_is_off():
    detail = {}
    _call(detail=detail)
    assert all(d["p11_factor"] == pytest.approx(1.0) for d in detail.values())


def test_p11_factor_records_the_discount_actually_applied():
    """A fed maker must show a factor below 1, and making_raw must be the POST-discount value."""
    cp = {1: {9: 1000.0}, 2: {8: 1.0, 7: 1.0}, 3: {6: 1.0, 5: 1.0}}
    plain, detail = {}, {}
    base = debeta_scores(CB, CS, AL, floor=median_abs_floor(AL), w_make=0.30, detail=plain)
    disc = debeta_scores(CB, CS, AL, floor=median_abs_floor(AL), w_make=0.30,
                         cp=cp, p11_strength=1.0, detail=detail)
    assert detail[1]["p11_factor"] < 1.0
    assert detail[1]["making_raw"] <= plain[1]["making_raw"] + 1e-12
    # and the detail still recombines to the discounted score
    assert detail[1]["making_rank"] * 0.30 + detail[1]["skill_rank"] * 0.70 == pytest.approx(disc[1], abs=1e-12)


def test_empty_input_returns_empty_and_leaves_detail_empty():
    detail = {"stale": "must be cleared or absent"}
    out = debeta_scores({}, {}, {}, detail=detail)
    assert out == {}
    assert "stale" not in detail or detail == {}

def test_detail_counts_qualifying_books():
    """skill_books = books whose |alpha| clears the floor: the coverage number a miner can act on.
    Without it the dashboards showed the KAPPA leg's book count, which is a constant penalty-floor
    artifact (e.g. 80 for every idle miner) and says nothing about de-beta."""
    detail = {}
    _call(detail=detail)
    floor = median_abs_floor(AL)
    for uid, d in detail.items():
        assert d["skill_books"] == sum(1 for a in AL[uid] if abs(a) >= floor)
