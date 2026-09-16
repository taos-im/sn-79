# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Only positive directional skill earns skill rank, and only making of real size earns making rank.

Without these rules a skill of exactly zero outranks every trader who lost money, because a tie
block takes its minimum rank and most of the pool is non-positive. Negligible two-sided quoting is
rewarded the same way on the making side, since few agents make at all. Both effects pay for an
absence of exposure rather than for a result.

Two rules, both applied before ranking and both leaving the raw legs in `detail` untouched:
skill is clamped at 0 (every non-positive skill ties at rank 0), and making below
making_floor_scale times the median positive making enters the rank as 0.
"""
from pathlib import Path

import pytest

from taos.im.validator.debeta import (_rank01, debeta_scores, kappa_floored, making_magnitude_floor,
                                      median_abs_floor)

DEV = Path(__file__).resolve().parents[1]

# Four books of alpha per uid, well above any floor. uid 1 consistently positive, uid 2 consistently
# negative, uid 3 mildly negative, uid 4 has no qualifying books (skill exactly 0), uid 5 positive.
AL = {
    1: [10.0, 11.0, 9.0, 12.0],
    2: [-10.0, -11.0, -9.0, -12.0],
    3: [-1.0, -1.5, -0.5, -1.2],
    4: [],
    5: [3.0, 4.0, 3.5, 3.2],
}
# Two-sided capture: uid 1 large, uid 2 large, uid 3 tiny, uid 4 tiny, uid 5 none.
CB = {1: {0: 100.0}, 2: {0: 80.0}, 3: {0: 1.0}, 4: {0: 0.5}, 5: {}}
CS = {1: {0: 100.0}, 2: {0: 80.0}, 3: {0: 1.0}, 4: {0: 0.5}, 5: {}}


def _run(**kw):
    detail = {}
    scores = debeta_scores(CB, CS, AL, floor=0.0, w_make=0.30, detail=detail, **kw)
    return scores, detail


def test_the_raw_skill_of_the_fixture_is_what_the_docstring_says():
    raw = {u: kappa_floored(AL[u], 0.0) for u in AL}
    assert raw[1] > 0 and raw[5] > 0
    assert raw[2] < 0 and raw[3] < 0
    assert raw[4] == 0.0


def test_non_positive_skill_ties_at_rank_zero():
    _, detail = _run()
    assert detail[2]["skill_rank"] == 0.0, "a consistently negative trader must not earn skill rank"
    assert detail[3]["skill_rank"] == 0.0, "less negative is still not skill"
    assert detail[4]["skill_rank"] == 0.0, "no qualifying book is exactly the case the floor was written for"
    # default scope 'positives': the two positive skills span the leg, the lower at 0, the higher at 1
    assert detail[1]["skill_rank"] == 1.0 and detail[5]["skill_rank"] == 0.0
    # rollback scope 'whole': the lower positive sits on the step above the three non-positives
    whole = {}
    debeta_scores(CB, CS, AL, floor=0.0, w_make=0.30, detail=whole, skill_rank_scope="whole")
    assert whole[5]["skill_rank"] == pytest.approx(3 / 4) and whole[1]["skill_rank"] == 1.0


def test_the_raw_skill_leg_is_still_the_measured_value():
    _, detail = _run()
    assert detail[2]["skill_raw"] < 0.0
    assert detail[3]["skill_raw"] < 0.0
    assert detail[4]["skill_raw"] == 0.0


def test_making_below_the_magnitude_floor_ranks_zero():
    _, detail = _run(making_floor_scale=0.5)
    # positive making: 200, 160, 2, 1 -> median 81 -> floor 40.5; two makers clear it
    assert detail[3]["making_rank"] == 0.0
    assert detail[4]["making_rank"] == 0.0
    assert detail[5]["making_rank"] == 0.0
    # default making scope 'positives': the two makers above the floor span the leg
    assert detail[1]["making_rank"] == 1.0 and detail[2]["making_rank"] == 0.0
    assert detail[3]["making_raw"] == pytest.approx(2.0), "the raw leg stays what was measured"
    # rollback scope 'whole': the lower maker sits on the step above the three zero makers
    whole = {}
    debeta_scores(CB, CS, AL, floor=0.0, w_make=0.30, detail=whole, making_floor_scale=0.5, making_rank_scope="whole")
    assert whole[2]["making_rank"] == pytest.approx(3 / 4) and whole[1]["making_rank"] == 1.0


def test_a_zero_scale_disables_the_making_floor():
    _, detail = _run(making_floor_scale=0.0)
    # all four makers enter the rank: 200, 160, 2, 1 -> 1, 2/3, 1/3, 0 under 'positives'
    assert detail[3]["making_rank"] == pytest.approx(1 / 3)
    assert detail[4]["making_rank"] == 0.0, "the smallest positive maker earns no making rank"
    assert detail[2]["making_rank"] == pytest.approx(2 / 3) and detail[1]["making_rank"] == 1.0
    whole = {}
    debeta_scores(CB, CS, AL, floor=0.0, w_make=0.30, detail=whole, making_floor_scale=0.0, making_rank_scope="whole")
    assert whole[4]["making_rank"] == pytest.approx(1 / 4), "under 'whole' it inherits the zero maker's block"


def test_making_magnitude_floor_is_a_multiple_of_the_positive_median():
    assert making_magnitude_floor([200.0, 160.0, 2.0, 1.0, 0.0, 0.0], 0.5) == pytest.approx(0.5 * 81.0)
    assert making_magnitude_floor([0.0, 0.0], 0.5) == 0.0
    assert making_magnitude_floor([5.0, 7.0], 0.0) == 0.0


def test_the_score_is_still_the_published_ranks_blended():
    scores, detail = _run(making_floor_scale=0.5)
    for u, s in scores.items():
        assert s == pytest.approx(0.30 * detail[u]["making_rank"] + 0.70 * detail[u]["skill_rank"])


def test_the_testnet_board_shape_is_corrected():
    """The measured shape: many negatives, a few zeros, few positives on skill; a couple of tiny
    two-sided quoters among a few large makers. The tiny quoters and the negative takers end at 0."""
    skill = {u: v for u, v in zip(range(23), [-0.65, -0.43, -0.29, -0.22, -0.15, -0.13, -0.11, -0.09,
                                                -0.086, -0.054, -0.042, -0.035, -0.3, 0.0, 0.0, 0.0,
                                                0.047, 0.12, 0.52, 3.85, -0.41, -0.54, -0.67])}
    making = {u: 0.0 for u in range(23)}
    making.update({17: 777.6, 13: 5.4, 20: 206.6, 14: 3.85, 21: 507.1, 22: 5.35, 7: 1060.9, 8: 60.0})
    floor = making_magnitude_floor(list(making.values()), 0.5)
    rm = _rank01([m if m >= floor else 0.0 for m in making.values()])
    rs = _rank01([max(0.0, s) for s in skill.values()])
    comb = {u: 0.30 * rm[u] + 0.70 * rs[u] for u in range(23)}
    assert comb[13] == 0.0 and comb[14] == 0.0, "the tiny two-sided quoters with zero skill score nothing"
    assert all(comb[u] == 0.0 for u in range(13) if making[u] == 0.0), "negative-skill pure takers score nothing"
    assert comb[19] > comb[18] > comb[16] > 0.0, "positive skill still orders the skill leg"
    assert sum(1 for v in comb.values() if v > 0) < 12


def test_reward_passes_the_dial_and_the_child_config_carries_it():
    reward_src = (DEV / "taos/im/validator/reward.py").read_text()
    shadow_src = (DEV / "taos/im/validator/scoring_shadow.py").read_text()
    config_src = (DEV / "taos/im/config/__init__.py").read_text()
    assert "making_floor_scale=float(getattr(dcfg, 'making_floor_scale', 0.0) or 0.0)" in reward_src
    assert "making_floor_scale=_num(getattr(_d, 'making_floor_scale', None), 0.0, float)" in shadow_src, (
        "the CUTOVER child scores from the config duck: a dial missing there is silently off in the "
        "authoritative scorer, and its default must match the config's"
    )
    assert '"--scoring.debeta.making_floor_scale"' in config_src


def test_the_making_floor_ships_off_by_default():
    """Operator decision the clamp is the necessary fix; the making floor took the testnet
    board's Gini from 0.64 to 0.74 for the removal of four small makers and stays off until observed."""
    import argparse
    from taos.im.config import add_im_validator_args
    parser = argparse.ArgumentParser()
    add_im_validator_args(object, parser)
    args, _ = parser.parse_known_args([])
    assert getattr(args, 'scoring.debeta.making_floor_scale') == 0.0


def test_the_skill_floor_helper_is_unchanged():
    assert median_abs_floor({1: [1.0, -3.0], 2: [2.0]}, scale=0.5) == pytest.approx(1.0)


def test_the_prune_drops_a_pair_once_its_fills_have_aged_out():
    """A (uid, book) pair whose fills have all left the window used to keep its key with a running
    total reduced to a floating-point residue. On a long-running validator those pairs come to
    outnumber the live ones, so the skill floor's median collapsed to a residue and switched itself
    off. The prune now removes the emptied pair from both maps; live pairs keep running == sum(kept)."""
    from taos.im.validator.debeta import prune_hist_2level, prune_hist_1level
    hist = {1: {0: {100: 4.0, 300: 2.0}, 1: {100: 0.7}}, 2: {0: {100: 0.3}}}
    running = {1: {0: 6.0, 1: 0.7}, 2: {0: 0.3}}
    prune_hist_2level(hist, running, threshold=250)
    assert hist == {1: {0: {300: 2.0}}}
    assert running == {1: {0: pytest.approx(2.0)}}
    h1 = {0: {100: 5.0}, 1: {100: 1.0, 300: 1.0}}
    r1 = {0: 5.0, 1: 2.0}
    prune_hist_1level(h1, r1, threshold=250)
    assert h1 == {1: {300: 1.0}} and r1 == {1: pytest.approx(1.0)}


def test_the_skill_floor_sees_only_pairs_with_fills_in_the_window():
    """After the prune the floor's pool holds the live pairs alone, so its median is theirs."""
    from taos.im.validator.debeta import prune_hist_2level, book_alphas_from_drift
    mtm = {1: {0: 4.0, 1: -6.0, 2: 5.0}, 2: {0: 3.0}}
    for u in range(10, 20):                       # pairs whose only fills are outside the window
        mtm[u] = {0: 1e-12, 1: 0.0}
    invsum = {u: {b: 0.0 for b in books} for u, books in mtm.items()}
    ts_of = lambda u: 300 if u in (1, 2) else 100
    mtm_hist = {u: {b: {ts_of(u): v} for b, v in books.items()} for u, books in mtm.items()}
    inv_hist = {u: {b: {ts_of(u): v} for b, v in books.items()} for u, books in invsum.items()}
    prune_hist_2level(mtm_hist, mtm, threshold=250)
    prune_hist_2level(inv_hist, invsum, threshold=250)
    alphas = book_alphas_from_drift(mtm, invsum, {0: 10, 1: 10, 2: 10}, {0: 0.0, 1: 0.0, 2: 0.0})
    assert set(alphas) == {1, 2}
    assert median_abs_floor(alphas, scale=0.5) == pytest.approx(2.25)


def test_the_skill_pool_is_the_books_a_uid_filled_inside_the_window():
    """A miner holding a static position on a book carries an alpha entry for it on every trade, and
    that alpha is exactly zero by the invariant. On a long-running validator such held-not-traded
    pairs were half of all pairs, so the floor collapsed to a rounding residue and kappa counted books
    the miner never traded. The pool is restricted to books with fills inside the window, read from
    the windowed capture maps."""
    from taos.im.validator.debeta import traded_book_alphas
    by_book = {1: {0: 4.0, 1: -6.0, 2: 5.0, 3: 0.0, 4: 0.0}, 2: {0: 3.0, 5: 0.0}, 3: {0: 0.0, 1: 0.0}}
    cb = {1: {0: 1.0, 2: 1.0}, 2: {0: 1.0}}
    cs = {1: {1: 1.0}, 2: {}, 3: {}}
    pool = traded_book_alphas(by_book, cb, cs)
    assert pool == {1: {0: 4.0, 1: -6.0, 2: 5.0}, 2: {0: 3.0}, 3: {}}
    alphas = {u: list(b.values()) for u, b in pool.items()}
    assert median_abs_floor(alphas, scale=0.5) == pytest.approx(2.25)
    assert median_abs_floor({u: list(b.values()) for u, b in by_book.items()}, scale=0.5) == 0.0
    assert kappa_floored(alphas[3], 2.25) == 0.0
