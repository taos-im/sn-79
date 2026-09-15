# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The skill leg ranks positive skill among positive skill.

Under the earlier clamp every non-positive skill tied at rank 0, so the smallest positive skill
inherited the rank of the whole non-positive block, which is most of the pool. A negligible skill
therefore collected a mid score, and the uids whose
skill sign flips between boards moved 0.48 in score doing so: the square waves on the De-beta Score
chart. Ranking among positives (`positives`, the default scope) cut the near-zero share to 0.4 per cent
and the flip move to 0.23 with the top 20 all but unchanged (17.8 of 20). `whole` keeps the clamp rule
for rollback. Both scopes are pinned here; the operator dial is --scoring.debeta.skill_rank_scope.
"""
import argparse
from pathlib import Path

import pytest

from taos.im.validator.debeta import SKILL_RANK_SCOPES, _rank01, debeta_scores, rank_skill

DEV = Path(__file__).resolve().parents[1]

# 8 uids: five non-positive (three negative, two exactly zero), three positive of very different size.
SKILL = [-0.65, -0.11, -0.03, 0.0, 0.0, 0.008, 0.55, 3.7]


def test_non_positive_skill_ranks_zero_under_both_scopes():
    for scope in SKILL_RANK_SCOPES:
        r = rank_skill(SKILL, scope)
        assert r[:5] == [0.0] * 5, scope


def test_whole_puts_the_smallest_positive_on_the_block_step():
    """The rollback rule, as measured: 5 non-positive of 8 -> the 0.008 skill ranks 5/7."""
    r = rank_skill(SKILL, "whole")
    assert r[5] == pytest.approx(5 / 7)
    assert r[6] == pytest.approx(6 / 7) and r[7] == pytest.approx(1.0)


def test_positives_spreads_the_positive_skills_from_zero_to_one():
    r = rank_skill(SKILL, "positives")
    assert r[5] == 0.0, "the smallest positive skill earns no skill rank"
    assert r[6] == pytest.approx(0.5)
    assert r[7] == pytest.approx(1.0)


def test_positives_ranks_are_independent_of_the_non_positive_block_size():
    """Adding negative traders to the pool must not move the positives' ranks (under 'whole' it did)."""
    base = rank_skill([0.008, 0.55, 3.7], "positives")
    with_negatives = rank_skill([-1.0] * 40 + [0.0] * 5 + [0.008, 0.55, 3.7], "positives")
    assert with_negatives[-3:] == base
    whole = rank_skill([-1.0] * 40 + [0.0] * 5 + [0.008, 0.55, 3.7], "whole")
    assert whole[-3] == pytest.approx(45 / 47), "under whole the smallest positive inherits the block"


def test_a_sign_flip_near_zero_moves_to_the_bottom_rung_not_the_middle():
    before = rank_skill([-1.0, -0.5, -0.001, 0.4, 0.9], "positives")
    after = rank_skill([-1.0, -0.5, 0.001, 0.4, 0.9], "positives")
    assert before[2] == 0.0 and after[2] == 0.0, "0.001 is the lowest positive and ranks 0"
    assert rank_skill([-1.0, -0.5, 0.001, 0.4, 0.9], "whole")[2] == pytest.approx(0.5), "the step it used to jump to"


def test_ties_among_positives_share_the_minimum_rank():
    assert rank_skill([0.0, 0.5, 0.5, 1.0], "positives") == [0.0, 0.0, 0.0, 1.0]


def test_unknown_scope_is_refused():
    with pytest.raises(ValueError):
        rank_skill(SKILL, "median")


CB = {1: {0: 5.0}, 2: {0: 1.0}, 3: {}, 4: {}}
CS = {1: {0: 5.0}, 2: {0: 1.0}, 3: {}, 4: {}}
# kappa-of-alpha is scale-free (MAD-normalised), so the positive shapes must differ, not just their size,
# or two uids tie on skill and share a rank.
AL = {1: [-1.0, -1.1, -0.9, -1.2], 2: [0.5, -0.3, 0.4, -0.35], 3: [1.0, 1.1, 0.9, 1.2, 0.2], 4: [3.0, 3.1, 2.9, 3.2]}


def test_debeta_scores_defaults_to_positives_and_publishes_matching_ranks():
    detail = {}
    scores = debeta_scores(CB, CS, AL, floor=0.0, w_make=0.30, detail=detail)
    assert detail[1]["skill_raw"] < 0 and detail[1]["skill_rank"] == 0.0, "negative skill earns nothing"
    # kappa-of-alpha is magnitude-blind, so the ORDER of the three positive skills is whatever the
    # consistency measure says; the scope rule is about the ranks that order receives.
    positives = sorted((u for u in (2, 3, 4)), key=lambda u: detail[u]["skill_raw"])
    assert all(detail[u]["skill_raw"] > 0 for u in positives)
    assert detail[positives[0]]["skill_rank"] == 0.0, "the smallest positive skill earns nothing on the skill leg"
    assert detail[positives[1]]["skill_rank"] == pytest.approx(0.5)
    assert detail[positives[2]]["skill_rank"] == pytest.approx(1.0)
    for u, s in scores.items():
        assert s == pytest.approx(0.30 * detail[u]["making_rank"] + 0.70 * detail[u]["skill_rank"])


def test_debeta_scores_whole_reproduces_the_clamp_rule():
    detail = {}
    debeta_scores(CB, CS, AL, floor=0.0, w_make=0.30, detail=detail, skill_rank_scope="whole")
    positives = sorted((u for u in (2, 3, 4)), key=lambda u: detail[u]["skill_raw"])
    # one non-positive of four: the smallest positive inherits the block and ranks 1/3
    assert detail[positives[0]]["skill_rank"] == pytest.approx(1 / 3)
    assert detail[1]["skill_rank"] == 0.0


def test_the_dial_reaches_the_scorer_and_the_child_and_defaults_to_positives():
    reward_src = (DEV / "taos/im/validator/reward.py").read_text()
    shadow_src = (DEV / "taos/im/validator/scoring_shadow.py").read_text()
    assert "skill_rank_scope=str(getattr(dcfg, 'skill_rank_scope', None) or 'positives')" in reward_src
    assert "skill_rank_scope=str(_d_scope if (_d_scope := getattr(_d, 'skill_rank_scope', None)) is not None" in shadow_src, (
        "the CUTOVER child scores from the config duck: a dial missing there silently reverts the child"
    )
    from taos.im.config import add_im_validator_args
    parser = argparse.ArgumentParser()
    add_im_validator_args(object, parser)
    args, _ = parser.parse_known_args([])
    assert getattr(args, "scoring.debeta.skill_rank_scope") == "positives"
    with pytest.raises(SystemExit):
        parser.parse_known_args(["--scoring.debeta.skill_rank_scope", "median"])


def test_rank01_is_unchanged():
    """The whole-pool rank both legs use under scope 'whole' (see test_debeta_making_rank_scope for making)."""
    assert _rank01([0.0, 0.0, 2.0, 1.0]) == [0.0, 0.0, 1.0, 2 / 3]
