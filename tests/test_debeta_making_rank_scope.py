# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The making leg ranks positive making among positive making, as the skill leg does.

Most of the pool makes nothing on both networks. Under the whole-pool rank every zero maker tied at
0 and the smallest positive maker inherited the rank of that whole block, so negligible two-sided
capture was materially overpaid. Ranking among positives (`positives`, the default scope) removes
that while leaving the top of the board substantially unchanged. `whole` keeps the old rule for
rollback. The operator dial is --scoring.debeta.making_rank_scope.
"""
import argparse
from pathlib import Path

import pytest

from taos.im.validator.debeta import SKILL_RANK_SCOPES, _rank01, debeta_scores, rank_making, rank_skill

DEV = Path(__file__).resolve().parents[1]

# 8 uids: five zero makers, three positive makers of very different size.
MAKING = [0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 60.0, 4262.0]


def test_zero_making_ranks_zero_under_both_scopes():
    for scope in SKILL_RANK_SCOPES:
        r = rank_making(MAKING, scope)
        assert r[:5] == [0.0] * 5, scope


def test_whole_puts_the_smallest_positive_maker_on_the_block_step():
    """The rollback rule: with most of the pool at zero, a negligible making inherits the block's rank."""
    r = rank_making(MAKING, "whole")
    assert r[5] == pytest.approx(5 / 7)
    assert r[6] == pytest.approx(6 / 7) and r[7] == pytest.approx(1.0)


def test_positives_spreads_the_positive_makers_from_zero_to_one():
    r = rank_making(MAKING, "positives")
    assert r[5] == 0.0, "the smallest positive making earns no making rank"
    assert r[6] == pytest.approx(0.5)
    assert r[7] == pytest.approx(1.0)


def test_positives_ranks_are_independent_of_the_zero_block_size():
    base = rank_making([0.02, 60.0, 4262.0], "positives")
    with_zeros = rank_making([0.0] * 22 + [0.02, 60.0, 4262.0], "positives")
    assert with_zeros[-3:] == base
    whole = rank_making([0.0] * 22 + [0.02, 60.0, 4262.0], "whole")
    assert whole[-3] == pytest.approx(22 / 24), "under whole the smallest positive inherits the block"


def test_a_lone_positive_ranks_one_on_both_legs():
    """A one-value rank is 0, which would pay the only maker (or the only skilled trader) nothing."""
    assert rank_making([0.0, 0.0, 12.5], "positives") == [0.0, 0.0, 1.0]
    assert rank_skill([-0.2, 0.0, 0.7], "positives") == [0.0, 0.0, 1.0]
    assert rank_making([0.0, 0.0, 12.5], "whole") == [0.0, 0.0, 1.0]


def test_the_two_legs_share_one_rule():
    vals = [0.0, 0.0, 0.3, 7.0, 1.0]
    for scope in SKILL_RANK_SCOPES:
        assert rank_making(vals, scope) == rank_skill(vals, scope)


def test_unknown_scope_is_refused_and_named():
    with pytest.raises(ValueError, match="making_rank_scope"):
        rank_making(MAKING, "median")


# Four uids: 4 quotes 20 on both sides, 3 quotes 5, 2 quotes 0.5, 1 makes nothing. Alphas spread the skill
# leg so the score check exercises both ranks.
CB = {1: {}, 2: {0: 0.5}, 3: {0: 5.0}, 4: {0: 20.0}}
CS = {1: {}, 2: {0: 0.5}, 3: {0: 5.0}, 4: {0: 20.0}}
AL = {1: [3.0, 3.1, 2.9, 3.2], 2: [1.0, 1.1, 0.9, 1.2, 0.2], 3: [0.5, -0.3, 0.4, -0.35], 4: [-1.0, -1.1, -0.9, -1.2]}


def test_debeta_scores_defaults_to_positives_and_publishes_matching_ranks():
    detail = {}
    scores = debeta_scores(CB, CS, AL, floor=0.0, w_make=0.30, detail=detail)
    assert detail[1]["making_raw"] == 0.0 and detail[1]["making_rank"] == 0.0, "no two-sided capture, no making rank"
    assert detail[2]["making_rank"] == 0.0, "the smallest positive maker earns nothing on the making leg"
    assert detail[3]["making_rank"] == pytest.approx(0.5)
    assert detail[4]["making_rank"] == pytest.approx(1.0)
    for u, s in scores.items():
        assert s == pytest.approx(0.30 * detail[u]["making_rank"] + 0.70 * detail[u]["skill_rank"])


def test_debeta_scores_whole_reproduces_the_old_rule():
    detail = {}
    debeta_scores(CB, CS, AL, floor=0.0, w_make=0.30, detail=detail, making_rank_scope="whole")
    # one zero maker of four: the smallest positive inherits the block and ranks 1/3
    assert detail[2]["making_rank"] == pytest.approx(1 / 3)
    assert detail[1]["making_rank"] == 0.0
    assert detail[4]["making_rank"] == pytest.approx(1.0)


def test_a_negligible_maker_inherits_the_zero_block_rank():
    """A board of 22 zero makers and 10 positive ones spanning four orders of magnitude, down to a
    maker so small it is negligible. Under `whole` that maker inherits the zero block's rank and is
    paid for it; under `positives` it ranks last among the makers, which is what it is."""
    making = [0.0] * 22 + [4262.0, 2250.0, 1097.0, 361.0, 209.0, 74.0, 4.9, 1.95, 0.5, 0.02]
    whole = rank_making(making, "whole")
    assert whole[-1] == pytest.approx(22 / 31), "the cliff a whole-pool ranking produces"
    pos = rank_making(making, "positives")
    assert pos[-1] == 0.0 and pos[-2] == pytest.approx(1 / 9) and pos[22] == 1.0
    assert all(r == 0.0 for r in pos[:22])


def test_the_dial_reaches_the_scorer_and_the_child_and_defaults_to_positives():
    reward_src = (DEV / "taos/im/validator/reward.py").read_text()
    shadow_src = (DEV / "taos/im/validator/scoring_shadow.py").read_text()
    assert "making_rank_scope=str(getattr(dcfg, 'making_rank_scope', None) or 'positives')" in reward_src
    assert "making_rank_scope=str(_d_mscope if (_d_mscope := getattr(_d, 'making_rank_scope', None)) is not None" in shadow_src, (
        "the CUTOVER child scores from the config duck: a dial missing there silently reverts the child"
    )
    from taos.im.config import add_im_validator_args
    parser = argparse.ArgumentParser()
    add_im_validator_args(object, parser)
    args, _ = parser.parse_known_args([])
    assert getattr(args, "scoring.debeta.making_rank_scope") == "positives"
    with pytest.raises(SystemExit):
        parser.parse_known_args(["--scoring.debeta.making_rank_scope", "median"])


def test_rank01_still_ties_at_the_block_minimum():
    assert _rank01([0.0, 0.0, 2.0, 1.0]) == [0.0, 0.0, 1.0, 2 / 3]
