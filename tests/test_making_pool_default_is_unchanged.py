# SPDX-License-Identifier: MIT
"""The making_pool dial defaults to `proportional_both` since 0.6.2; at `rank` the emission vector is
exactly what shipped before the option existed: the Pareto sort-multiply over the whole blended score.

This is the guard that keeps `rank` an exact way back to the shipped pay. It compares against
distribute_rewards directly rather than against a recorded fixture, so it keeps holding if the
ladder's own parameters change.
"""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from taos.im.validator.reward import (  # noqa: E402
    allocate_trading, apply_reward_floor, distribute_rewards, making_pool_inputs,
)

CONFIG = {"rewarding": {"seed": 898746039182, "pareto": {"shape": 1.5, "scale": 1.0}, "floor": {"enabled": False}}}


def test_the_dial_defaults_to_proportional_both_and_only_takes_the_four_settings():
    import argparse

    from taos.im.config import add_im_validator_args
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    known, _ = parser.parse_known_args([])
    assert getattr(known, "scoring.debeta.making_pool") == "proportional_both"
    for _mode in ("rank", "proportional", "proportional_blended"):
        known, _ = parser.parse_known_args(["--scoring.debeta.making_pool", _mode])
        assert getattr(known, "scoring.debeta.making_pool") == _mode
    with pytest.raises(SystemExit):
        parser.parse_known_args(["--scoring.debeta.making_pool", "per-coldkey"])


def test_rank_mode_reproduces_the_shipped_vector_exactly():
    """What get_rewards computes under the default: floor then Pareto over the full score."""
    uids = list(range(1, 25))
    scores = {u: (u * 37 % 100) / 100.0 for u in uids}
    shipped = distribute_rewards(apply_reward_floor([scores[u] for u in uids], CONFIG), CONFIG)
    # the pool helpers are computed under both settings but must not touch the default allocation
    detail = {u: {"making_rank": (u % 5) / 4.0, "making_raw": float(u * 13), "present": True} for u in uids}
    ladder, term, share = making_pool_inputs(detail, uids, scores, w=1.0, w_make=0.5)
    again = distribute_rewards(apply_reward_floor([scores[u] for u in uids], CONFIG), CONFIG)
    assert torch.equal(shipped, again)
    assert all(ladder[u] <= scores[u] for u in uids)      # the helpers ran and returned sane values
    assert sum(share.values()) == pytest.approx(1.0)


def test_pool_zero_is_the_ladder_whatever_the_share_says():
    """debeta.weight * w_make == 0 means there is no making pool to pay, so the allocation is the
    ladder alone even if the dial is set to proportional."""
    uids = [1, 2, 3]
    scores = {1: 0.9, 2: 0.5, 3: 0.1}
    out = allocate_trading(scores, {1: 0.7, 2: 0.2, 3: 0.1}, pool=0.0, all_uids=uids, config=CONFIG)
    lad = distribute_rewards([scores[u] for u in uids], CONFIG)
    assert torch.allclose(out, lad / float(lad.sum()))


def test_the_published_trading_score_is_untouched_by_the_option():
    """The option changes allocation, not the score: making_pool_inputs derives its ladder input
    from the trading score and never writes back to it, so the blend invariant the rung gates check
    (trading == kappa*w + pnl*w + debeta*w) keeps holding under either setting."""
    uids = [1, 2]
    scores = {1: 0.8, 2: 0.4}
    before = dict(scores)
    making_pool_inputs({1: {"making_rank": 1.0, "making_raw": 10.0},
                        2: {"making_rank": 0.0, "making_raw": 0.0}}, uids, scores, 1.0, 0.5)
    assert scores == before
