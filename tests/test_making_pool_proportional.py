# SPDX-License-Identifier: MIT
"""The proportional making pool: what it pays, and the property it exists for.

The Pareto ladder pays rank POSITIONS with a steep top, so an operator whose accounts occupy the
top positions collects many top-of-curve weights whatever its aggregate service. Paying the making
leg in proportion to captured spread makes an operator's total equal its share of the liquidity
provided, so splitting one strategy across more accounts gains nothing. That last sentence is the
cloning-invariance test below, and it is the reason the option exists.
"""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from taos.im.validator.reward import (  # noqa: E402
    allocate_trading, distribute_rewards, making_pool_inputs,
)

CONFIG = {"rewarding": {"seed": 898746039182, "pareto": {"shape": 1.5, "scale": 1.0}, "floor": {"enabled": False}}}


def _detail(rank, raw, present=True):
    return {"making_rank": rank, "making_raw": raw, "present": present}


def test_inputs_remove_the_making_leg_from_the_ladder_and_share_the_capture():
    uids = [1, 2, 3]
    detail = {1: _detail(1.0, 900.0), 2: _detail(0.5, 100.0), 3: _detail(0.0, 0.0)}
    trading = {1: 0.80, 2: 0.45, 3: 0.20}
    ladder, term, share = making_pool_inputs(detail, uids, trading, w=1.0, w_make=0.5)
    assert term == {1: 0.5, 2: 0.25, 3: 0.0}
    assert ladder == pytest.approx({1: 0.30, 2: 0.20, 3: 0.20})
    assert share == pytest.approx({1: 0.9, 2: 0.1, 3: 0.0})
    assert sum(share.values()) == pytest.approx(1.0)


def test_an_absent_uid_pays_nothing_into_the_pool_and_takes_nothing_from_it():
    uids = [1, 2]
    detail = {1: _detail(1.0, 500.0, present=False), 2: _detail(0.5, 500.0)}
    _, _, share = making_pool_inputs(detail, uids, {1: 0.9, 2: 0.4}, w=1.0, w_make=0.5)
    assert share[1] == 0.0 and share[2] == pytest.approx(1.0)


def test_a_uid_with_no_capture_gets_only_the_skill_side():
    uids = [1, 2]
    share = {1: 1.0, 2: 0.0}
    out = allocate_trading({1: 0.2, 2: 0.9}, share, pool=0.5, all_uids=uids, config=CONFIG)
    assert float(out.sum()) == pytest.approx(1.0)
    # uid 2 captured nothing, so its whole entitlement comes from the ladder half
    ladder_only = distribute_rewards([0.2, 0.9], CONFIG)
    ladder_only = ladder_only / float(ladder_only.sum())
    assert float(out[1]) == pytest.approx(0.5 * float(ladder_only[1]), rel=1e-6)


def _ladder_norm(scores, uids):
    """The normalised Pareto ladder on its own, which is what allocate_trading mixes the pool with.
    Equal scores do NOT split evenly: each uid draws its own Pareto sample, so the ladder part has
    to be computed rather than assumed."""
    lad = distribute_rewards([scores[u] for u in uids], CONFIG)
    return lad / float(lad.sum())


@pytest.mark.parametrize("pool", [0.0, 0.25, 0.5, 1.0])
def test_the_pool_takes_exactly_its_share_of_emission(pool):
    uids = [1, 2, 3, 4]
    scores = {u: 0.5 for u in uids}
    share = {1: 0.4, 2: 0.3, 3: 0.3, 4: 0.0}
    out = allocate_trading(scores, share, pool=pool, all_uids=uids, config=CONFIG)
    lad = _ladder_norm(scores, uids)
    assert float(out.sum()) == pytest.approx(1.0)
    for i, u in enumerate(uids):
        assert float(out[i]) == pytest.approx((1.0 - pool) * float(lad[i]) + pool * share[u], abs=1e-7)
    # the three that captured spread hold the whole pool between them
    captured = sum(float(out[i]) - (1.0 - pool) * float(lad[i]) for i in range(3))
    assert captured == pytest.approx(pool, abs=1e-6)


def test_nothing_captured_anywhere_leaves_the_ladder_whole_rather_than_burning_the_pool():
    uids = [1, 2]
    out = allocate_trading({1: 0.7, 2: 0.3}, {1: 0.0, 2: 0.0}, pool=0.5, all_uids=uids, config=CONFIG)
    ladder = distribute_rewards([0.7, 0.3], CONFIG)
    ladder = ladder / float(ladder.sum())
    assert torch.allclose(out, ladder)


def test_cloning_one_maker_across_more_uids_does_not_change_the_operator_total():
    """The property the option exists for. An operator captures the same 600 units of spread, first
    through one uid and then split across six. Its pool money is identical, so cloning gains it
    nothing; under the ladder the six would occupy six top positions instead of one."""
    pool = 0.5
    rest = {u: 120.0 for u in range(10, 20)}          # the rest of the board, unchanged in both worlds
    detail_one = {1: _detail(1.0, 600.0), **{u: _detail(0.5, v) for u, v in rest.items()}}
    detail_many = {**{u: _detail(0.5, 100.0) for u in range(1, 7)},
                   **{u: _detail(0.5, v) for u, v in rest.items()}}
    uids_one = [1] + list(rest)
    uids_many = list(range(1, 7)) + list(rest)
    _, _, share_one = making_pool_inputs(detail_one, uids_one, {u: 0.5 for u in uids_one}, 1.0, 0.5)
    _, _, share_many = making_pool_inputs(detail_many, uids_many, {u: 0.5 for u in uids_many}, 1.0, 0.5)
    # the operator's share of captured spread is the same either way
    assert sum(share_one[u] for u in [1]) == pytest.approx(sum(share_many[u] for u in range(1, 7)), abs=1e-9)

    scores_one = {u: 0.5 for u in uids_one}
    scores_many = {u: 0.5 for u in uids_many}
    out_one = allocate_trading(scores_one, share_one, pool, uids_one, CONFIG)
    out_many = allocate_trading(scores_many, share_many, pool, uids_many, CONFIG)
    lad_one, lad_many = _ladder_norm(scores_one, uids_one), _ladder_norm(scores_many, uids_many)
    pool_one = sum(float(out_one[uids_one.index(u)]) - (1 - pool) * float(lad_one[uids_one.index(u)])
                   for u in [1])
    pool_many = sum(float(out_many[uids_many.index(u)]) - (1 - pool) * float(lad_many[uids_many.index(u)])
                    for u in range(1, 7))
    assert pool_one == pytest.approx(pool_many, abs=1e-6)
    assert pool_one == pytest.approx(pool * 600.0 / (600.0 + sum(rest.values())), abs=1e-6)


def test_the_ladder_still_rewards_the_skill_leg_superlinearly():
    """Skill keeps the Pareto ladder: a better predictor earns far more than a slightly worse one,
    which is the property the making leg is losing on purpose."""
    uids = list(range(1, 11))
    scores = {u: 0.5 + 0.01 * u for u in uids}
    out = allocate_trading(scores, {u: 0.0 for u in uids}, pool=0.0, all_uids=uids, config=CONFIG)
    top, near = float(out.max()), sorted(float(x) for x in out)[-2]
    assert top > 1.5 * near
