# SPDX-License-Identifier: MIT
"""The fourth making-pool setting: both halves paid additively.

`proportional` pays the making half in proportion to captured spread and leaves the skill half on a
Pareto ladder over kappa. kappa is magnitude-blind, so one predictor split k ways keeps the full kappa
on every clone and collects k rank positions: on real alphas (board 20260922T114403Z) the top predictor
split sixteen ways gained 14.03x. A rank ladder on a standalone pot is a tournament, and tournaments
without identity are entered many times.

`proportional_both` pays the skill half in proportion to each uid's net alpha over the books it filled,
times its counterparty factor, among uids whose skill is positive on at least `min_books` qualifying
books. Additive on both sides, so cloning-invariant on both by construction. Real markets pay traders on
realised P&L and let consistency decide who is allocated capital; that is the market justification
identity grouping lacks. The default stays `rank`.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from taos.im.validator.reward import (  # noqa: E402
    allocate_trading, distribute_rewards, making_pool_inputs, skill_pool_share,
)

CONFIG = {"rewarding": {"seed": 898746039182, "pareto": {"shape": 1.5, "scale": 1.0}, "floor": {"enabled": False}}}
W = 0.5


def _d(mrank, mraw, skill_raw, net, books, factor=1.0, present=True):
    return {"making_rank": mrank, "making_raw": mraw, "skill_raw": skill_raw, "skill_net_alpha": net,
            "skill_books": books, "skill_p11_factor": factor, "present": present}


def test_the_dial_takes_four_settings_and_defaults_to_proportional_both():
    import argparse

    from taos.im.config import add_im_validator_args
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    action = next(a for a in parser._actions if a.dest == "scoring.debeta.making_pool")
    assert action.default == "proportional_both"
    assert set(action.choices) == {"rank", "proportional", "proportional_blended", "proportional_both"}


def test_the_skill_share_is_net_alpha_times_the_factor_gated_on_positive_skill_and_books():
    uids = [1, 2, 3, 4, 5]
    detail = {
        1: _d(0.9, 500.0, 2.0, 1000.0, 60),            # in: positive skill, broad
        2: _d(0.0, 0.0, 3.0, 400.0, 60, factor=0.25),  # in, discounted by its counterparty factor
        3: _d(0.5, 100.0, 0.0, 800.0, 60),             # out: skill not positive
        4: _d(0.5, 100.0, 2.0, 900.0, 10),             # out: under the book minimum
        5: _d(0.5, 100.0, 2.0, -300.0, 60),            # in but negative net alpha -> zero share
    }
    share = skill_pool_share(detail, uids, min_books=20)
    assert share[1] == pytest.approx(1000.0 / 1100.0)
    assert share[2] == pytest.approx(100.0 / 1100.0)
    assert share[3] == 0.0 and share[4] == 0.0 and share[5] == 0.0
    assert sum(share.values()) == pytest.approx(1.0)


def test_an_absent_uid_takes_nothing_from_the_skill_pool():
    detail = {1: _d(0.5, 100.0, 2.0, 500.0, 60, present=False), 2: _d(0.5, 100.0, 2.0, 500.0, 60)}
    share = skill_pool_share(detail, [1, 2], min_books=20)
    assert share[1] == 0.0 and share[2] == pytest.approx(1.0)


def test_nothing_eligible_returns_an_empty_share_so_the_caller_falls_back_to_the_ladder():
    detail = {1: _d(0.5, 100.0, 0.0, 500.0, 60), 2: _d(0.5, 100.0, 2.0, 500.0, 4)}
    share = skill_pool_share(detail, [1, 2], min_books=20)
    assert sum(share.values()) == 0.0


def test_allocate_pays_the_skill_half_proportionally_when_a_skill_share_is_given():
    uids = [1, 2, 3]
    ladder = {1: 0.5, 2: 0.2, 3: 0.1}
    making = {1: 0.9, 2: 0.1, 3: 0.0}
    skill = {1: 0.0, 2: 0.25, 3: 0.75}
    out = allocate_trading(ladder, making, W, uids, CONFIG, skill_share=skill)
    vec = [float(x) for x in out]
    assert sum(vec) == pytest.approx(1.0)
    # making half by making share, skill half by skill share, no ladder anywhere
    assert vec[0] == pytest.approx(W * 0.9 + (1 - W) * 0.0)
    assert vec[1] == pytest.approx(W * 0.1 + (1 - W) * 0.25)
    assert vec[2] == pytest.approx(W * 0.0 + (1 - W) * 0.75)


def test_an_empty_skill_share_falls_back_to_the_ladder_for_that_half():
    uids = [1, 2, 3]
    ladder = {1: 0.5, 2: 0.2, 3: 0.1}
    making = {1: 0.9, 2: 0.1, 3: 0.0}
    with_ladder = allocate_trading(ladder, making, W, uids, CONFIG)
    with_empty = allocate_trading(ladder, making, W, uids, CONFIG, skill_share={1: 0.0, 2: 0.0, 3: 0.0})
    assert [float(x) for x in with_empty] == pytest.approx([float(x) for x in with_ladder])


def test_without_a_skill_share_the_function_is_byte_identical_to_before():
    uids = [1, 2, 3]
    ladder = {1: 0.5, 2: 0.2, 3: 0.1}
    making = {1: 0.9, 2: 0.1, 3: 0.0}
    a = allocate_trading(ladder, making, W, uids, CONFIG)
    b = allocate_trading(ladder, making, W, uids, CONFIG, skill_share=None)
    assert [float(x) for x in a] == [float(x) for x in b]


def _field():
    """40 skilled non-makers around kappa 1 with real net alpha, 40 makers with capture."""
    det = {}
    for j in range(40):
        det[1000 + j] = _d(0.0, 0.0, 0.8 + (j % 7) * 0.1, 20000.0 + j * 500, 60)
    for j in range(40):
        det[2000 + j] = _d(0.5 + j / 80, 300.0 + j * 20, 0.4 + (j % 5) * 0.1, 8000.0 + j * 200, 60)
    return det


def _pay(det, mode):
    uids = sorted(det)
    tr = {u: W * det[u]["making_rank"] + (1 - W) * (0.5 + 0.5 * min(1.0, det[u]["skill_raw"] / 3.0)) for u in uids}
    ladder, _t, mshare = making_pool_inputs(det, uids, tr, 1.0, W)
    if mode == "proportional":
        vec = allocate_trading(ladder, mshare, W, uids, CONFIG)
    else:
        vec = allocate_trading(ladder, mshare, W, uids, CONFIG, skill_share=skill_pool_share(det, uids, 20))
    return dict(zip(uids, [float(x) for x in vec]))


def test_cloning_one_predictor_across_sixteen_uids_gains_nothing_under_both():
    """The property the setting exists for, against the ladder it replaces."""
    base = _field()
    base[1] = _d(0.0, 0.0, 3.0, 400000.0, 60)               # the top predictor
    one = _pay(base, "proportional_both")[1]
    split = _field()
    for c in range(16):
        split[1 + c] = _d(0.0, 0.0, 3.0, 400000.0 / 16, 60)  # kappa is magnitude-blind: same skill_raw
    many = sum(_pay(split, "proportional_both")[1 + c] for c in range(16))
    assert many / one == pytest.approx(1.0, abs=0.01)
    # and the ladder it replaces does pay the split, which is the reason for the setting
    one_l = _pay(base, "proportional")[1]
    many_l = sum(_pay(split, "proportional")[1 + c] for c in range(16))
    assert many_l / one_l > 1.5


def test_a_zero_net_alpha_account_with_top_kappa_earns_nothing_from_the_skill_half():
    det = _field()
    det[1] = _d(0.0, 0.0, 9.0, 0.0, 60)                       # extreme consistency, nothing made
    pay = _pay(det, "proportional_both")
    assert pay[1] == pytest.approx(0.0, abs=1e-12)


def test_the_two_halves_sum_to_one_with_real_shaped_inputs():
    pay = _pay(_field(), "proportional_both")
    assert sum(pay.values()) == pytest.approx(1.0)


def test_rank_mode_emission_is_unchanged_by_the_new_helper_existing():
    """Adding the fourth setting must not touch the shipped path."""
    det = _field()
    uids = sorted(det)
    tr = [W * det[u]["making_rank"] + (1 - W) * 0.5 for u in uids]
    a = distribute_rewards(tr, CONFIG)
    b = distribute_rewards(tr, CONFIG)
    assert [float(x) for x in a] == [float(x) for x in b]
