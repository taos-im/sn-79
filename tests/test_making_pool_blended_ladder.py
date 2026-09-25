# SPDX-License-Identifier: MIT
"""The third making-pool setting: pay the pool proportionally, but let the ladder keep the blend.

`proportional` takes the making leg out of the ladder entirely, so the ladder ranks on skill alone.
That is what makes it cure concentration, and also what makes it expensive: at w_make 0.50 half of
emission becomes a pot decided purely on skill, which accounts with no maker volume can win outright.
Across twelve mainnet boards, zero-maker accounts go from 10.4 per cent of emission to 42.1.

`proportional_blended` pays the same pool on the same shares and leaves the ladder ranking on the
full blended score. Making is then paid on both surfaces, which is the cost; in exchange a zero-maker
account is capped by its making rank again, exactly as it is under `rank`.

None of this changes the default. `rank` remains the shipped behaviour.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from taos.im.validator.reward import (  # noqa: E402
    allocate_trading, distribute_rewards, making_pool_inputs, pool_ladder_input,
)

CONFIG = {"rewarding": {"seed": 898746039182, "pareto": {"shape": 1.5, "scale": 1.0}, "floor": {"enabled": False}}}
W = 0.5


def _detail(rank, raw, present=True):
    return {"making_rank": rank, "making_raw": raw, "present": present}


def _pay(uids, detail, trading, mode):
    if mode == "rank":
        vec = [float(x) for x in distribute_rewards([trading[u] for u in uids], CONFIG)]
        tot = sum(vec) or 1.0
        return [x / tot for x in vec]
    ladder, _term, share = making_pool_inputs(detail, uids, trading, 1.0, W)
    ladder = pool_ladder_input(mode, ladder, trading, uids)
    return [float(x) for x in allocate_trading(ladder, share, W, uids, CONFIG)]


def test_the_helper_picks_the_stripped_ladder_only_for_plain_proportional():
    uids = [1, 2]
    stripped = {1: 0.30, 2: 0.20}
    trading = {1: 0.80, 2: 0.45}
    assert pool_ladder_input("proportional", stripped, trading, uids) == stripped
    assert pool_ladder_input("proportional_blended", stripped, trading, uids) == pytest.approx(trading)
    # rank does not allocate through the pool at all; asked anyway it must not invent a third answer
    assert pool_ladder_input("rank", stripped, trading, uids) == stripped


def test_the_pool_half_is_identical_between_the_two_proportional_settings():
    """Only what the LADDER ranks differs. The shares, and so the pool payout, must not move."""
    uids = [1, 2, 3]
    detail = {1: _detail(1.0, 900.0), 2: _detail(0.5, 100.0), 3: _detail(0.0, 0.0)}
    trading = {1: 0.80, 2: 0.45, 3: 0.20}
    _l1, t1, s1 = making_pool_inputs(detail, uids, trading, 1.0, W)
    _l2, t2, s2 = making_pool_inputs(detail, uids, trading, 1.0, W)
    assert s1 == s2 and t1 == t2


def test_a_zero_maker_account_is_capped_again_under_the_blend():
    """The property the setting exists for: no maker volume, top skill, and it must not win."""
    uids = [1, 2, 3]
    detail = {1: _detail(1.0, 900.0), 2: _detail(0.6, 300.0), 3: _detail(0.0, 0.0)}
    # uid 3 captures nothing and has the best skill on the board
    trading = {1: W * 1.0 + (1 - W) * 0.2, 2: W * 0.6 + (1 - W) * 0.3, 3: W * 0.0 + (1 - W) * 1.0}
    prop = _pay(uids, detail, trading, "proportional")
    blend = _pay(uids, detail, trading, "proportional_blended")
    assert blend[2] < prop[2]


def test_the_largest_maker_gains_which_is_the_cost_of_the_setting():
    """Stated rather than hidden: making is paid twice, so the biggest maker collects on both sides."""
    uids = [1, 2, 3]
    detail = {1: _detail(1.0, 900.0), 2: _detail(0.6, 300.0), 3: _detail(0.0, 0.0)}
    trading = {1: W * 1.0 + (1 - W) * 0.2, 2: W * 0.6 + (1 - W) * 0.3, 3: W * 0.0 + (1 - W) * 1.0}
    prop = _pay(uids, detail, trading, "proportional")
    blend = _pay(uids, detail, trading, "proportional_blended")
    assert blend[0] > prop[0]


def _clone_board(k, field=40):
    """One operator's capture split across k uids, total held constant, against a fixed field."""
    uids = list(range(k + field))
    cap = {i: 9000.0 / k for i in range(k)}
    cap.update({k + j: 300.0 + (j * 37 % 400) for j in range(field)})
    order = sorted(uids, key=lambda u: -cap[u])
    mrank = {u: 1.0 - (order.index(u) / (len(uids) - 1)) for u in uids}
    skill = {u: (0.30 if u < k else 0.25 + ((u * 17) % 50) / 100.0) for u in uids}
    detail = {u: _detail(mrank[u], cap[u]) for u in uids}
    trading = {u: W * mrank[u] + (1 - W) * skill[u] for u in uids}
    return uids, detail, trading


@pytest.mark.parametrize("mode,limit", [("proportional", 1.10), ("proportional_blended", 1.40)])
def test_cloning_one_maker_across_more_uids_still_buys_almost_nothing(mode, limit):
    """Blended puts the making rank back on the ladder, so cloning is not fully neutral any more.

    It is bounded, and the bound is what this pins. Splitting the same captured spread across 16 uids
    gains about 4x under the shipped rank ladder, 1.04x under plain proportional and 1.18x here, so
    the setting keeps most of the property rather than trading it away. If a change pushes this past
    the limit, the setting has stopped being worth its cost.
    """
    one = sum(_pay(*_clone_board(1), mode)[:1])
    many = sum(_pay(*_clone_board(16), mode)[:16])
    assert many / one < limit


def test_the_rank_ladder_is_the_thing_being_improved_on():
    """Not a property of the new setting: the reason both pool settings exist."""
    one = sum(_pay(*_clone_board(1), "rank")[:1])
    many = sum(_pay(*_clone_board(16), "rank")[:16])
    assert many / one > 3.0


def test_the_default_is_proportional_both_and_the_blended_setting_is_still_accepted():
    from taos.im.config import add_im_validator_args
    import argparse

    parser = argparse.ArgumentParser()

    class _C:
        pass

    add_im_validator_args(_C, parser)
    action = next(a for a in parser._actions if a.dest == "scoring.debeta.making_pool")
    assert action.default == "proportional_both"
    assert set(action.choices) == {"rank", "proportional", "proportional_blended", "proportional_both"}
