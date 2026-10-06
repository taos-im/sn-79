# SPDX-License-Identifier: MIT
"""Under the flat rule an asset class's weight in each half of the pool is whatever share of the field's
capture and alpha it happens to produce, so a 64+64 layout hands a new class about half the pool on day
one and the share floats with activity. scoring.debeta.class_weights fixes it: each class's share vector
is normalised within the class and the classes are mixed at the dial, so a new class can be started at
5 per cent and stepped up once it has been observed. A class with no credit in a half contributes
nothing and the others' weights are renormalised, as the flat rule already does when nothing is captured.
The book-to-class map travels in the simulation config so the scoring child pays the same vector."""
import argparse
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taos.im.config import add_im_validator_args  # noqa: E402
import taos.im.validator.reward as reward  # noqa: E402


def _detail(making, skill, books=30):
    return {
        u: {"making_raw": making.get(u, 0.0), "present": True, "skill_raw": max(0.0, skill.get(u, 0.0)),
            "skill_net_alpha": skill.get(u, 0.0), "skill_books": books, "skill_p11_factor": 1.0}
        for u in set(making) | set(skill)}


def test_the_weights_parse_normalise_and_an_empty_spec_means_flat():
    assert reward.class_weights_from("") == []
    assert reward.class_weights_from(None) == []
    assert reward.class_weights_from("0.95,0.05") == [0.95, 0.05]
    assert reward.class_weights_from("19, 1") == [0.95, 0.05]
    assert reward.class_weights_from([1.0]) == [1.0]
    with pytest.raises(ValueError):
        reward.class_weights_from("0.5,-0.5")


def test_the_books_of_each_class_feed_only_that_class():
    cb = {1: {0: 1.0, 1: 2.0, 96: 3.0}, 2: {96: 4.0}}
    cs = {1: {0: 1.0, 1: 1.0, 96: 3.0}, 2: {96: 4.0}}
    pool = {1: {0: 0.5, 96: -0.2}, 2: {96: 0.7}}
    classes = reward.split_by_class(cb, cs, pool, {0: 0, 1: 0, 96: 1})
    assert sorted(classes) == [0, 1]
    assert classes[0]["capture_buy"] == {1: {0: 1.0, 1: 2.0}}
    assert classes[1]["capture_buy"] == {1: {96: 3.0}, 2: {96: 4.0}}
    assert classes[0]["pool"] == {1: {0: 0.5}} and classes[1]["pool"] == {1: {96: -0.2}, 2: {96: 0.7}}
    assert classes[0]["n_books"] == 2 and classes[1]["n_books"] == 1


def test_a_uid_only_in_the_small_class_is_paid_at_most_the_dial_and_shares_sum_to_one():
    base = _detail({1: 10.0, 2: 30.0}, {1: 5.0, 2: 15.0})
    new = _detail({3: 8.0, 2: 2.0}, {3: 4.0, 2: 4.0})
    making, skill = reward.class_pool_shares({0: base, 1: new}, [0.95, 0.05], [1, 2, 3], {0: 20, 1: 20})
    assert making[3] == pytest.approx(0.05 * 0.8) and skill[3] == pytest.approx(0.05 * 0.5)
    assert making[2] == pytest.approx(0.95 * 0.75 + 0.05 * 0.2)
    assert sum(making.values()) == pytest.approx(1.0) and sum(skill.values()) == pytest.approx(1.0)


def test_a_class_with_no_credit_in_a_half_hands_its_weight_to_the_others():
    base = _detail({1: 10.0}, {1: 5.0})
    empty = _detail({3: 0.0}, {3: 0.0})
    making, skill = reward.class_pool_shares({0: base, 1: empty}, [0.95, 0.05], [1, 3], {0: 20, 1: 20})
    assert making[1] == pytest.approx(1.0) and skill[1] == pytest.approx(1.0)
    assert making[3] == 0.0 and skill[3] == 0.0


def test_one_class_at_full_weight_is_the_flat_rule():
    base = _detail({1: 10.0, 2: 30.0}, {1: 5.0, 2: 15.0})
    making, skill = reward.class_pool_shares({0: base}, [1.0], [1, 2], {0: 20})
    assert making == {1: pytest.approx(0.25), 2: pytest.approx(0.75)}
    assert skill == {1: pytest.approx(0.25), 2: pytest.approx(0.75)}


def test_the_bar_applies_per_class():
    base = _detail({1: 10.0}, {1: 5.0}, books=30)
    new = _detail({3: 8.0}, {3: 4.0}, books=3)          # under the small class's own bar of 4
    making, skill = reward.class_pool_shares({0: base, 1: new}, [0.95, 0.05], [1, 3], {0: 20, 1: 4})
    assert skill[3] == 0.0 and skill[1] == pytest.approx(1.0)
    assert making[3] == pytest.approx(0.05)


def test_the_pool_vector_honours_the_mixed_shares():
    import torch
    cfg = {"rewarding": {"seed": 1, "pareto": {"shape": 1.5, "scale": 1.0}, "floor": {"enabled": False}}}
    detail = _detail({1: 10.0, 2: 30.0, 3: 0.0}, {1: 5.0, 2: 15.0, 3: 0.0})
    uids = [1, 2, 3]
    mixed = ({1: 0.5, 2: 0.4, 3: 0.1}, {1: 0.2, 2: 0.7, 3: 0.1})
    vec = reward.pool_pay_vector("proportional_both", 0.5, detail, uids, {u: 1.0 for u in uids},
                                 {u: 1.0 for u in uids}, mixed[0], cfg, 20, class_shares=mixed)
    assert torch.allclose(vec, torch.FloatTensor([0.35, 0.55, 0.10]), atol=1e-6)


def test_the_map_the_weights_and_the_shares_reach_both_pay_paths():
    import taos.im.validator.scoring_shadow as shadow

    assert "'class_of_book'" in inspect.getsource(reward.build_simulation_config_dict)
    assert "'class_weights'" in inspect.getsource(reward.build_scoring_config)
    assert "_debeta_class_shares" in inspect.getsource(reward.compute_debeta_scores)
    for fn in (reward.get_rewards, shadow.shadow_score):
        assert "class_shares=" in inspect.getsource(fn)


def test_the_default_is_the_production_split_and_the_flat_rule_everywhere_it_does_not_fit():
    # Since 1 October 2026 the release default is the two-background production split; the flat rule is what a
    # single market (no class map) and any layout with another background count get from it.
    from taos.im.validator.reward import class_weights_for_layout

    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    known, _ = parser.parse_known_args([])
    default = getattr(known, "scoring.debeta.class_weights")
    assert default == "0.95,0.05"
    assert class_weights_for_layout(default, {}) == []
    assert class_weights_for_layout(default, {0: 0, 1: 1, 2: 2}) == []
    assert class_weights_for_layout(default, {0: 0, 1: 1}) == [0.95, 0.05]
