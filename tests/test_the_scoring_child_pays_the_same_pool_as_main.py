# SPDX-License-Identifier: MIT
"""The scoring child and main build the emission vector through one function, so a pool setting cannot be paid
one way by main and another by the child. The child handled 'proportional' only and fell through to the Pareto
ladder under 'proportional_both', so in cutover mode main adopted ladder vectors after every restart until the
child's VERIFY mismatches suspended it (mainnet, 28 September 2026, 15:05 to 15:20 UTC).

Each round's trading vector is also normalised to sum 1 before the moving average. The ladder's output is not
(about 390 per round on 257 uids) and the pool's is, so without it a ladder round outweighed hundreds of pool
rounds and the weights kept the ladder's shape until enough pool rounds had accumulated to displace it.
"""
import inspect
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taos.im.validator.reward as reward  # noqa: E402
import taos.im.validator.scoring_shadow as shadow  # noqa: E402

CFG = {"rewarding": {"seed": 898746039182, "pareto": {"shape": 1.5, "scale": 1.0}, "floor": {"enabled": False}}}
UIDS = list(range(8))


def _inputs():
    detail = {u: {"present": True, "skill_raw": 0.5 if u < 5 else 0.0, "skill_books": 25,
                  "skill_net_alpha": 10.0 * (u + 1), "skill_p11_factor": 1.0,
                  "making_rank": u / 7, "making_raw": float(u)} for u in UIDS}
    trading = {u: 0.1 * u for u in UIDS}
    ladder, _term, share = reward.making_pool_inputs(detail, UIDS, trading, 1.0, 0.5)
    return detail, trading, ladder, share


def test_proportional_both_pays_the_mixed_shares_with_no_ladder():
    detail, trading, ladder, share = _inputs()
    v = reward.pool_pay_vector("proportional_both", 0.5, detail, UIDS, ladder, trading, share, CFG, 20)
    skill = reward.skill_pool_share(detail, UIDS, 20)
    tot = sum(share.values())
    expect = [0.5 * skill[u] + 0.5 * share[u] / tot for u in UIDS]
    assert torch.allclose(v, torch.tensor(expect), atol=1e-6)


def test_rank_is_the_ladder():
    detail, trading, ladder, share = _inputs()
    v = reward.pool_pay_vector("rank", 0.5, detail, UIDS, ladder, trading, share, CFG, 20)
    assert torch.allclose(v, reward.distribute_rewards([trading[u] for u in UIDS], CFG))


def test_main_and_the_child_both_build_the_vector_through_it():
    for fn in (reward.get_rewards, shadow.shadow_score):
        src = inspect.getsource(fn)
        assert "pool_pay_vector(" in src, fn.__name__
        assert "distribute_rewards(" not in src, f"{fn.__name__} builds a ladder of its own"


def test_each_rounds_trading_vector_enters_the_moving_average_normalised():
    from taos.common.neurons.validator import BaseValidatorNeuron
    v = SimpleNamespace(scores=torch.zeros(4), gentrx_scores=torch.zeros(4), device="cpu",
                        config=SimpleNamespace(neuron=SimpleNamespace(moving_average_alpha=0.5)))
    BaseValidatorNeuron.update_scores(v, torch.tensor([300.0, 100.0, 0.0, 0.0]), [0, 1, 2, 3])
    assert abs(float(v.scores.sum()) - 0.5) < 1e-6
    assert torch.allclose(v.scores, torch.tensor([0.375, 0.125, 0.0, 0.0]))
    BaseValidatorNeuron.update_scores(v, torch.tensor([0.0, 0.0, 0.0, 1.0]), [0, 1, 2, 3])
    assert torch.allclose(v.scores, torch.tensor([0.1875, 0.0625, 0.0, 0.5])), \
        "a normalised round and a ladder-scale round carry equal mass"


def test_an_all_zero_round_is_left_alone():
    from taos.common.neurons.validator import BaseValidatorNeuron
    v = SimpleNamespace(scores=torch.tensor([0.5, 0.5]), gentrx_scores=torch.zeros(2), device="cpu",
                        config=SimpleNamespace(neuron=SimpleNamespace(moving_average_alpha=0.5)))
    BaseValidatorNeuron.update_scores(v, torch.zeros(2), [0, 1])
    assert torch.allclose(v.scores, torch.tensor([0.25, 0.25]))


def test_scores_saved_at_ladder_scale_load_at_unit_sum_with_the_same_weights():
    import inspect as _i
    import taos.im.validator.persistence as persistence
    src = _i.getsource(persistence._load_validator_state)
    assert "self.scores = self.scores / _ssum" in src
    scores = torch.tensor([120.0, 60.0, 20.0, 0.0])
    before = torch.nn.functional.normalize(scores, p=1, dim=0)
    rescaled = scores / float(scores.sum())
    assert abs(float(rescaled.sum()) - 1.0) < 1e-6
    assert torch.allclose(torch.nn.functional.normalize(rescaled, p=1, dim=0), before)
