"""The skill share the pool pays each uid is published beside the making share. A count of uids with skill taken on the
flat all-books rank nets a uid's alpha across classes and is not what pays; under the per-class rule the paid skill
share is the class-mixed share, smoothed by the track-record EMA (smoothed_skill_share). The rule pool_pay_vector pays
by is stated once (reward.paid_skill_share); both scorers, main and the scoring service, note its normalized vector on
each uid's kappa_values as 'skill_share' (note_skill_share) the way they note 'making_share', and the report publishes
it as the miner gauge debeta_skill_share beside debeta_making_share. Pay is unchanged: pool_pay_vector given the shared
function's vector pays what it paid before."""
import os

import torch

from taos.im.validator import reward

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UIDS = [0, 1, 2, 3]
CLASS = ({0: 0.4, 1: 0.3, 2: 0.3, 3: 0.0}, {0: 0.1, 1: 0.6, 2: 0.3, 3: 0.0})
SMOOTH = {0: 0.2, 1: 0.5, 2: 0.3, 3: 0.0}


def test_the_paid_skill_share_follows_pool_pay_vectors_rule():
    assert reward.paid_skill_share("proportional", 0.5, {}, UIDS, 4, CLASS, SMOOTH) is None, "skill not paid apart"
    assert reward.paid_skill_share("proportional_both", 0.0, {}, UIDS, 4, CLASS, SMOOTH) is None, "no pool"
    assert reward.paid_skill_share("proportional_both", 0.5, {}, UIDS, 4, CLASS, SMOOTH) is SMOOTH, "smoothed first"
    assert reward.paid_skill_share("proportional_both", 0.5, {}, UIDS, 4, CLASS, None) == CLASS[1]
    detail = {0: {"skill_raw": 1.0, "skill_books": 9}, 1: {"skill_raw": 3.0, "skill_books": 9}}
    flat = reward.paid_skill_share("proportional_both", 0.5, detail, UIDS, 4, None, None)
    assert flat == reward.skill_pool_share(detail, UIDS, 4), "the flat share without classes"


def test_pool_pay_vector_pays_the_same_given_the_shared_rules_vector():
    cfg = {"rewarding": {"seed": 1, "pareto": {"shape": 1.5, "scale": 1.0}, "floor": {"enabled": False}}}
    making = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    ladder = {u: float(u) for u in UIDS}
    trading = {u: 1.0 + u for u in UIDS}
    for skill in (SMOOTH, None):
        before = reward.pool_pay_vector("proportional_both", 0.5, {}, UIDS, ladder, trading, dict(making), cfg, 4,
                                        class_shares=CLASS, skill_share=skill)
        paid = reward.paid_skill_share("proportional_both", 0.5, {}, UIDS, 4, CLASS, skill)
        after = reward.pool_pay_vector("proportional_both", 0.5, {}, UIDS, ladder, trading, dict(making), cfg, 4,
                                       class_shares=CLASS, skill_share=paid)
        assert torch.equal(before, after)


def test_each_uid_notes_its_normalized_paid_skill_share():
    kv = {0: {}, 1: {}, 2: {}}
    reward.note_skill_share(kv, {0: 1.0, 1: 3.0, 2: 0.0, 3: 4.0}, UIDS)
    assert kv == {0: {"skill_share": 0.125}, 1: {"skill_share": 0.375}, 2: {"skill_share": 0.0}}, "uid 3 has no row"
    reward.note_skill_share(kv, None, UIDS)
    assert all(r["skill_share"] is None for r in kv.values()), "skill not paid apart: nothing to publish"


def test_both_scorers_note_it_and_the_report_publishes_it():
    src = {p: open(os.path.join(ROOT, p)).read() for p in (
        "taos/im/validator/reward.py", "taos/im/validator/scoring_shadow.py", "taos/im/validator/report.py")}
    assert src["taos/im/validator/reward.py"].count("note_skill_share(") >= 2, "defined and called by main's scorer"
    assert "note_skill_share(shadow.kappa_values" in src["taos/im/validator/scoring_shadow.py"]
    rep = src["taos/im/validator/report.py"]
    assert "'debeta_making_share', 'debeta_skill_share', 'debeta_ladder_input'," in rep, "a declared label"
    assert "'skill_share': kappa_values.get('skill_share') if kappa_values else None," in rep
    assert "('skill_share', 'debeta_skill_share')," in rep, "a miner gauge, evicted when absent"
    assert "debeta_skill_share=(m['skill_share'] if m.get('skill_share') is not None else 0.0)," in rep
