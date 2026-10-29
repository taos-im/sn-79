# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Both shrinkage aggregations of a miner's per-book alphas are published beside the present skill and pay nothing
(0.6.6), so the choice between them is made on the next-window alpha test.

The self-normalising form shrinks a miner's mean per-book alpha by its own standard error, alpha_hat * alpha_hat^2 /
(alpha_hat^2 + se^2), with no floor, no bar beyond the two books a standard error needs, and its sign kept. The
empirical Bayes form shrinks it by the pool's between-miner variance, B_i = tau2 / (tau2 + se_i^2) with tau2 = max(0,
Var_i(alpha_hat_i) - mean_i(se_i^2)). Both read the per-book alphas the present skill reads (the skill pool, after the
notional hurdle when it is on), the empirical Bayes pool is each asset class's own under a multi-asset layout, and the
report publishes debeta_skill_shrunk_self and debeta_skill_shrunk_eb per uid and skill_shrunk_self and skill_shrunk_eb
per class.
"""
import inspect
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from debeta_isolated import isolated  # noqa: E402
from taos.im.validator import debeta, report, reward  # noqa: E402
from taos.im.validator.debeta import shrunk_skill_eb, shrunk_skill_self  # noqa: E402

CONSISTENT = [10.0, 10.5, 9.5, 10.0]
NOISY = [30.0, -20.0, 25.0, -15.0]


def test_a_consistent_miner_keeps_nearly_all_of_its_mean():
    # mean 10, sample variance 0.5 / 3, se^2 = 1/24: 10 * 100 / (100 + 1/24)
    assert shrunk_skill_self(CONSISTENT) == pytest.approx(1000.0 / (100.0 + 1.0 / 24.0), rel=1e-12)
    assert shrunk_skill_self(CONSISTENT) > 0.999 * 10.0


def test_a_noisy_miner_shrinks():
    # mean 5, sample variance 2050 / 3, se^2 = 2050 / 12: 5 * 25 / (25 + 2050 / 12) = 0.6383
    assert shrunk_skill_self(NOISY) == pytest.approx(125.0 / (25.0 + 2050.0 / 12.0), rel=1e-12)
    assert shrunk_skill_self(NOISY) < 0.2 * 5.0


def test_the_sign_is_kept_and_nothing_is_floored():
    assert shrunk_skill_self([-x for x in CONSISTENT]) == pytest.approx(-shrunk_skill_self(CONSISTENT), rel=1e-12)
    assert shrunk_skill_self([1e-9, 1e-9]) == pytest.approx(1e-9, rel=1e-12), "no magnitude floor"
    assert shrunk_skill_self([1.0, 3.0]) == pytest.approx(1.6, rel=1e-12), "two books are enough: 2 * 4 / (4 + 1)"


def test_one_book_and_no_alpha_are_guarded():
    assert shrunk_skill_self([7.0]) == 0.0
    assert shrunk_skill_self([]) == 0.0
    assert shrunk_skill_self([0.0, 0.0, 0.0]) == 0.0


def test_the_empirical_bayes_form_on_a_hand_computed_pool():
    # means 10, 5, -5 (uid 4 has one book and stays out of the moments); se^2 1/24, 4100/24, 1; the sample variance of
    # the means is 175/3 and their mean se^2 is 1375/24, so tau2 = 25/24
    got = shrunk_skill_eb({1: CONSISTENT, 2: NOISY, 3: [-4.0, -6.0], 4: [7.0]})
    assert got[1] == pytest.approx(10.0 * 25.0 / 26.0, rel=1e-12)
    assert got[2] == pytest.approx(5.0 / 165.0, rel=1e-12)
    assert got[3] == pytest.approx(-125.0 / 49.0, rel=1e-12)
    assert got[4] == 0.0
    assert got[1] / 10.0 > got[2] / 5.0, "the noisy miner shrinks harder than the consistent one"


def test_tau2_floored_at_zero_shrinks_everything_to_zero():
    # means 1 and 0 vary by 0.5, less than their mean se^2 of 16: the pool explains nothing, uid 1's positive mean too
    assert shrunk_skill_eb({1: [5.0, -3.0], 2: [4.0, -4.0]}) == {1: 0.0, 2: 0.0}
    assert shrunk_skill_eb({1: CONSISTENT, 2: [3.0]}) == {1: 0.0, 2: 0.0}, "a pool of one has no between-miner variance"
    assert shrunk_skill_eb({}) == {}


# per-book alphas set directly (alpha = mtm when the inventory time-sum is zero), books 0-3 in class 0 and 4-5 in class 1
ALPHAS = {
    1: {0: 10.0, 1: 10.5, 2: 9.5, 3: 10.0, 4: 11.0, 5: 9.0},
    2: {0: 30.0, 1: -20.0, 2: 25.0, 3: -15.0, 4: 28.0, 5: -22.0},
    3: {0: -4.0, 1: -6.0, 2: -5.0, 3: -5.0, 4: -4.0, 5: -6.0},
    4: {0: 1.0},
    5: {0: 2.0, 1: 3.0, 2: 2.0, 3: 3.0, 4: 2.0, 5: 3.0},
}
CLASS_OF_BOOK = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1}


def _validator(class_weights="", hurdle_bps=0.0, notional=None):
    cfg = SimpleNamespace(**isolated(enabled=True, w_make=0.30, centered_window=15, floor_scale=0.0, min_books=1,
                                     making_floor_scale=0.0, skill_rank_scope="positives", making_rank_scope="positives",
                                     presence_gate=0, class_weights=class_weights, skill_hurdle_bps=hurdle_bps))
    v = SimpleNamespace(config=SimpleNamespace(scoring=SimpleNamespace(debeta=cfg)))
    v.debeta_mtm = {u: dict(b) for u, b in ALPHAS.items()}
    v.debeta_invsum = {u: {k: 0.0 for k in b} for u, b in ALPHAS.items()}
    v.debeta_invn = {b: 1 for b in CLASS_OF_BOOK}
    v.debeta_drift = {b: 0.0 for b in CLASS_OF_BOOK}
    cb, cs = defaultdict(dict), defaultdict(dict)
    for u, books in ALPHAS.items():
        for b in books:
            cb[u][b] = 1.0 + 0.1 * u
            cs[u][b] = 1.0
    v.capture_buy_sums, v.capture_sell_sums = cb, cs
    v.debeta_notional = notional or {}
    v.miner_presence = {}
    if class_weights:
        v.simulation_config = {"class_of_book": dict(CLASS_OF_BOOK)}
    return v


def _lists(pool):
    return {u: list(b.values()) for u, b in pool.items()}


def test_the_scorer_publishes_both_forms_on_the_pool_the_present_skill_reads():
    v = _validator()
    assert reward.compute_debeta_scores(v)
    eb = shrunk_skill_eb(_lists(ALPHAS))
    for u, books in ALPHAS.items():
        assert v.debeta_detail[u]["skill_shrunk_self"] == pytest.approx(shrunk_skill_self(list(books.values())), rel=1e-12)
        assert v.debeta_detail[u]["skill_shrunk_eb"] == pytest.approx(eb[u], rel=1e-12)


def test_under_the_hurdle_both_forms_read_the_books_it_kept():
    # a hurdle of 10,000 bp keeps a book only where |alpha| clears its whole notional of 5: uid 3 keeps its four books
    # at 5 and 6, uid 5 keeps none
    notional = {u: {b: 5.0 for b in books} for u, books in ALPHAS.items()}
    v = _validator(hurdle_bps=10_000.0, notional=notional)
    assert reward.compute_debeta_scores(v)
    kept = debeta.hurdle_filter(ALPHAS, notional, 10_000.0)
    assert sorted(kept[3].values()) == [-6.0, -6.0, -5.0, -5.0] and kept[5] == {}
    eb = shrunk_skill_eb(_lists(kept))
    for u in ALPHAS:
        assert v.debeta_detail[u]["skill_shrunk_self"] == pytest.approx(shrunk_skill_self(list(kept[u].values())), rel=1e-12)
        assert v.debeta_detail[u]["skill_shrunk_eb"] == pytest.approx(eb[u], rel=1e-12)
    assert v.debeta_detail[5]["skill_shrunk_self"] == 0.0


def test_each_asset_class_shrinks_against_its_own_pool():
    v = _validator(class_weights="books")
    assert reward.compute_debeta_scores(v)
    parts = reward.split_by_class(ALPHAS, ALPHAS, ALPHAS, CLASS_OF_BOOK)
    for c, part in parts.items():
        eb = shrunk_skill_eb(_lists(part["pool"]))
        for u, books in part["pool"].items():
            got = v._debeta_class_details[c][u]
            assert got["skill_shrunk_self"] == pytest.approx(shrunk_skill_self(list(books.values())), rel=1e-12)
            assert got["skill_shrunk_eb"] == pytest.approx(eb[u], rel=1e-12)
    flat = v.debeta_detail
    assert any(abs(v._debeta_class_details[1][u]["skill_shrunk_eb"] - flat[u]["skill_shrunk_eb"]) > 1e-9
               for u in v._debeta_class_details[1]), "the class pool is not the board's"


def test_the_forms_pay_nothing(monkeypatch):
    base = _validator(class_weights="books")
    scores = reward.compute_debeta_scores(base)
    monkeypatch.setattr(reward, "shrunk_skill_self", lambda a: 1e9)
    monkeypatch.setattr(reward, "shrunk_skill_eb", lambda pool: {u: -1e9 for u in pool})
    moved = _validator(class_weights="books")
    assert reward.compute_debeta_scores(moved) == scores
    assert moved._debeta_class_shares == base._debeta_class_shares

    def strip(d):
        return {u: {k: x for k, x in dd.items() if not k.startswith("skill_shrunk_")} for u, dd in d.items()}

    assert strip(moved.debeta_detail) == strip(base.debeta_detail)
    for c in base._debeta_class_details:
        assert strip(moved._debeta_class_details[c]) == strip(base._debeta_class_details[c])
    assert moved.debeta_detail[1]["skill_shrunk_self"] == 1e9


def test_the_carrier_and_the_report_publish_both_gauges():
    from tests.test_kappa_skip_under_debeta import _validator_data

    vd = _validator_data(debeta_scores={1: 0.8, 2: 0.2}, weight=0.25, kappa_values={1: None, 2: None})
    vd["debeta_detail"][1].update(skill_shrunk_self=1.25, skill_shrunk_eb=-0.5)
    reward.score_uid(vd, 1)
    assert vd["kappa_values"][1]["skill_shrunk_self"] == 1.25 and vd["kappa_values"][1]["skill_shrunk_eb"] == -0.5
    rsrc = inspect.getsource(report)
    for key in ("skill_shrunk_self", "skill_shrunk_eb"):
        assert f"('{key}', 'debeta_{key}')" in rsrc, f"the miner gauge debeta_{key}"
        assert f"'{key}': kappa_values.get('{key}') if kappa_values else None" in rsrc
        assert (key, key) in report.CLASS_DETAIL_GAUGES, f"the class gauge {key}"


def test_the_class_gauges_carry_both_forms():
    duck = SimpleNamespace(prometheus_agent_class_gauges="AC", simulation=None,
                           debeta_class_details={0: {5: {"skill_raw": 0.1, "skill_shrunk_self": 0.4, "skill_shrunk_eb": 0.2}}})
    updates = []
    report.publish_agent_class_gauges(duck, updates, "hot", 79, "sim-1")
    rows = {u[7]: u[1] for u in updates}
    assert rows["skill_shrunk_self"] == 0.4 and rows["skill_shrunk_eb"] == 0.2 and rows["skill"] == 0.1
