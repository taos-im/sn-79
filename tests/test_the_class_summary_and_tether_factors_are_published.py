# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator publishes what the two-class layout needs observed: a class_gauges family (weight, bar, books,
makers, skilled uids, credit and effective pool weight per asset class) and two more per-uid miner gauges, the
S3' tether factor and the maker's background share of fill volume, so the board monitor can read the class split,
the tether's bite and a griefed honest maker (high background share, low P11 factor) directly."""
import inspect


def test_the_class_summary_is_built_from_the_class_details():
    from taos.im.validator.reward import class_summary

    details = {
        0: {1: {"making_raw": 10.0, "skill_raw": 0.5}, 2: {"making_raw": 30.0, "skill_raw": 0.0}},
        1: {9: {"making_raw": 0.0, "skill_raw": 0.0}},
    }
    s = class_summary(details, [0.95, 0.05], {0: 20, 1: 4}, {0: 96, 1: 32})
    assert s[0]["weight"] == 0.95 and s[0]["bar"] == 20 and s[0]["n_books"] == 96
    assert s[0]["n_makers"] == 2 and s[0]["n_skilled"] == 1 and s[0]["making_credit"] == 40.0
    assert s[1]["n_makers"] == 0 and s[1]["making_credit"] == 0.0
    # a quiet class hands its weight back: the class with credit carries the whole half
    assert s[0]["making_weight_effective"] == 1.0 and s[1]["making_weight_effective"] == 0.0
    assert s[0]["skill_weight_effective"] == 1.0
    # a class whose skill values are all negative has no skill credit and carries no skill weight
    s2 = class_summary({0: {1: {"making_raw": 1.0, "skill_raw": 0.3}}, 1: {9: {"making_raw": 2.0, "skill_raw": -0.4}}}, [0.95, 0.05], {0: 20, 1: 4}, {0: 96, 1: 32})
    assert s2[1]["skill_credit"] == 0.0 and s2[1]["skill_weight_effective"] == 0.0 and s2[1]["making_weight_effective"] == 0.05


def test_the_scorer_records_the_summary_and_the_two_factors_for_the_report():
    from taos.im.validator import reward

    src = inspect.getsource(reward)
    assert "self._debeta_class_summary = class_summary(" in src
    assert "uid_kappa['s3_factor'] = _d.get('making_s3_factor') if _d else None" in src
    assert "uid_kappa['bg_share'] = _d.get('bg_share') if _d else None" in src


def test_the_report_publishes_the_class_gauges_and_the_two_miner_gauges():
    from taos.im.validator import report

    src = inspect.getsource(report)
    i = src.index("_SnapshotCollector('class_gauges'")
    assert "'asset_class'" in src[i:i + 400] and "'class_gauge_name'" in src[i:i + 400]
    assert "data.get('debeta_class_summary')" in src
    for g in ("debeta_s3_factor", "debeta_bg_share"):
        assert f"'{g}'" in src and f"{g}=" in src
    for k in ("('s3_factor', 'debeta_s3_factor')", "('bg_share', 'debeta_bg_share')"):
        assert k in src


def test_the_validator_hands_the_summary_to_the_report():
    from taos.im.neurons import validator

    src = inspect.getsource(validator)
    assert "'debeta_class_summary': getattr(self, '_debeta_class_summary', None) or {}" in src
