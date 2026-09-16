# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A book's activity factor starts at exactly 0.0 and becomes 1.0 (or more) the first time one of its
round trips is seen inside the latest sampling bucket at a scoring tick. A book that carries a kappa has
realized round trips inside the lookback, so it is active by definition; the 0.0 marker can only
survive on such a book when scoring paused across a bucket, and left in place it weights a real kappa
to the worst possible value on every book. The weighting step seeds that marker at neutral and
persists it. A decayed factor (tiny but nonzero) is a measurement and is left alone."""
import pytest

import taos.im.validator.reward as reward


def _inputs(activity):
    uid = 7
    config = {
        'kappa': {'normalization_min': -1.0, 'normalization_max': 1.0, 'lookback': 10_000, 'pnl': {}},
        'activity': {'capital_turnover_cap': 1.0, 'trade_volume_sampling_interval': 1_000,
                     'decay_grace_period': 100, 'impact': 0.0, 'decay_rate': 0.0},
        'interval': 1_000,
        'max_inactive_books_ratio': 1.0,
    }
    simulation_config = {'miner_wealth': 1.0, 'volumeDecimals': 4, 'book_count': 3, 'book_ids': [0, 1, 2]}
    kappa_values = {uid: {'books': {0: 0.2, 1: -0.5, 2: None}}}
    activity_factors = {uid: dict(activity)}
    pnl_factors = {uid: {0: 1.0, 1: 1.0, 2: 1.0}}
    return uid, kappa_values, activity_factors, pnl_factors, config, simulation_config


def test_a_book_with_a_kappa_and_the_never_observed_marker_is_scored_at_neutral_weight():
    uid, kv, af, pf, cfg, sim = _inputs({0: 0.0, 1: 0.0, 2: 0.0})
    reward.calculate_kappa_score(uid, kv, af, pf, {}, {uid: {}}, cfg, sim, 100_000)
    weighted = kv[uid]['books_weighted']
    assert weighted[0] == pytest.approx(0.6)     # normalized 0.2 on [-1, 1], factor seeded to 1
    assert weighted[1] == pytest.approx(0.25)    # normalized -0.5, likewise
    assert weighted[2] is None                   # no kappa: nothing to seed, nothing to weight
    assert af[uid] == {0: 1.0, 1: 1.0, 2: 0.0}   # the seed persists; the kappa-less book keeps its marker


def test_a_decayed_factor_is_a_measurement_and_is_not_seeded():
    uid, kv, af, pf, cfg, sim = _inputs({0: 1e-6, 1: 0.0, 2: 0.0})
    reward.calculate_kappa_score(uid, kv, af, pf, {}, {uid: {}}, cfg, sim, 100_000)
    weighted = kv[uid]['books_weighted']
    assert weighted[0] == pytest.approx(0.6e-6)
    assert af[uid][0] == pytest.approx(1e-6)
    assert weighted[1] == pytest.approx(0.25) and af[uid][1] == 1.0
