# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Trading is a FLAT weighted sum (kappa.weight + pnl.weight + debeta.weight = 1, validated at
init). A component is computed iff its own weight is nonzero; kappa at weight 0 must skip its
10s+ batch WITHOUT breaking reporting: kappa_values doubles as the per-uid reporting carrier, so
skipped uids need explicit stub entries, not absence and not stale last-cycle values.
"""
import pytest

import taos.im.validator.reward as reward


def _validator_data(*, debeta_scores, weight=None, kappa_values=None, parallel_workers=0):
    uids = [1, 2]
    return {
        'config': {'scoring': {
            'kappa': {'tau': 0.0, 'lookback': 10_000, 'normalization_min': -1.0,
                      'normalization_max': 1.0, 'min_lookback': 1, 'min_realized_observations': 1,
                      'parallel_workers': parallel_workers,
                      'weight': 1.0 - (weight or 0.0), 'reward_cores': [0],
                      'normalization': {}},
            'pnl': {'weight': 0.0},
            'gentrx': {},
            'debeta': ({'weight': weight} if weight is not None else {}),
            'interval': 1_000,
            'max_inactive_books_ratio': 1.0,
        }},
        'uids': uids,
        'deregistered_uids': [],
        'simulation_config': {'grace_period': 0, 'book_count': 2, 'miner_wealth': 1.0,
                              'book_ids': [0, 1], 'publish_interval': 1_000},
        'kappa_values': dict(kappa_values or {}),
        'kappa_cache': {},
        'realized_pnl_history': {u: {} for u in uids},
        'activity_factors': {},
        'pnl_factors': {},
        'roundtrip_volumes': {},
        'simulation_timestamp': 100_000,
        'debeta_scores': dict(debeta_scores),
        'debeta_detail': {u: {'making_raw': 0.0, 'making_rank': 0.0, 'skill_raw': 0.0,
                              'skill_rank': 0.0, 'p11_factor': 1.0} for u in uids},
    }


def _forbid_kappa(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("kappa computation ran while de-beta replace was active")
    monkeypatch.setattr(reward, 'kappa_3', _boom)
    monkeypatch.setattr(reward, 'batch_kappa_3', _boom)


def test_kappa_skipped_when_debeta_replaces(monkeypatch):
    _forbid_kappa(monkeypatch)
    vd = _validator_data(debeta_scores={1: 0.8, 2: 0.2}, weight=1.0)
    trading, _ = reward.score_uids(vd)
    assert trading == {1: 0.8, 2: 0.2}


def test_skipped_uids_get_truthy_stub_entries(monkeypatch):
    """Reporting keys off `if kappa_values.get(uid)`; without a stub the de-beta decomposition
    never reaches the dashboards for exactly the miners de-beta scores."""
    _forbid_kappa(monkeypatch)
    vd = _validator_data(debeta_scores={1: 0.8, 2: 0.2}, weight=1.0)
    reward.score_uids(vd)
    for uid in (1, 2):
        entry = vd['kappa_values'][uid]
        assert entry and entry.get('skipped') is True
        assert entry.get('books') == {}
        assert entry.get('debeta_score') == pytest.approx(vd['debeta_scores'][uid])
        assert entry.get('making_rank') is not None


def test_stale_entries_are_replaced_not_reported(monkeypatch):
    """A skip cycle must not leave last-cycle kappa numbers where dashboards read fresh ones."""
    _forbid_kappa(monkeypatch)
    stale = {1: {'books': {0: 1.7}, 'kappa': 0.9}, 2: {'books': {1: -0.2}}}
    vd = _validator_data(debeta_scores={1: 0.5, 2: 0.5}, weight=1.0, kappa_values=stale)
    reward.score_uids(vd)
    assert vd['kappa_values'][1].get('skipped') is True
    assert vd['kappa_values'][1].get('books') == {}


def test_empty_map_on_a_blend_rung_computes_kappa():
    """Warming/exception (empty map) on a blend rung, e.g. weights (0.5, 0, 0.5): the legacy
    components carry nonzero weight, so kappa must compute and the score renormalizes onto it."""
    vd = _validator_data(debeta_scores={}, weight=0.5)
    trading, _ = reward.score_uids(vd)
    assert set(trading) == {1, 2}
    for uid in (1, 2):
        assert not (vd['kappa_values'].get(uid) or {}).get('skipped')


def test_empty_map_at_full_replacement_scores_zero_without_resurrecting_kappa(monkeypatch):
    """(0, 0, 1) while the de-beta map is warming: there is no legacy to fall back to (its weights
    are zero by validation), so scores are honestly 0 and the kappa batch stays skipped rather
    than being resurrected for a component the operator explicitly zeroed."""
    _forbid_kappa(monkeypatch)
    vd = _validator_data(debeta_scores={}, weight=1.0)
    trading, _ = reward.score_uids(vd)
    for uid in (1, 2):
        assert trading[uid] == 0.0
        assert (vd['kappa_values'].get(uid) or {}).get('skipped') is True


def test_default_weight_is_zero_legacy_scores_with_decomposition():
    """No weight configured = the old 'disabled' emissions, but the decomposition still publishes:
    there is no boolean any more, and weight defaults to 0.0 (deploy-safe rehearsal)."""
    vd = _validator_data(debeta_scores={1: 0.9, 2: 0.9})
    trading, _ = reward.score_uids(vd)
    for uid in (1, 2):
        assert trading[uid] != pytest.approx(0.9)
        assert not (vd['kappa_values'].get(uid) or {}).get('skipped')
        assert vd['kappa_values'][uid].get('debeta_score') == pytest.approx(0.9)


def test_kappa_weight_zero_skips_the_kappa_batch(monkeypatch):
    """kappa.weight=0 (e.g. weights (0, 0.5, 0.5)) makes the 10s+ batch dead computation in every
    path; it must be skipped with the same stub-entry reporting the full-replacement rung uses."""
    _forbid_kappa(monkeypatch)
    vd = _validator_data(debeta_scores={1: 0.4, 2: 0.4}, weight=0.5)
    vd['config']['scoring']['kappa']['weight'] = 0.0
    vd['config']['scoring']['pnl']['weight'] = 0.5
    trading, _ = reward.score_uids(vd)
    # trading = 0*kappa + 0.5*pnl(=0 here) + 0.5*0.4
    assert trading[1] == pytest.approx(0.5 * 0.4)
    assert vd['kappa_values'][1].get('skipped') is True
    assert vd['kappa_values'][1].get('debeta_score') == pytest.approx(0.4)
