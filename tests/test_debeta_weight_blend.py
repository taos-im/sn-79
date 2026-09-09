# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Trading is a FLAT weighted sum over three same-level weights, validated to sum to 1 at init:

    trading = kappa.weight * kappa + pnl.weight * pnl + debeta.weight * debeta

debeta.weight defaults to 0.0: deploying is rehearsal by default (legacy emissions, decomposition
always computed and published); (0, 0, 1) is the characterized full replacement, and a component
is computed iff its own weight is nonzero. An empty de-beta map renormalizes onto the legacy
components so warming never reads as a scale change.

Across the full 0.6.0 board through the real floor+Pareto pipeline, the transition is
front-loaded (weight 0.25 already reallocates 53% of the pot) but nobody is floor-zeroed until
weight ~0.5, so the dial exists for a stepped migration (0.1 -> 0.25 -> 0.5 -> 1.0) and for
instant rollback. The kappa-3 computation is skipped ONLY at weight >= 1.0: any blend needs the legacy score live.
"""
import pytest

import taos.im.validator.reward as reward
from tests.test_kappa_skip_under_debeta import _validator_data, _forbid_kappa


def _with_weight(vd, w):
    vd['config']['scoring']['debeta']['weight'] = w
    vd['config']['scoring']['kappa']['weight'] = 1.0 - w
    return vd


def test_weight_one_is_full_replace_and_skips_kappa(monkeypatch):
    _forbid_kappa(monkeypatch)
    vd = _with_weight(_validator_data(debeta_scores={1: 0.8, 2: 0.2}), 1.0)
    trading, _ = reward.score_uids(vd)
    assert trading == {1: 0.8, 2: 0.2}


def test_partial_weight_computes_kappa(monkeypatch):
    calls = []
    real = reward.kappa_3
    monkeypatch.setattr(reward, 'kappa_3', lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    vd = _with_weight(_validator_data(debeta_scores={1: 0.8, 2: 0.2}), 0.5)
    reward.score_uids(vd)
    assert calls, "a blend needs the legacy score: kappa-3 must run at weight < 1.0"


def test_blend_arithmetic(monkeypatch):
    monkeypatch.setattr(reward, 'calculate_kappa_score', lambda **kw: 0.8)
    vd = _with_weight(_validator_data(debeta_scores={1: 0.4, 2: 0.4}), 0.25)
    trading, _ = reward.score_uids(vd)
    # flat: 0.75*kappa + 0*pnl + 0.25*debeta
    assert trading[1] == pytest.approx(0.75 * 0.8 + 0.25 * 0.4)


def test_empty_map_renormalizes_scale_onto_legacy(monkeypatch):
    """Warming with weights (0.75, 0, 0.25) must NOT shrink everyone's score by 25%: the EMA
    standing would read the warmup as a skill collapse. The de-beta share renormalizes onto the
    configured legacy components."""
    monkeypatch.setattr(reward, 'calculate_kappa_score', lambda **kw: 0.8)
    vd = _with_weight(_validator_data(debeta_scores={}), 0.25)
    trading, _ = reward.score_uids(vd)
    assert trading[1] == pytest.approx(0.8)
    # warming must still publish the dial: the live blend invariant reconstructs the
    # renormalized composition from it
    assert vd['kappa_values'][1].get('debeta_weight') == pytest.approx(0.25)


def test_weight_zero_keeps_legacy_score_but_reports_the_decomposition(monkeypatch):
    """weight 0 is the rehearsal mode: emissions stay legacy while dashboards already show the
    de-beta legs, so miners can see what the new score WOULD pay before it carries weight."""
    monkeypatch.setattr(reward, 'calculate_kappa_score', lambda **kw: 0.8)
    vd = _with_weight(_validator_data(debeta_scores={1: 0.4, 2: 0.4}), 0.0)
    trading, _ = reward.score_uids(vd)
    assert trading[1] == pytest.approx(0.8)
    assert vd['kappa_values'][1].get('debeta_score') == pytest.approx(0.4)
    assert vd['kappa_values'][1].get('debeta_weight') == pytest.approx(0.0)


def test_weight_clamped_to_unit_interval(monkeypatch):
    _forbid_kappa(monkeypatch)
    vd = _with_weight(_validator_data(debeta_scores={1: 0.8, 2: 0.2}), 1.5)
    trading, _ = reward.score_uids(vd)
    assert trading == {1: 0.8, 2: 0.2}


def test_config_dict_carries_weight():
    """build_scoring_config must ship the weight or score_uids silently full-replaces."""
    import inspect
    src = inspect.getsource(reward.build_scoring_config)
    assert "'weight'" in src
