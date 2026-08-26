# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Tests for the soft score floor (concentration lever).

`apply_reward_floor` tapers below-percentile trading scores toward zero before the
Pareto allocation, so a fleet of merely-adequate UIDs stops earning while genuine
performers are preserved. It must be a strict no-op when disabled (default), a smooth
taper (not a cliff) when on, and ownership-agnostic (acts on the score vector only).
"""
import numpy as np

from taos.im.validator.reward import apply_reward_floor

OFF = {"rewarding": {"floor": {"enabled": False}}}


def _cfg(percentile=50.0, softness=0.5):
    return {"rewarding": {"floor": {"enabled": True, "percentile": percentile, "softness": softness}}}


def test_disabled_is_exact_noop():
    r = [0.1, 0.5, 0.9, 0.0, 0.3]
    assert apply_reward_floor(r, OFF) is r
    assert apply_reward_floor(r, {}) is r          # missing config -> no-op


def test_below_median_tapered_above_preserved():
    # 10 active scores 0.1..1.0; median threshold = 0.55
    r = [0.1 * i for i in range(1, 11)]
    out = apply_reward_floor(r, _cfg(percentile=50.0, softness=0.5))
    # top score unchanged (>= threshold)
    assert out[-1] == r[-1]
    # bottom scores fully floored to 0 (below thr*(1-softness) = 0.275)
    assert out[0] == 0.0 and out[1] == 0.0
    # total mass strictly reduced (bottom half attenuated)
    assert sum(out) < sum(r)


def test_soft_is_not_a_cliff():
    # a score in the taper band is partially attenuated, not zeroed or untouched
    r = [0.1 * i for i in range(1, 11)]              # thr=0.55, lo=0.275 at softness .5
    out = apply_reward_floor(r, _cfg(percentile=50.0, softness=0.5))
    band = out[3]                                     # r=0.4, between lo and thr
    assert 0.0 < band < 0.4


def test_softer_taper_keeps_more_than_sharper():
    r = [0.1 * i for i in range(1, 11)]
    soft = sum(apply_reward_floor(r, _cfg(softness=1.0)))
    sharp = sum(apply_reward_floor(r, _cfg(softness=0.1)))
    assert soft > sharp                               # gentler taper attenuates less


def test_concentration_effect_on_farm_like_vector():
    # 60 genuine miners above the median + a 40-UID mediocre fleet below it. The taper acts on
    # position relative to the median, so a vector like this is the one that exercises it.
    honest = list(np.linspace(0.5, 1.0, 60))
    fleet = [0.2] * 40                                # below thr*(1-softness) -> floored
    r = honest + fleet
    out = apply_reward_floor(r, _cfg(percentile=50.0, softness=0.5))
    fleet_after = sum(out[60:])
    top_after = sum(out[:60][-8:])                    # genuine top 8 untouched
    assert fleet_after == 0.0                          # fleet mass gutted to zero
    assert top_after == sum(r[:60][-8:])


def test_edge_cases_noop():
    assert apply_reward_floor([1.0], _cfg()) == [1.0]          # <2 active
    assert apply_reward_floor([0.0, 0.0, 0.0], _cfg()) == [0.0, 0.0, 0.0]
    # zeros stay zero, never pushed negative
    out = apply_reward_floor([0.0, 0.2, 0.4, 0.6, 0.8], _cfg())
    assert all(v >= 0.0 for v in out)
