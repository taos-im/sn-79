# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The Kappa3 PnL factor reads --scoring.kappa.pnl.impact where the scoring config carries it, config['kappa']
['pnl_impact']; a config without the dial leaves every book neutral. Shipped pay is unchanged: the Kappa3 Score is
not computed at the shipped Kappa3 Weight of 0."""
from taos.im.validator.reward import _compute_pnl_factors

DAY = 86_400_000_000_000


def _factors(config):
    history = {7: {5 * DAY: {0: 50_000.0, 1: -50_000.0}}}
    return _compute_pnl_factors(
        uid=7, pnl_factors={}, normalized_kappas={0: 0.5, 1: 0.5}, config=config, lookback=DAY,
        simulation_config={"miner_wealth": 100_000.0}, realized_pnl_history=history, lookback_threshold=0)


def test_the_factor_follows_the_impact_the_config_carries():
    f = _factors({"kappa": {"pnl_impact": 1.0}})
    assert f[0] > 1.0, "a book with a realized gain is boosted at impact 1"
    assert f[0] == 1.5 and f[1] == 0.5


def test_a_config_without_the_dial_is_neutral():
    assert _factors({"kappa": {}}) == {0: 1.0, 1: 1.0}
    assert _factors({"kappa": {"pnl_impact": 0.0}}) == {0: 1.0, 1: 1.0}
