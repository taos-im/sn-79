# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A simulation start shifts the alternative mark's and the maker-liquidity legs before any update has run.

The validator loads those legs' histories from its saved state, but their running accumulators were created only
inside the trade update, so a start or resume that reached the validator before its first update after a load raised
AttributeError in the history shift and the start was never handled. The shift now creates whatever of that state
is missing before it shifts.
"""
import pytest

import test_scoring_shadow as tss
from taos.im.validator import trade

ALT_ACC = ("debeta_mtm_alt", "debeta_invsum_alt", "debeta_inv_alt", "debeta_invn_alt", "debeta_drift_alt",
           "debeta_pfirst_alt", "debeta_plast_alt", "debeta_mark_state_alt")
LIQ_ACC = ("debeta_maker_buy_notional", "debeta_maker_sell_notional", "debeta_at_best", "debeta_best_size")


def _loaded_validator():
    """A validator as a load leaves it: the legs' histories present, their accumulators not yet created."""
    v = tss._fresh_main()
    for st in tss._ROUNDS:
        tss.update_trade_volumes(v, st)
    for name in ALT_ACC + LIQ_ACC:
        if hasattr(v, name):
            delattr(v, name)
    for name in ("debeta_invn_alt_hist", "debeta_drift_alt_hist"):
        if hasattr(v, name):
            delattr(v, name)
    assert hasattr(v, "debeta_mtm_alt_hist") or hasattr(v, "debeta_at_best_hist")
    return v


def _shift(v):
    trade.shift_simulation_histories(
        v, old_ts=4 * tss._S, new_ts=0, book_count=tss._BOOKS, volume_decimals=4, lookback=10800 * tss._S,
        volume_assessment_period=3600 * tss._S, miner_wealth=50000.0, effective_max_uids=tss._UIDS)


def test_a_start_before_the_first_update_shifts_without_raising():
    v = _loaded_validator()
    _shift(v)
    for name in ("debeta_mtm_alt", "debeta_invsum_alt", "debeta_maker_buy_notional", "debeta_at_best"):
        assert hasattr(v, name), name


def test_the_next_update_after_such_a_start_still_runs():
    v = _loaded_validator()
    _shift(v)
    tss.update_trade_volumes(v, tss._state(1 * tss._S, [(1, 2, 0, 2.0, 100.0, 0)]))


@pytest.mark.parametrize("init", ["init_alt_mark_state", "init_liquidity_state"])
def test_the_initialisers_leave_existing_state_alone(init):
    v = tss._fresh_main()
    for st in tss._ROUNDS:
        tss.update_trade_volumes(v, st)
    before = {n: getattr(v, n) for n in ALT_ACC + LIQ_ACC if hasattr(v, n)}
    getattr(trade, init)(v)
    assert all(getattr(v, n) is obj for n, obj in before.items())
