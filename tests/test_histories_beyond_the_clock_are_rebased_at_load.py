# SPDX-License-Identifier: MIT
"""A validator that restarts after the engine has already opened a new run never receives the start
event that rebases its histories, so the previous run's last window stays in every structure with
timestamps ahead of the new clock. Those entries sort after everything the new run writes, never leave
a window, and are counted by every windowed sum. The load-time guard treats any stamp ahead of the
clock as a missed run change: it moves those entries onto the new time base and rebuilds the running
sums, and leaves the current run's entries exactly as they were.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taos.im.validator.trade import rebase_entries_beyond_clock  # noqa: E402

H = 3600_000_000_000
OLD_END = 24 * H          # the previous run ended on its own clock at 24 h
NOW = 4 * H               # the new run's clock when the state is loaded
LOOKBACK = 3 * H


class _V:
    pass


def _state():
    v = _V()
    v.debeta_mtm_hist = {1: {2: {23 * H: 100.0, OLD_END: 50.0, H: 7.0, 3 * H: 9.0}}}
    v.debeta_invsum_hist = {1: {2: {OLD_END: 1000.0, 3 * H: 12.0}}}
    v.debeta_capbuy_hist = {1: {2: {OLD_END - H // 2: 1.5, 2 * H: 0.5}}}
    v.debeta_capsell_hist = {1: {2: {2 * H: 0.25}}}
    v.debeta_cp_hist = {1: {2: {OLD_END: 3.0, 3 * H: 1.0}}}
    v.debeta_heldn_hist = {1: {2: {3 * H: 2.0}}}
    v.debeta_heldinv_hist = {1: {2: {3 * H: 4.0}}}
    v.debeta_helddrift_hist = {1: {2: {3 * H: 0.5}}}
    v.debeta_notional_hist = {1: {2: {3 * H: 300.0}}}
    v.debeta_invn_hist = {2: {OLD_END: 5.0, 3 * H: 1.0}}
    v.debeta_drift_hist = {2: {OLD_END: -2.0, 3 * H: 0.1}}
    v.realized_pnl_history = {1: {OLD_END - H: {2: 11.0}, 3 * H: {2: 1.0}}}
    v.inventory_history = {1: {H // 10: {2: 0.0}, OLD_END: {2: 640.0}, 3 * H: {2: -20.0}}}
    v.trade_volumes = {1: {2: {"total": {OLD_END: 9.0, 3 * H: 1.0}, "maker": {OLD_END: 9.0}, "taker": {3 * H: 1.0}, "self": {}}}}
    v.roundtrip_volumes = {1: {2: {OLD_END: 4.0, 3 * H: 2.0}}}
    dd = lambda: collections.defaultdict(lambda: collections.defaultdict(float))  # noqa: E731
    for name in ("debeta_mtm", "debeta_invsum", "capture_buy_sums", "capture_sell_sums", "debeta_cp",
                 "debeta_heldn", "debeta_heldinv", "debeta_helddrift", "debeta_notional"):
        setattr(v, name, dd())
    v.debeta_invn = collections.defaultdict(float)
    v.debeta_drift = collections.defaultdict(float)
    v.debeta_inv = collections.defaultdict(lambda: collections.defaultdict(float), {2: collections.defaultdict(float, {1: 12.5})})
    v.debeta_plast = {2: 300.0}
    v.debeta_pfirst = {}
    v.debeta_mark_state = {}
    return v


def _stamps(d):
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, val in x.items():
                if isinstance(k, int) and k > 10**11:
                    out.append(k)
                walk(val)
    walk(d)
    return out


def test_nothing_stays_ahead_of_the_clock_and_the_current_run_is_untouched():
    v = _state()
    moved = rebase_entries_beyond_clock(v, NOW, lookback=LOOKBACK)
    assert moved > 0
    for name in ("debeta_mtm_hist", "debeta_invsum_hist", "debeta_capbuy_hist", "debeta_cp_hist", "debeta_invn_hist",
                 "debeta_drift_hist", "realized_pnl_history", "inventory_history", "trade_volumes", "roundtrip_volumes"):
        assert all(ts <= NOW for ts in _stamps(getattr(v, name))), name
    assert v.debeta_mtm_hist[1][2][H] == 7.0 and v.debeta_mtm_hist[1][2][3 * H] == 9.0
    assert v.inventory_history[1][3 * H] == {2: -20.0}
    assert v.trade_volumes[1][2]["total"][3 * H] == 1.0


def test_the_previous_run_lands_before_the_new_run_started():
    v = _state()
    rebase_entries_beyond_clock(v, NOW, lookback=LOOKBACK)
    # the previous run's end maps onto the new run's start, so its last window sits at or below zero;
    # the inventory history, measured from the run's own start, drops it instead
    assert set(v.inventory_history[1]) == {H // 10, 3 * H}
    assert (OLD_END - H) - OLD_END in v.realized_pnl_history[1]
    assert 0 in v.trade_volumes[1][2]["maker"] and v.trade_volumes[1][2]["maker"][0] == 9.0


def test_debeta_windows_drop_what_the_new_clock_left_behind_and_sums_follow():
    v = _state()
    rebase_entries_beyond_clock(v, NOW, lookback=LOOKBACK)
    # at 4 h with a 3 h window, entries at or below zero are outside the window and gone; the entry
    # on the window's edge is kept, as the update path keeps it
    assert set(v.debeta_mtm_hist[1][2]) == {H, 3 * H}
    assert set(v.debeta_invn_hist[2]) == {3 * H} and set(v.debeta_drift_hist[2]) == {3 * H}
    assert v.debeta_mtm[1][2] == 16.0
    assert v.debeta_invsum[1][2] == 12.0
    assert v.capture_buy_sums[1][2] == 0.5 and v.capture_sell_sums[1][2] == 0.25
    assert v.debeta_invn[2] == 1.0 and v.debeta_drift[2] == 0.1


def test_a_state_already_on_the_clock_is_left_alone():
    v = _state()
    for name in ("debeta_mtm_hist", "debeta_invsum_hist", "debeta_capbuy_hist", "debeta_cp_hist", "debeta_invn_hist",
                 "debeta_drift_hist", "realized_pnl_history", "inventory_history", "roundtrip_volumes"):
        d = getattr(v, name)

        def strip(x):
            if isinstance(x, dict):
                for k in [k for k in x if isinstance(k, int) and k > NOW]:
                    del x[k]
                for val in x.values():
                    strip(val)
        strip(d)
    strip(v.trade_volumes)
    before = repr(v.inventory_history)
    assert rebase_entries_beyond_clock(v, NOW, lookback=LOOKBACK) == 0
    assert repr(v.inventory_history) == before


def test_the_carried_positions_are_subtracted_when_the_previous_run_end_is_supplied():
    v = _state()
    # 12.5 held now = 10 carried from the previous run + 2.5 built since this run opened
    rebase_entries_beyond_clock(v, NOW, lookback=LOOKBACK, inventory_correction={"2": {"1": 10.0}})
    assert abs(v.debeta_inv[2][1] - 2.5) < 1e-12
    assert v.debeta_plast == {2: 300.0}


def test_without_a_correction_the_reconstruction_state_is_reset_as_the_run_change_would():
    v = _state()
    rebase_entries_beyond_clock(v, NOW, lookback=LOOKBACK)
    assert dict(v.debeta_inv) == {} and v.debeta_plast == {} and v.debeta_pfirst == {} and v.debeta_mark_state == {}


def test_a_state_on_the_clock_keeps_its_positions():
    v = _state()
    for name in ("debeta_mtm_hist", "debeta_invsum_hist", "debeta_capbuy_hist", "debeta_cp_hist", "debeta_invn_hist",
                 "debeta_drift_hist", "realized_pnl_history", "inventory_history", "roundtrip_volumes", "trade_volumes"):
        d = getattr(v, name)

        def strip(x):
            if isinstance(x, dict):
                for k in [k for k in x if isinstance(k, int) and k > NOW]:
                    del x[k]
                for val in x.values():
                    strip(val)
        strip(d)
    rebase_entries_beyond_clock(v, NOW, lookback=LOOKBACK)
    assert v.debeta_inv[2][1] == 12.5 and v.debeta_plast == {2: 300.0}


def test_the_correction_is_gated_on_the_hotkey_that_held_the_slot():
    v = _state()
    v.hotkeys = ["hk-0", "hk-a", "hk-b-new"]     # uid 1 still held by hk-a; uid 2 changed hands
    v.debeta_inv[2][2] = 3.0
    rebase_entries_beyond_clock(v, NOW, lookback=LOOKBACK,
                                inventory_correction={"positions": {"2": {"1": 10.0, "2": 3.0}}, "hotkeys": {"1": "hk-a", "2": "hk-b-old"}})
    assert abs(v.debeta_inv[2][1] - 2.5) < 1e-12      # slot 1 still held by hk-a: corrected
    assert v.debeta_inv[2][2] == 3.0                   # slot 2 changed hands: left alone


def test_a_uid_outside_the_metagraph_on_both_sides_is_corrected():
    v = _state()
    v.hotkeys = ["hk-0"]
    v.debeta_inv[2][256] = 4.0
    rebase_entries_beyond_clock(v, NOW, lookback=LOOKBACK,
                                inventory_correction={"positions": {"2": {"256": 1.5}}, "hotkeys": {"0": "hk-0"}})
    assert abs(v.debeta_inv[2][256] - 2.5) < 1e-12
