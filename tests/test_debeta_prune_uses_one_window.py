# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""De-beta's two legs must be pruned on ONE retention window.

Making inputs (capture_buy_sums, capture_sell_sums, debeta_cp) were pruned on
`scoring.activity.trade_volume_assessment_period` (24h) while skill inputs (mtm, invsum, invn,
drift) used `scoring.kappa.lookback` (3h) -- each de-beta leg inheriting the window of the legacy
leg it replaced. Exactly 8x apart, and de-beta replaces both legs, so the split was an artifact of
implementation rather than a design choice.

Across all 23 windows of a full board, averaged over every sliding position, with
the shipped per-book making rule: making alignment against a size-free two-sidedness reference
(two-sided book fraction x per-book balance) falls MONOTONICALLY as the window lengthens, from
+0.175 at 2h to +0.067 at 24h. The leg whose entire purpose is two-sidedness was pruned on the
window that detects it least well, because over a long window a directional miner accumulates
offsetting flow across time and presents as balanced. Moving making to the 3h window improves that
alignment 2.6x and the inventory-exposure gate slightly (+0.018), and costs only stability, which
was already 0.820 in practice because skill carries 1-w_make = 0.70 of the weight on the 3h window.

This pins the wiring, not the value: every de-beta prune must read the SAME threshold, so the
two-window structure cannot be reintroduced by editing one line. The legacy paths (trade volumes,
roundtrip volumes, _volume_seen_tids) keep the activity window and are deliberately untouched.
"""
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "taos" / "im" / "validator" / "trade.py"
_DEBETA_STRUCTS = (
    "debeta_capbuy_hist",
    "debeta_capsell_hist",
    "debeta_cp_hist",
    "debeta_mtm_hist",
    "debeta_invsum_hist",
    "debeta_invn_hist",
    "debeta_drift_hist",
    "debeta_heldn_hist",
    "debeta_heldinv_hist",
    "debeta_helddrift_hist",
    "debeta_notional_hist",
)


def _prune_calls():
    """Every prune_hist_* call, as (struct, threshold_expression)."""
    src = _SRC.read_text()
    out = []
    for m in re.finditer(r"prune_hist_\dlevel\(\s*([^,]+?),\s*([^,]+?),\s*([^)]+?)\)", src, re.S):
        out.append((m.group(1).strip(), m.group(3).strip()))
    return out


def _shift_calls():
    """Every shift_hist_* call, as (struct, threshold_expression).

    The simulation-boundary shift carried the same making/skill split as the live prune, and a
    guard that inspected only the prune left the boundary free to reintroduce the 24h window on
    making. Both paths are pinned.
    """
    src = _SRC.read_text()
    out = []
    for m in re.finditer(
        r"shift_hist_\dlevel\(\s*([^,]+?),\s*([^,]+?),\s*([^,]+?),\s*([^,]+?),\s*([^)]+?)\)", src, re.S
    ):
        out.append((m.group(1).strip(), m.group(5).strip()))
    return out


def test_the_boundary_shift_uses_one_window_for_both_legs():
    calls = _shift_calls()
    assert calls, f"no shift_hist_* calls found in {_SRC}"
    thresholds = {t for struct, t in calls if any(s in struct for s in _DEBETA_STRUCTS)}
    assert len(thresholds) == 1, (
        f"the simulation-boundary shift splits de-beta across {sorted(thresholds)}; it must match "
        "the live prune and use one window for both legs"
    )


def test_every_debeta_prune_uses_the_same_threshold():
    calls = _prune_calls()
    assert calls, f"no prune_hist_* calls found in {_SRC}"
    thresholds = {t for struct, t in calls if any(s in struct for s in _DEBETA_STRUCTS)}
    assert len(thresholds) == 1, (
        f"de-beta legs are pruned on {len(thresholds)} different windows {sorted(thresholds)}; "
        "de-beta replaces kappa+PnL and must use one retention window for both legs"
    )


def test_no_debeta_prune_reads_the_legacy_activity_window():
    offenders = [
        struct
        for struct, t in _prune_calls()
        if any(s in struct for s in _DEBETA_STRUCTS) and "volume_prune_threshold" in t
    ]
    assert not offenders, (
        f"{offenders} pruned on the legacy activity window (24h). Making alignment is worst at that "
        "window; de-beta must prune on the kappa lookback."
    )


def test_the_debeta_threshold_is_the_kappa_lookback():
    src = _SRC.read_text()
    thresholds = {t for struct, t in _prune_calls() if any(s in struct for s in _DEBETA_STRUCTS)}
    name = thresholds.pop()
    assign = re.search(rf"{re.escape(name)}\s*=\s*(.+)", src)
    assert assign, f"could not find where {name} is assigned"
    assert "scoring.kappa.lookback" in assign.group(1), (
        f"{name} is {assign.group(1).strip()!r}; expected it derived from scoring.kappa.lookback"
    )


def test_legacy_volume_paths_keep_the_activity_window():
    """The legacy trade/roundtrip volume retention is NOT part of this change."""
    src = _SRC.read_text()
    assert "volume_prune_threshold = timestamp - self.config.scoring.activity.trade_volume_assessment_period" in src
    assert "_volume_seen_tids" in src and "volume_prune_threshold}" in src
