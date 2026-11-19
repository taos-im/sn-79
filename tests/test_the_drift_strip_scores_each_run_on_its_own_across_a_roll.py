# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""At a run change the de-beta histories are shifted onto the new clock so the window spans the boundary, and every
inventory starts flat; the windowed finalizer then stripped the whole window with one mean inventory and one drift, so
the old run's mean inventory met the new run's price move: a score term from no trade (a uid with no fill in the new run
could carry a large published net alpha from it).

The split finalizer strips each side of the seam on its own from the histories the window already keeps (the old
run's entries sit before the seam on the new clock, the new run's at or after it) and sums the two strips. The toy
from the card: hold +100 for 1,000 prints at a flat price, roll, stay flat for 100 prints while the price falls 2.0:
each run alone scores 0, the one-strip window +181.8, the split 0.
"""
import pathlib

from taos.im.validator import debeta

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEAM = 10_000


def _toy():
    """Histories of the card's toy on the new clock: the old run's 1,000 prints at ts 1..1000 (before the seam), the
    new run's 100 prints at ts SEAM..SEAM+99 with the price falling 0.02 a print."""
    mtm, invsum, invn, drift = {7: {0: {}}}, {7: {0: {}}}, {0: {}}, {0: {}}
    for ts in range(1, 1001):
        mtm[7][0][ts] = 0.0
        invsum[7][0][ts] = 100.0
        invn[0][ts] = 1
        drift[0][ts] = 0.0
    for i in range(100):
        ts = SEAM + i
        mtm[7][0][ts] = 0.0          # flat in the new run: no inventory, no mtm
        invsum[7][0][ts] = 0.0
        invn[0][ts] = 1
        drift[0][ts] = -0.02
    return mtm, invsum, invn, drift


def _sums(h2):
    return {u: {b: sum(d.values()) for b, d in bb.items()} for u, bb in h2.items()}


def test_one_strip_over_the_seam_pays_the_old_inventory_the_new_runs_move_and_the_split_does_not():
    mtm, invsum, invn, drift = _toy()
    one = debeta.book_alphas_by_book(_sums(mtm), _sums(invsum), {b: sum(d.values()) for b, d in invn.items()},
                                     {b: sum(d.values()) for b, d in drift.items()})
    assert abs(one[7][0] - 181.8) < 0.05, one
    split = debeta.book_alphas_split(mtm, invsum, invn, drift, SEAM)
    assert abs(split[7][0]) < 1e-9, split


def test_without_a_seam_inside_the_window_the_split_equals_the_one_strip():
    mtm, invsum, invn, drift = _toy()
    for h in (mtm[7][0], invsum[7][0], invn[0], drift[0]):
        for ts in [t for t in h if t < SEAM]:
            del h[ts]
    one = debeta.book_alphas_by_book(_sums(mtm), _sums(invsum), {b: sum(d.values()) for b, d in invn.items()},
                                     {b: sum(d.values()) for b, d in drift.items()})
    split = debeta.book_alphas_split(mtm, invsum, invn, drift, SEAM)
    assert abs(split[7][0] - one[7][0]) < 1e-9
    assert debeta.book_alphas_split(mtm, invsum, invn, drift, None) == split, "no seam recorded: the one strip"


def test_each_side_is_stripped_on_its_own_when_both_hold_inventory_and_moves():
    mtm, invsum, invn, drift = {7: {0: {}}}, {7: {0: {}}}, {0: {}}, {0: {}}
    # old run: hold 10 over four prints as the price climbs 1.0 a print: mtm 40, mean inv 10, drift 4 -> alpha 0
    for ts in range(1, 5):
        mtm[7][0][ts] = 10.0
        invsum[7][0][ts] = 10.0
        invn[0][ts] = 1
        drift[0][ts] = 1.0
    # new run: hold 2 over two prints as the price falls 3.0 a print, having bought ahead of a 1.0 rise: alpha 1.0
    mtm[7][0][SEAM] = -6.0 + 1.0
    invsum[7][0][SEAM] = 2.0
    invn[0][SEAM] = 1
    drift[0][SEAM] = -3.0
    mtm[7][0][SEAM + 1] = -6.0
    invsum[7][0][SEAM + 1] = 2.0
    invn[0][SEAM + 1] = 1
    drift[0][SEAM + 1] = -3.0
    split = debeta.book_alphas_split(mtm, invsum, invn, drift, SEAM)
    assert abs(split[7][0] - 1.0) < 1e-9, "0 from the old run plus (-11 - 2 x -6) = 1.0 from the new"
    only = debeta.book_alphas_split(mtm, invsum, invn, drift, SEAM, only={7: {1}})
    assert only == {7: {}}, "`only` keeps the pairs it names, as the one-strip finalizer does"


def test_the_held_variant_splits_the_same_way():
    mtm, heldinv, heldn, helddrift = {7: {0: {}}}, {7: {0: {}}}, {7: {0: {}}}, {7: {0: {}}}
    for ts in range(1, 1001):
        mtm[7][0][ts] = 0.0
        heldinv[7][0][ts] = 100.0
        heldn[7][0][ts] = 1
        helddrift[7][0][ts] = 0.0
    held = debeta.book_alphas_held_split(mtm, heldinv, heldn, helddrift, SEAM)
    assert abs(held[7][0]) < 1e-9
    one = debeta.book_alphas_held_by_book(_sums(mtm), _sums(heldinv), _sums(heldn), _sums(helddrift))
    assert abs(held[7][0] - one[7][0]) < 1e-9


def test_the_run_change_records_the_seam_and_the_state_carries_it():
    trade_src = (ROOT / "taos" / "im" / "validator" / "trade.py").read_text()
    assert "self.debeta_seam_ts = new_ts" in trade_src, "the shift records the boundary on the new clock"
    assert "seam_nudge(" in trade_src, "an old entry landing exactly on the seam is moved just before it"
    persistence_src = (ROOT / "taos" / "im" / "validator" / "persistence.py").read_text()
    assert persistence_src.count("debeta_seam_ts") >= 2, "saved and restored"
    reward_src = (ROOT / "taos" / "im" / "validator" / "reward.py").read_text()
    assert "book_alphas_split(" in reward_src and "'seam_split'" in reward_src
    cfg = (ROOT / "taos" / "im" / "config" / "__init__.py").read_text()
    i = cfg.index('"--scoring.debeta.seam_split"')
    assert "default=1" in cfg[i:i + 1400]


def test_an_old_entry_on_the_seam_is_nudged_before_it():
    from taos.im.validator.trade import seam_nudge
    h2 = {7: {0: {SEAM - 5: 1.0, SEAM: 2.0}}}
    h1 = {0: {SEAM: 3, SEAM - 1: 1}}
    seam_nudge([h2], [h1], SEAM)
    assert h2[7][0] == {SEAM - 5: 1.0, SEAM - 1: 2.0}
    assert h1[0] == {SEAM - 1: 4}
