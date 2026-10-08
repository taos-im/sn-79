# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The published book carries a fixed number of levels per side; the miner's reconstruction carries every level the
events built. When the published side is full, a reconstructed level deeper than its worst published level is not a
discrepancy: the publisher simply did not print it (on the 7 October capture 1740 of the 'other' mismatches were
levels entering or leaving the published window). When the published side is short of the depth, every level is
printed and a deeper reconstructed level is a real disagreement."""
from types import SimpleNamespace

from taos.im.protocol.book import L2Snapshot, LevelInfo

CFG = SimpleNamespace(volumeDecimals=4)


def _snap(bids, asks):
    return L2Snapshot(timestamp=0,
                      bids={p: LevelInfo.model_construct(price=p, quantity=q, orders=None) for p, q in bids.items()},
                      asks={p: LevelInfo.model_construct(price=p, quantity=q, orders=None) for p, q in asks.items()})


def test_a_deeper_reconstructed_level_is_not_a_discrepancy_when_the_published_side_is_full():
    recon = _snap({100.0: 1.0, 99.0: 1.0, 98.0: 1.0, 97.0: 5.0}, {101.0: 1.0, 102.0: 1.0, 103.0: 1.0, 104.0: 7.0})
    target = _snap({100.0: 1.0, 99.0: 1.0, 98.0: 1.0}, {101.0: 1.0, 102.0: 1.0, 103.0: 1.0})
    matched, discrepancies, existing = recon.compare(target, CFG, depth=3)
    assert matched, discrepancies
    assert existing == {"bid": {}, "ask": {}}


def test_the_same_level_is_a_discrepancy_when_the_published_side_is_short_of_the_depth():
    recon = _snap({100.0: 1.0, 99.0: 1.0, 97.0: 5.0}, {})
    target = _snap({100.0: 1.0, 99.0: 1.0}, {})
    matched, discrepancies, _ = recon.compare(target, CFG, depth=3)
    assert not matched and len(discrepancies) == 1 and "97.0" in discrepancies[0]


def test_without_a_depth_the_comparison_is_as_before():
    recon = _snap({100.0: 1.0, 97.0: 5.0}, {})
    target = _snap({100.0: 1.0}, {})
    matched, discrepancies, _ = recon.compare(target, CFG)
    assert not matched and len(discrepancies) == 1


def test_a_level_inside_the_published_window_still_counts():
    recon = _snap({100.0: 1.0, 99.0: 2.0, 98.0: 1.0}, {})
    target = _snap({100.0: 1.0, 99.0: 1.0, 98.0: 1.0}, {})
    matched, discrepancies, _ = recon.compare(target, CFG, depth=3)
    assert not matched and "99.0" in discrepancies[0]
