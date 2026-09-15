# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Equal raw values must map to equal ranks, or uid order decides emissions.

On a live board (3-window retention, 251 uids), 222 miners share
making_raw == 0.0 exactly, and positional tie-breaking spread them across making_rank
0.000..0.884, worth w_make * 0.884 = 0.265 of combined score assigned by uid order alone.

The rule under test: ties share their block's MINIMUM rank. Zero making earns zero
relative making credit, and the ordering among non-makers is then decided by skill only.
"""
import pytest

from taos.im.validator.debeta import _rank01, debeta_scores, median_abs_floor


def test_equal_values_get_equal_ranks():
    r = _rank01([5.0, 1.0, 5.0, 3.0, 5.0])
    assert r[0] == r[2] == r[4]


def test_tie_block_takes_its_minimum_rank():
    # 0.0 x3 then 7.0, 9.0: zeros at positions 0-2 -> all rank 0.0; 7.0 at 3/4; 9.0 at 4/4.
    r = _rank01([0.0, 7.0, 0.0, 9.0, 0.0])
    assert r[0] == r[2] == r[4] == 0.0
    assert r[1] == pytest.approx(3 / 4)
    assert r[3] == pytest.approx(4 / 4)


def test_strictly_greater_value_gets_strictly_greater_rank():
    vals = [0.0, 0.0, 1.0, 2.0, 2.0, 5.0]
    r = _rank01(vals)
    for i in range(len(vals)):
        for j in range(len(vals)):
            if vals[i] > vals[j]:
                assert r[i] > r[j]


def test_all_equal_all_rank_zero():
    assert _rank01([4.2, 4.2, 4.2]) == [0.0, 0.0, 0.0]


def test_input_order_never_changes_a_score():
    """The lottery test: permuting uid insertion order must leave every score unchanged."""
    cb = {1: {0: 5.0}, 2: {0: 0.0}, 3: {0: 0.0}, 4: {0: 2.0}}
    cs = {1: {0: 5.0}, 2: {0: 3.0}, 3: {0: 4.0}, 4: {0: 2.0}}
    al = {1: [1.0, 1.2, 0.9, 1.1], 2: [0.0] * 4, 3: [0.0] * 4, 4: [2.0, 1.8, 2.1, 1.9]}
    floor = median_abs_floor(al)
    fwd = debeta_scores(cb, cs, al, floor=floor, w_make=0.30)
    rev = debeta_scores(
        dict(reversed(cb.items())), dict(reversed(cs.items())),
        dict(reversed(al.items())), floor=floor, w_make=0.30,
    )
    assert fwd == rev


def test_zero_makers_share_one_making_rank():
    """uids 2 and 3 both have zero balanced making; their making ranks must be identical."""
    cb = {1: {0: 5.0}, 2: {0: 0.0}, 3: {0: 4.0}}
    cs = {1: {0: 5.0}, 2: {0: 3.0}, 3: {0: 0.0}}
    al = {1: [1.0, 1.2, 0.9, 1.1], 2: [0.5] * 4, 3: [0.5] * 4}
    detail = {}
    debeta_scores(cb, cs, al, floor=median_abs_floor(al), w_make=0.30, detail=detail)
    assert detail[2]["making_raw"] == detail[3]["making_raw"] == 0.0
    assert detail[2]["making_rank"] == detail[3]["making_rank"] == 0.0
