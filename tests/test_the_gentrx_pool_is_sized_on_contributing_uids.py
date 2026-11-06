# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The training pool is sized and split on the uids that contribute to it.

A uid sizes and shares the training pool only while its GenTRX EMA holds at least GENTRX_EMA_FLOOR; a uid under the
floor neither counts toward the pool's size nor takes a share of it."""
import torch

from taos.common.neurons.validator import GENTRX_EMA_FLOOR, compute_two_pool_allocation

SHARE, TARGET = 0.05, 20


def _alloc(gentrx):
    w, summary = compute_two_pool_allocation(torch.zeros(gentrx.numel()), gentrx, burn_ratio=0.0,
                                             gentrx_simulation_share=SHARE, n_target=TARGET)
    return w, summary


def test_one_contributor_beside_lapsed_uids_sizes_the_pool_alone():
    scores = torch.zeros(TARGET)
    scores[0] = 0.5
    scores[1:11] = 1e-43
    w, summary = _alloc(scores)
    assert summary["n_active"] == 1
    assert abs(summary["gentrx_alloc"] - SHARE / TARGET) < 1e-12
    assert float(w[1:11].sum()) == 0.0, "a lapsed uid takes nothing from the pool"


def test_a_uid_at_the_floor_counts_and_one_under_it_does_not():
    assert GENTRX_EMA_FLOOR == 0.01
    scores = torch.zeros(TARGET)
    scores[0], scores[1], scores[2] = 0.5, GENTRX_EMA_FLOOR, GENTRX_EMA_FLOOR * 0.99
    w, summary = _alloc(scores)
    assert summary["n_active"] == 2 and float(w[2]) == 0.0 and float(w[1]) > 0.0


def test_the_contributors_split_the_pool_in_proportion():
    scores = torch.zeros(TARGET)
    scores[0], scores[1] = 0.3, 0.1
    w, summary = _alloc(scores)
    assert summary["n_active"] == 2
    assert abs(float(w[0]) / float(w[1]) - 3.0) < 1e-6
