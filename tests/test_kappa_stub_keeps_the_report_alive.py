# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A kappa_values stub must carry every key the report reads, and the report must never index one.

A fresh registration with no realized PnL makes kappa_3 return None, and score_uid's
reporting-carrier guard replaces that None with {'books': {}}: truthy, but with no median. The report
worker's per-agent stats then read kappa_values['median'] and raise KeyError on every cycle. The
validator itself keeps going, so scoring, miner queries and weight setting continue at their normal
rate while every miner, book and simulation gauge stops advancing, and the dashboards show a frozen
network.

Two defences, both pinned: the stub has the full reporting shape (the same keys as the zeroed slot in
trade.py and the init default in validator.py), and the report reads the median with .get.
"""
from pathlib import Path

import pytest

from taos.im.validator import reward
from tests.test_kappa_skip_under_debeta import _validator_data

DEV = Path(__file__).resolve().parents[1]
REPORT_SRC = (DEV / "taos/im/validator/report.py").read_text()
REWARD_SRC = (DEV / "taos/im/validator/reward.py").read_text()

# Every kappa_values key the report worker's per-agent block reads.
REPORT_READS = ("median", "penalty", "activity_weighted_normalized_median", "score")


def _report_reads(kv):
    """The report's per-agent expressions, as written: they must not raise on any carrier."""
    return {
        "kappa": kv.get("median") if kv else None,
        "kappa_penalty": kv.get("penalty") if kv else None,
        "kappa_score": kv.get("score") if kv else None,
    }


def test_the_stub_has_the_full_reporting_shape():
    stub = reward.kappa_stub()
    for key in REPORT_READS:
        assert key in stub, f"the report reads {key!r}; a stub without it kills the report worker"
    assert stub["books"] == {} and stub["books_weighted"] == {} and stub["median"] is None
    assert "skipped" not in stub and reward.kappa_stub(skipped=True)["skipped"] is True


def test_a_uid_kappa_could_not_score_gets_a_readable_carrier_at_a_nonzero_rung():
    # kappa_3 returns None for a freshly reset uid with no realized PnL.
    vd = _validator_data(debeta_scores={1: 0.8, 2: 0.2}, weight=0.25, kappa_values={1: None, 2: None})
    reward.score_uid(vd, 1)
    entry = vd["kappa_values"][1]
    assert entry, "the carrier must exist so the decomposition and the dial are published"
    for key in REPORT_READS:
        assert key in entry, f"{key!r} missing: this is the KeyError that froze testnet"
    assert entry["debeta_weight"] == pytest.approx(0.25)
    assert _report_reads(entry)["kappa"] is None


def test_kappa_weight_zero_stubs_are_readable_too(monkeypatch):
    vd = _validator_data(debeta_scores={1: 0.8, 2: 0.2}, weight=1.0)
    reward.score_uids(vd)
    for uid in (1, 2):
        entry = vd["kappa_values"][uid]
        assert entry.get("skipped") is True
        for key in REPORT_READS:
            assert key in entry
        _report_reads(entry)


def test_the_report_reads_the_median_defensively_and_no_bare_stub_literal_remains():
    assert "kappa_values.get('median') if kappa_values else None" in REPORT_SRC
    assert "kappa_values['median']" not in REPORT_SRC, "indexing a carrier is what killed the report worker"
    assert "kappa_values[uid] = {'books': {}}" not in REWARD_SRC, "the median-less stub is back"
    assert "kappa_values[uid] = {'books': {}, 'books_weighted': {}, 'skipped': True}" not in REWARD_SRC, (
        "every stub must come from kappa_stub() so the shape cannot drift again"
    )
