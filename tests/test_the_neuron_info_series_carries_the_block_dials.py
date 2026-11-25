# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The block rule's dials are published with the other scoring dials on the neuron_info series, so the board's dials
line and the dashboards' Scoring Config table show whether the rule runs and at what bar."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_the_neuron_info_series_stays_under_the_store_label_cap import _scoring_keys  # noqa: E402


def test_the_published_scoring_dials_name_the_block_rules_three():
    keys = set(_scoring_keys())
    for name in ("debeta_block_min_uids", "debeta_block_min_notional", "debeta_block_impact_window_s"):
        assert name in keys, name
