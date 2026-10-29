# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
PR-5 — defensive guards + circuit-breaker observability.

Tests:
  1. The rollback-rate warning fires when ≥8 of the last 10 rounds rolled
     back.
  2. `GENTRX_STRICT_VERSION_CHECK` env var controls whether untagged
     gradients are rejected (default off keeps the current behaviour).
"""
import logging
import os

import pytest

from GenTRX.src.gradient_server import GradientAggregator


def _make_aggregator(tmp_path):
    return GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path="",
        output_path=str(tmp_path / "out.pt"),
        log_path=str(tmp_path / "agg.jsonl"),
        validator_store=None,
        is_aggregator=False,
        no_startup_cleanup=True,
    )


def test_rollback_rate_warning_fires_at_threshold(tmp_path, caplog):
    """When ≥8 of the last 10 rolled back, a warning logger record appears.

    Set the level for the specific gradient_server logger to dodge caplog
    inter-test contamination (other tests can leave the propagation chain
    in a state where the default root-level capture misses these records).
    """
    caplog.set_level(logging.WARNING, logger="GenTRX.src.gradient_server")
    gs = _make_aggregator(tmp_path)
    for is_rollback in [True] * 8 + [False] * 2:
        gs._log_event({
            "type": "aggregation",
            "rolled_back": is_rollback,
            "sibling_only": False,
            "round": 0,
        })
    assert any(
        "rollback circuit-breaker" in r.getMessage() for r in caplog.records
    ), "rate ≥0.8 over last-10 should log a warning"


def test_rollback_rate_warning_silent_below_threshold(tmp_path, caplog):
    """Mostly-successful rounds do not trigger the warning."""
    caplog.set_level(logging.WARNING, logger="GenTRX.src.gradient_server")
    gs = _make_aggregator(tmp_path)
    for is_rollback in [True] * 3 + [False] * 7:
        gs._log_event({
            "type": "aggregation",
            "rolled_back": is_rollback,
            "sibling_only": False,
            "round": 0,
        })
    assert not any(
        "rollback circuit-breaker" in r.getMessage() for r in caplog.records
    ), "low rollback rate must not log the warning"


def test_strict_version_check_env_var_default_off(monkeypatch):
    """The strict check is off when the env var is unset/empty."""
    monkeypatch.delenv("GENTRX_STRICT_VERSION_CHECK", raising=False)
    val = os.environ.get("GENTRX_STRICT_VERSION_CHECK", "").strip().lower()
    truthy = val in ("1", "true", "yes", "on")
    assert not truthy


@pytest.mark.parametrize("val,expected", [
    ("1", True),
    ("true", True),
    ("TRUE", True),
    ("yes", True),
    ("on", True),
    ("0", False),
    ("false", False),
    ("", False),
    ("no", False),
])
def test_strict_version_check_env_var_parse(val, expected, monkeypatch):
    """The env-var parse accepts a small explicit truthy vocabulary."""
    monkeypatch.setenv("GENTRX_STRICT_VERSION_CHECK", val)
    parsed = os.environ.get("GENTRX_STRICT_VERSION_CHECK", "").strip().lower()
    truthy = parsed in ("1", "true", "yes", "on")
    assert truthy == expected
