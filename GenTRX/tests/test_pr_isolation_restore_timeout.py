# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""GenTRX startup isolation: `_restore_written_parquets` wall-clock budget.

At startup the gradient server lists the validator's data shard to rebuild
its parquet registry. A paginated S3 LIST can stall in a socket read that
the client-level read timeout does not always break, and the restore runs
before uvicorn binds, so a stalled LIST would wedge startup.

Each S3 list call runs in an abandonable daemon thread under one budget
(`GENTRX_RESTORE_BUDGET_S`, default 60s). The registry itself is only
written on the calling thread, so an abandoned worker cannot touch shared
state. Cost of a timeout: the un-scanned pages re-accumulate over the next
window.

These tests pin the contract:
  1. Happy path: a fast bucket is restored inside the budget.
  2. A stuck list_books returns after the budget with a warning and leaves
     the registry empty (no deadlock on the stuck worker).
  3. A budget spent part way through the books stops the scan.
  4. `GENTRX_RESTORE_BUDGET_S=0` skips the scan without listing anything.
  5. An unparseable budget falls back to the default instead of raising.
  6. An empty bucket returns cleanly.
"""

import logging
import threading
import time
from pathlib import Path

import pytest

from GenTRX.src.gradient_server import GradientAggregator

_LOGGER = "GenTRX.src.gradient_server"
_PAGE = "00000000-00000300.parquet"


def _make_aggregator(tmp_path: Path, store=None) -> GradientAggregator:
    return GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path="",
        output_path=str(tmp_path / "out.pt"),
        log_path=str(tmp_path / "agg.jsonl"),
        validator_store=store,
        is_aggregator=False,
        no_startup_cleanup=True,
    )


class _Store:
    """Validator-store stand-in: the two list calls plus the sim marker."""

    def __init__(self, books=(), list_books_delay=0.0, list_data_delay=0.0):
        self.books = list(books)
        self.list_books_delay = list_books_delay
        self.list_data_delay = list_data_delay
        self.list_books_calls = 0
        self.list_data_calls = 0
        self.list_books_started = threading.Event()

    def list_books(self, _uid):
        self.list_books_calls += 1
        self.list_books_started.set()
        time.sleep(self.list_books_delay)
        return list(self.books)

    def list_data(self, _uid, _book_id):
        self.list_data_calls += 1
        time.sleep(self.list_data_delay)
        return [_PAGE, "not-a-page.txt"]

    def get_sim_marker(self, _uid):
        return None


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records]


def test_restore_happy_path_returns_in_budget(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    monkeypatch.delenv("GENTRX_RESTORE_BUDGET_S", raising=False)
    agg = _make_aggregator(tmp_path, _Store(books=[0, 1]))

    t0 = time.perf_counter()
    agg._restore_written_parquets()
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, f"happy-path restore should finish fast, took {elapsed:.2f}s"
    assert set(agg._written_parquets) == {0, 1}
    assert all(len(v) == 1 for v in agg._written_parquets.values())
    assert not any("parquet restore" in m for m in _messages(caplog))


def test_restore_timeout_does_not_deadlock(tmp_path, caplog, monkeypatch):
    """A list_books that outlives the budget is abandoned: the restore
    returns promptly, logs a warning and restores nothing."""
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    monkeypatch.setenv("GENTRX_RESTORE_BUDGET_S", "0.5")
    store = _Store(books=[0], list_books_delay=5.0)
    agg = _make_aggregator(tmp_path, store)

    t0 = time.perf_counter()
    agg._restore_written_parquets()
    elapsed = time.perf_counter() - t0

    assert store.list_books_started.is_set(), "worker thread should have started"
    # Allow slack for scheduling, but never the 5s the stuck call sleeps.
    assert elapsed < 2.5, f"restore should return after the budget, took {elapsed:.2f}s"
    assert any("list_books exceeded" in m for m in _messages(caplog))
    assert agg._written_parquets == {}


def test_restore_budget_spent_mid_scan_stops(tmp_path, caplog, monkeypatch):
    """Once the budget is spent between books the scan stops; the books
    already listed stay in the registry."""
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    monkeypatch.setenv("GENTRX_RESTORE_BUDGET_S", "0.5")
    store = _Store(books=list(range(10)), list_data_delay=0.2)
    agg = _make_aggregator(tmp_path, store)

    t0 = time.perf_counter()
    agg._restore_written_parquets()
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"restore should stop near the budget, took {elapsed:.2f}s"
    assert 0 < len(agg._written_parquets) < 10
    assert store.list_data_calls < 10
    assert any(
        "budget spent" in m or "exceeded remaining budget" in m for m in _messages(caplog)
    )


def test_restore_budget_zero_skips_call_entirely(tmp_path, caplog, monkeypatch):
    """`GENTRX_RESTORE_BUDGET_S=0` is the operator switch for skipping the
    startup S3 walk: nothing is listed and a warning says so."""
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    monkeypatch.setenv("GENTRX_RESTORE_BUDGET_S", "0")
    store = _Store(books=[0])
    agg = _make_aggregator(tmp_path, store)

    t0 = time.perf_counter()
    agg._restore_written_parquets()
    elapsed = time.perf_counter() - t0

    assert store.list_books_calls == 0, "no S3 list may run when the budget is 0"
    assert elapsed < 0.5
    assert agg._written_parquets == {}
    assert any("parquet restore disabled" in m for m in _messages(caplog))


def test_restore_budget_env_override_parses(tmp_path, monkeypatch):
    """The env var is read on each call. A bad value falls back to the
    default rather than raising out of startup; a good value is used."""
    store = _Store(books=[0])
    agg = _make_aggregator(tmp_path, store)

    monkeypatch.setenv("GENTRX_RESTORE_BUDGET_S", "not-a-number")
    agg._restore_written_parquets()
    assert set(agg._written_parquets) == {0}

    agg._written_parquets.clear()
    monkeypatch.setenv("GENTRX_RESTORE_BUDGET_S", "0.1")
    agg._restore_written_parquets()
    assert set(agg._written_parquets) == {0}


def test_restore_empty_bucket_returns_cleanly(tmp_path, monkeypatch):
    monkeypatch.delenv("GENTRX_RESTORE_BUDGET_S", raising=False)
    store = _Store(books=[])
    agg = _make_aggregator(tmp_path, store)

    agg._restore_written_parquets()

    assert store.list_books_calls == 1
    assert store.list_data_calls == 0
    assert agg._written_parquets == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
