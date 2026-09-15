"""Storage-path safety properties for the gradient server.

These tests pin down behaviour that the storage layer must satisfy
regardless of implementation strategy:

- A parquet flush is concurrency-safe: rows appended to the underlying
  list while the flush runs are not orphaned.
- The flushed parquet is time-ordered even when rows arrived out of
  order on the wire.
- A failed flush retains rows for the next attempt.
- Sim restart is detected via sim_id (config block) and ESE markers,
  not via backwards-time heuristics that would fire on benign WAN
  reordering of state ticks.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest

from GenTRX.src.util.schema import LOB_DEPTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(ts: int, mid_price: int = 100, qty: int = 1) -> dict:
    """Row dict matching the schema produced by `_process_tick`."""
    r = {
        "timestamp": ts,
        "order_type": 0,
        "rel_price": 0,
        "volume_int": qty,
        "volume_dec": 0.0,
        "interval_ns": 0,
        "mid_price": mid_price,
        "time_of_day_s": 0,
        "mid_price_delta": 0,
    }
    for i in range(LOB_DEPTH):
        r[f"lob_ask_vol_{i + 1}"] = 0.0
        r[f"lob_bid_vol_{i + 1}"] = 0.0
    return r


def _config_tick(ts: int, sim_id: str | None) -> dict:
    return {
        "step": 1,
        "ts": ts,
        "books": {},
        "config": {
            "priceDecimals": 8,
            "volumeDecimals": 8,
            "simulation_id": sim_id,
        },
    }


@pytest.fixture
def aggregator(tmp_path):
    from GenTRX.src.gradient_server import GradientAggregator

    agg = GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path=str(tmp_path / "val"),
        output_path=str(tmp_path / "out.pt"),
    )
    agg.validator_store = MagicMock()
    agg.validator_store.put_data.return_value = None
    # Pin the local-mirror cache to pytest tmp_path so flushes don't leak
    # gentrx_s3_cache_* directories into the system tempdir on each run.
    agg._s3_cache_dir = tmp_path / "s3_cache"
    return agg


# ---------------------------------------------------------------------------
# Parquet flush
# ---------------------------------------------------------------------------


def test_flush_preserves_rows_appended_during_put(aggregator):
    """Rows appended to `_pending_rows` while `put_data` is in flight must
    survive the flush — rebinding the slot to `[]` would orphan them."""
    book_id = 0
    aggregator._pending_rows[book_id] = [_row(ts) for ts in (100, 200, 300)]

    def slow_put_data(validator_uid, *, book_id, filename, data):
        # Simulate _process_tick appending a row from another thread
        # while the (slow) S3 PUT is in progress.
        aggregator._pending_rows[book_id].append(_row(400))

    aggregator.validator_store.put_data = MagicMock(side_effect=slow_put_data)

    aggregator._flush_book_parquet(book_id)

    remaining_ts = [r["timestamp"] for r in aggregator._pending_rows[book_id]]
    assert 400 in remaining_ts, "row appended during put_data was orphaned"
    for stale in (100, 200, 300):
        assert stale not in remaining_ts


def test_flush_produces_time_ordered_parquet(aggregator):
    """Out-of-order arrivals are sorted by timestamp at flush time so the
    parquet stream is monotonic for downstream readers."""
    book_id = 0
    aggregator._pending_rows[book_id] = [_row(ts) for ts in (100, 300, 200, 400)]

    captured: dict = {}

    def capture(validator_uid, *, book_id, filename, data):
        captured["data"] = data
        captured["filename"] = filename

    aggregator.validator_store.put_data = MagicMock(side_effect=capture)
    aggregator._flush_book_parquet(book_id)

    table = pq.read_table(io.BytesIO(captured["data"]))
    timestamps = [int(t.value) for t in table["timestamp"]]
    assert timestamps == sorted(timestamps) == [100, 200, 300, 400]


def test_flush_does_not_raise_on_concurrent_mutation(aggregator):
    """The flush builds many columns over the row list; a concurrent
    mutation must not produce a column-length mismatch."""
    book_id = 0
    aggregator._pending_rows[book_id] = [_row(ts) for ts in range(100, 200)]

    def mutate_during_put(validator_uid, *, book_id, filename, data):
        # Simulate steady-state appends during S3 PUT.
        for ts in range(200, 220):
            aggregator._pending_rows[book_id].append(_row(ts))

    aggregator.validator_store.put_data = MagicMock(side_effect=mutate_during_put)

    # Must not raise (column lengths must agree).
    aggregator._flush_book_parquet(book_id)
    aggregator.validator_store.put_data.assert_called_once()


def test_flush_failure_retains_rows(aggregator):
    """If `put_data` raises, the rows stay so the next attempt can retry."""
    book_id = 0
    rows = [_row(ts) for ts in (100, 200, 300)]
    aggregator._pending_rows[book_id] = list(rows)

    aggregator.validator_store.put_data = MagicMock(
        side_effect=RuntimeError("simulated S3 outage")
    )

    aggregator._flush_book_parquet(book_id)

    timestamps = [r["timestamp"] for r in aggregator._pending_rows[book_id]]
    assert timestamps == [100, 200, 300]


# ---------------------------------------------------------------------------
# Disk-primary mirror: scoring reads hit local disk instead of round-tripping
# back through S3 for data this process just wrote.
# ---------------------------------------------------------------------------


def test_flush_mirrors_parquet_to_local_cache(aggregator):
    """After a successful flush, the parquet exists on local disk at the same
    path `_fetch_s3_book_files` would download to, and is registered in
    `_s3_cached_files` so the cache lookup short-circuits the S3 download."""
    book_id = 0
    aggregator._pending_rows[book_id] = [_row(ts) for ts in (100, 200, 300)]

    captured: dict = {}

    def capture(validator_uid, *, book_id, filename, data):
        captured["filename"] = filename
        captured["data"] = data

    aggregator.validator_store.put_data = MagicMock(side_effect=capture)

    aggregator._flush_book_parquet(book_id)

    fname = captured["filename"]
    cache_key = f"{book_id}/{fname}"
    assert cache_key in aggregator._s3_cached_files, (
        "flushed parquet was not registered in the local cache map"
    )
    local_path = aggregator._s3_cached_files[cache_key]
    assert local_path.is_file(), "mirror path was registered but file is missing"
    assert local_path.read_bytes() == captured["data"], (
        "local mirror bytes differ from what was sent to S3"
    )
    # Path mirrors the shape `_fetch_s3_book_files` constructs.
    assert local_path.parent.name == "intervals"
    assert local_path.parent.parent.name == str(book_id)


def test_flush_mirror_failure_does_not_block_publish(aggregator, monkeypatch):
    """A local-disk write failure (e.g. disk full) must not break the S3
    publish or leave rows un-flushed — the mirror is best-effort."""
    book_id = 0
    aggregator._pending_rows[book_id] = [_row(ts) for ts in (100, 200, 300)]

    def explode(*_a, **_kw):
        raise OSError("simulated disk full")

    monkeypatch.setattr(Path, "write_bytes", explode)

    aggregator._flush_book_parquet(book_id)

    aggregator.validator_store.put_data.assert_called_once()
    assert aggregator._pending_rows[book_id] == [], (
        "rows must clear when the S3 publish succeeded, regardless of mirror outcome"
    )
    assert aggregator._s3_cached_files == {}, (
        "mirror failure must not pollute the cache map"
    )


# ---------------------------------------------------------------------------
# Cache-dir lifecycle: deterministic path, restart re-use, sim-end wipe,
# rolling age eviction.
# ---------------------------------------------------------------------------


def test_cache_dir_is_deterministic_and_under_output(tmp_path):
    """Cache dir is `<output_path>.parent/s3_cache/` — stable across runs,
    no tempdir leak in /tmp."""
    from GenTRX.src.gradient_server import GradientAggregator

    agg = GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path=str(tmp_path / "val"),
        output_path=str(tmp_path / "out.pt"),
    )
    agg._s3_cache_dir = None
    cache_dir = agg._get_s3_cache_dir()
    assert cache_dir == tmp_path / "s3_cache"
    assert cache_dir.is_dir()
    # Re-call returns the same path and does not raise on existing dir.
    assert agg._get_s3_cache_dir() == cache_dir


def test_warm_cache_skips_s3_download_on_restart(aggregator, tmp_path):
    """A parquet already on disk under the deterministic cache dir is
    registered without a redundant S3 GET on the first read."""
    book_id = 7
    fname = "00000000-00000300.parquet"
    cache_dir = aggregator._get_s3_cache_dir()
    local_path = cache_dir / str(book_id) / "intervals" / fname
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"already-on-disk")

    aggregator.validator_store.list_data = MagicMock(return_value=[fname])
    aggregator.validator_store.get_data = MagicMock(
        side_effect=AssertionError("get_data must not be called when file exists locally")
    )

    files = aggregator._fetch_s3_book_files(str(book_id))

    assert files == [local_path]
    cache_key = f"{book_id}/{fname}"
    assert aggregator._s3_cached_files.get(cache_key) == local_path
    aggregator.validator_store.get_data.assert_not_called()


def test_run_data_cleanup_wipes_local_mirror(aggregator):
    """Sim-end cleanup deletes S3 data/<uid>/ AND the local mirror, plus
    clears the in-memory cache map and loader cache."""
    book_id = 3
    fname = "00000000-00000300.parquet"
    cache_dir = aggregator._get_s3_cache_dir()
    local_path = cache_dir / str(book_id) / "intervals" / fname
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"stale-from-prev-sim")
    aggregator._s3_cached_files[f"{book_id}/{fname}"] = local_path
    aggregator._loader_cache[("dummy_key",)] = "stale loader"
    aggregator.validator_store.delete_prefix = MagicMock(return_value=5)

    aggregator._data_cleanup_pending = True
    removed = aggregator._run_data_cleanup()

    aggregator.validator_store.delete_prefix.assert_called_once_with(
        f"data/{aggregator._validator_uid}/"
    )
    assert removed == 5
    assert not local_path.exists(), "local mirror file should be deleted"
    assert not cache_dir.exists(), "cache dir should be removed"
    assert aggregator._s3_cached_files == {}
    assert aggregator._loader_cache == {}
    assert aggregator._data_cleanup_pending is False


def test_prune_s3_cache_evicts_old_files(aggregator):
    """Files older than retention are deleted; fresh files are kept; the
    `_s3_cached_files` map drops entries pointing at unlinked files."""
    import os as _os
    import time as _time

    aggregator.s3_cache_retention_hours = 1.0
    cache_dir = aggregator._get_s3_cache_dir()

    old = cache_dir / "1" / "intervals" / "old.parquet"
    fresh = cache_dir / "1" / "intervals" / "fresh.parquet"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(b"old")
    fresh.write_bytes(b"fresh")
    aggregator._s3_cached_files["1/old.parquet"] = old
    aggregator._s3_cached_files["1/fresh.parquet"] = fresh

    # Push old's mtime two hours into the past.
    long_ago = _time.time() - 2 * 3600
    _os.utime(old, (long_ago, long_ago))

    removed = aggregator._prune_s3_cache()

    assert removed == 1
    assert not old.exists()
    assert fresh.is_file()
    assert "1/old.parquet" not in aggregator._s3_cached_files
    assert "1/fresh.parquet" in aggregator._s3_cached_files


def test_prune_s3_cache_disabled_is_noop(aggregator):
    """Retention <= 0 disables rolling eviction. Sim-end wipe is unaffected
    (covered by test_run_data_cleanup_wipes_local_mirror)."""
    import os as _os
    import time as _time

    aggregator.s3_cache_retention_hours = 0
    cache_dir = aggregator._get_s3_cache_dir()
    ancient = cache_dir / "1" / "intervals" / "ancient.parquet"
    ancient.parent.mkdir(parents=True, exist_ok=True)
    ancient.write_bytes(b"ancient")
    long_ago = _time.time() - 365 * 24 * 3600
    _os.utime(ancient, (long_ago, long_ago))

    assert aggregator._prune_s3_cache() == 0
    assert ancient.is_file()


# ---------------------------------------------------------------------------
# Restart detection (sim_id / ESE / restored)
# ---------------------------------------------------------------------------


def test_same_sim_id_does_not_reset(aggregator):
    aggregator._sim_id = "sim_X"
    aggregator._pending_rows[0] = [_row(100)]

    aggregator._process_tick(_config_tick(ts=200, sim_id="sim_X"))

    assert aggregator._pending_rows.get(0) == [_row(100)]
    assert aggregator._sim_id == "sim_X"
    assert not aggregator._data_cleanup_pending


def test_changed_sim_id_resets(aggregator):
    aggregator._sim_id = "sim_X"
    aggregator._pending_rows[0] = [_row(100)]

    aggregator._process_tick(_config_tick(ts=200, sim_id="sim_Y"))

    assert aggregator._pending_rows == {}
    assert aggregator._sim_id == "sim_Y"
    assert aggregator._data_cleanup_pending


def test_restored_sim_id_mismatch_resets(aggregator):
    """Cross-process restart with a different sim than the staging file:
    the first config bind must wipe restored-but-stale state.

    This exercises the startup-cleanup path specifically (`_sim_id is None`
    at the moment of the mismatched tick). The constructor's
    `no_startup_cleanup` defaults to True as an op-safety
    net, which suppresses this path — flip it off for this test so the
    wipe fires as designed. Runtime rollovers (`test_changed_sim_id_resets`)
    still work with the safe default because they don't take this branch.
    """
    aggregator._no_startup_cleanup = False
    aggregator._sim_id = None
    aggregator._restored_sim_id = "sim_old"
    aggregator._pending_rows[0] = [_row(100)]

    aggregator._process_tick(_config_tick(ts=200, sim_id="sim_new"))

    assert aggregator._pending_rows == {}
    assert aggregator._sim_id == "sim_new"
    assert aggregator._restored_sim_id is None
    assert aggregator._data_cleanup_pending


def test_restored_sim_id_match_does_not_reset(aggregator):
    """Cross-process restart with the same sim as the staging file: the
    restored buffer is preserved and accumulation continues."""
    aggregator._sim_id = None
    aggregator._restored_sim_id = "sim_X"
    aggregator._pending_rows[0] = [_row(100)]

    aggregator._process_tick(_config_tick(ts=200, sim_id="sim_X"))

    assert aggregator._pending_rows.get(0) == [_row(100)]
    assert aggregator._sim_id == "sim_X"
    assert aggregator._restored_sim_id is None
    assert not aggregator._data_cleanup_pending


def test_out_of_order_tick_does_not_reset(aggregator):
    """A tick whose timestamp is older than what we've already processed
    is benign WAN reordering, not a sim restart, and must not wipe state."""
    aggregator._sim_id = "sim_X"
    aggregator._max_timestamp_ns = 500
    aggregator._pending_rows[0] = [_row(100), _row(200)]

    tick = {"step": 1, "ts": 300, "books": {}}
    aggregator._process_tick(tick)

    assert len(aggregator._pending_rows[0]) == 2
    assert aggregator._sim_id == "sim_X"
    assert not aggregator._data_cleanup_pending


def test_ese_marker_resets(aggregator):
    aggregator._sim_id = "sim_X"
    aggregator._max_timestamp_ns = 500
    aggregator._pending_rows[0] = [_row(100)]

    tick = {"step": 1, "ts": 600, "sim_events": ["ESE"], "books": {}}
    aggregator._process_tick(tick)

    assert aggregator._pending_rows == {}
    assert aggregator._data_cleanup_pending


# ---------------------------------------------------------------------------
# Replay dedup: == only, late arrivals (<) still process
# ---------------------------------------------------------------------------


def test_dedup_drops_exact_duplicate(aggregator):
    aggregator._sim_id = "sim_X"
    aggregator._last_seen_sim_ts = 100
    aggregator._max_timestamp_ns = 100
    aggregator._process_tick({"step": 5, "ts": 100, "books": {}})
    assert aggregator._dedup_drops_total == 1
    assert aggregator._last_seen_sim_ts == 100


def test_dedup_does_not_drop_late_straggler(aggregator):
    """A reorder-buffer drain with sim_ts < last_seen must still process (C1)."""
    aggregator._sim_id = "sim_X"
    aggregator._last_seen_sim_ts = 500
    aggregator._process_tick({"step": 7, "ts": 300, "books": {}})
    assert aggregator._dedup_drops_total == 0


def test_dedup_first_tick_processed(aggregator):
    """First tick with a sim_id is processed, not deduped."""
    aggregator._sim_id = None
    aggregator._last_seen_sim_ts = 0
    aggregator._process_tick(_config_tick(ts=100, sim_id="sim_X"))
    assert aggregator._dedup_drops_total == 0
    assert aggregator._last_seen_sim_ts == 100
    assert aggregator._sim_id == "sim_X"


def test_pre_bind_tick_without_sim_id_dropped(aggregator):
    """Pre-bind tick without sim_id is refused (no book processing, no state update)."""
    aggregator._sim_id = None
    aggregator._last_seen_sim_ts = 0
    aggregator._process_tick({"step": 1, "ts": 100, "books": {}})
    assert aggregator._pre_bind_drops == 1
    assert aggregator._last_seen_sim_ts == 0
    assert aggregator._sim_id is None


def test_dedup_sim_swap_bypasses(aggregator):
    aggregator._sim_id = "sim_X"
    aggregator._last_seen_sim_ts = 500
    aggregator._process_tick(_config_tick(ts=50, sim_id="sim_Y"))
    assert aggregator._dedup_drops_total == 0
    assert aggregator._sim_id == "sim_Y"


def test_dedup_advances(aggregator):
    aggregator._sim_id = "sim_X"
    aggregator._last_seen_sim_ts = 0
    for ts in (100, 200, 300):
        aggregator._process_tick({"step": 1, "ts": ts, "books": {}})
    assert aggregator._dedup_drops_total == 0
    assert aggregator._last_seen_sim_ts == 300


def test_last_seen_sim_ts_persists_across_save_restore(aggregator, tmp_path):
    """C2: _last_seen_sim_ts survives restart via pending_rows.msgpack."""
    aggregator._sim_id = "sim_X"
    aggregator._last_seen_sim_ts = 1234
    aggregator._pending_rows[0] = [_row(1000)]
    aggregator._pending_staging_path = tmp_path / "pending.msgpack"
    aggregator._save_pending_rows()

    fresh_path = aggregator._pending_staging_path
    aggregator._last_seen_sim_ts = 0
    aggregator._pending_rows.clear()
    aggregator._pending_staging_path = fresh_path

    aggregator._restore_pending_rows()
    assert aggregator._last_seen_sim_ts == 1234
