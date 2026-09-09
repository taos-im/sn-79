"""Tests for grad-server restart resilience.

Pin the rule: sim_id is the only authority on sim identity. A wipe fires
only on positive mismatch between the live cfg.simulation_id and either
the local staging file's sim_id or the bucket marker's sim_id. Absence
of either is "unknown lineage" and the default is preserve.

Run: pytest GenTRX/tests/test_restart_resilience.py -v
"""

import pytest


def _make_aggregator(tmp_path, validator_store=None):
    from GenTRX.src.gradient_server import GradientAggregator

    # `no_startup_cleanup=False` re-enables the wipe path — the whole
    # point of this file is to exercise sim-id-mismatch wipe behaviour.
    # The constructor default is True (op-safety),
    # which would otherwise suppress the very code paths under test.
    return GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path=str(tmp_path / "val"),
        output_path=str(tmp_path / "out.pt"),
        books_per_miner=1,
        interval=60,
        window_ns=50,
        warmup_rounds=0,
        rollback=False,
        validator_store=validator_store,
        no_startup_cleanup=False,
    )


class _FakeStore:
    """Minimal stand-in for GradientStore — covers sim marker + cleanup."""
    def __init__(self):
        self.marker: str | None = None
        self.deleted_prefixes: list[str] = []

    def get_sim_marker(self, uid):
        return self.marker

    def put_sim_marker(self, uid, sim_id):
        self.marker = sim_id

    def list_books(self, uid):
        return []

    def list_data(self, uid, book_id):
        return []

    def delete_prefix(self, prefix):
        self.deleted_prefixes.append(prefix)
        return 0


# ---------------------------------------------------------------------------
# Save/restore round-trip
# ---------------------------------------------------------------------------


def test_save_restore_preserves_agg_round_and_assignments(tmp_path):
    src = _make_aggregator(tmp_path)
    src._sim_id = "SIM_A"
    src._agg_round = 7
    src._assignments[2] = {
        "round": 7,
        "books": ["0", "1"],
        "data": ["data/0/x.parquet", "data/1/y.parquet"],
        "ts_start": 100,
        "ts_end": 200,
        "model_version": 3,
        "_state": "DELIVERED",
        "_delivered_at": 1.0,
        "_created_at": 0.0,
        "_gradient_data": b"HEAVY_BYTES_SHOULD_NOT_PERSIST",
        "_score": 0.42,
    }
    src._prev_round_assignments[3] = {
        "round": 6,
        "books": ["2"],
        "data": ["data/2/z.parquet"],
        "ts_start": 0,
        "ts_end": 100,
        "model_version": 3,
        "_state": "GRADIENT_IN",
        "_gradient_data": b"HEAVY_BYTES_2",
    }
    src._save_pending_rows()

    dst = _make_aggregator(tmp_path)
    dst._restore_pending_rows()

    assert dst._restored_sim_id == "SIM_A"
    assert dst._agg_round == 7
    assert 2 in dst._assignments
    assert dst._assignments[2]["_state"] == "DELIVERED"
    assert dst._assignments[2]["_gradient_data"] is None
    assert 3 in dst._prev_round_assignments
    assert 6 in dst._pending_aggregation_rounds


def test_save_without_sim_id_is_skipped(tmp_path):
    """Save is a no-op when _sim_id is None; staging files stay identity-paired."""
    src = _make_aggregator(tmp_path)
    assert src._sim_id is None
    src._agg_round = 5
    src._pending_rows[0] = [{"timestamp": 1000}]
    src._max_timestamp_ns = 1000
    src._assignments[2] = {
        "round": 5, "books": ["0"], "data": [],
        "ts_start": 0, "ts_end": 100, "model_version": 1,
        "_state": "DELIVERED", "_delivered_at": 0.0, "_created_at": 0.0,
    }
    src._save_pending_rows()
    assert not src._pending_staging_path.exists()

    dst = _make_aggregator(tmp_path)
    dst._restore_pending_rows()
    assert dst._restored_sim_id is None
    assert dst._agg_round == 0
    assert dst._assignments == {}
    assert dst._pending_rows == {}
    assert dst._max_timestamp_ns == 0


# ---------------------------------------------------------------------------
# Cfg-bind: orphan rule
# ---------------------------------------------------------------------------


def _tick(ts: int, *, sim_id: str | None = None, sim_events: list[str] | None = None):
    t: dict = {"ts": ts, "books": {}}
    if sim_id is not None:
        t["config"] = {
            "priceDecimals": 8,
            "volumeDecimals": 8,
            "simulation_id": sim_id,
        }
    if sim_events:
        t["sim_events"] = sim_events
    return t


def test_no_wipe_when_no_prior_identity(tmp_path):
    """Unknown lineage: bind to incoming sim_id without wiping (preserve default)."""
    store = _FakeStore()
    agg = _make_aggregator(tmp_path, validator_store=store)
    assert agg._restored_sim_id is None
    assert agg._bucket_sim_id is None

    agg._process_tick(_tick(1000, sim_id="FRESH_SIM"))

    assert agg._sim_id == "FRESH_SIM"
    assert agg._sim_epoch == 0
    assert agg._data_cleanup_pending is False
    assert store.marker == "FRESH_SIM"


def test_wipe_on_staged_sim_id_mismatch(tmp_path):
    agg = _make_aggregator(tmp_path)
    agg._no_startup_cleanup = False  # opt into startup cleanup to test the wipe path
    agg._restored_sim_id = "OLD_SIM"

    agg._process_tick(_tick(1000, sim_id="NEW_SIM"))

    assert agg._sim_id == "NEW_SIM"
    assert agg._sim_epoch == 1
    assert agg._data_cleanup_pending is True


def test_startup_mismatch_preserves_by_default(tmp_path):
    """Default (--no-startup-cleanup): a positive sim_id mismatch at STARTUP
    (first bind, _sim_id is None) preserves data — only a rollover detected
    while running wipes. Pins the data-preservation default."""
    agg = _make_aggregator(tmp_path)  # no_startup_cleanup defaults True
    agg._restored_sim_id = "OLD_SIM"

    agg._process_tick(_tick(1000, sim_id="NEW_SIM"))

    assert agg._sim_id == "NEW_SIM"
    assert agg._data_cleanup_pending is False  # preserved, not wiped


def test_no_wipe_when_staged_sim_id_matches(tmp_path):
    agg = _make_aggregator(tmp_path)
    agg._restored_sim_id = "SAME_SIM"

    agg._process_tick(_tick(1000, sim_id="SAME_SIM"))

    assert agg._sim_id == "SAME_SIM"
    assert agg._sim_epoch == 0
    assert agg._data_cleanup_pending is False


def test_wipe_on_bucket_marker_mismatch(tmp_path):
    """Bucket marker says a different sim; wipe."""
    store = _FakeStore()
    store.marker = "OLD_BUCKET_SIM"
    agg = _make_aggregator(tmp_path, validator_store=store)
    agg._no_startup_cleanup = False  # opt into startup cleanup to test the wipe path
    agg._bucket_sim_id = store.marker

    agg._process_tick(_tick(1000, sim_id="NEW_SIM"))

    assert agg._sim_id == "NEW_SIM"
    assert agg._sim_epoch == 1
    assert agg._data_cleanup_pending is True
    assert store.marker == "NEW_SIM"


def test_no_wipe_when_bucket_marker_matches(tmp_path):
    store = _FakeStore()
    store.marker = "SAME_SIM"
    agg = _make_aggregator(tmp_path, validator_store=store)
    agg._bucket_sim_id = store.marker

    agg._process_tick(_tick(1000, sim_id="SAME_SIM"))

    assert agg._sim_id == "SAME_SIM"
    assert agg._sim_epoch == 0
    assert agg._data_cleanup_pending is False


# ---------------------------------------------------------------------------
# ESS handling
# ---------------------------------------------------------------------------


def test_ess_with_new_sim_id_rebinds(tmp_path):
    agg = _make_aggregator(tmp_path)
    agg._process_tick(_tick(1000, sim_id="SIM_A"))
    assert agg._sim_id == "SIM_A"
    epoch_a = agg._sim_epoch

    agg._process_tick(_tick(2000, sim_id="SIM_B", sim_events=["ESS"]))
    assert agg._sim_id == "SIM_B"
    assert agg._sim_epoch == epoch_a + 1


def test_duplicate_ess_for_same_sim_is_idempotent(tmp_path):
    """ESS replay with the same sim_id is a no-op."""
    agg = _make_aggregator(tmp_path)
    agg._process_tick(_tick(1000, sim_id="SIM_A"))
    agg._max_timestamp_ns = 5_000_000_000
    agg._pending_rows[0] = [{"timestamp": 1000}]
    agg._pending_interval_start[0] = 1000

    agg._process_tick(_tick(2000, sim_id="SIM_A", sim_events=["ESS"]))

    assert agg._sim_id == "SIM_A"
    assert agg._sim_epoch == 0
    assert agg._data_cleanup_pending is False
    assert agg._max_timestamp_ns >= 5_000_000_000
    assert agg._pending_rows.get(0)
