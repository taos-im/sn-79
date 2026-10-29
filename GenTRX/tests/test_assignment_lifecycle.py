"""Tests for GradientAggregator assignment lifecycle.

Verifies the event-driven assignment state machine:
PENDING -> DATA_READY -> DELIVERED -> GRADIENT_IN -> SCORED

These tests use a mocked store so they run without S3.
"""


import pytest


@pytest.fixture
def aggregator(tmp_path):
    """Create a GradientAggregator with mocked store."""
    from GenTRX.src.gradient_server import GradientAggregator

    ckpt_path = tmp_path / "ckpt.pt"
    output_path = tmp_path / "out.pt"

    agg = GradientAggregator(
        checkpoint_path=str(ckpt_path),
        val_data_path=str(tmp_path / "val"),
        output_path=str(output_path),
        books_per_miner=2,
        interval=60,
        window_ns=50,  # tiny so max_ts > window_ns
    )
    agg._max_timestamp_ns = 200  # fake "data exists"
    # Populate in-memory parquet registry covering the full time range.
    # Must cover [0, 200] so any beta-sampled window finds overlapping data.
    for bid in range(4):
        agg._written_parquets[bid] = [
            ("00000000-00000200.parquet", 0, 200),
        ]
    return agg


def test_get_assignment_creates_on_demand_for_arbitrary_uid(aggregator):
    """When a UID requests an assignment that doesn't exist, one is created."""
    # Initially no assignments
    assert 5 not in aggregator._assignments

    # Request an assignment for uid 5
    aggregator.get_assignment(5)

    # An assignment should now exist for uid 5
    assert 5 in aggregator._assignments
    a = aggregator._assignments[5]
    assert a["round"] == aggregator._agg_round
    # data should have been resolved (data_keys from list_data mock)
    assert a.get("_state") in ("DATA_READY", "DELIVERED")


def test_get_assignment_returns_none_when_no_books_known(aggregator):
    """No books known anywhere → no assignment is created at all.

    With the n_books fallback removed, the aggregator refuses to invent
    assignments for fake books. It just waits for state to arrive.
    """
    aggregator._written_parquets = {}      # no parquets flushed
    aggregator._pending_rows = {}          # no state ticks observed yet
    aggregator._max_timestamp_ns = 200

    result = aggregator.get_assignment(0)
    assert result is None
    assert 0 not in aggregator._assignments


def test_assignment_pending_when_books_seen_but_no_parquets(aggregator):
    """Books known from state ticks but no parquets yet → PENDING assignment."""
    aggregator._written_parquets = {}
    aggregator._pending_rows = {0: [], 1: [], 2: [], 3: []}  # books seen via ticks
    aggregator._max_timestamp_ns = 200

    result = aggregator.get_assignment(0)
    assert result is None
    # Assignment was created (books were known) but data isn't ready yet
    assert 0 in aggregator._assignments
    assert aggregator._assignments[0]["_state"] == "PENDING"


def test_get_assignment_returns_data_when_ready(aggregator):
    """When data is ready, get_assignment returns it and transitions to DELIVERED."""
    result = aggregator.get_assignment(0)
    assert result is not None
    assert result.get("books")
    assert result.get("data")
    assert aggregator._assignments[0]["_state"] == "DELIVERED"
    assert aggregator._assignments[0]["_delivered_at"] is not None


def test_get_assignment_returns_none_after_delivered(aggregator):
    """Second call for the same UID in same round should not re-deliver."""
    aggregator.get_assignment(0)  # first call: DELIVERED
    result = aggregator.get_assignment(0)  # second call
    # Still DELIVERED, but get_assignment only returns DATA_READY
    assert result is None


def test_metagraph_uids_work(aggregator):
    """Assignments can be requested for arbitrary metagraph UIDs (e.g. 2, 3)."""
    # Validator scenario: miners are at metagraph uids 2 and 3
    result_2 = aggregator.get_assignment(2)
    result_3 = aggregator.get_assignment(3)

    assert result_2 is not None
    assert result_3 is not None
    # Different UIDs should get different assignments (different beta sample)
    assert 2 in aggregator._assignments
    assert 3 in aggregator._assignments


def test_round_advances_creates_new_assignments(aggregator):
    """When a new round starts, get_assignment creates fresh assignments."""
    aggregator.get_assignment(0)
    old_round = aggregator._assignments[0]["round"]

    # Simulate round advance
    aggregator._agg_round += 1

    # Request again for same UID
    aggregator.get_assignment(0)
    new_round = aggregator._assignments[0]["round"]

    assert new_round == old_round + 1


def test_aggregator_initial_state(aggregator):
    """Newly created aggregator has no assignments."""
    assert aggregator._assignments == {}
    assert aggregator._agg_round == 0
