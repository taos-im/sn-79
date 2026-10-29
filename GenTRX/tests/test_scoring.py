"""Tests for GenTRX gradient scoring publication.

Pins the contract for `_deliver_scores` / `_latest_scores`:

  - Every UID with an assignment in the current round appears in the
    published payload, including miners that did not submit a gradient.
  - Non-submitters appear with `score=0.0` and `accepted=False`, even
    when 0.0 happens to exceed `min_score`.
  - Submitters carry the stamped score; `accepted` follows the threshold.
  - A round where nobody submits still publishes a fresh payload for
    the current round; the previous payload does not stick around.
  - Across rounds, a miner that stops submitting drops to 0 within one
    round (the bug this suite was written to catch).

These tests sidestep torch by replacing `_score_and_aggregate` on the
instance with a stub that reads pre-stamped `_score` off each
submitter's assignment and calls `_deliver_scores` directly.

Run: pytest GenTRX/tests/test_scoring.py -v
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


@pytest.fixture
def aggregator(tmp_path):
    from GenTRX.src.gradient_server import GradientAggregator

    agg = GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path=str(tmp_path / "val"),
        output_path=str(tmp_path / "out.pt"),
        books_per_miner=1,
        interval=60,
        window_ns=50,
        warmup_rounds=0,
        rollback=False,
    )
    agg._max_timestamp_ns = 200
    for bid in range(4):
        agg._written_parquets[bid] = [("00000000-00000200.parquet", 0, 200)]
    return agg


def _plant_assignment(agg, uid, round_id, *, submitted, score=None, books=None):
    """Place an assignment in the aggregator's `_assignments` map.

    `submitted=True` simulates a miner that uploaded a gradient: state
    GRADIENT_IN with a non-empty `_gradient_data` and an optional
    pre-stamped score (the fake scorer reads this).

    `submitted=False` simulates a miner that was delivered an assignment
    but never sent a gradient back: state DELIVERED.
    """
    books = books or [str(uid)]
    assignment = {
        "round": round_id,
        "books": books,
        "data": [f"data/{bid}/intervals/x.parquet" for bid in books],
        "model_version": 1,
        "_state": "GRADIENT_IN" if submitted else "DELIVERED",
        "_delivered_at": 0.0,
    }
    if submitted:
        assignment["_gradient_data"] = b"FAKE_GRADIENT"
        if score is not None:
            assignment["_score"] = score
            assignment["_score_own"] = score
            assignment["_score_held"] = score
            assignment["_overfitting"] = False
    agg._assignments[uid] = assignment
    return assignment


def _install_fake_scorer(agg):
    """Replace `_score_and_aggregate` with a stub that bypasses torch.

    Reads `_score` off each pending assignment and calls `_deliver_scores`
    with the same shape the real path produces. Preserves the public
    contract under test without dragging in the model.
    """

    def fake(pending, round_assignments):
        scored = []
        for uid, _win, a, _grad in pending:
            s = a.get("_score", 0.0)
            scored.append((uid, agg._agg_round, s, b"comp", a))
        threshold = agg._effective_min_score
        accepted = [t for t in scored if t[2] > threshold]
        rejected = [t for t in scored if t[2] <= threshold]
        agg._deliver_scores(scored, accepted, rejected, threshold, round_assignments)

    agg._score_and_aggregate = fake


# ---------------------------------------------------------------------------
# Server-side: _deliver_scores via _aggregate_round
# ---------------------------------------------------------------------------


def test_payload_lists_every_assigned_uid(aggregator):
    _install_fake_scorer(aggregator)
    aggregator._agg_round = 1
    _plant_assignment(aggregator, 1, 1, submitted=True, score=0.5)
    _plant_assignment(aggregator, 2, 1, submitted=True, score=0.2)
    _plant_assignment(aggregator, 3, 1, submitted=False)

    aggregator._aggregate_round()

    payload = aggregator._latest_scores
    assert payload is not None
    assert payload["round"] == 1
    assert set(payload["scores"].keys()) == {"1", "2", "3"}


def test_submitter_keeps_stamped_score(aggregator):
    _install_fake_scorer(aggregator)
    aggregator._agg_round = 1
    _plant_assignment(aggregator, 1, 1, submitted=True, score=0.42)

    aggregator._aggregate_round()
    entry = aggregator._latest_scores["scores"]["1"]
    assert entry["score"] == pytest.approx(0.42)
    assert entry["accepted"] is True


def test_non_submitter_is_zero_and_unaccepted(aggregator):
    """Score=0.0 > min_score (-0.1) would falsely flip accepted=True for a
    non-submitter without the explicit submitter-only gate. Pin it."""
    _install_fake_scorer(aggregator)
    aggregator._agg_round = 1
    _plant_assignment(aggregator, 1, 1, submitted=True, score=0.5)
    _plant_assignment(aggregator, 2, 1, submitted=False)

    aggregator._aggregate_round()
    entry = aggregator._latest_scores["scores"]["2"]
    assert entry["score"] == 0.0
    assert entry["accepted"] is False


def test_rejected_submitter_is_not_accepted(aggregator):
    _install_fake_scorer(aggregator)
    aggregator._agg_round = 1
    _plant_assignment(aggregator, 1, 1, submitted=True, score=-0.5)

    aggregator._aggregate_round()
    entry = aggregator._latest_scores["scores"]["1"]
    assert entry["score"] == pytest.approx(-0.5)
    assert entry["accepted"] is False


def test_no_submitters_round_still_publishes(aggregator):
    """When pending is empty, _latest_scores must still update for the
    current round with zeros for every assigned UID. Without this, a
    quiet round leaves the previous payload as 'latest' forever."""
    _install_fake_scorer(aggregator)
    aggregator._agg_round = 1
    _plant_assignment(aggregator, 1, 1, submitted=False)
    _plant_assignment(aggregator, 2, 1, submitted=False)

    aggregator._aggregate_round()
    payload = aggregator._latest_scores
    assert payload is not None
    assert payload["round"] == 1
    assert set(payload["scores"].keys()) == {"1", "2"}
    for uid in ("1", "2"):
        assert payload["scores"][uid]["score"] == 0.0
        assert payload["scores"][uid]["accepted"] is False
    assert payload["n_scored"] == 0
    assert payload["n_accepted"] == 0


def test_stopped_miner_drops_to_zero_next_round(aggregator):
    """The regression this suite exists for: train a few rounds with two
    miners, then have one stop submitting. Their published score must
    drop to 0 in the very next round, not linger at the last value."""
    _install_fake_scorer(aggregator)

    aggregator._agg_round = 1
    _plant_assignment(aggregator, 1, 1, submitted=True, score=0.7)
    _plant_assignment(aggregator, 2, 1, submitted=True, score=0.3)
    aggregator._aggregate_round()
    assert aggregator._latest_scores["scores"]["2"]["score"] == pytest.approx(0.3)

    aggregator._agg_round = 2
    aggregator._assignments.clear()
    aggregator._prev_round_assignments.clear()
    _plant_assignment(aggregator, 1, 2, submitted=True, score=0.6)
    _plant_assignment(aggregator, 2, 2, submitted=False)
    aggregator._aggregate_round()

    payload = aggregator._latest_scores
    assert payload["round"] == 2
    assert payload["scores"]["1"]["score"] == pytest.approx(0.6)
    assert payload["scores"]["2"]["score"] == 0.0
    assert payload["scores"]["2"]["accepted"] is False


def test_quiet_round_overwrites_prior_payload(aggregator):
    """Round 1 had a submitter at 0.9. Round 2 had no submitters at all.
    The payload's round number must advance to 2 and the score collapse
    to 0, instead of round 1's payload sticking around."""
    _install_fake_scorer(aggregator)

    aggregator._agg_round = 1
    _plant_assignment(aggregator, 1, 1, submitted=True, score=0.9)
    aggregator._aggregate_round()
    assert aggregator._latest_scores["round"] == 1
    assert aggregator._latest_scores["scores"]["1"]["score"] == pytest.approx(0.9)

    aggregator._agg_round = 2
    aggregator._assignments.clear()
    aggregator._prev_round_assignments.clear()
    _plant_assignment(aggregator, 1, 2, submitted=False)
    aggregator._aggregate_round()

    assert aggregator._latest_scores["round"] == 2
    assert aggregator._latest_scores["scores"]["1"]["score"] == 0.0
    assert aggregator._latest_scores["scores"]["1"]["accepted"] is False


def test_n_counters_reflect_submitters_only(aggregator):
    _install_fake_scorer(aggregator)
    aggregator._agg_round = 1
    _plant_assignment(aggregator, 1, 1, submitted=True, score=0.5)    # accepted
    _plant_assignment(aggregator, 2, 1, submitted=True, score=-0.5)   # rejected
    _plant_assignment(aggregator, 3, 1, submitted=False)              # no-show

    aggregator._aggregate_round()
    payload = aggregator._latest_scores
    assert payload["n_scored"] == 2
    assert payload["n_accepted"] == 1
    assert payload["n_rejected"] == 1


def test_positive_score_excluded_from_accepted_is_not_rewarded(aggregator):
    """Reward-leak guard: a submitter with score > threshold that was dropped
    from the accepted list (dedup duplicate or version-mismatch) must report
    accepted=False. The reward path keys purely on this flag, so `score >
    threshold` alone is NOT sufficient to be paid."""
    aggregator._agg_round = 1
    a1 = _plant_assignment(aggregator, 1, 1, submitted=True, score=0.9)
    a2 = _plant_assignment(aggregator, 2, 1, submitted=True, score=0.8)
    round_assignments = [(1, a1), (2, a2)]
    scored = [(1, 1, 0.9, b"c", a1), (2, 1, 0.8, b"c", a2)]
    threshold = aggregator._effective_min_score
    # uid 2 dropped as a duplicate despite a positive score above threshold.
    accepted = [(1, 1, 0.9, b"c", a1)]
    rejected = [(2, 1, 0.8, {**a2, "_duplicate_of": 1})]
    aggregator._deliver_scores(scored, accepted, rejected, threshold, round_assignments)

    scores = aggregator._latest_scores["scores"]
    assert scores["1"]["accepted"] is True
    assert scores["2"]["accepted"] is False          # not paid
    assert scores["2"]["score"] == pytest.approx(0.8)  # score still reported


def test_pushed_val_books_are_round_keyed(aggregator):
    """The validator-pushed held-out split is authoritative and per-round;
    a round with no pushed split falls back to a (non-empty) local derivation."""
    aggregator._pushed_val_books[5] = frozenset({"3"})
    aggregator._pushed_val_books[6] = frozenset({"1"})
    assert aggregator._get_val_books(5) == {"3"}
    assert aggregator._get_val_books(6) == {"1"}
    assert aggregator._get_val_books(7)  # derived fallback, non-empty


def test_held_baseline_key_tracks_forward_window(aggregator):
    """The held-out baseline cache key must key on the FORWARD ranges the
    loader was built from, not the raw training ranges.

    Regression: two gradients with the SAME raw [ts_start, ts_end] scored while
    _max_timestamp_ns has advanced get DIFFERENT forward windows (and different
    loaders). Keying on raw ranges collided them onto one cached baseline and
    scored a miner as baseline_A - loss_after_B across mismatched held-out data.
    """
    aggregator._forward_scoring = True
    raw = [(0, 500)]

    aggregator._max_timestamp_ns = 1000
    fwd1 = aggregator._forward_val_ranges(raw)
    assert fwd1 == [(500, 1000)]
    key1 = aggregator._held_baseline_key(3, aggregator._version, fwd1)

    aggregator._max_timestamp_ns = 2000
    fwd2 = aggregator._forward_val_ranges(raw)
    assert fwd2 == [(500, 2000)]
    key2 = aggregator._held_baseline_key(3, aggregator._version, fwd2)

    # Distinct forward windows must not share a baseline entry.
    assert key1 != key2
    # Same forward window (same round/version/max_ts) reuses the same entry.
    key2_again = aggregator._held_baseline_key(
        3, aggregator._version, aggregator._forward_val_ranges(raw)
    )
    assert key2_again == key2


# ---------------------------------------------------------------------------
# Validator side: receive_scores reflects payload faithfully
# ---------------------------------------------------------------------------


class _SilentHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a, **kw):
        pass


@pytest.fixture
def http_url():
    srv = HTTPServer(("127.0.0.1", 0), _SilentHandler)
    port = srv.server_port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def test_receive_scores_reflects_zero_for_stopped_miner(http_url):
    """When the server publishes round N+1 with miner 2 at 0.0, the
    validator's in-memory score for miner 2 must read 0.0 — i.e. the
    server-authoritative model holds end-to-end."""
    from GenTRX.src.service import GenTRXService
    from GenTRX.src.state_packager import StatePackager

    s = GenTRXService(
        packager=StatePackager(),
        gradient_server_url=http_url,
        miner_uids=[1, 2],
    )

    s.receive_scores({
        "round": 1, "n_scored": 2, "n_accepted": 2,
        "scores": {
            "1": {"score": 0.7, "accepted": True, "books": ["0"]},
            "2": {"score": 0.3, "accepted": True, "books": ["1"]},
        },
    })
    assert s.get_scores()[2]["score"] == pytest.approx(0.3)

    s.receive_scores({
        "round": 2, "n_scored": 1, "n_accepted": 1,
        "scores": {
            "1": {"score": 0.6, "accepted": True, "books": ["0"]},
            "2": {"score": 0.0, "accepted": False, "books": ["1"]},
        },
    })
    assert s.get_scores()[2]["score"] == 0.0
    assert s.get_scores()[2]["accepted"] is False
