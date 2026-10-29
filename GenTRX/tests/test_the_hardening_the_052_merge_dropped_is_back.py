"""The gradient server's hardening holds.

  * dedup pruning: `_processed_grad_keys` / `_failed_grad_keys` keep one key per miner per round and are pruned;
  * the locks: `_assignments_lock` and `_scoring_cache_lock` are taken where the API path and the round loop act on
    an assignment or the scoring cache;
  * the per-miner boto3 client cache: a client is reused across gradient polls;
  * the aggregation: the local aggregate is not decompressed and compressed again.

The lock tests observe the lock being held at the point the code acts, not merely that it exists.
"""

import pathlib
import threading

from GenTRX.src.gradient_server import GradientAggregator

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "gradient_server.py"


def _aggregator(tmp_path):
    return GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path="",
        output_path=str(tmp_path / "out.pt"),
        log_path=str(tmp_path / "agg.jsonl"),
        validator_store=None,
        is_aggregator=False,
        no_startup_cleanup=True,
    )


class _StubBucketInfo:
    def __init__(self, endpoint_url, bucket_name, access_key_id):
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.access_key_id = access_key_id
        self.secret_access_key = "secret-x"
        self.region = "auto"


class _RecordingLock:
    """A context manager standing in for a plain Lock, counting entries."""

    def __init__(self):
        self.entries = 0
        self.held = False
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        self.entries += 1
        self.held = True
        return self

    def __exit__(self, *exc):
        self.held = False
        self._lock.release()
        return False


# -- dedup pruning ------------------------------------------------------------------------------------------------


def test_prune_dedup_keys_drops_old_rounds(tmp_path):
    """Keys for rounds older than the retention window are removed, recent keys survive (30741709's own test)."""
    gs = _aggregator(tmp_path)
    gs._agg_round = 100
    gs._processed_grad_keys = {
        "miner_1/round_99",
        "miner_2/round_95",
        "miner_3/round_94",
        "miner_4/round_0",
        "malformed_key",
    }
    gs._failed_grad_keys = {"miner_5/round_99", "miner_6/round_10"}

    gs._prune_dedup_keys()

    assert "miner_1/round_99" in gs._processed_grad_keys
    assert "miner_2/round_95" in gs._processed_grad_keys
    assert "miner_3/round_94" not in gs._processed_grad_keys
    assert "miner_4/round_0" not in gs._processed_grad_keys
    assert "malformed_key" in gs._processed_grad_keys, "a key whose round cannot be read is kept"
    assert "miner_5/round_99" in gs._failed_grad_keys
    assert "miner_6/round_10" not in gs._failed_grad_keys


def test_a_new_rounds_assignments_prune_the_dedup_keys(tmp_path):
    """Both round paths (the loop's own advance and the validator's POST /round) create the new round's
    assignments, so that is where the sets are pruned, once per round."""
    gs = _aggregator(tmp_path)
    gs._agg_round = 100
    gs._processed_grad_keys = {"miner_1/round_99", "miner_3/round_10"}
    gs._failed_grad_keys = {"miner_6/round_10"}
    gs._discover_books = lambda: ["book_0"]
    gs._refresh_miner_buckets = lambda: None
    gs._select_books_for_miner = lambda uid: ["book_0"]
    gs._make_assignment_dict = lambda books: {"round": gs._agg_round, "_state": "PENDING", "books": books, "data": []}
    gs._apply_pages = lambda a, uid: False
    gs._create_round_assignments()
    assert gs._processed_grad_keys == {"miner_1/round_99"} and gs._failed_grad_keys == set(), (
        "a new round left the old rounds' dedup keys in place: the sets grow for the process's life"
    )


# -- the locks ----------------------------------------------------------------------------------------------------


def test_get_assignment_creates_and_transitions_under_the_assignments_lock(tmp_path):
    gs = _aggregator(tmp_path)
    held = []

    def create(uid):
        held.append(gs._assignments_lock._is_owned())
        gs._assignments[uid] = {"round": gs._agg_round, "_state": "DATA_READY", "books": [], "data": []}

    gs._create_assignment_for = create
    out = gs.get_assignment(3)
    assert out is not None and held == [True], "the on-demand create ran outside _assignments_lock"


def test_data_readiness_promotes_under_the_assignments_lock(tmp_path):
    gs = _aggregator(tmp_path)
    held = []
    gs._assignments = {1: {"round": 0, "_state": "PENDING", "books": [], "data": []}}

    def apply_pages(a, uid):
        held.append(gs._assignments_lock._is_owned())
        return False

    gs._apply_pages = apply_pages
    gs._check_data_readiness()
    assert held == [True], "the PENDING sweep ran outside _assignments_lock"


def test_round_assignments_are_created_under_the_assignments_lock(tmp_path):
    gs = _aggregator(tmp_path)
    held = []
    gs._discover_books = lambda: ["book_0"]
    gs._refresh_miner_buckets = lambda: None
    gs._select_books_for_miner = lambda uid: ["book_0"]
    gs._make_assignment_dict = lambda books: {"round": gs._agg_round, "_state": "PENDING", "books": books, "data": []}

    def apply_pages(a, uid):
        held.append(gs._assignments_lock._is_owned())
        return False

    gs._apply_pages = apply_pages
    gs._create_round_assignments()
    assert held and all(held), "a round's assignments were created outside _assignments_lock"


def test_the_scoring_cache_is_read_and_cleared_under_its_lock(tmp_path):
    gs = _aggregator(tmp_path)
    rec = _RecordingLock()
    gs._scoring_cache_lock = rec
    gs._scoring_cache = {"round": 7}
    assert gs._get_scoring_cache(7) == {"round": 7}
    assert rec.entries == 1, "a cached read did not take _scoring_cache_lock"
    gs._clear_scoring_cache()
    assert rec.entries == 2 and gs._scoring_cache is None, "the clear did not take _scoring_cache_lock"


# -- the per-miner client cache -----------------------------------------------------------------------------------


def test_miner_s3_client_is_cached(tmp_path, monkeypatch):
    """Two calls with the same bucket triple return the same client; another triple gets a fresh one."""
    gs = _aggregator(tmp_path)
    built = {"n": 0}

    class _StubClient:
        def __init__(self, *a, **k):
            built["n"] += 1

    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: _StubClient(*a, **k))
    alpha = _StubBucketInfo("https://s3.a", "bucket-a", "key-a")
    beta = _StubBucketInfo("https://s3.b", "bucket-b", "key-b")
    c1 = gs._get_miner_s3_client(alpha)
    c2 = gs._get_miner_s3_client(alpha)
    c3 = gs._get_miner_s3_client(beta)
    assert c1 is c2 and c1 is not c3
    assert built["n"] == 2


def test_miner_s3_client_evicted_on_credential_rotation(tmp_path, monkeypatch):
    """The key includes the access key, so a rotated credential builds a fresh client."""
    gs = _aggregator(tmp_path)
    built = {"n": 0}

    class _StubClient:
        def __init__(self, *a, **k):
            built["n"] += 1

    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: _StubClient(*a, **k))
    gs._get_miner_s3_client(_StubBucketInfo("https://s3.x", "bucket-x", "key-v1"))
    gs._get_miner_s3_client(_StubBucketInfo("https://s3.x", "bucket-x", "key-v2"))
    assert built["n"] == 2


def test_the_gradient_poll_takes_its_client_from_the_cache():
    src = SRC.read_text(encoding="utf-8")
    assert src.count("boto3.client(") == 1, "a miner's client is built in one place, the cache"
    i = src.index("def _get_miner_s3_client")
    assert src.index("boto3.client(", i) < src.index("\n    def ", i + 1)
    assert "self._get_miner_s3_client(bucket_info)" in src


# -- the aggregation round trip -----------------------------------------------------------------------------------


def _cg(values, uid, top_k_frac=1.0):
    import torch

    from GenTRX.src.gradient import GradientDelta, GradientMetadata, compress

    md = GradientMetadata(miner_uid=uid, window_id=0, steps_trained=1, model_v_trained=1)
    return compress(GradientDelta(delta={"w": torch.tensor(values)}, metadata=md), top_k_frac=top_k_frac)


def test_the_aggregate_of_two_or_more_is_read_without_the_round_trip():
    import torch

    from GenTRX.src.gradient import aggregate, decompress
    from GenTRX.src.gradient_server import _aggregate_delta

    agg = aggregate([_cg([1.0, 2.0, 3.0, 4.0], 1), _cg([3.0, 4.0, 5.0, 6.0], 2)])
    got = _aggregate_delta(agg, 2)
    assert torch.equal(got.delta["w"], decompress(agg).delta["w"])


def test_a_single_sparse_gradient_still_goes_through_decompress():
    """aggregate() returns a lone gradient as sent, top-k and in magnitude order, so its values are not the dense
    tensor: reshaping them, as 30741709's skip did for every round, garbles a one-gradient round."""
    import torch

    from GenTRX.src.gradient import aggregate, decompress
    from GenTRX.src.gradient_server import _aggregate_delta

    lone = aggregate([_cg([0.1, -9.0, 0.2, 7.0], 1, top_k_frac=0.5)])
    got = _aggregate_delta(lone, 1)
    assert torch.equal(got.delta["w"], decompress(lone).delta["w"])
    assert torch.equal(got.delta["w"], torch.tensor([0.0, -9.0, 0.0, 7.0]))


def test_the_local_aggregate_skips_the_decompress_compress_round_trip():
    src = SRC.read_text(encoding="utf-8")
    assert "compress(decompress(local_agg_dense)" not in src
    assert "_aggregate_delta(local_agg_dense, len(accepted))" in src
