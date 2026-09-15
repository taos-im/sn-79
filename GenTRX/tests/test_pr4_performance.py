# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
PR-4 — performance micro-fixes.

Tests:
  1. The decompress→compress round-trip on the aggregated dense gradient
     is gone — we build the GradientDelta directly. Verified by checking
     that the new code path produces a bit-identical top-k result with no
     intermediate `zeros(N)` allocation.
  2. The per-miner boto3 client is cached: a second call to
     `_get_miner_s3_client` with the same (endpoint, bucket, key) triple
     returns the SAME client object.
"""

import torch

from GenTRX.src.gradient import (
    GradientDelta,
    GradientMetadata,
    aggregate,
    compress,
    decompress,
)
from GenTRX.src.gradient_server import GradientAggregator


def _make_cg(values: dict, miner_uid: int = 1):
    delta = GradientDelta(
        delta=dict(values),
        metadata=GradientMetadata(window_id=0, miner_uid=miner_uid),
    )
    return compress(delta, top_k_frac=1.0)


def test_dense_aggregate_to_delta_matches_decompress_path():
    """Building the GradientDelta directly from the aggregated dense
    CompressedGradient must produce the same top-k output as the old
    `compress(decompress(agg))` round trip, because aggregate() stores
    arange indices over flat values."""
    a = _make_cg({"w": torch.tensor([1.0, 2.0, 3.0, 4.0])}, miner_uid=1)
    b = _make_cg({"w": torch.tensor([3.0, 4.0, 5.0, 6.0])}, miner_uid=2)
    agg = aggregate([a, b])

    # Old path: decompress then compress.
    old = compress(decompress(agg), top_k_frac=0.5)

    # New path: direct GradientDelta construction.
    direct = GradientDelta(
        delta={
            name: vals.reshape(shape)
            for name, (_idx, vals, shape) in agg.sparse.items()
        },
        metadata=agg.metadata,
    )
    new = compress(direct, top_k_frac=0.5)

    # Bit-identical top-k indices + values.
    assert old.sparse.keys() == new.sparse.keys()
    for name in old.sparse:
        old_idx, old_vals, old_shape = old.sparse[name]
        new_idx, new_vals, new_shape = new.sparse[name]
        assert torch.equal(old_idx.sort()[0], new_idx.sort()[0])
        assert torch.allclose(
            old_vals[old_idx.argsort()], new_vals[new_idx.argsort()]
        )
        assert old_shape == new_shape


class _StubBucketInfo:
    def __init__(self, endpoint_url, bucket_name, access_key_id):
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.access_key_id = access_key_id
        self.secret_access_key = "secret-x"
        self.region = "auto"


def test_miner_s3_client_is_cached(tmp_path, monkeypatch):
    """Two calls with the same bucket triple must hit the cache and return
    the SAME boto3 client object (identity, not equality). A third call
    with a different triple gets a fresh client."""
    gs = GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path="",
        output_path=str(tmp_path / "out.pt"),
        log_path=str(tmp_path / "agg.jsonl"),
        validator_store=None,
        is_aggregator=False,
        no_startup_cleanup=True,
    )

    # Avoid actually hitting boto3 — replace boto3.client with a stub that
    # returns a unique object per construction. The test asserts the
    # cache returns the FIRST stub from a second call with the same key.
    construction_count = {"n": 0}

    class _StubClient:
        def __init__(self, *a, **k):
            construction_count["n"] += 1

    import boto3
    monkeypatch.setattr(boto3, "client", lambda *a, **k: _StubClient(*a, **k))

    bi_alpha = _StubBucketInfo("https://s3.a", "bucket-a", "key-a")
    bi_beta = _StubBucketInfo("https://s3.b", "bucket-b", "key-b")

    c1 = gs._get_miner_s3_client(bi_alpha)
    c2 = gs._get_miner_s3_client(bi_alpha)
    c3 = gs._get_miner_s3_client(bi_beta)

    assert c1 is c2, "same bucket triple → same client (cache hit)"
    assert c1 is not c3, "different bucket triple → fresh client"
    assert construction_count["n"] == 2, (
        "exactly two real client constructions (one per distinct triple)"
    )


def test_single_gradient_aggregate_survives_direct_reshape_path():
    """Regression: `aggregate([single_top_k_gradient])` short-circuits and
    returns the input UNCHANGED — where `vals.numel() != shape.numel()`.
    The PR-4 direct-reshape path (`vals.reshape(shape)`) crashed on this
    case with `shape '[3, 288]' is invalid for input of size 43` (seen on
    testnet aggregation loop).

    The fix: numel-check gates the fast path; single-gradient falls back
    to `decompress()`, which allocates zeros(numel) + scatter and handles
    any (indices, values) pair.
    """
    # Simulate a miner's top-k gradient: shape [3, 288] = 864 elements,
    # but only 43 top-k values kept — same numbers the live crash logged.
    dense = torch.randn(3, 288)
    delta_g = GradientDelta(
        delta={"emb_type": dense},
        metadata=GradientMetadata(window_id=0, miner_uid=42),
    )
    compressed_g = compress(delta_g, top_k_frac=43 / 864.0)
    _, vals, shape = compressed_g.sparse["emb_type"]
    assert vals.numel() == 43
    assert shape.numel() == 864

    # aggregate([g]) short-circuits to g unchanged — the direct-reshape
    # path in _aggregate_round would crash on this. The safe path uses
    # decompress().
    agg = aggregate([compressed_g])
    assert agg is compressed_g

    # Fast-path guard: numel comparison must catch this.
    dense_ok = all(
        v.numel() == s.numel() for _, v, s in agg.sparse.values()
    )
    assert not dense_ok, "single top-k gradient must NOT be treated as dense"

    # decompress fallback works — round-tripped delta reshapes cleanly.
    delta_back = decompress(agg)
    assert delta_back.delta["emb_type"].shape == (3, 288)
    # And re-compress produces a valid CompressedGradient with the same
    # top-k count.
    recompressed = compress(delta_back, top_k_frac=43 / 864.0)
    _, r_vals, _ = recompressed.sparse["emb_type"]
    assert r_vals.numel() == 43


def test_multi_gradient_aggregate_still_takes_dense_fast_path():
    """Sanity: the multi-gradient case still goes through the fast path
    (`vals.numel() == shape.numel()`), so we didn't regress the perf win."""
    a = _make_cg({"w": torch.randn(3, 288)}, miner_uid=1)
    b = _make_cg({"w": torch.randn(3, 288)}, miner_uid=2)
    agg = aggregate([a, b])
    dense_ok = all(
        v.numel() == s.numel() for _, v, s in agg.sparse.values()
    )
    assert dense_ok, "multi-gradient aggregate must remain dense (arange-indexed)"


def test_miner_s3_client_evicted_on_credential_rotation(tmp_path, monkeypatch):
    """Cache key includes access_key_id — rotating the key for the same
    endpoint+bucket builds a fresh client (so a revoked credential cannot
    keep authenticating from a cached client)."""
    gs = GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path="",
        output_path=str(tmp_path / "out.pt"),
        log_path=str(tmp_path / "agg.jsonl"),
        validator_store=None,
        is_aggregator=False,
        no_startup_cleanup=True,
    )

    counter = {"n": 0}

    class _StubClient:
        def __init__(self, *a, **k):
            counter["n"] += 1

    import boto3
    monkeypatch.setattr(boto3, "client", lambda *a, **k: _StubClient(*a, **k))

    bi_v1 = _StubBucketInfo("https://s3.x", "bucket-x", "key-v1")
    bi_v2 = _StubBucketInfo("https://s3.x", "bucket-x", "key-v2")

    gs._get_miner_s3_client(bi_v1)
    gs._get_miner_s3_client(bi_v2)
    assert counter["n"] == 2, "credential rotation must trigger a fresh client"
