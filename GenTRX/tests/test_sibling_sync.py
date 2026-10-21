"""Siblings keep scoring on uid 0's canonical model.

Regression: a sibling only pulled uid 0's checkpoint when it had no local
file, or from the heartbeat-loss fallback; rounds closed by POST /round never
synced. With a validator bucket configured, the own-bucket block also resumed
or uploaded a v1 for the sibling, and --is-aggregator defaulted to True.

uid 0 uploads a full checkpoint (latest.json) every --checkpoint-interval
versions and a delta + head.json for every version; a sibling follows the head.

Run: pytest GenTRX/tests/test_sibling_sync.py -v
"""

import argparse
import io
import runpy
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from GenTRX.src.version import TRAIN_REGIME_VERSION


def _ckpt_bytes(state_dict=None, n_types=None):
    from dataclasses import asdict

    from GenTRX.src.model import ModelConfig, OrderModel
    from GenTRX.src.tokenizer import TokenizerConfig

    cfg = ModelConfig() if n_types is None else ModelConfig(n_types=n_types)
    buf = io.BytesIO()
    torch.save(
        {
            "model_state_dict": state_dict if state_dict is not None else OrderModel(cfg).state_dict(),
            "model_config": asdict(cfg),
            "tokenizer_config": asdict(TokenizerConfig()),
            "train_regime_version": TRAIN_REGIME_VERSION,
        },
        buf,
    )
    return buf.getvalue()


def _weights(path):
    return torch.load(str(path), map_location="cpu", weights_only=True)["model_state_dict"]


def _same_weights(a, b):
    return a.keys() == b.keys() and all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)


class _Uid0:
    """uid 0's bucket, publishing a real model the way the aggregator does."""

    def __init__(self, seed=0):
        from GenTRX.src.model import ModelConfig, OrderModel

        torch.manual_seed(seed)
        self.model = OrderModel(ModelConfig())
        self.head = 0
        self.latest = 0
        self.full = {}
        self.deltas = {}
        self.downloads = []
        self.head_reads = 0
        self.fail_checkpoint = 0

    def publish_full(self, version, data=None):
        self.full[version] = data if data is not None else _ckpt_bytes(self.model.state_dict())
        self.latest = self.head = version

    def publish_delta(self, version):
        from GenTRX.src.gradient import compress, extract_delta, serialize, snapshot_state

        before = snapshot_state(self.model)
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        comp = compress(extract_delta(before, snapshot_state(self.model)), top_k_frac=1.0)
        self.deltas[version] = serialize(comp)
        self.head = version

    def state(self):
        return self.model.state_dict()

    def get_head_version(self, uid):
        assert uid == 0
        self.head_reads += 1
        return self.head

    def get_latest_version(self, uid):
        return self.latest

    def get_latest_meta(self, uid):
        return {"version": self.latest, "train_regime_version": TRAIN_REGIME_VERSION}

    def get_checkpoint(self, uid, version):
        self.downloads.append(version)
        if self.fail_checkpoint:
            self.fail_checkpoint -= 1
            raise ConnectionError("transient")
        return self.full[version]

    def get_version_delta(self, uid, version):
        return self.deltas.get(version)


class _OwnBucket:
    """The sibling's own validator bucket; records every call made to it."""

    def __init__(self, head_version=40):
        self.head_version = head_version
        self.calls = []

    def __getattr__(self, name):
        def _call(*args, **kwargs):
            self.calls.append(name)
            return self.head_version

        return _call


class _FakeChain:
    def get_bucket(self, uid):
        assert uid == 0
        return SimpleNamespace(
            endpoint_url="http://uid0",
            bucket_name="uid0-bucket",
            access_key_id="ak",
            secret_access_key="sk",
            region="auto",
        )


@pytest.fixture
def uid0(monkeypatch):
    store = _Uid0()
    monkeypatch.setattr("GenTRX.src.gradient_store.GradientStore", lambda **kwargs: store)
    return store


def _kwargs(tmp_path, is_aggregator=False, validator_store=None, same_path=False):
    kw = dict(
        checkpoint_path=str(tmp_path / "best.pt"),
        val_data_path=str(tmp_path / "val"),
        output_path=None if same_path else str(tmp_path / "latest.pt"),
        books_per_miner=1,
        interval=60,
        window_ns=50,
        warmup_rounds=0,
        rollback=False,
        validator_store=validator_store,
        is_aggregator=is_aggregator,
        no_startup_cleanup=True,
    )
    return kw


def _make(tmp_path, **kwargs):
    from GenTRX.src.gradient_server import GradientAggregator

    agg = GradientAggregator(**_kwargs(tmp_path, **kwargs))
    agg._chain = _FakeChain()
    return agg


def _start(agg):
    """Run start() without the background loop or the parquet restores."""
    agg._aggregation_loop = lambda: None
    agg._restore_pending_rows = lambda: None
    agg._restore_written_parquets = lambda: None
    agg.start()
    agg._thread.join(timeout=5)
    agg._running = False


def _wait_for(cond, timeout=10.0):
    deadline = time.time() + timeout
    while not cond() and time.time() < deadline:
        time.sleep(0.01)
    return cond()


# ---------------------------------------------------------------------------
# Following uid 0's head
# ---------------------------------------------------------------------------


def test_follows_head_by_replaying_deltas_from_full_checkpoint(tmp_path, uid0):
    uid0.publish_full(6)
    for v in (7, 8, 9):
        uid0.publish_delta(v)
    agg = _make(tmp_path)

    assert agg._sync_from_uid0() is True

    assert uid0.downloads == [6]
    assert agg._canonical_version == 9
    assert agg._version == 9
    assert _same_weights(_weights(agg.checkpoint_path), uid0.state())


def test_advances_from_local_canonical_via_deltas(tmp_path, uid0):
    uid0.publish_full(6)
    agg = _make(tmp_path)
    assert agg._sync_from_uid0() is True
    uid0.publish_delta(7)
    uid0.publish_delta(8)

    assert agg._sync_from_uid0() is True

    assert uid0.downloads == [6]  # no second full download
    assert agg._canonical_version == 8
    assert _same_weights(_weights(agg.checkpoint_path), uid0.state())


def test_delta_gap_falls_back_to_full_checkpoint(tmp_path, uid0):
    uid0.publish_full(6)
    agg = _make(tmp_path)
    assert agg._sync_from_uid0() is True
    for v in range(7, 13):
        uid0.publish_delta(v)
    uid0.publish_full(12)
    del uid0.deltas[8]  # pruned

    assert agg._sync_from_uid0() is True

    assert uid0.downloads == [6, 12]
    assert agg._canonical_version == 12
    assert _same_weights(_weights(agg.checkpoint_path), uid0.state())


def test_follows_uid0_when_numbering_restarts(tmp_path, uid0):
    uid0.publish_full(9)
    agg = _make(tmp_path)
    assert agg._sync_from_uid0() is True

    restarted = _Uid0(seed=1)
    restarted.publish_full(1)
    uid0.__dict__.update(restarted.__dict__)

    assert agg._sync_from_uid0() is True
    assert agg._canonical_version == 1
    assert agg._version == 1
    assert _same_weights(_weights(agg.checkpoint_path), uid0.state())


@pytest.mark.parametrize("bad", ["corrupt", "n_types"])
def test_unusable_download_is_not_installed(tmp_path, uid0, bad):
    uid0.publish_full(6)
    agg = _make(tmp_path)
    assert agg._sync_from_uid0() is True
    good = _weights(agg.checkpoint_path)

    uid0.publish_full(12, data=b"not a checkpoint" if bad == "corrupt" else _ckpt_bytes(n_types=3))
    assert agg._sync_from_uid0() is False

    assert agg._canonical_version == 6
    assert agg._load_canonical_version() == 6
    assert _same_weights(_weights(agg.checkpoint_path), good)


def test_scoring_reloads_synced_model(tmp_path, uid0):
    agg = _make(tmp_path)
    agg.checkpoint_path.write_bytes(_ckpt_bytes())
    stale = agg._get_scoring_cache(12)
    assert stale is not None

    uid0.publish_full(6)
    uid0.publish_delta(7)
    assert agg._sync_from_uid0() is True

    fresh = agg._get_scoring_cache(12)  # same round: must not reuse the cache
    assert fresh is not stale
    got = {k: v.cpu() for k, v in fresh["model"].state_dict().items()}
    assert _same_weights(got, uid0.state())


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def test_startup_syncs_over_stale_local_checkpoint(tmp_path, uid0):
    agg = _make(tmp_path)
    agg.checkpoint_path.write_bytes(_ckpt_bytes())
    agg._set_canonical_version(3)
    uid0.publish_full(6)
    uid0.publish_delta(7)

    restarted = _make(tmp_path)
    _start(restarted)

    assert restarted._canonical_version == 7
    assert restarted._version == 7
    assert restarted._load_canonical_version() == 7
    assert _same_weights(_weights(restarted.checkpoint_path), uid0.state())


def test_startup_skips_download_when_current(tmp_path, uid0):
    uid0.publish_full(6)
    uid0.publish_delta(7)
    assert _make(tmp_path)._sync_from_uid0() is True

    restarted = _make(tmp_path)
    _start(restarted)

    assert uid0.downloads == [6]
    assert restarted._canonical_version == 7
    assert restarted._version == 7


def test_startup_without_local_checkpoint_ignores_stale_record(tmp_path, uid0):
    uid0.publish_full(6)
    agg = _make(tmp_path)
    assert agg._sync_from_uid0() is True
    agg.checkpoint_path.unlink()  # e.g. quarantined seed

    restarted = _make(tmp_path)
    _start(restarted)

    assert uid0.downloads == [6, 6]
    assert _same_weights(_weights(restarted.checkpoint_path), uid0.state())


def test_sibling_skips_own_bucket_resume_and_upload(tmp_path, uid0):
    own = _OwnBucket(head_version=40)
    uid0.publish_full(6)
    uid0.publish_delta(7)
    agg = _make(tmp_path, validator_store=own)

    _start(agg)

    assert "get_latest_existing_version" not in own.calls
    assert "get_head_version" not in own.calls
    assert "put_checkpoint" not in own.calls
    assert agg._version == 7  # uid 0's version, not the own bucket's 40


def test_sibling_without_uid0_does_not_upload_v1(tmp_path, monkeypatch):
    own = _OwnBucket(head_version=0)
    agg = _make(tmp_path, validator_store=own)
    agg._chain = None
    monkeypatch.setattr("GenTRX.src.chain.BucketInfo.from_aggregator_env", staticmethod(lambda: None))

    _start(agg)

    assert agg.checkpoint_path.exists()  # fresh-init fallback
    assert "put_checkpoint" not in own.calls
    assert agg._version == 0


def test_aggregator_still_resumes_from_own_bucket(tmp_path, uid0):
    own = _OwnBucket(head_version=40)
    agg = _make(tmp_path, is_aggregator=True, validator_store=own)
    local = _ckpt_bytes()
    agg.checkpoint_path.write_bytes(local)
    uid0.publish_full(99)

    _start(agg)

    assert "get_latest_existing_version" in own.calls
    assert agg._version == 40
    assert uid0.downloads == []
    assert agg.checkpoint_path.read_bytes() == local


# ---------------------------------------------------------------------------
# Per-round sync
# ---------------------------------------------------------------------------


def test_round_sync_waits_for_drain_and_retries_until_uid0_moves(tmp_path, uid0):
    uid0.publish_full(6)
    agg = _make(tmp_path)
    assert agg._sync_from_uid0() is True
    agg._canonical_retry_s = 3600.0
    agg._agg_round = 10

    agg._pending_aggregation_rounds = {9}
    assert agg._sync_sibling_for_round() is False  # round 9 still drains
    assert uid0.head_reads == 1

    agg._pending_aggregation_rounds = set()
    assert agg._sync_sibling_for_round() is False  # uid 0 has not published yet
    reads = uid0.head_reads
    assert agg._sync_sibling_for_round() is False  # throttled
    assert uid0.head_reads == reads

    agg._canonical_retry_s = 0.0
    agg._canonical_retry_at = 0.0
    uid0.publish_delta(7)
    assert agg._sync_sibling_for_round() is True
    assert agg._canonical_version == 7

    uid0.publish_delta(8)
    assert agg._sync_sibling_for_round() is False  # synced for round 10
    agg._agg_round = 11
    assert agg._sync_sibling_for_round() is True
    assert agg._canonical_version == 8


def test_round_sync_retries_after_transient_error(tmp_path, uid0):
    agg = _make(tmp_path)
    agg._canonical_retry_s = 0.0
    agg._agg_round = 10
    uid0.publish_full(6)
    uid0.fail_checkpoint = 1

    assert agg._sync_sibling_for_round() is False
    assert agg._canonical_version == 0
    assert agg._sync_sibling_for_round() is True
    assert agg._canonical_version == 6


def test_post_round_drain_then_sync(tmp_path, uid0):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from GenTRX.src.gradient_server import create_gradient_router

    kw = _kwargs(tmp_path)
    router, agg = create_gradient_router(round_grace_s=0.0, **kw)
    agg._chain = _FakeChain()
    client = TestClient(FastAPI())
    client.app.include_router(router)

    uid0.publish_full(6)
    assert agg._sync_from_uid0() is True
    agg._agg_round = 5
    agg._canonical_synced_round = 5
    agg._assignments[1] = {"round": 5, "books": ["0"], "model_version": 6, "_state": "DELIVERED"}

    events = []
    real_sync = agg._sync_from_uid0
    agg._aggregate_round = lambda: events.append(("aggregate", agg._agg_round))

    def _sync():
        ok = real_sync()
        events.append(("sync", agg._agg_round, ok))
        return ok

    agg._sync_from_uid0 = _sync
    agg._loop_sleep_s = 0.01
    agg._canonical_retry_s = 0.01
    agg._running = True
    thread = threading.Thread(target=agg._aggregation_loop, daemon=True)
    thread.start()
    try:
        uid0.publish_delta(7)
        assert client.post("/gentrx/round", json={"round": 6, "assignments": {}}).status_code == 200
        assert _wait_for(lambda: ("sync", 6, True) in events)
        assert events[0] == ("aggregate", 5)  # the closing round drains first
        assert agg._canonical_version == 7

        # Round 7 closes nothing; uid 0 publishes only after our first attempts.
        assert client.post("/gentrx/round", json={"round": 7, "assignments": {}}).status_code == 200
        assert _wait_for(lambda: ("sync", 7, False) in events)
        uid0.publish_delta(8)
        assert _wait_for(lambda: ("sync", 7, True) in events)
    finally:
        agg._running = False
        thread.join(timeout=5)

    assert agg._agg_round == 7
    assert agg._version == 8
    assert _same_weights(_weights(agg.checkpoint_path), uid0.state())


def test_aggregation_loop_never_syncs_aggregator(tmp_path):
    agg = _make(tmp_path, is_aggregator=True)
    calls = []
    agg._sync_from_uid0 = lambda: calls.append(1) or False
    agg._loop_sleep_s = 0.01
    agg._running = True
    thread = threading.Thread(target=agg._aggregation_loop, daemon=True)
    thread.start()
    time.sleep(0.2)
    agg._running = False
    thread.join(timeout=5)

    assert calls == []


# ---------------------------------------------------------------------------
# Local save after a sibling's own aggregation
# ---------------------------------------------------------------------------


def _local_ckpt_dict():
    return torch.load(io.BytesIO(_ckpt_bytes()), map_location="cpu", weights_only=True)


def test_sibling_local_save_keeps_canonical_version(tmp_path, uid0):
    uid0.publish_full(6)
    agg = _make(tmp_path)
    assert agg._sync_from_uid0() is True

    with agg._lock:
        agg._save_local_checkpoint(_local_ckpt_dict())

    assert agg.output_path.exists()
    assert agg._version == 6
    assert agg._canonical_version == 6
    assert agg._sync_from_uid0() is False  # best.pt is still uid 0's v6


def test_sibling_local_save_over_synced_file_forces_repull(tmp_path, uid0):
    uid0.publish_full(6)
    agg = _make(tmp_path, same_path=True)
    assert agg.output_path == agg.checkpoint_path
    assert agg._sync_from_uid0() is True

    with agg._lock:
        agg._save_local_checkpoint(_local_ckpt_dict())

    assert agg._version == 6
    assert agg._canonical_version == 0
    assert agg._load_canonical_version() == 0
    assert agg._sync_from_uid0() is True
    assert uid0.downloads == [6, 6]
    assert _same_weights(_weights(agg.checkpoint_path), uid0.state())


def test_aggregator_local_save_is_a_new_version(tmp_path):
    agg = _make(tmp_path, is_aggregator=True)
    agg._version = 6

    with agg._lock:
        agg._save_local_checkpoint(_local_ckpt_dict())

    assert agg._version == 7


# ---------------------------------------------------------------------------
# Role flags on the real CLI parser
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "flags, want",
    [([], False), (["--is-aggregator"], True), (["--no-is-aggregator"], False)],
)
def test_cli_role_defaults_to_sibling(monkeypatch, flags, want):
    class _Parsed(Exception):
        pass

    parsed = []
    real_parse_args = argparse.ArgumentParser.parse_args

    def _capture(self, *args, **kwargs):
        parsed.append(real_parse_args(self, *args, **kwargs))
        raise _Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _capture)
    monkeypatch.setattr(sys, "argv", ["gradient_server", "--checkpoint", "c.pt", "--val-data", "v", *flags])

    with pytest.raises(_Parsed):
        runpy.run_module("GenTRX.src.gradient_server", run_name="__main__")

    assert parsed[0].is_aggregator is want
