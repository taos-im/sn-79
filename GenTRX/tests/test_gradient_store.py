"""Tests for GradientStore — key formatting, proposal round-trip, checkpoint versioning.

Uses a mock S3 client (no real S3/MinIO needed). Tests verify that the correct
keys are generated and that put/get round-trips work at the API level.

Run: pytest GenTRX/tests/test_gradient_store.py -v
"""

import io

import pytest

from GenTRX.src.gradient_store import GradientStore, create_validator_store_from_env


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeExceptions:
    """Mock boto3 client.exceptions for NoSuchKey etc."""
    class NoSuchKey(Exception):
        pass

class FakeS3:
    """In-memory S3 mock — stores objects as {key: bytes}."""

    def __init__(self):
        self._objects: dict[str, bytes] = {}
        self.exceptions = _FakeExceptions()

    def put_object(self, Bucket, Key, Body, **kw):
        self._objects[f"{Bucket}/{Key}"] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key, **kw):
        full = f"{Bucket}/{Key}"
        if full not in self._objects:
            raise self.exceptions.NoSuchKey(f"Key not found: {Key}")
        return {"Body": io.BytesIO(self._objects[full])}

    def head_object(self, Bucket, Key, **kw):
        full = f"{Bucket}/{Key}"
        if full not in self._objects:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": len(self._objects[full])}

    def list_objects_v2(self, Bucket, Prefix="", **kw):
        contents = []
        for key in self._objects:
            bkt, obj_key = key.split("/", 1)
            if bkt == Bucket and obj_key.startswith(Prefix):
                contents.append({"Key": obj_key, "Size": len(self._objects[key])})
        return {"Contents": contents, "KeyCount": len(contents)}

    def get_paginator(self, operation):
        s3 = self

        class FakePaginator:
            def paginate(self_, **kw):
                yield s3.list_objects_v2(**kw)

        return FakePaginator()


@pytest.fixture
def store():
    """GradientStore with a mocked S3 client."""
    s = GradientStore(
        endpoint_url="http://fake:9000",
        bucket="test-bucket",
        access_key="key",
        secret_key="secret",
    )
    fake = FakeS3()
    s._sync_client = fake
    return s, fake


# ---------------------------------------------------------------------------
# Key format tests
# ---------------------------------------------------------------------------


def test_checkpoint_key_format(store):
    s, _ = store
    key = s._key("checkpoints/{uid}/v{version:05d}.pt", uid=1, version=42)
    assert key == "checkpoints/1/v00042.pt"


def test_gradient_key_format(store):
    s, _ = store
    key = s.get_gradient_key(7, 1234)
    assert key == "gradients/7/00001234.grad"


def test_proposal_key_format(store):
    s, _ = store
    key = s._key("proposals/{uid}/{block:08d}.grad", uid=2, block=99)
    assert key == "proposals/2/00000099.grad"


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_put_get_checkpoint(store):
    s, _ = store
    data = b"fake checkpoint data"
    s.put_checkpoint(validator_uid=0, version=1, data=data)
    result = s.get_checkpoint(validator_uid=0, version=1)
    assert result == data


def test_get_latest_version(store):
    s, _ = store
    s.put_checkpoint(validator_uid=0, version=1, data=b"v1")
    s.put_checkpoint(validator_uid=0, version=2, data=b"v2")
    # latest.json should point to version 2
    assert s.get_latest_version(validator_uid=0) == 2


def test_get_latest_version_empty(store):
    s, _ = store
    assert s.get_latest_version(validator_uid=0) == 0


# ---------------------------------------------------------------------------
# Proposal round-trip
# ---------------------------------------------------------------------------


def test_put_get_proposal(store):
    s, _ = store
    data = b"compressed gradient delta"
    key = s.put_proposal(validator_uid=1, round_id=5, data=data)
    assert "00000005" in key
    result = s.get_proposal(validator_uid=1, round_id=5)
    assert result == data


def test_get_proposal_not_found(store):
    s, _ = store
    result = s.get_proposal(validator_uid=1, round_id=999)
    assert result is None


# ---------------------------------------------------------------------------
# Gradient round-trip
# ---------------------------------------------------------------------------


def test_put_get_gradient(store):
    s, _ = store
    data = b"gradient bytes"
    s.put_gradient(miner_uid=3, round_id=10, data=data)
    result = s.get_gradient("gradients/3/00000010.grad")
    assert result == data


def test_list_round_gradients_found(store):
    s, _ = store
    s.put_gradient(miner_uid=0, round_id=7, data=b"grad")
    keys = s.list_round_gradients(miner_uid=0, round_id=7)
    assert len(keys) == 1
    assert "00000007" in keys[0]


def test_list_round_gradients_empty(store):
    s, _ = store
    keys = s.list_round_gradients(miner_uid=0, round_id=999)
    assert keys == []


# ---------------------------------------------------------------------------
# Data (parquets)
# ---------------------------------------------------------------------------


def test_put_get_data(store):
    s, _ = store
    data = b"parquet bytes"
    s.put_data(validator_uid=0, book_id=3, filename="00000000-00000300.parquet", data=data)
    result = s.get_data(validator_uid=0, book_id=3, filename="00000000-00000300.parquet")
    assert result == data


def test_list_data(store):
    s, _ = store
    s.put_data(validator_uid=0, book_id=1, filename="a.parquet", data=b"a")
    s.put_data(validator_uid=0, book_id=1, filename="b.parquet", data=b"b")
    filenames = s.list_data(validator_uid=0, book_id=1)
    assert set(filenames) == {"a.parquet", "b.parquet"}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_create_validator_store_from_env_missing_vars():
    """Returns None when env vars are not set."""
    import os
    # Clear any existing vars
    env_backup = {}
    for k in list(os.environ.keys()):
        if k.startswith("GENTRX_VALIDATOR_S3"):
            env_backup[k] = os.environ.pop(k)
    try:
        result = create_validator_store_from_env()
        assert result is None
    finally:
        os.environ.update(env_backup)


def test_create_validator_store_from_env_with_vars():
    """Returns a GradientStore when env vars are set."""
    import os
    env = {
        "GENTRX_VALIDATOR_S3_ENDPOINT_URL": "http://localhost:9000",
        "GENTRX_VALIDATOR_S3_BUCKET": "test",
        "GENTRX_VALIDATOR_S3_READ_ACCESS_KEY": "key",
        "GENTRX_VALIDATOR_S3_READ_SECRET_KEY": "secret",
    }
    old = {}
    for k, v in env.items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        result = create_validator_store_from_env(mode="read")
        assert result is not None
        assert result.bucket == "test"
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Checkpoint version stamps in latest.json
# ---------------------------------------------------------------------------


def test_put_checkpoint_stamps_meta_in_latest(store):
    s, _ = store
    s.put_checkpoint(
        validator_uid=0, version=1, data=b"v1",
        meta={"train_regime_version": 2, "taos_spec_version": 44},
    )
    meta = s.get_latest_meta(validator_uid=0)
    assert meta["version"] == 1
    assert meta["train_regime_version"] == 2
    assert meta["taos_spec_version"] == 44


def test_get_latest_meta_empty(store):
    s, _ = store
    assert s.get_latest_meta(validator_uid=0) == {}


def test_put_checkpoint_without_meta_has_no_stamp(store):
    s, _ = store
    s.put_checkpoint(validator_uid=0, version=1, data=b"v1")
    meta = s.get_latest_meta(validator_uid=0)
    assert meta["version"] == 1
    assert "train_regime_version" not in meta


def test_repair_preserves_version_stamp(store):
    s, fake = store
    # v1 written with a stamp; latest.json points at v1.
    s.put_checkpoint(
        validator_uid=0, version=1, data=b"v1",
        meta={"train_regime_version": 2},
    )
    # An orphan v2.pt appears on disk without updating latest.json.
    fake._objects["test-bucket/checkpoints/0/v00002.pt"] = b"v2"
    best = s.get_latest_existing_version(validator_uid=0)
    assert best == 2
    meta = s.get_latest_meta(validator_uid=0)
    assert meta["version"] == 2
    assert meta["train_regime_version"] == 2
