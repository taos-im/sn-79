# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""S3-compatible storage for GenTRX distributed training.

All keys live under a two-axis prefix so a single bucket can hold both
networks (mainnet, testnet) and both training modes (simulation, exchange):

    gentrx/<network>/<mode>/...

Two bucket types, each with own credentials:

Validator bucket (typically one per validator, but multiple validators can share):
    gentrx/<net>/<mode>/checkpoints/<uid>/v{version:05d}.pt
    gentrx/<net>/<mode>/checkpoints/<uid>/latest.json   (pointer)
    gentrx/<net>/<mode>/data/<uid>/{book_id}/intervals/{ddHHMMSS}-{ddHHMMSS}.parquet
    gentrx/<net>/<mode>/proposals/<uid>/{block:08d}.grad

Per-miner bucket (typically one per miner, but multiple miners can share):
    gentrx/<net>/<mode>/gradients/<uid>/{block:08d}.grad

Scores are served in-memory via HTTP (GET /gentrx/scores), not written to S3.

network is derived from the connected subtensor: "finney" maps to "mainnet",
everything else (test, local, custom wss endpoints) maps to "testnet". mode
is operator-selected (--mode); only "simulation" has a working data path
today, "exchange" reserves the prefix for future exchange-data training.

Usage:
    # Build the prefix from the connected chain + chosen mode
    prefix = gentrx_prefix(network_from_subtensor(sub.network), "simulation")
    store = GradientStore(..., prefix=prefix)

    # Server: upload checkpoint after aggregation (uid = this validator's uid)
    store.put_checkpoint(my_uid, version=5, data=checkpoint_bytes)

    # Miner: pull a specific checkpoint version named in an assignment
    # (uid = aggregator's uid, typically 0 on mainnet)
    data = store.get_checkpoint(aggregator_uid, assignment["model_version"])

    # Miner: upload its own gradient (round_id is block // blocks_per_round)
    store.put_gradient(miner_uid=7, round_id=42, data=grad_bytes)
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Layout axes
NETWORK_MAINNET = "mainnet"
NETWORK_TESTNET = "testnet"
NETWORK_LOCALNET = "localnet"
# SN-79's canonical netuids — used as a deterministic fallback when the
# subtensor's `network` field is "unknown" (e.g. operator passed only
# --subtensor.chain_endpoint without --subtensor.network).
NETUID_MAINNET = 79
NETUID_TESTNET = 366
MODE_SIMULATION = "simulation"
MODE_EXCHANGE = "exchange"
ALLOWED_MODES = (MODE_SIMULATION, MODE_EXCHANGE)


def network_from_subtensor(name: str | None, netuid: int | None = None) -> str:
    """Map a bittensor network/endpoint identifier to mainnet/testnet/localnet.

    Resolution order (highest priority first):
      1. GENTRX_NETWORK env var — explicit operator override
         ("mainnet" / "testnet" / "localnet"). Required only when none of the
         signals below disambiguate (e.g. private testnet node).
      2. name == "finney" or "finney.opentensor.ai" URL (excluding
         "test.finney") → mainnet.
      3. name == "test" / "local" / test.finney.opentensor.ai / local loopback
         endpoint → testnet.
      4. **netuid mapping** — deterministic when the operator passed only
         --subtensor.chain_endpoint and bittensor leaves `network="unknown"`:
            netuid 79  → mainnet
            netuid 366 → testnet
            anything else → localnet (private/dev chain)
      5. Any other wss:// or ws:// endpoint with no netuid hint → mainnet
         (operator's own node, almost always mainnet).
      6. None / unrecognised → testnet.
    """
    import os as _os
    env = _os.environ.get("GENTRX_NETWORK", "").strip().lower()
    if env in ("mainnet", "main"):
        return NETWORK_MAINNET
    if env in ("testnet", "test"):
        return NETWORK_TESTNET
    if env in ("localnet", "local"):
        return NETWORK_LOCALNET
    # Explicit network names take precedence over netuid (operator was explicit).
    if name:
        if name == "finney":
            return NETWORK_MAINNET
        if name in ("test", "local"):
            return NETWORK_TESTNET
        # Known public endpoints — test.finney check must be explicit because
        # "finney.opentensor.ai" is a substring of "test.finney.opentensor.ai"
        if "finney.opentensor.ai" in name:
            return NETWORK_TESTNET if "test.finney" in name else NETWORK_MAINNET
        # Local loopback — always testnet (localnet)
        if name.startswith(("ws://127.", "ws://localhost", "wss://127.", "wss://localhost")):
            return NETWORK_TESTNET
        # Custom private-chain hosts (e.g. wss://localnet.example.org):
        # classify by host substring so netuid 79 on a seeded localnet is not
        # misread as mainnet by the netuid fallback below.
        _low = name.lower()
        if "localnet" in _low:
            return NETWORK_LOCALNET
        if "testnet" in _low:
            return NETWORK_TESTNET
    # Netuid-based fallback for ambiguous / "unknown" network names. Triggered
    # when the operator passed only --subtensor.chain_endpoint <custom-url>.
    if netuid is not None:
        if netuid == NETUID_MAINNET:
            return NETWORK_MAINNET
        if netuid == NETUID_TESTNET:
            return NETWORK_TESTNET
        return NETWORK_LOCALNET
    # Any other explicit wss:// / ws:// endpoint with no netuid hint → mainnet
    if name and name.startswith(("wss://", "ws://")):
        return NETWORK_MAINNET
    return NETWORK_TESTNET


def network_from_config(subtensor_config: Any, netuid: int | None = None) -> str:
    """Resolve the bucket-prefix network from a bittensor subtensor config.

    A local chain_endpoint overrides the `network` field; otherwise the
    `network` field decides via network_from_subtensor, with the optional
    `netuid` arg supplying a deterministic fallback for "unknown" networks.
    """
    if subtensor_config is None:
        return network_from_subtensor(None, netuid=netuid)
    chain_endpoint = getattr(subtensor_config, "chain_endpoint", "") or ""
    if any(m in chain_endpoint for m in ("localhost", "127.0.0.1", "::1")):
        return NETWORK_TESTNET
    # Custom private-chain endpoints hosted on a real domain (e.g.
    # wss://localnet.example.org) must be classified by the endpoint
    # host, NOT the netuid fallback below: a mainnet-SEEDED localnet keeps
    # netuid 79, which the fallback misreads as mainnet and points the bucket
    # prefix at gentrx/mainnet/... instead of the correct gentrx/localnet/....
    _ep = chain_endpoint.lower()
    if "localnet" in _ep:
        return NETWORK_LOCALNET
    if "testnet" in _ep:
        return NETWORK_TESTNET
    return network_from_subtensor(
        getattr(subtensor_config, "network", None), netuid=netuid
    )


def gentrx_prefix(network: str, mode: str) -> str:
    """Return the canonical key prefix for the (network, mode) shard.

    Always trailing slash so callers can concatenate with downstream
    prefixes (`checkpoints/`, `gradients/`, ...) directly.
    """
    if mode not in ALLOWED_MODES:
        raise ValueError(
            f"mode must be one of {ALLOWED_MODES}, got {mode!r}"
        )
    return f"gentrx/{network}/{mode}/"

# Keys / prefixes
#
# Layout — per-miner bucket, so miner UID is implicit. All keys are
# nested under gentrx/<network>/<mode>/ via the GradientStore prefix:
#   gentrx/<net>/<mode>/gradients/<uid>/00001234.grad   ← miner writes
#   gentrx/<net>/<mode>/checkpoints/, data/, ...  ← validator writes
#
# round_id == block // blocks_per_round (Templar-style block-keyed paths)
#
# IAM policy per miner:    Allow PutObject on gentrx/*/*/gradients/<self-uid>/*
# IAM policy per validator: Allow *        on gentrx/*/*/{checkpoints,data,proposals,scores}/<self-uid>/*
#                          Allow GetObject on gentrx/*/*/gradients/*/*
#                                          (read miner gradients across uids)
_CKPT_PREFIX = "checkpoints/{uid}/"
_CKPT_KEY = "checkpoints/{uid}/v{version:05d}.pt"
_LATEST_KEY = "checkpoints/{uid}/latest.json"
_GRAD_KEY = "gradients/{uid}/{block:08d}.grad"
_GRAD_PRUNE_PREFIX = "gradients/{uid}/"
_GRAD_BLOCK_PATTERN = "{block:08d}.grad"
_ALL_GRADS_PREFIX = "gradients/"
_SCORES_KEY = "scores/{block:08d}.json"  # retained for diagnostics tooling
_SCORES_PREFIX = "scores/"               # not written in production (HTTP-only scores)
_PROPOSAL_KEY = "proposals/{uid}/{block:08d}.grad"
_PROPOSALS_PREFIX = "proposals/"
_PROPOSAL_PRUNE_PREFIX = "proposals/{uid}/"
# Version-keyed canonical delta a miner applies to advance v(n-1) → v(n).
# Keyed by model version (not round), since version only bumps on accepted rounds.
_DELTA_KEY = "deltas/{uid}/v{version:05d}.grad"
_DELTA_PREFIX = "deltas/{uid}/"
# Head pointer: current model version, written every version. latest.json keeps
# pointing at the latest baseline checkpoint (uploaded every K versions).
_HEAD_KEY = "checkpoints/{uid}/head.json"
_DATA_PREFIX = "data/{uid}/{book_id}/intervals/"
_DATA_BOOKS_PREFIX = "data/{uid}/"
_DATA_KEY = "data/{uid}/{book_id}/intervals/{filename}"
_DATA_SIM_MARKER = "data/{uid}/.sim_id"


class GradientStore:
    """S3-compatible storage for checkpoints, gradients, and data.

    Supports both sync and async operations. Async requires aiobotocore
    (used in production). Sync uses boto3 (simpler, for testing/scripts).
    """

    def __init__(
        self,
        endpoint_url: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "auto",
        prefix: str = "",
    ) -> None:
        """Initialize store.

        Args:
            endpoint_url: S3 endpoint (e.g. "https://acct.r2.cloudflarestorage.com"
                          or Hippius S3 endpoint)
            bucket: Bucket name
            access_key: S3 access key ID
            secret_key: S3 secret access key
            region: AWS region (default "auto" for R2/Hippius)
            prefix: Optional key prefix (e.g. "gentrx/" for shared buckets)
        """
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.prefix = prefix
        self._sync_client = None

    def _key(self, template: str, **kwargs: Any) -> str:
        """Build a full S3 key with optional prefix."""
        return self.prefix + template.format(**kwargs)

    # ------------------------------------------------------------------
    # Sync client (boto3)
    # ------------------------------------------------------------------

    def _get_sync_client(self):
        if self._sync_client is None:
            import boto3
            from botocore.config import Config

            self._sync_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    # Bound socket waits so a stuck LIST response (seen on
                    # localnet minio during paginated `_restore_written_parquets`
                    # at startup) can't wedge the whole process. botocore
                    # default is unbounded; without these the gradient server
                    # would hang on `_get_sync_client().get_paginator(...)`
                    # forever instead of failing the LIST and proceeding.
                    connect_timeout=15,
                    read_timeout=60,
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                ),
            )
        return self._sync_client

    def _put_with_retry(
        self,
        key: str,
        data: bytes,
        max_attempts: int = 6,
        base_delay: float = 1.0,
    ) -> None:
        """Upload `data` to `key` with exponential backoff.

        Retries on transient S3/R2 errors: SlowDown, SlowDownWrite,
        IncompleteBody, ServiceUnavailable, RequestTimeout, and any
        ClientError whose code contains "Throttl" or "Slow".
        """
        client = self._get_sync_client()
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                client.put_object(Bucket=self.bucket, Key=key, Body=data)
                return
            except Exception as exc:
                last_exc = exc
                code = ""
                try:
                    code = exc.response["Error"]["Code"]  # type: ignore[attr-defined]
                except Exception:
                    pass
                transient = any(t in code for t in (
                    "SlowDown", "Throttl", "ServiceUnavailable",
                    "RequestTimeout", "IncompleteBody",
                )) or "SlowDown" in str(exc) or "IncompleteBody" in str(exc)
                if not transient or attempt == max_attempts - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "S3 put %s attempt %d/%d failed (%s), retrying in %.1fs",
                    key, attempt + 1, max_attempts, code or exc, delay,
                )
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def put_checkpoint(
        self,
        validator_uid: int | str,
        version: int,
        data: bytes,
        meta: dict | None = None,
    ) -> str:
        """Upload a checkpoint and update this validator's latest pointer.

        `meta` (e.g. train/spec version stamps) is merged into latest.json so
        compatibility can be checked without downloading the full checkpoint.
        """
        import json

        key = self._key(_CKPT_KEY, uid=validator_uid, version=version)
        self._put_with_retry(key, data)
        logger.info(
            "Uploaded checkpoint v%d (%d MB)", version, len(data) // (1024 * 1024)
        )

        latest = json.dumps({"version": version, "key": key, **(meta or {})}).encode()
        latest_key = self._key(_LATEST_KEY, uid=validator_uid)
        self._put_with_retry(latest_key, latest)

        return key

    def get_latest_meta(self, validator_uid: int | str) -> dict:
        """Return the parsed latest.json pointer (version + any stamps). {} if none."""
        import json

        client = self._get_sync_client()
        try:
            resp = client.get_object(
                Bucket=self.bucket,
                Key=self._key(_LATEST_KEY, uid=validator_uid),
            )
            return json.loads(resp["Body"].read())
        except client.exceptions.NoSuchKey:
            return {}
        except Exception as exc:
            logger.debug("Failed to read latest.json meta: %s", exc)
            return {}

    def get_latest_version(self, validator_uid: int | str) -> int:
        """Poll for the latest checkpoint version under <validator_uid>. Returns 0 if none."""
        import json

        client = self._get_sync_client()
        try:
            resp = client.get_object(
                Bucket=self.bucket,
                Key=self._key(_LATEST_KEY, uid=validator_uid),
            )
            data = json.loads(resp["Body"].read())
            return data.get("version", 0)
        except client.exceptions.NoSuchKey:
            return 0
        except Exception as exc:
            logger.debug("Failed to read latest.json: %s", exc)
            return 0

    def get_checkpoint(self, validator_uid: int | str, version: int) -> bytes:
        """Download a checkpoint owned by <validator_uid> at the given version."""
        client = self._get_sync_client()
        key = self._key(_CKPT_KEY, uid=validator_uid, version=version)
        resp = client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def get_latest_existing_version(self, validator_uid: int | str) -> int:
        """Scan checkpoints/<uid>/ for the highest existing version. Returns 0 if none.

        Repairs latest.json if it disagrees with what's actually on disk.
        """
        import json

        client = self._get_sync_client()
        prefix = self._key(_CKPT_PREFIX, uid=validator_uid)
        versions: list[int] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".pt"):
                    continue
                stem = key.rsplit("/", 1)[-1]  # "v00081.pt"
                try:
                    versions.append(int(stem[1:-3]))  # strip leading "v" and ".pt"
                except (ValueError, IndexError):
                    pass
        if not versions:
            return 0
        best = max(versions)
        latest = self.get_latest_version(validator_uid)
        if latest != best:
            ckpt_key = self._key(_CKPT_KEY, uid=validator_uid, version=best)
            # Preserve any version stamps already in latest.json; only the
            # pointer is stale, not the regime the checkpoints were saved under.
            meta = {
                k: v
                for k, v in self.get_latest_meta(validator_uid).items()
                if k not in ("version", "key")
            }
            payload = json.dumps({"version": best, "key": ckpt_key, **meta}).encode()
            self._put_with_retry(self._key(_LATEST_KEY, uid=validator_uid), payload)
            logger.info("Repaired stale latest.json: %d → %d", latest, best)
        return best

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------

    def put_gradient(self, miner_uid: int, round_id: int, data: bytes) -> str:
        """Upload a compressed gradient for a round.

        UID is part of the key path so several miners can share a bucket
        (benchmark agents pooled under one operator's R2 account, etc.)
        without colliding on the same round file.
        """
        key = self._key(_GRAD_KEY, uid=miner_uid, block=round_id)
        self._put_with_retry(key, data)
        logger.debug(
            "Uploaded gradient: miner=%d round=%d key=%s (%.1f KB)",
            miner_uid,
            round_id,
            key,
            len(data) / 1024,
        )
        return key

    def get_gradient_key(self, miner_uid: int, block_id: int) -> str:
        """Return the S3 key for a miner's gradient for a given block."""
        return self._key(_GRAD_KEY, uid=miner_uid, block=block_id)

    def list_round_gradients(self, miner_uid: int, round_id: int) -> list[str]:
        """Return the gradient key for `(miner_uid, round_id)` if it exists."""
        client = self._get_sync_client()
        key = self._key(_GRAD_KEY, uid=miner_uid, block=round_id)
        try:
            client.head_object(Bucket=self.bucket, Key=key)
            return [key]
        except Exception:
            return []

    def get_gradient(self, key: str) -> bytes:
        """Download a gradient by its S3 key."""
        client = self._get_sync_client()
        resp = client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def get_round_gradients(self, miner_uid: int, round_id: int) -> list[tuple[str, bytes]]:
        """Download all gradients for `(miner_uid, round_id)`. Returns [(key, data), ...]."""
        keys = self.list_round_gradients(miner_uid, round_id)
        results = []
        for key in keys:
            try:
                data = self.get_gradient(key)
                results.append((key, data))
            except Exception as exc:
                logger.warning("Failed to download %s: %s", key, exc)
        return results

    # ------------------------------------------------------------------
    # Proposals (aggregation-of-aggregations)
    # ------------------------------------------------------------------

    def put_proposal(self, validator_uid: int | str, round_id: int, data: bytes) -> str:
        """Upload a proposed gradient delta for a round.

        UID is part of the path so several validators can share a bucket
        without colliding on the same round file. Aggregator reads each
        sibling's proposal at the sibling's own uid path.
        """
        key = self._key(_PROPOSAL_KEY, uid=validator_uid, block=round_id)
        self._put_with_retry(key, data)
        logger.debug("Uploaded proposal for round %d (%.1f KB)", round_id, len(data) / 1024)
        return key

    def get_proposal(self, validator_uid: int | str, round_id: int) -> bytes | None:
        """Download a proposed gradient delta for a round. Returns None if not found."""
        client = self._get_sync_client()
        try:
            key = self._key(_PROPOSAL_KEY, uid=validator_uid, block=round_id)
            resp = client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Canonical version deltas (miner model-advance) + head pointer
    # ------------------------------------------------------------------

    def put_version_delta(self, validator_uid: int | str, version: int, data: bytes) -> str:
        """Upload the canonical delta that advanced the model to `version`."""
        key = self._key(_DELTA_KEY, uid=validator_uid, version=version)
        self._put_with_retry(key, data)
        logger.debug("Uploaded version delta v%d (%.1f KB)", version, len(data) / 1024)
        return key

    def get_version_delta(self, validator_uid: int | str, version: int) -> bytes | None:
        """Download the canonical delta for `version`. None if absent (e.g. pruned)."""
        client = self._get_sync_client()
        try:
            key = self._key(_DELTA_KEY, uid=validator_uid, version=version)
            resp = client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except Exception:
            return None

    def prune_version_deltas(self, validator_uid: int | str, keep: int) -> int:
        """Keep the newest `keep` version deltas under <uid>; delete older. Returns deleted count."""
        if keep <= 0:
            return 0
        client = self._get_sync_client()
        prefix = self._key(_DELTA_PREFIX, uid=validator_uid)
        versions: list[tuple[int, str]] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                stem = obj["Key"].rsplit("/", 1)[-1]
                try:
                    versions.append((int(stem[1:-5]), obj["Key"]))  # strip "v" and ".grad"
                except (ValueError, IndexError):
                    pass
        versions.sort()
        stale = versions[:-keep] if len(versions) > keep else []
        for _, key in stale:
            try:
                client.delete_object(Bucket=self.bucket, Key=key)
            except Exception as exc:
                logger.debug("Failed to delete stale delta %s: %s", key, exc)
        return len(stale)

    def put_head_version(self, validator_uid: int | str, version: int, meta: dict | None = None) -> None:
        """Write the head-pointer (current model version), updated every version."""
        import json

        key = self._key(_HEAD_KEY, uid=validator_uid)
        self._put_with_retry(key, json.dumps({"version": version, **(meta or {})}).encode())

    def get_head_version(self, validator_uid: int | str) -> int:
        """Read the head-pointer version. 0 if absent."""
        return int(self.get_head_meta(validator_uid).get("version", 0))

    def get_head_meta(self, validator_uid: int | str) -> dict:
        """Read the head-pointer JSON (version + optional state_hash). {} if absent."""
        import json

        client = self._get_sync_client()
        try:
            resp = client.get_object(Bucket=self.bucket, Key=self._key(_HEAD_KEY, uid=validator_uid))
            return json.loads(resp["Body"].read())
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Training data (parquets)
    # ------------------------------------------------------------------

    def put_data(
        self, validator_uid: int | str, book_id: int, filename: str, data: bytes
    ) -> str:
        """Upload a parquet file for a book under <validator_uid>'s data shard."""
        key = self._key(_DATA_KEY, uid=validator_uid, book_id=book_id, filename=filename)
        self._put_with_retry(key, data)
        logger.debug(
            "Uploaded data: book=%d %s (%.1f KB)", book_id, filename, len(data) / 1024
        )
        return key

    def get_data(
        self, validator_uid: int | str, book_id: int, filename: str
    ) -> bytes:
        """Download a parquet file from <validator_uid>'s data shard."""
        client = self._get_sync_client()
        key = self._key(_DATA_KEY, uid=validator_uid, book_id=book_id, filename=filename)
        resp = client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def list_data(self, validator_uid: int | str, book_id: int) -> list[str]:
        """List parquet filenames under <validator_uid>'s data shard for one book."""
        client = self._get_sync_client()
        prefix = self._key(_DATA_PREFIX, uid=validator_uid, book_id=book_id)
        filenames = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                # Extract filename from full key
                fname = obj["Key"].split("/")[-1]
                filenames.append(fname)
        return filenames

    def put_sim_marker(self, validator_uid: int | str, sim_id: str) -> None:
        """Write data/<uid>/.sim_id marker so future processes can check bucket lineage."""
        key = self._key(_DATA_SIM_MARKER, uid=validator_uid)
        self._put_with_retry(key, sim_id.encode("utf-8"))

    def get_sim_marker(self, validator_uid: int | str) -> str | None:
        """Read data/<uid>/.sim_id; return None if missing or empty."""
        client = self._get_sync_client()
        key = self._key(_DATA_SIM_MARKER, uid=validator_uid)
        try:
            resp = client.get_object(Bucket=self.bucket, Key=key)
        except Exception:
            return None
        raw = resp["Body"].read()
        if not raw:
            return None
        return raw.decode("utf-8").strip() or None

    def list_books(self, validator_uid: int | str) -> list[int]:
        """List book IDs that have data under <validator_uid>'s data shard."""
        client = self._get_sync_client()
        prefix = self._key(_DATA_BOOKS_PREFIX, uid=validator_uid)
        book_ids = set()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
            Delimiter="/",
        ):
            for cp in page.get("CommonPrefixes", []):
                # e.g. "data/<uid>/3/" → extract 3
                part = cp["Prefix"].rstrip("/").split("/")[-1]
                try:
                    book_ids.add(int(part))
                except ValueError:
                    pass
        return sorted(book_ids)

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under `prefix` in this bucket. Returns count deleted.

        Used for the sim-transition `data/` cleanup: the aggregator-owned
        bucket wipes its stale parquets on sim end so the new sim starts
        with a clean tree. Batched in groups of 1000 (S3 `delete_objects`
        hard limit). No-op and returns 0 if nothing matches.
        """
        client = self._get_sync_client()
        full_prefix = self._key(prefix)
        deleted = 0
        paginator = client.get_paginator("list_objects_v2")
        batch: list[dict] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                batch.append({"Key": obj["Key"]})
                if len(batch) >= 1000:
                    client.delete_objects(
                        Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True}
                    )
                    deleted += len(batch)
                    batch = []
        if batch:
            client.delete_objects(
                Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True}
            )
            deleted += len(batch)
        return deleted

    def prune_keep_latest(
        self, prefix: str, keep: int, suffix: str = ""
    ) -> int:
        """Keep the `keep` newest objects under `prefix`, delete the rest.

        Newest is determined by lexicographic key sort — works for our
        zero-padded round / version filenames (e.g. `00000042.grad`,
        `v00010.pt`). Pass `suffix` to filter by extension so e.g.
        `checkpoints/latest.json` is left alone when pruning `.pt` files.

        keep<=0 disables pruning entirely (no-op, returns 0). Used by:
          - miner write bucket → `gradients/`
          - aggregator validator → `checkpoints/` and `proposals/`
        """
        if keep <= 0:
            return 0
        client = self._get_sync_client()
        full_prefix = self._key(prefix)
        keys: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if suffix and not key.endswith(suffix):
                    continue
                keys.append(key)
        if len(keys) <= keep:
            return 0
        keys.sort()  # zero-padded names sort chronologically
        to_delete = keys[:-keep]
        deleted = 0
        for i in range(0, len(to_delete), 1000):
            batch = [{"Key": k} for k in to_delete[i : i + 1000]]
            client.delete_objects(
                Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True}
            )
            deleted += len(batch)
        return deleted



# ---------------------------------------------------------------------------
# Multi-bucket gradient collection (production: per-miner buckets)
# ---------------------------------------------------------------------------


def collect_miner_gradients(
    miner_buckets: dict[int, Any],
    round_id: int,
    region: str = "auto",
    prefix: str = "",
) -> list[tuple[int, bytes]]:
    """Read gradients from per-miner S3 buckets.

    In production, each miner owns their own S3 bucket and commits read
    credentials on-chain. The validator uses those credentials to pull
    gradients.

    Args:
        miner_buckets: {uid: BucketInfo} from GenTRXChain.get_miner_buckets()
        round_id: Aggregation round to collect
        region: S3 region for client creation
        prefix: Key prefix in miner buckets, e.g.
            "gentrx/<network>/<mode>/" — typically built via
            gentrx_prefix(...). Required; pass "" only for tests against
            an unprefixed bucket layout.

    Returns:
        List of (miner_uid, gradient_bytes) for successful reads.
    """
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    results: list[tuple[int, bytes]] = []
    for uid, bucket_info in miner_buckets.items():
        grad_key = f"{prefix}gradients/{uid}/{round_id:08d}.grad"
        try:
            client = boto3.client(
                "s3",
                endpoint_url=bucket_info.endpoint_url,
                aws_access_key_id=bucket_info.access_key_id,
                aws_secret_access_key=bucket_info.secret_access_key,
                region_name=region,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 2, "mode": "adaptive"},
                    connect_timeout=5,
                    read_timeout=10,
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                ),
            )
            bucket_name = bucket_info.bucket_name
            resp = client.get_object(Bucket=bucket_name, Key=grad_key)
            data = resp["Body"].read()
            results.append((uid, data))
            logger.debug(
                "Read gradient from miner %d bucket (%.1f KB)",
                uid,
                len(data) / 1024,
            )
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                # No gradient for this round — normal
                continue
            logger.debug("Miner %d gradient read failed: %s", uid, exc)
        except Exception as exc:
            logger.debug("Miner %d gradient read failed: %s", uid, exc)

    if results:
        logger.info(
            "Collected %d gradients from miner buckets (round %d)",
            len(results),
            round_id,
        )
    return results


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_validator_store_from_env(
    mode: str = "read",
    prefix: str = "",
) -> GradientStore | None:
    """Create this validator's own bucket store from GENTRX_VALIDATOR_S3_* env vars.

    Points at the bucket the validator owns and writes to (checkpoints/,
    data/, proposals/). Used by every validator for its own bucket;
    miners do not set these.

    Required env vars:
        GENTRX_VALIDATOR_S3_ENDPOINT_URL  (or derived from account_id for R2/Hippius)
        GENTRX_VALIDATOR_S3_BUCKET
        GENTRX_VALIDATOR_S3_{READ|WRITE}_ACCESS_KEY
        GENTRX_VALIDATOR_S3_{READ|WRITE}_SECRET_KEY

    Args:
        mode: "read" or "write" (which credential pair to load).
        prefix: Key prefix to apply on this store. Pass the result of
            gentrx_prefix(network, training_mode) so reads/writes land
            under gentrx/<network>/<training_mode>/...

    Returns None if required vars are missing.
    """
    if mode not in ("read", "write"):
        raise ValueError(f"mode must be 'read' or 'write', got {mode!r}")
    return _store_from_env_prefix(
        "GENTRX_VALIDATOR_S3",
        access_suffix=f"{mode.upper()}_ACCESS_KEY",
        secret_suffix=f"{mode.upper()}_SECRET_KEY",
        prefix=prefix,
    )


def create_aggregator_store_from_env(
    prefix: str = "",
) -> GradientStore | None:
    """Create a read-only store for uid-0's aggregator bucket from GENTRX_AGGREGATOR_S3_* env vars.

    Used as a chain-discovery fallback by miners and sibling validators
    so that checkpoint reads keep working when the chain commitment has
    not propagated yet. Only read credentials are expected here; write
    creds live on the uid-0 operator host.

    Required env vars:
        GENTRX_AGGREGATOR_S3_BUCKET
        GENTRX_AGGREGATOR_S3_ENDPOINT_URL  (or derived from account_id for R2/Hippius)
        GENTRX_AGGREGATOR_S3_READ_ACCESS_KEY  (or GENTRX_AGGREGATOR_S3_ACCESS_KEY)
        GENTRX_AGGREGATOR_S3_READ_SECRET_KEY  (or GENTRX_AGGREGATOR_S3_SECRET_KEY)

    Returns None if required vars are missing.
    """
    return _store_from_env_prefix(
        "GENTRX_AGGREGATOR_S3",
        access_suffix="READ_ACCESS_KEY",
        secret_suffix="READ_SECRET_KEY",
        prefix=prefix,
    )


def _store_from_env_prefix(
    env_prefix: str,
    access_suffix: str = "ACCESS_KEY",
    secret_suffix: str = "SECRET_KEY",
    prefix: str = "",
) -> GradientStore | None:
    """Generic store factory from {env_prefix}_* env vars."""
    import os

    endpoint = os.environ.get(f"{env_prefix}_ENDPOINT_URL")
    bucket = os.environ.get(f"{env_prefix}_BUCKET")
    access = os.environ.get(f"{env_prefix}_{access_suffix}")
    secret = os.environ.get(f"{env_prefix}_{secret_suffix}")
    # Fall back to unprefixed access/secret if mode-specific not set
    # (useful for localnet where read==write)
    if not access:
        access = os.environ.get(f"{env_prefix}_ACCESS_KEY")
    if not secret:
        secret = os.environ.get(f"{env_prefix}_SECRET_KEY")
    region = os.environ.get(f"{env_prefix}_REGION", "auto")

    if not all([endpoint, bucket, access, secret]):
        return None

    return GradientStore(
        endpoint_url=endpoint,
        bucket=bucket,
        access_key=access,
        secret_key=secret,
        region=region,
        prefix=prefix,
    )
