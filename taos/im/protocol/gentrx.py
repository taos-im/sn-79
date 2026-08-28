# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import json
import typing

import bittensor as bt
from pydantic import ConfigDict, model_validator


class GenTRXAssignment(bt.Synapse):
    """Pushed from validator to miner once per GenTRX aggregation round.

    Carries the book IDs, timestamp window, and pre-resolved S3 data keys the
    miner should download and train on.  Miners that don't run GenTRXAgent
    return a non-200 status which the validator silently ignores.

    Fields match the dict returned by GradientAggregator.get_assignment().
    """

    model_config = ConfigDict(protected_namespaces=())

    # Bind the assignment payload into the signed body_hash so the miner can
    # detect tampering of the bucket/credential/window fields in transit (a MITM
    # could otherwise redirect training to a malicious bucket). The validator
    # dendrite signs body_hash; the miner axon's default verify recomputes it.
    required_hash_fields: typing.ClassVar[tuple[str, ...]] = (
        "round",
        "model_version",
        "books",
        "ts_start",
        "ts_end",
        "data",
        "data_source",
        "data_endpoint",
        "data_bucket",
        "data_access_key",
        "data_secret_key",
        "validator_uid",
    )

    # bittensor 10.2's parent validator does `values["name"] = ...` and crashes
    # when the dendrite ships the synapse as a raw-bytes body. Decode here.
    @model_validator(mode="before")
    @classmethod
    def set_name_type(cls, values):
        """Coerce the assignment's name field to a string before validation."""
        if isinstance(values, (bytes, bytearray)):
            try:
                values = json.loads(values)
            except (json.JSONDecodeError, ValueError):
                return values
        if isinstance(values, dict):
            values["name"] = cls.__name__
        return values

    round: int = 0
    model_version: int = 0
    books: list[str] = []
    ts_start: int = 0
    ts_end: int = 0
    data: list[str] = []        # S3 keys for training parquets
    data_source: str = "s3"     # "s3" or "local"

    # Validator's data bucket 
    # Enables per-validator data buckets access without chain commitments.
    data_endpoint: str = ""
    data_bucket: str = ""
    data_access_key: str = ""
    data_secret_key: str = ""

    # UID of the validator that issued this assignment.
    validator_uid: int = -1

    # Validator advice for this round (e.g. {"top_k_frac": 0.1}). The miner uses
    # it to adapt gradient sparsity; empty = use the miner's configured default.
    advice: dict = {}
