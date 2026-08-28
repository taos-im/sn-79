# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Payload compression utilities: lz4/zlib/zstd + Base64 encoding for synapse
data, with parallel batching support via ThreadPoolExecutor.
"""
import zstandard as zstd
import zlib
import lz4.frame
import pybase64
import base64
import msgspec
from typing import Literal
from concurrent.futures import ThreadPoolExecutor

compressors = {
    "zlib": zlib.compress,
    "lz4": lz4.frame.compress,
    "zstd": lambda raw, level: zstd.ZstdCompressor(level=level).compress(raw),
}

decompressors = {
    "zlib": zlib.decompress,
    "lz4": lz4.frame.decompress,
    "zstd": lambda raw: zstd.ZstdDecompressor().decompress(raw),
}

json_encoder = msgspec.json.Encoder()
msgpack_encoder = msgspec.msgpack.Encoder()

def compress(
    payload,
    level: int = 1,
    engine: Literal["zlib", "lz4", "zstd"] = "lz4",
    version: int = 45,
) -> str | None:
    """
    Compress a payload using either JSON (legacy, version < 45)
    or Msgpack (version >= 45), wrapped in Base64 text.

    Args:
        payload: The object to compress.
        level: Compression level.
        engine: Compression engine name.
        version: Protocol version selecting the codec.

    Returns:
        str: The Base64-wrapped compressed payload.
    """
    try:
        if version < 45:
            raw = json_encoder.encode(payload)
        else:
            raw = msgpack_encoder.encode(payload)

        compressed = compressors[engine](raw, level)
        return base64.standard_b64encode(compressed).decode("ascii")
    except Exception as ex:
        print(f"Failed to compress! {ex}")
        return None

def decompress(
    payload: str | dict,
    engine: Literal["zlib", "lz4", "zstd"] = "lz4",
    version: int = 45,
) -> dict | None:
    """
    Decompress payload using the correct codec depending on version.
    - version < 45 → JSON
    - version >= 45 → Msgpack
    Supports Base64-encoded transport, and old dict container format.

    Args:
        payload: The compressed payload.
        engine: Compression engine name.
        version: Protocol version selecting the codec.

    Returns:
        The decompressed object.
    """
    try:
        if isinstance(payload, str):
            decoded = pybase64.b64decode(payload)
            raw = decompressors[engine](decoded)

            if version < 45:
                return msgspec.json.decode(raw)
            return msgspec.msgpack.decode(raw)

        else:
            # Legacy container with 'payload' and 'books'
            decoded_main = pybase64.b64decode(payload["payload"])
            raw_main = decompressors[engine](decoded_main)
            if version < 45:
                decompressed_payload = msgspec.json.decode(raw_main)
            else:
                decompressed_payload = msgspec.msgpack.decode(raw_main)

            if payload.get("books"):
                decoded_books = pybase64.b64decode(payload["books"])
                raw_books = decompressors[engine](decoded_books)
                if version < 45:
                    books = msgspec.json.decode(raw_books)
                else:
                    books = msgspec.msgpack.decode(raw_books)
            else:
                books = {}

            return {"books": books, **decompressed_payload}

    except Exception as ex:
        print(f"Failed to decompress! {ex}")
        return None

def compress_batch(axon_synapses: dict, batch, compressed_books: str, level: int = 1, engine: str = "lz4", version: int = 45) -> dict:
    """
    Compress payload fields for a batch of synapse UIDs in place.

    Replaces the accounts, notices, config, and response fields on each synapse
    with a compressed dict containing 'books' and 'payload' keys.

    Args:
        axon_synapses (dict): Mapping of UID to synapse objects (mutated in place).
        batch: Iterable of UIDs in this batch to compress.
        compressed_books (str): Pre-compressed books blob shared across all UIDs.
        level (int): Compression level. Defaults to 1.
        engine (str): Compression engine ('lz4', 'zlib', or 'zstd'). Defaults to 'lz4'.
        version (int): Protocol version; < 45 uses JSON, >= 45 uses Msgpack. Defaults to 45.

    Returns:
        dict: The same `axon_synapses` mapping with compressed payloads applied.
    """
    for uid in batch:
        axon_synapses[uid].books = None
        payload = {
            "pools":    getattr(axon_synapses[uid], 'pools', None),
            "accounts": axon_synapses[uid].accounts,
            "notices":  axon_synapses[uid].notices,
            "config":   axon_synapses[uid].config,
            "response": axon_synapses[uid].response,
        }
        if hasattr(axon_synapses[uid], 'pools'):
            axon_synapses[uid].pools = None
        axon_synapses[uid].accounts = None
        axon_synapses[uid].notices  = None
        axon_synapses[uid].config   = None
        axon_synapses[uid].response = None
        axon_synapses[uid].compressed = {
            "books": compressed_books,
            "payload": compress(payload, level=level, engine=engine, version=version),
        }
    return axon_synapses

def batch_compress(
    axon_synapses: dict,
    compressed_books: str,
    batches: list[list[int]],
    level: int = 1,
    engine: str = "lz4",
    version: int = 45,
) -> dict:
    """
    Compress synapse payloads for multiple UID batches in parallel threads.

    Args:
        axon_synapses (dict): Mapping of UID to synapse objects (mutated in place).
        compressed_books (str): Pre-compressed books blob shared across all UIDs.
        batches (list[list[int]]): List of UID batches; one thread per batch.
        level (int): Compression level. Defaults to 1.
        engine (str): Compression engine ('lz4', 'zlib', or 'zstd'). Defaults to 'lz4'.
        version (int): Protocol version; < 45 uses JSON, >= 45 uses Msgpack. Defaults to 45.

    Returns:
        dict: Merged mapping of all UIDs to their compressed synapse objects.
    """
    compressed_batches = []
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        tasks = [
            pool.submit(compress_batch, axon_synapses, batch, compressed_books,  level, engine, version)
            for batch in batches
        ]
        for task in tasks:
            compressed_batches.append(task.result())
    compressed_synapses = {k: v for d in compressed_batches for k, v in d.items()}
    return compressed_synapses
