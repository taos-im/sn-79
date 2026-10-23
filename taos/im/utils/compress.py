# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Payload compression utilities: lz4/zlib/zstd + Base64 encoding for synapse
data, with parallel batching support via ThreadPoolExecutor.
"""
import binascii
import functools
import zstandard as zstd
import zlib
import lz4.frame
import pybase64
import base64
import msgspec
from typing import Annotated, Literal
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


# A miner's reply is read under bounds the validator sets: a reply carries at most one instruction list, each
# book at its instruction cap, and nothing else the validator uses.
REPLY_BYTES_PER_INSTRUCTION = 512   # about twice the largest encoded instruction
REPLY_BYTES_FIXED = 64 * 1024       # the response's own fields and the container around it
REPLY_BOOKS_MAX_BYTES = 1024        # a stock miner's books part is an encoded nil
# Instructions per book read beyond the per-book limit: an over-limit reply is read whole and the query loop keeps the
# first max_instructions_per_book per book, as 0.6.3 did; only a reply past this allowance is refused.
REPLY_OVERSUBMIT_PER_BOOK = 64


class ReplyRefused(ValueError):
    """A miner's reply the validator does not read; the message names the bound it passed."""


def reply_bounds(book_count: int, max_instructions_per_book: int) -> tuple[int, int]:
    """The instruction cap and the decompressed byte cap for one miner reply."""
    instructions = int(book_count) * (int(max_instructions_per_book) + REPLY_OVERSUBMIT_PER_BOOK)
    return instructions, instructions * REPLY_BYTES_PER_INSTRUCTION + REPLY_BYTES_FIXED


def _encoded_cap(max_bytes: int) -> int:
    """The longest base64 text of a compressed form of at most max_bytes (compression can add a little to
    input it cannot shrink)."""
    return (max_bytes + max_bytes // 64 + 1024) * 4 // 3 + 4


def bounded_decompress(data: bytes, engine: str, max_bytes: int) -> bytes:
    """Decompress data, refusing it as soon as the output passes max_bytes. A size declared inside the data (a
    zstd frame's content size) is never used to allocate. Data that is not a frame of the engine is refused like any
    other unreadable reply, rather than raising the engine's own error (RuntimeError, zlib.error, ZstdError)."""
    try:
        return _bounded_decompress(data, engine, max_bytes)
    except ReplyRefused:
        raise
    except (RuntimeError, ValueError, zlib.error, zstd.ZstdError) as ex:
        raise ReplyRefused(f"does not decompress as {engine}: {ex}")


def _bounded_decompress(data: bytes, engine: str, max_bytes: int) -> bytes:
    if engine == "zlib":
        d = zlib.decompressobj()
        out = d.decompress(data, max_bytes + 1)
        if len(out) > max_bytes or d.unconsumed_tail:
            raise ReplyRefused(f"decompresses past {max_bytes} bytes")
        if not d.eof:
            raise ReplyRefused("zlib stream is incomplete")
        return out
    if engine == "lz4":
        d = lz4.frame.LZ4FrameDecompressor()
        out = d.decompress(data, max_length=max_bytes + 1)
        if len(out) > max_bytes:
            raise ReplyRefused(f"decompresses past {max_bytes} bytes")
        if not d.eof:
            raise ReplyRefused("lz4 frame is incomplete")
        return out
    if engine == "zstd":
        reader = zstd.ZstdDecompressor().stream_reader(data)
        chunks, total = [], 0
        while True:
            chunk = reader.read(min(1 << 20, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ReplyRefused(f"decompresses past {max_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    raise ReplyRefused(f"unknown compression engine {engine!r}")


class _ReplyEnvelope(msgspec.Struct):
    """The one field of a reply payload the validator reads; every other field is skipped unread."""
    response: msgspec.Raw = msgspec.Raw()


@functools.lru_cache(maxsize=16)
def _instruction_counter(max_instructions: int, json_codec: bool):
    """A decoder that reads a response's instruction list as raw spans, refusing it past max_instructions from
    the array's declared length."""
    shape = msgspec.defstruct("_ReplyInstructions", [
        ("instructions", Annotated[list[msgspec.Raw], msgspec.Meta(max_length=max_instructions)], []),
    ])
    return msgspec.json.Decoder(shape) if json_codec else msgspec.msgpack.Decoder(shape)


def _decode_response(raw: bytes, json_codec: bool):
    return msgspec.json.decode(raw) if json_codec else msgspec.msgpack.decode(raw)


def _b64(text: str) -> bytes:
    try:
        return pybase64.b64decode(text)
    except (binascii.Error, ValueError) as ex:
        raise ReplyRefused(f"not base64: {ex}")


def _decode_part(text: str, engine: str, json_codec: bool, max_bytes: int):
    if len(text) > _encoded_cap(max_bytes):
        raise ReplyRefused(f"encoded text of {len(text)} characters passes its bound")
    raw = bounded_decompress(_b64(text), engine, max_bytes)
    try:
        return msgspec.json.decode(raw) if json_codec else msgspec.msgpack.decode(raw)
    except msgspec.DecodeError as ex:
        raise ReplyRefused(f"does not decode: {ex}")


def decode_reply(payload, *, engine: str, version: int, max_bytes: int, max_instructions: int) -> dict | None:
    """A miner reply's response, read under bounds; None when the reply carries no response.

    payload is the reply's compressed field: the base64 text, or the two-part container a stock miner sends
    (its books part an encoded nil). The text length is checked before decoding, the decompressed size while
    decompressing, and the instruction count from the array header before the response is decoded. Only the
    response is decoded. Raises ReplyRefused, naming the bound, for a reply the validator does not read."""
    if engine not in decompressors:
        raise ReplyRefused(f"unknown compression engine {engine!r}")
    if isinstance(payload, str):
        main, books = payload, None
    elif isinstance(payload, dict):
        main, books = payload.get("payload"), payload.get("books")
    else:
        raise ReplyRefused(f"compressed field of type {type(payload).__name__}")
    if not isinstance(main, str) or not (books is None or isinstance(books, str)):
        raise ReplyRefused("container parts are not text")
    json_codec = version < 45
    if books:
        try:
            part = _decode_part(books, engine, json_codec, REPLY_BOOKS_MAX_BYTES)
        except ReplyRefused as ex:
            raise ReplyRefused(f"books part: {ex}")
        if part not in (None, {}, []):
            raise ReplyRefused("books part is not empty")
    if len(main) > _encoded_cap(max_bytes):
        raise ReplyRefused(f"encoded text of {len(main)} characters passes its bound")
    raw = bounded_decompress(_b64(main), engine, max_bytes)
    codec = msgspec.json if json_codec else msgspec.msgpack
    try:
        response = bytes(codec.decode(raw, type=_ReplyEnvelope).response)
    except msgspec.DecodeError as ex:
        raise ReplyRefused(f"payload does not decode: {ex}")
    if response in (b"", b"\xc0", b"null"):
        return None
    try:
        _instruction_counter(int(max_instructions), json_codec).decode(response)
    except msgspec.DecodeError as ex:
        raise ReplyRefused(f"instructions refused at a cap of {max_instructions}: {ex}")
    try:
        return _decode_response(response, json_codec)
    except msgspec.DecodeError as ex:
        raise ReplyRefused(f"response does not decode: {ex}")


def read_reply(synapse, *, version: int, max_bytes: int, max_instructions: int) -> str | None:
    """Read a miner's reply into synapse.response under the bounds of decode_reply, for any state synapse that
    carries compressed, compression_engine and response. Only the response is read; the reply's other fields
    stay unset. Returns None when the reply was read, or the reason it was refused (the synapse is then left
    without a response)."""
    try:
        response = decode_reply(synapse.compressed, engine=synapse.compression_engine, version=version,
                                max_bytes=max_bytes, max_instructions=max_instructions)
    except ReplyRefused as ex:
        synapse.compressed = None
        return str(ex)
    synapse.compressed = None
    if response is None:
        return None
    try:
        synapse.response = response
    except Exception as ex:
        return f"response does not validate: {str(ex)[:300]}"
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
        axon_synapses[uid].compression_engine = engine
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
