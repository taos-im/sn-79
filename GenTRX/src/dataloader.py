# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""DataLoader for order stream parquets — per-field inputs and labels.

Lazy chunk-indexed loading: only parquet metadata is read at init time.
File contents are loaded, tokenized, and cached on demand via LRU.

Uses ChunkSampler for training: shuffles file order each epoch, then yields
sequential windows within each file. This gives near-random training with
optimal cache utilization (one file loaded at a time, no thrashing).

"""

from __future__ import annotations

import bisect
import json
import logging
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from GenTRX.src.tokenizer import OrderTokenizer, TokenizerConfig

logger = logging.getLogger(__name__)

# Field names that are both input embeddings and prediction targets
PRED_FIELDS = [
    "order_types",
    "price_bins",
    "vol_int_bins",
    "vol_dec_bins",
    "interval_bins",
]
# Label key mapping: field array name → model head name
LABEL_KEYS = {
    "order_types": "order_type",
    "price_bins": "price",
    "vol_int_bins": "vol_int",
    "vol_dec_bins": "vol_dec",
    "interval_bins": "interval",
}
# Conditioning fields (input only, not predicted)
COND_FIELDS = ["lob_volumes", "time_of_day", "mid_deltas"]

ALL_FIELDS = PRED_FIELDS + COND_FIELDS


def _discover_parquets(
    data_dir: Path,
    split: str | None = None,
    max_books: int | None = None,
) -> list[Path]:
    """Find order parquets for a given split.

    Supports three data layouts:
      1. Sim layout with splits.json: <data_dir>/<book_id>/intervals/*.parquet
         Uses splits.json at data_dir level for train/val/test split selection.
      2. Live layout (GenTRXAgent): <data_dir>/book_NNNN/intervals/*.parquet
         Auto-discovered when no splits.json present. Book dirs matching
         ``book_*`` with an ``intervals/`` subdirectory are found automatically.
         When ``split`` is requested, books are sorted and split 80/10/10.
      3. Flat layout: <data_dir>/**/*.parquet (legacy fallback)

    Args:
        max_books: If set, only use the first N book_ids from the split.
                   Useful for quick training runs on a subset of the data.
    """
    splits_path = data_dir / "splits.json"

    # --- Layout 1: splits.json present ---
    if splits_path.exists() and split is not None:
        manifest = json.loads(splits_path.read_text())
        splits_data = manifest.get("splits", manifest)
        if split not in splits_data:
            raise ValueError(
                f"Split '{split}' not in {splits_path} (available: {list(splits_data)})"
            )
        book_ids = splits_data[split].get("book_ids", [])
        if max_books is not None:
            book_ids = book_ids[:max_books]
            logger.info(
                "Using %d/%d books for split '%s'",
                len(book_ids),
                len(splits_data[split].get("book_ids", [])),
                split,
            )
        files = []
        for bid in book_ids:
            book_dir = data_dir / str(bid) / "intervals"
            files.extend(sorted(book_dir.glob("*.parquet")))
        return files

    # --- Layout 2: live data (book_NNNN/intervals/*.parquet) ---
    book_dirs = sorted(
        d
        for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("book_") and (d / "intervals").is_dir()
    )
    if book_dirs:
        if split is not None:
            book_dirs = _auto_split(book_dirs, split)
        if max_books is not None:
            book_dirs = book_dirs[:max_books]
            logger.info(
                "Live layout: using %d books for split '%s'", len(book_dirs), split
            )
        files = []
        for bd in book_dirs:
            files.extend(sorted((bd / "intervals").glob("*.parquet")))
        if files:
            return files

    # --- Layout 3: flat fallback ---
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquets found under {data_dir}")
    return files


def _auto_split(
    book_dirs: list[Path], split: str, train_frac: float = 0.8, val_frac: float = 0.1
) -> list[Path]:
    """Split sorted book dirs into train/val/test (80/10/10 by default)."""
    n = len(book_dirs)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    if split == "train":
        return book_dirs[:n_train]
    elif split == "val":
        return book_dirs[n_train : n_train + n_val]
    elif split == "test":
        return book_dirs[n_train + n_val :]
    else:
        logger.warning("Unknown split '%s', returning all books", split)
        return book_dirs


class OrderDataset(Dataset):
    """Chunk-indexed sliding-window dataset with LRU cache.

    Init reads only parquet metadata (row counts) — O(n_files) but no data loaded.
    __getitem__ lazily loads + tokenizes the needed file, caching the last
    ``max_cached`` files in memory. Consecutive windows from the same file
    are a cache hit.

    Each window is seq_len orders. Labels are the next-step fields (shifted by 1).
    """

    def __init__(
        self,
        parquet_files: list[Path],
        seq_len: int = 1024,
        tokenizer: OrderTokenizer | None = None,
        max_cached: int = 8,
        asset_refs: dict[str, dict] | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.tokenizer = tokenizer or OrderTokenizer()
        self._max_cached = max_cached
        # Optional per-file cross-asset reference: {str(path): ref}. When set,
        # each file is tokenized with its own asset_ref (mixed-pair training).
        self._asset_refs = asset_refs

        if not parquet_files:
            raise FileNotFoundError("No parquet files provided")

        # Index phase: read metadata only (fast, no column data loaded)
        self._files: list[Path] = []
        self._file_lengths: list[int] = []
        self._cum_offsets: list[int] = [0]

        for f in parquet_files:
            meta = pq.read_metadata(f)
            n_rows = meta.num_rows
            if n_rows == 0:
                continue
            self._files.append(f)
            self._file_lengths.append(n_rows)
            self._cum_offsets.append(self._cum_offsets[-1] + n_rows)

        self.total_orders = self._cum_offsets[-1]
        # Need seq_len + 1 consecutive orders (seq_len input + 1 shifted label)
        self.n_windows = max(0, self.total_orders - self.seq_len)

        if self.n_windows == 0:
            raise ValueError(
                f"Not enough data: {self.total_orders} orders < seq_len {self.seq_len}"
            )

        # LRU cache: file_idx → {field_name: np.ndarray}
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(
            "Indexed %d files, %d orders, %d windows (seq_len=%d, cache=%d files)",
            len(self._files),
            self.total_orders,
            self.n_windows,
            self.seq_len,
            self._max_cached,
        )

    def _load_chunk(self, file_idx: int) -> dict[str, np.ndarray]:
        """Load, tokenize, and cache a single parquet file."""
        if file_idx in self._cache:
            self._cache.move_to_end(file_idx)
            self._cache_hits += 1
            return self._cache[file_idx]

        self._cache_misses += 1
        # Read via pyarrow (system malloc, no caching pool) and convert
        # columns to numpy upfront. Using polars's pl.read_parquet retains
        # buffers in its bundled mimalloc pool and leaks ~600 MB per cycle
        # in the miner training loop.
        table = pq.read_table(self._files[file_idx])
        cols: dict[str, np.ndarray] = {
            name: table.column(name).to_numpy(zero_copy_only=False)
            for name in table.column_names
        }
        del table
        ref = (
            self._asset_refs.get(str(self._files[file_idx]))
            if self._asset_refs else None
        )
        encoded = self.tokenizer.encode_columns(cols, asset_ref=ref)
        del cols

        # Evict oldest if cache is full
        while len(self._cache) >= self._max_cached:
            self._cache.popitem(last=False)

        self._cache[file_idx] = encoded
        return encoded

    @property
    def cache_hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0

    def _resolve_file(self, global_idx: int) -> tuple[int, int]:
        """Map global order index → (file_idx, local_offset). O(log n_files)."""
        file_idx = bisect.bisect_right(self._cum_offsets, global_idx) - 1
        local_offset = global_idx - self._cum_offsets[file_idx]
        return file_idx, local_offset

    def _get_window(self, start: int, length: int) -> dict[str, np.ndarray]:
        """Extract a contiguous window of `length` orders starting at global index `start`.

        Handles the case where the window spans two consecutive files.
        """
        file_idx, local_start = self._resolve_file(start)
        chunk = self._load_chunk(file_idx)
        file_len = self._file_lengths[file_idx]

        end_local = local_start + length

        if end_local <= file_len:
            # Window fits entirely in one file
            return {k: v[local_start:end_local] for k, v in chunk.items()}

        # Window spans into the next file — concatenate the tail + head
        tail = {k: v[local_start:] for k, v in chunk.items()}
        remaining = length - (file_len - local_start)
        next_chunk = self._load_chunk(file_idx + 1)
        head = {k: v[:remaining] for k, v in next_chunk.items()}

        return {k: np.concatenate([tail[k], head[k]]) for k in chunk}

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # We need seq_len + 1 orders: [idx .. idx+seq_len] for input + shifted labels
        window = self._get_window(idx, self.seq_len + 1)
        sl = self.seq_len

        batch: dict[str, torch.Tensor] = {}

        # .copy() each slice before torch.as_tensor so the resulting
        # tensor owns its own buffer instead of viewing into the cached
        # chunk. Without this, evicting a chunk from self._cache leaves
        # its underlying ndarray alive as long as any in-flight tensor
        # holds a view of it — defeating the max_cached bound.

        # Input fields: positions [0 .. seq_len-1]
        for k in PRED_FIELDS:
            arr = window[k][:sl].copy()
            batch[k] = torch.as_tensor(arr, dtype=torch.long)

        # Conditioning: positions [0 .. seq_len-1]
        for k in COND_FIELDS:
            arr = window[k][:sl].copy()
            batch[k] = torch.as_tensor(
                arr, dtype=torch.float32 if arr.ndim > 1 else torch.long
            )

        # Optional multi-scale regime conditioning (present iff tokenizer emits it)
        if "regime" in window:
            batch["regime"] = torch.as_tensor(window["regime"][:sl].copy(), dtype=torch.float32)

        # Labels: positions [1 .. seq_len] (next-step prediction)
        for k in PRED_FIELDS:
            arr = window[k][1 : sl + 1].copy()
            batch[f"label_{LABEL_KEYS[k]}"] = torch.as_tensor(arr, dtype=torch.long)

        return batch


def discover_intervals(
    data_dir: Path,
    split: str | None = None,
    max_books: int | None = None,
) -> dict[str, dict[str, list[Path]]]:
    """Group parquet files by interval name across books.

    Returns:
        {interval_name: {book_id: [parquet_path]}}
        e.g., {"00000000-00010000": {"0": [Path(...)], "1": [Path(...)]}}

    Interval names are the parquet filenames (consistent across books).
    """
    splits_path = data_dir / "splits.json"
    if splits_path.exists() and split is not None:
        manifest = json.loads(splits_path.read_text())
        splits_data = manifest.get("splits", manifest)
        if split not in splits_data:
            raise ValueError(f"Split '{split}' not in {splits_path}")
        book_ids = splits_data[split].get("book_ids", [])
        if max_books is not None:
            book_ids = book_ids[:max_books]
    else:
        book_ids = [
            d.name
            for d in sorted(data_dir.iterdir())
            if d.is_dir() and (d / "intervals").is_dir()
        ]
        if max_books:
            book_ids = book_ids[:max_books]

    intervals: dict[str, dict[str, list[Path]]] = {}
    for bid in book_ids:
        book_dir = data_dir / str(bid) / "intervals"
        if not book_dir.is_dir():
            continue
        for f in sorted(book_dir.glob("*.parquet")):
            iname = f.name
            if iname not in intervals:
                intervals[iname] = {}
            intervals[iname][str(bid)] = [f]

    logger.info(
        "Discovered %d intervals across %d books (split=%s)",
        len(intervals),
        len(book_ids),
        split,
    )
    return intervals


class ChunkSampler(Sampler[int]):
    """Sampler that shuffles file order but yields sequential indices within each file.

    This ensures that consecutive __getitem__ calls access the same cached file,
    eliminating the cache thrashing caused by fully random global shuffling.

    Within each file, windows are also shuffled for training randomness.
    Each epoch re-shuffles both file order and intra-file window order.
    """

    def __init__(self, dataset: OrderDataset, shuffle: bool = True) -> None:
        self._dataset = dataset
        self._shuffle = shuffle

    def __len__(self) -> int:
        ds = self._dataset
        return sum(max(0, flen - ds.seq_len) for flen in ds._file_lengths)

    def __iter__(self) -> Iterator[int]:
        ds = self._dataset
        seq_len = ds.seq_len

        # Build per-file window index ranges
        # file i owns global indices [cum_offsets[i] .. cum_offsets[i] + file_lengths[i] - seq_len - 1]
        file_ranges: list[list[int]] = []
        for i, flen in enumerate(ds._file_lengths):
            start = ds._cum_offsets[i]
            # Exclude windows that would span into the next file (a different book).
            n_windows = max(0, flen - seq_len)
            file_ranges.append(list(range(start, start + n_windows)))

        if self._shuffle:
            # Shuffle file order
            file_order = list(range(len(file_ranges)))
            random.shuffle(file_order)

            indices: list[int] = []
            for fi in file_order:
                windows = file_ranges[fi]
                random.shuffle(windows)
                indices.extend(windows)
        else:
            # Sequential: files in order, windows in order
            indices = []
            for windows in file_ranges:
                indices.extend(windows)

        return iter(indices)


class InterleaveSampler(ChunkSampler):
    """Block-shuffle across files so any prefix (a step-capped run) spans all files,
    not just the first few — fixes ChunkSampler's file-by-file front-loading while
    keeping per-block cache locality. With ``recency_alpha`` > 0, blocks from later
    files (assumed newer; caller orders files chronologically) are drawn earlier, so a
    budget-limited round over-samples recent data while still covering the whole set.
    ``recency_alpha`` = 0 reproduces a plain uniform block shuffle.
    """

    def __init__(self, dataset, shuffle: bool = True, block: int = 1024,
                 recency_alpha: float = 0.0) -> None:
        super().__init__(dataset, shuffle)
        self._block = max(1, block)
        self._recency_alpha = float(recency_alpha)

    def __iter__(self) -> Iterator[int]:
        ds = self._dataset
        seq = ds.seq_len
        nf = max(1, len(ds._file_lengths) - 1)
        blocks: list[tuple[int, list[int]]] = []
        for i, flen in enumerate(ds._file_lengths):
            start = ds._cum_offsets[i]
            nw = max(0, flen - seq)
            if nw == 0:
                continue
            w = list(range(start, start + nw))
            if self._shuffle:
                random.shuffle(w)
            for j in range(0, nw, self._block):
                blocks.append((i, w[j:j + self._block]))
        if self._shuffle:
            if self._recency_alpha > 0:
                # Efraimidis-Spirakis weighted shuffle: key = u^(1/weight), sort desc;
                # weight rises with file recency so recent blocks tend to the front.
                def _key(b: tuple[int, list[int]]) -> float:
                    weight = math.exp(self._recency_alpha * (b[0] / nf))
                    return random.random() ** (1.0 / weight)
                blocks.sort(key=_key, reverse=True)
            else:
                random.shuffle(blocks)
        indices: list[int] = []
        for _, b in blocks:
            indices.extend(b)
        return iter(indices)


def create_dataloaders(
    data_dir: str | Path,
    seq_len: int = 1024,
    batch_size: int = 32,
    tokenizer_config: TokenizerConfig | None = None,
    train_split: str = "train",
    val_split: str = "val",
    num_workers: int = 0,
    max_cached: int = 3,
    max_books: int | None = None,
) -> tuple[DataLoader, DataLoader, OrderTokenizer]:
    """Create train and val dataloaders from a sim data directory.

    Expects data_dir to contain splits.json mapping book_ids to splits,
    with <book_id>/intervals/*.parquet underneath.

    Args:
        max_books: If set, only use the first N book_ids per split.
                   Useful for quick training runs on a subset of the data.
    """
    data_path = Path(data_dir)
    tokenizer = OrderTokenizer(tokenizer_config)

    train_files = _discover_parquets(data_path, train_split, max_books=max_books)
    val_files = _discover_parquets(data_path, val_split, max_books=max_books)

    train_ds = OrderDataset(train_files, seq_len, tokenizer, max_cached=max_cached)
    val_ds = OrderDataset(val_files, seq_len, tokenizer, max_cached=max_cached)

    # ChunkSampler: shuffles file order + intra-file window order each epoch,
    # but yields sequential windows per file → cache stays hot (one file at a time).
    train_sampler = ChunkSampler(train_ds, shuffle=True)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=ChunkSampler(val_ds, shuffle=False),
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, tokenizer
