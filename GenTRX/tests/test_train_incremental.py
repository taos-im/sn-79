"""train_incremental: budget governs how far training gets, one delta over all
loaders trained, optional fixed-step cap.

Run: pytest GenTRX/tests/test_train_incremental.py -v
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from torch.utils.data import DataLoader

from GenTRX.src.dataloader import OrderDataset, ChunkSampler
from GenTRX.src.distributed import train_incremental, WindowConfig
from GenTRX.src.model import OrderModel, ModelConfig
from GenTRX.src.tokenizer import OrderTokenizer
from GenTRX.src.util.schema import LOB_DEPTH, order_stream_schema

SEQ = 16
BATCH = 4


def _write_parquet(path, n=200):
    rows = []
    for i in range(n):
        r = {
            "timestamp": 1_000_000_000 * (i + 1),
            "order_type": i % 3,
            "rel_price": (-1) ** i * (i % 50),
            "volume_int": i % 64,
            "volume_dec": (i % 7) / 7.0,
            "interval_ns": 1_000 * (i % 100),
            "mid_price": 100_000 + (i % 20),
            "time_of_day_s": (i * 13) % 86400,
            "mid_price_delta": (i % 11) - 5,
        }
        for j in range(LOB_DEPTH):
            r[f"lob_ask_vol_{j + 1}"] = float((i + j) % 50) + 1.0
            r[f"lob_bid_vol_{j + 1}"] = float((i + j) % 40) + 1.0
        rows.append(r)
    cols = {
        "timestamp": pa.array([r["timestamp"] for r in rows], type=pa.timestamp("ns")),
        "order_type": np.array([r["order_type"] for r in rows], dtype=np.int8),
        "rel_price": np.array([r["rel_price"] for r in rows], dtype=np.int32),
        "volume_int": np.array([r["volume_int"] for r in rows], dtype=np.int32),
        "volume_dec": np.array([r["volume_dec"] for r in rows], dtype=np.float32),
        "interval_ns": np.array([r["interval_ns"] for r in rows], dtype=np.int64),
        "mid_price": np.array([r["mid_price"] for r in rows], dtype=np.int64),
        "time_of_day_s": np.array([r["time_of_day_s"] for r in rows], dtype=np.int32),
        "mid_price_delta": np.array([r["mid_price_delta"] for r in rows], dtype=np.int64),
    }
    for i in range(LOB_DEPTH):
        cols[f"lob_ask_vol_{i + 1}"] = np.array(
            [r[f"lob_ask_vol_{i + 1}"] for r in rows], dtype=np.float64
        )
        cols[f"lob_bid_vol_{i + 1}"] = np.array(
            [r[f"lob_bid_vol_{i + 1}"] for r in rows], dtype=np.float64
        )
    pq.write_table(pa.table(cols, schema=order_stream_schema()), str(path))
    return path


@pytest.fixture
def loaders(tmp_path):
    tok = OrderTokenizer()
    out = []
    for k in range(3):
        f = _write_parquet(tmp_path / f"page{k}.parquet")
        ds = OrderDataset([f], seq_len=SEQ, tokenizer=tok, max_cached=1)
        out.append(DataLoader(ds, batch_size=BATCH, sampler=ChunkSampler(ds, shuffle=True)))
    return out


def _model():
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64,
                      film_layers=(0,), film_d_cond=16)
    return OrderModel(cfg)


def test_budget_zero_trains_nothing(loaders):
    """budget_s=0 stops before the first step (no time)."""
    delta = train_incremental(_model(), loaders, WindowConfig(budget_s=0.0), "cpu")
    assert delta.metadata.steps_trained == 0


def test_step_cap_is_exact(loaders):
    """n_steps caps the total steps regardless of available batches."""
    delta = train_incremental(_model(), loaders, WindowConfig(n_steps=5, budget_s=None), "cpu")
    assert delta.metadata.steps_trained == 5


def test_no_budget_trains_all_loaders(loaders):
    """With no budget and no cap, every batch of every loader is trained once."""
    total_batches = sum(len(dl) for dl in loaders)
    # n_steps=0 → no cap (the WindowConfig default of 100 would cap otherwise).
    delta = train_incremental(_model(), loaders, WindowConfig(n_steps=0, budget_s=None), "cpu")
    assert delta.metadata.steps_trained == total_batches
    assert delta.norm > 0  # weights actually moved
