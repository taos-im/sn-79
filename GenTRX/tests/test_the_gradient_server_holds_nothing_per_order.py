"""The gradient server keeps nothing per order.

_process_tick keeps no per-book map keyed by order id, so the server's memory does not grow with the orders a run
places.

Run: pytest GenTRX/tests/test_the_gradient_server_holds_nothing_per_order.py -v
"""

FIRST_ID = 7_000_000_000


def _make_aggregator(tmp_path):
    from GenTRX.src.gradient_server import GradientAggregator

    return GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path=str(tmp_path / "val"),
        output_path=str(tmp_path / "out.pt"),
        books_per_miner=1,
        interval=60,
        window_ns=50,
        warmup_rounds=0,
        rollback=False,
    )


def _tick(ts, ids):
    events = [{"y": "o", "s": k % 2, "p": 300.0 + 0.01 * (k % 5), "t": ts + k, "i": oid, "q": 1.0}
              for k, oid in enumerate(ids)]
    book = {"bids": [[299.9, 5.0]], "asks": [[300.1, 5.0]], "events": events}
    cfg = {"simulation_id": "SIM_A", "priceDecimals": 2, "volumeDecimals": 4}
    return {"ts": ts, "books": {"0": book, "1": dict(book)}, "config": cfg, "sim_id": "SIM_A"}


def _keys_in_order_range(obj, depth=0):
    """Every order id found as a dict key in obj, down to four levels."""
    found = set()
    if depth > 4:
        return found
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, int) and k >= FIRST_ID:
                found.add(k)
            found |= _keys_in_order_range(v, depth + 1)
    return found


def test_order_events_leave_no_per_order_state(tmp_path):
    agg = _make_aggregator(tmp_path)
    for n in range(4):
        ids = range(FIRST_ID + n * 500, FIRST_ID + (n + 1) * 500)
        agg._process_tick(_tick(1_000_000_000 * (n + 1), ids))
    held = set()
    for value in vars(agg).values():
        held |= _keys_in_order_range(value)
    assert not held, f"{len(held)} order ids held after the ticks were processed"
