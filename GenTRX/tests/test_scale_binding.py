"""Tests for price/volume scale binding in the gradient server.

Pin the rule: the live sim's priceDecimals/volumeDecimals bind the scale when
present; a tick that reaches the server with no decimals falls back to the
canonical simulation_0.xml values (pd=2, vd=4) and logs a warning, never a
silently-wrong scale.

Run: pytest GenTRX/tests/test_scale_binding.py -v
"""

from GenTRX.src.util.schema import DEFAULT_PRICE_DECIMALS, DEFAULT_VOLUME_DECIMALS


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


def _tick(*, sim_id, price_decimals=None, volume_decimals=None, ts=1000):
    cfg: dict = {"simulation_id": sim_id}
    if price_decimals is not None:
        cfg["priceDecimals"] = price_decimals
    if volume_decimals is not None:
        cfg["volumeDecimals"] = volume_decimals
    return {"ts": ts, "books": {}, "config": cfg, "sim_id": sim_id}


def test_scale_binds_from_explicit_decimals(tmp_path):
    agg = _make_aggregator(tmp_path)
    agg._process_tick(_tick(sim_id="SIM_A", price_decimals=2, volume_decimals=4))
    assert agg._price_scale == 10**2
    assert agg._vol_scale == 10**4


def test_explicit_decimals_override_canonical_default(tmp_path):
    agg = _make_aggregator(tmp_path)
    agg._process_tick(_tick(sim_id="SIM_B", price_decimals=3, volume_decimals=5))
    assert agg._price_scale == 10**3
    assert agg._vol_scale == 10**5
