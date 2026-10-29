"""Per-asset scale-invariant normalization for cross-asset GenTRX.

Maps raw order fields into a SHARED space so one tokenizer/model can span assets that
differ by orders of magnitude in order size and in book width:

    volume:  v_norm = log10( (vol_int + vol_dec) / median_qty_asset )
    price:   p_norm = ( rel_price_ticks / mid_ticks * 1e4 ) / price_scale_bps_asset
             (tick cancels in bps: rel_px*tick / (mid*tick) = rel_px/mid)

Per-asset REFERENCE = {median_qty, price_scale_bps} (+ pair_decimals for the engine
boundary). Forward (encode) + inverse (decode/inference) so generation can reconstruct
raw ticks/qty from a normalized bin. int+dec stays the parquet/engine storage; only the
model tokenizes the combined, normalized value.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def compute_reference(qty: np.ndarray, rel_price_ticks: np.ndarray, mid_ticks: np.ndarray,
                      pair_decimals: int, price_scale_pct: float = 90.0) -> dict:
    """Per-asset reference from a sample of the asset's order stream.
    median_qty: median of positive order qty. price_scale_bps: the price_scale_pct-th
    percentile of |rel_price in bps| (book half-width); divisor that maps a typical
    book offset to ~1.0 in normalized units (so wide-book assets align with tight-book ones)."""
    qty = np.asarray(qty, float)
    m = mid_ticks.astype(float)
    qpos = qty[qty > 0]
    median_qty = float(np.median(qpos)) if qpos.size else 1.0
    ok = m > 0
    bps = np.abs(rel_price_ticks.astype(float)[ok] / m[ok] * 1e4)
    bps = bps[np.isfinite(bps)]
    price_scale_bps = float(np.percentile(bps, price_scale_pct)) if bps.size else 1.0
    return {"median_qty": max(median_qty, EPS),
            "price_scale_bps": max(price_scale_bps, EPS),
            "pair_decimals": int(pair_decimals)}


# --- volume ---------------------------------------------------------------
def vol_to_norm(qty: np.ndarray, ref: dict) -> np.ndarray:
    return np.log10(np.clip(np.asarray(qty, float), EPS, None) / ref["median_qty"])


def vol_from_norm(v_norm: np.ndarray, ref: dict) -> np.ndarray:
    return ref["median_qty"] * np.power(10.0, np.asarray(v_norm, float))


# --- price ----------------------------------------------------------------
def price_to_norm(rel_price_ticks: np.ndarray, mid_ticks: np.ndarray, ref: dict) -> np.ndarray:
    m = np.asarray(mid_ticks, float)
    bps = np.where(m > 0, np.asarray(rel_price_ticks, float) / np.where(m > 0, m, 1.0) * 1e4, 0.0)
    return bps / ref["price_scale_bps"]


def price_from_norm(p_norm: np.ndarray, mid_ticks: np.ndarray, ref: dict) -> np.ndarray:
    """Back to integer rel_price ticks (round) given the row's mid (ticks)."""
    bps = np.asarray(p_norm, float) * ref["price_scale_bps"]
    return np.rint(bps / 1e4 * np.asarray(mid_ticks, float)).astype(np.int64)


if __name__ == "__main__":  # generic round-trip self-test (no external data, no real references)
    rng = np.random.default_rng(0)
    n = 10000
    print(f"{'median_qty':>12}{'pscale_bps':>11}{'v_norm[p1,p50,p99]':>26}{'|p_norm| p99':>13}{'roundtrip':>11}")
    for med, pscale, pdz in ((1.0, 10.0, 2), (10.0, 20.0, 2)):
        qty = np.abs(rng.lognormal(np.log(med), 1.5, n)) + EPS
        mid = np.full(n, 30000.0)
        rp = rng.normal(0.0, pscale, n) / 1e4 * mid           # rel_price ticks ~ pscale bps
        ref = compute_reference(qty, rp, mid, pdz)
        vn = vol_to_norm(qty, ref)
        pn = price_to_norm(rp, mid, ref)
        q_rt = vol_from_norm(vn, ref)
        rp_rt = price_from_norm(pn, mid, ref)
        rt = np.allclose(q_rt, qty, rtol=1e-6) and np.median(np.abs(rp_rt - rp)) <= 1

        def q(a, p):
            return round(float(np.percentile(a, p)), 2)

        print(f"{med:>12.4g}{pscale:>11.2f}{f'[{q(vn,1)},{q(vn,50)},{q(vn,99)}]':>26}"
              f"{q(np.abs(pn),99):>13}{('OK' if rt else 'FAIL'):>11}")
