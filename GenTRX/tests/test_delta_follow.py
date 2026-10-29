"""Delta-follow checkpoint sharing: a miner advances its model by applying the
server's canonical per-version deltas, and falls back when one is missing.

Run: pytest GenTRX/tests/test_delta_follow.py -v
"""

import torch

from GenTRX.src.distributed import apply_version_deltas, model_state_hash
from GenTRX.src.gradient import (
    GradientMetadata,
    snapshot_state,
    extract_delta,
    compress,
    serialize,
    decompress,
    apply_gradient,
)
from GenTRX.src.model import OrderModel, ModelConfig


def _tiny():
    return OrderModel(ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64,
                                  film_layers=(0,), film_d_cond=16))


def _meta(v):
    return GradientMetadata(window_id=0, miner_uid=0, steps_trained=1,
                            loss_before=0.0, loss_after=0.0, loss_trajectory=[0.0],
                            model_v_trained=v)


def _make_deltas():
    """Return (theta0, {1: bytes, 2: bytes}, comps) by perturbing a model twice."""
    torch.manual_seed(0)
    m = _tiny()
    theta0 = snapshot_state(m)
    comps = {}
    prev = theta0
    for v in (1, 2):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        cur = snapshot_state(m)
        comp = compress(extract_delta(prev, cur, _meta(v)), top_k_frac=0.5)
        comps[v] = comp
        prev = cur
    store_bytes = {v: serialize(c) for v, c in comps.items()}
    return theta0, store_bytes, comps


class _FakeStore:
    def __init__(self, by_version):
        self._d = by_version

    def get_version_delta(self, uid, version):
        return self._d.get(version)


def test_apply_version_deltas_matches_manual_apply():
    theta0, store_bytes, comps = _make_deltas()
    expected = {n: p.shape for n, p in _tiny().named_parameters()}

    ref = _tiny()
    ref.load_state_dict(theta0)
    apply_gradient(ref, decompress(comps[1]))
    apply_gradient(ref, decompress(comps[2]))

    test = _tiny()
    test.load_state_dict(theta0)
    reached = apply_version_deltas(test, _FakeStore(store_bytes), 0, 0, 2, expected)

    assert reached == 2
    rs, ts = ref.state_dict(), test.state_dict()
    for k in rs:
        assert torch.allclose(rs[k], ts[k], atol=1e-6), k


def test_missing_delta_stops_at_gap():
    theta0, store_bytes, _ = _make_deltas()
    expected = {n: p.shape for n, p in _tiny().named_parameters()}
    partial = _FakeStore({1: store_bytes[1]})  # v2 absent (pruned)

    test = _tiny()
    test.load_state_dict(theta0)
    reached = apply_version_deltas(test, partial, 0, 0, 2, expected)
    assert reached == 1  # advanced to 1, then hit the gap


def test_state_hash_is_deterministic_and_sensitive():
    torch.manual_seed(1)
    m = _tiny()
    h1 = model_state_hash(m)
    assert h1 == model_state_hash(m)  # stable
    with torch.no_grad():
        next(m.parameters()).add_(1.0)
    assert model_state_hash(m) != h1  # changes with the weights
