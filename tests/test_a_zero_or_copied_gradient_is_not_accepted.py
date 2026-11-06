# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The gradient server accepts, and so pays, only a gradient that carries an update of its own.

- The default minimum score is 0: an accepted gradient lowers the held-out loss (warmup still disables the minimum).
- A gradient whose norm is 0 is rejected, in warmup too: it changes nothing.
- A gradient with the same content as another uid's this round is rejected, in warmup too. Content is the update the
  scoring applies (gradient_content_hash), not its file or its encoding: the metadata names the uploader, and other
  index or value dtypes, permuted pairs or explicit zeros apply the same update. Of identical contents the single
  earliest upload, by the storage's own upload time, keeps it; when that earliest time is shared, or an upload has no
  time, no uid keeps it.
"""
import inspect
import pathlib
from types import SimpleNamespace

import pytest
import torch

from GenTRX.src import gradient_server
from GenTRX.src.gradient import CompressedGradient, GradientMetadata, serialize
from GenTRX.src.gradient_server import GradientAggregator, gradient_content_hash

_accept = GradientAggregator._is_gradient_acceptable
_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _comp(vals, uid=1, idx=(0, 2, 4), idx_dtype=torch.int64, val_dtype=torch.float32):
    sparse = {"w": (torch.tensor(idx, dtype=idx_dtype), torch.tensor(vals, dtype=val_dtype), torch.Size([8]))}
    return SimpleNamespace(sparse=sparse, metadata=SimpleNamespace(miner_uid=uid))


def _scored(*entries):
    out = []
    for uid, vals, uploaded in entries:
        c = _comp(vals, uid)
        a = {"_content_hash": gradient_content_hash(c), "_uploaded_at": uploaded, "_grad_norm": 1.0}
        out.append((uid, None, 0.5, c, a))
    return out


def _marks(scored):
    return {m: a.get("_duplicate_of") for m, _, _, _, a in scored}


def _mark(scored):
    agg = SimpleNamespace(_copy_drops_total=0)
    GradientAggregator._mark_copies(agg, scored)
    return agg


def test_the_content_hash_reads_the_tensors_not_the_uploader():
    assert gradient_content_hash(_comp([1.0, 2.0, 3.0], uid=1)) == gradient_content_hash(_comp([1.0, 2.0, 3.0], uid=2))
    assert gradient_content_hash(_comp([1.0, 2.0, 3.0])) != gradient_content_hash(_comp([4.0, 5.0, 6.0]))
    assert gradient_content_hash(_comp([1.0, 2.0, 3.0])) != gradient_content_hash(_comp([1.0, 2.0, 3.0], idx=(0, 2, 6)))


def test_a_re_encoded_copy_hashes_as_the_original():
    original = gradient_content_hash(_comp([1.0, 2.0, 3.0]))
    assert gradient_content_hash(_comp([1.0, 2.0, 3.0], idx_dtype=torch.int32)) == original
    assert gradient_content_hash(_comp([3.0, 1.0, 2.0], idx=(4, 0, 2))) == original
    assert gradient_content_hash(_comp([1.0, 2.0, 3.0], val_dtype=torch.float64)) == original
    assert gradient_content_hash(_comp([1.0, 2.0, 3.0], val_dtype=torch.bfloat16)) == original
    assert gradient_content_hash(_comp([1.0, 0.0, 2.0, 3.0], idx=(0, 1, 2, 4))) == original


def test_the_content_hash_reads_the_update_the_scoring_applies():
    """Of two pairs at one index the scoring applies the later, so the content is the later value."""
    original = gradient_content_hash(_comp([1.0, 2.0, 3.0]))
    assert gradient_content_hash(_comp([9.0, 1.0, 2.0, 3.0], idx=(0, 0, 2, 4))) == original
    assert gradient_content_hash(_comp([1.0, 9.0, 2.0, 3.0], idx=(0, 0, 2, 4))) != original


def test_the_earliest_upload_keeps_its_content_and_a_later_copy_is_marked():
    scored = _scored((5, [1.0, 2.0, 3.0], 200.0), (3, [1.0, 2.0, 3.0], 100.0), (7, [4.0, 5.0, 6.0], 50.0))
    agg = _mark(scored)
    assert _marks(scored) == {5: 3, 3: None, 7: None}
    assert agg._copy_drops_total == 1


def test_a_shared_earliest_upload_time_leaves_no_uid_the_content():
    scored = _scored((9, [1.0, 2.0, 3.0], 100.0), (4, [1.0, 2.0, 3.0], 100.0), (6, [1.0, 2.0, 3.0], 101.0))
    agg = _mark(scored)
    assert _marks(scored) == {9: 4, 4: 6, 6: 4}
    assert agg._copy_drops_total == 3


def test_without_an_upload_time_no_uid_keeps_the_content():
    scored = _scored((9, [1.0, 2.0, 3.0], None), (4, [1.0, 2.0, 3.0], None))
    _mark(scored)
    assert _marks(scored) == {9: 4, 4: 9}
    scored = _scored((9, [1.0, 2.0, 3.0], 100.0), (4, [1.0, 2.0, 3.0], None))
    _mark(scored)
    assert _marks(scored) == {9: 4, 4: 9}


@pytest.mark.parametrize("uploaded", [(100.0, 100.0), (None, 100.0)])
def test_a_copy_without_a_single_earliest_upload_is_not_accepted(uploaded):
    scored = _scored((9, [1.0, 2.0, 3.0], uploaded[0]), (4, [1.0, 2.0, 3.0], uploaded[1]))
    _mark(scored)
    gate = SimpleNamespace(_effective_min_score=0.0, _in_warmup=False)
    assert [_accept(gate, 0.5, a) for _m, _w, _s, _c, a in scored] == [False, False]


def test_a_copy_and_a_zero_norm_are_rejected_in_warmup_and_after():
    for agg in (SimpleNamespace(_effective_min_score=float("-inf"), _in_warmup=True),
                SimpleNamespace(_effective_min_score=0.0, _in_warmup=False)):
        assert _accept(agg, 0.9, {"_duplicate_of": 3}) is False
        assert _accept(agg, 0.9, {"_grad_norm": 0.0}) is False
        assert _accept(agg, 0.9, {"_grad_norm": 1e-3}) is True


def test_the_default_minimum_score_is_zero_everywhere_it_is_set():
    assert inspect.signature(GradientAggregator.__init__).parameters["min_score"].default == 0.0
    agg = SimpleNamespace(_effective_min_score=0.0, _in_warmup=False)
    assert _accept(agg, 0.0, {}) is False and _accept(agg, 1e-6, {}) is True
    for rel in ("GenTRX/src/gradient_server.py", "GenTRX/src/util/gen_config.py"):
        src = (_ROOT / rel).read_text()
        assert 'parser.add_argument("--min-score", type=float, default=0.0)' in src, rel


def test_the_round_marks_copies_before_it_accepts_and_keeps_the_upload_time():
    agg_src = inspect.getsource(GradientAggregator._aggregate_accepted)
    assert agg_src.index("self._mark_copies(scored)") < agg_src.index("accepted = [")
    assert '"LastModified"' in inspect.getsource(GradientAggregator._try_read_miner_gradient)
    assert 'a["_uploaded_at"]' in inspect.getsource(GradientAggregator._collect_round_gradients)


def _scoring_server(evaluations):
    """A server reduced to what _score_one_gradient reads, a small model and one serialized gradient for it; the
    evaluation is stubbed and records the assignment as it stands when the evaluation starts."""
    assignment = {"round": 1, "model_version": 0, "ts_start": 0, "ts_end": 10}

    def _evaluate(*_a, **_k):
        evaluations.append(dict(assignment))
        return 0.25

    server = SimpleNamespace(
        _max_gradient_bytes=10**9, _reject_untagged=False, max_grad_norm=None, _proof_batches=1,
        label_smooth_sigma=0.0, _agg_round=1, _version=1, max_val_batches=1, overfit_ratio=2.0,
        _build_miner_loader=lambda a, t, d: object(), _forward_val_ranges=lambda r: r,
        _build_val_loader_for_ranges=lambda r, t, d, round_id: None,
    )
    comp = CompressedGradient(
        sparse={"weight": (torch.tensor([0, 5]), torch.tensor([0.5, -0.5]), torch.Size([2, 4]))},
        metadata=GradientMetadata(miner_uid=3),
    )
    return server, torch.nn.Linear(4, 2), assignment, serialize(comp), _evaluate


def test_a_content_hash_that_fails_leaves_the_gradient_scored(monkeypatch):
    server, model, assignment, blob, evaluate = _scoring_server([])
    monkeypatch.setattr("GenTRX.src.distributed.evaluate_gradient", evaluate)

    def _fail(_comp):
        raise RuntimeError("hash")

    monkeypatch.setattr(gradient_server, "gradient_content_hash", _fail)
    result = GradientAggregator._score_one_gradient(server, 3, assignment, blob, model, None, "cpu")
    assert result is not None and result[0] == 0.25, "the gradient is scored"
    assert assignment["_score"] == 0.25
    assert assignment["_content_hash"] is None, "without a hash the copy check passes over it"


def test_the_content_hash_is_stamped_before_the_evaluation(monkeypatch):
    evaluations = []
    server, model, assignment, blob, evaluate = _scoring_server(evaluations)
    monkeypatch.setattr("GenTRX.src.distributed.evaluate_gradient", evaluate)
    GradientAggregator._score_one_gradient(server, 3, assignment, blob, model, None, "cpu")
    assert assignment["_content_hash"] is not None
    assert evaluations and evaluations[0].get("_content_hash") == assignment["_content_hash"]


def test_the_copies_are_published_per_round_and_in_total(tmp_path):
    agg = GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"), val_data_path=str(tmp_path / "val"),
        output_path=str(tmp_path / "out.pt"), books_per_miner=1, interval=60, window_ns=50,
        warmup_rounds=0, rollback=False,
    )
    assert agg._last_aggregation["n_copies"] == 0
    agg._log_event({"type": "aggregation", "round": 1, "n_copies": 2})
    assert agg._last_aggregation["n_copies"] == 2
    agg._copy_drops_total = 5
    agg._deliver_scores([], [], [], 0.0, [])
    assert agg._latest_scores["counters"]["copy_drops_total"] == 5
    src = inspect.getsource(GradientAggregator._aggregate_accepted)
    assert src.count('"n_copies": n_copies') == src.count('"n_version_mismatched": n_version_mismatched') > 0
