"""The aggregator scores and aggregates each round on the model it last published.

The server saves each published version to output_path and builds the next round's scoring model from
checkpoint_path. The launcher's defaults name two files (checkpoints/GenTRX/best.pt and latest.pt), so without a
link between them every round would restart from the boot checkpoint. A saved version also replaces
checkpoint_path, so the next round, and a restart, start from what was published.

Run: pytest GenTRX/tests/test_the_aggregator_scores_on_the_model_it_published.py -v
"""
from dataclasses import asdict

import torch


def _tiny_cfg():
    from GenTRX.src.model import ModelConfig

    return ModelConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=32)


def _ckpt(model, cfg, step):
    from GenTRX.src.tokenizer import TokenizerConfig

    return {"model_state_dict": model.state_dict(), "model_config": asdict(cfg),
            "tokenizer_config": asdict(TokenizerConfig()), "step": step, "loss": 1.0, "epoch": 0}


def _aggregator(tmp_path, is_aggregator=True):
    from GenTRX.src.gradient_server import GradientAggregator

    return GradientAggregator(
        checkpoint_path=str(tmp_path / "best.pt"),
        val_data_path=str(tmp_path / "val"),
        output_path=str(tmp_path / "latest.pt"),
        books_per_miner=1,
        interval=60,
        window_ns=50,
        warmup_rounds=0,
        rollback=False,
        validator_store=None,
        is_aggregator=is_aggregator,
    )


def _boot_and_trained(tmp_path):
    from GenTRX.src.model import OrderModel

    cfg = _tiny_cfg()
    boot = OrderModel(cfg)
    torch.save(_ckpt(boot, cfg, 0), tmp_path / "best.pt")
    trained = OrderModel(cfg)
    with torch.no_grad():
        for p in trained.parameters():
            p.add_(0.5)
    return cfg, boot, trained


def _same(a, b):
    return all(torch.equal(a[k].float(), b[k].float()) for k in a)


def test_the_next_round_scores_on_the_saved_version(tmp_path):
    cfg, boot, trained = _boot_and_trained(tmp_path)
    agg = _aggregator(tmp_path)
    agg._save_published_checkpoint(_ckpt(trained, cfg, 1))
    agg._clear_scoring_cache()
    cache = agg._get_scoring_cache(1)
    assert cache is not None
    scored = {k: v.cpu() for k, v in cache["model"].state_dict().items()}
    assert _same(scored, trained.state_dict()), "the next round scores on the published model, not the boot file"
    assert not _same(scored, boot.state_dict())


def test_both_paths_hold_the_published_version_for_a_restart(tmp_path):
    cfg, _boot, trained = _boot_and_trained(tmp_path)
    agg = _aggregator(tmp_path)
    agg._save_published_checkpoint(_ckpt(trained, cfg, 1))
    for name in ("latest.pt", "best.pt"):
        saved = torch.load(tmp_path / name, map_location="cpu", weights_only=False)
        assert saved["step"] == 1 and _same(saved["model_state_dict"], trained.state_dict()), name
    assert not list(tmp_path.glob("*.tmp")), "no temporary file is left behind"


def test_one_path_is_written_once(tmp_path):
    from GenTRX.src.gradient_server import GradientAggregator
    from GenTRX.src.model import OrderModel

    cfg = _tiny_cfg()
    torch.save(_ckpt(OrderModel(cfg), cfg, 0), tmp_path / "best.pt")
    agg = GradientAggregator(checkpoint_path=str(tmp_path / "best.pt"), val_data_path=str(tmp_path / "val"),
                             books_per_miner=1, interval=60, window_ns=50, warmup_rounds=0, rollback=False,
                             validator_store=None)
    trained = OrderModel(cfg)
    agg._save_published_checkpoint(_ckpt(trained, cfg, 1))
    saved = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert saved["step"] == 1 and sorted(p.name for p in tmp_path.glob("*.pt")) == ["best.pt"]


def test_the_round_saves_through_the_one_method():
    import inspect

    from GenTRX.src import gradient_server

    src = inspect.getsource(gradient_server.GradientAggregator._aggregate_accepted)
    assert "self._save_local_checkpoint(ckpt_dict)" in src
    assert "torch.save(ckpt_dict, tmp_path)" not in src
    # The aggregator's save reaches both paths through the one method; a sibling's save never touches the
    # published checkpoint, so its scoring model stays uid 0's.
    local = inspect.getsource(gradient_server.GradientAggregator._save_local_checkpoint)
    assert "if self.is_aggregator:" in local
    assert "self._save_published_checkpoint(ckpt_dict)" in local
