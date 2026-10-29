# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Training loop for order model — per-field heads.

Usage:
    from GenTRX.src.train import train, TrainConfig
    from GenTRX.src.util.paths import REPO_ROOT

    model, train_losses, val_losses = train(
        TrainConfig(data_dir=str(REPO_ROOT / "data" / "sim" / "20260218"))
    )

    # Resume from checkpoint:
    model, train_losses, val_losses = train(
        TrainConfig(
            data_dir=str(REPO_ROOT / "data" / "sim" / "20260218"),
            resume=str(REPO_ROOT / "checkpoints" / "GenTRX" / "best.pt"),
        )
    )

Pass absolute paths only. `TrainConfig.data_dir` and `resume` are
both filesystem paths read at process start; relative paths bind to
the shell's CWD and have surprised operators before.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from GenTRX.src.dataloader import create_dataloaders
from GenTRX.src.metrics import StepMetrics
from GenTRX.src.model import ModelConfig, OrderModel, compute_loss
from GenTRX.src.tokenizer import TokenizerConfig

if TYPE_CHECKING:
    from GenTRX.src.tokenizer import OrderTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    data_dir: str = "data/sim"
    ckpt_dir: str = "checkpoints/GenTRX"
    resume: str | None = None  # path to checkpoint for resuming
    seq_len: int = 512
    batch_size: int = 32
    epochs: int = 3
    lr: float = 3e-4
    min_lr: float = 1e-5  # cosine decay floor
    warmup_steps: int = 200  # linear warmup from 0 to lr
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 100
    val_interval: int = 500
    max_books: int | None = None  # limit books per split for quick runs
    max_steps: int | None = None  # stop after N steps (for quick test runs)
    patience: int | None = None  # early stop after N val checks with no improvement
    num_workers: int = 4  # DataLoader parallel workers
    # Ordinal-aware label smoothing (bins) for price/interval/volume heads;
    # order_type stays strict CE. 0.0 disables it (strict CE). Default matches
    # the live training regime so offline-pretrained bootstrap checkpoints are
    # in-regime. See `model.compute_loss`.
    label_smooth_sigma: float = 1.0
    # order_type class weights [bid, ask, cancel]; None = model default [2,4,0.5].
    order_type_weights: list[float] | None = None
    # Drop LOB / time-of-day / mid_delta conditioning (ablation) — predict from
    # the event sequence alone, to test whether conditioning is a memorization leak.
    use_conditioning: bool = True
    # Content-only / event-clock: drop the wall-clock interval field from input
    # (zeroed) and loss. Tests whether removing wall-clock time helps content.
    predict_interval: bool = True
    # Single integer volume: drop the fractional vol_dec head (real equity shares).
    predict_vol_dec: bool = True


def _excluded(predict_interval: bool, predict_vol_dec: bool) -> tuple[str, ...]:
    """Fields dropped from the loss (zeroed inputs in _forward_batch)."""
    excl = []
    if not predict_interval:
        excl.append("interval")
    if not predict_vol_dec:
        excl.append("vol_dec")
    return tuple(excl)


def _forward_batch(
    model: OrderModel,
    batch: dict[str, torch.Tensor],
    device: str,
    use_conditioning: bool = True,
    predict_interval: bool = True,
    predict_vol_dec: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Extract batch, run forward, return (logits_dict, labels_dict)."""
    ot = batch["order_types"].to(device)
    pb = batch["price_bins"].to(device)
    vi = batch["vol_int_bins"].to(device)
    vd = (batch["vol_dec_bins"].to(device) if predict_vol_dec
          else torch.zeros_like(batch["vol_dec_bins"]).to(device))
    ib = (batch["interval_bins"].to(device) if predict_interval
          else torch.zeros_like(batch["interval_bins"]).to(device))
    lob = batch["lob_volumes"].to(device) if use_conditioning else None
    tod = batch["time_of_day"].to(device) if use_conditioning else None
    md = batch["mid_deltas"].to(device) if use_conditioning else None
    rg = batch.get("regime")
    rg = rg.to(device) if (rg is not None and use_conditioning) else None

    logits = model(ot, pb, vi, vd, ib, lob, tod, md, regime=rg)

    labels = {
        "order_type": batch["label_order_type"].to(device),
        "price": batch["label_price"].to(device),
        "vol_int": batch["label_vol_int"].to(device),
        "vol_dec": batch["label_vol_dec"].to(device),
        "interval": batch["label_interval"].to(device),
    }
    return logits, labels


def save_checkpoint(
    path: Path | str,
    model: OrderModel,
    optimizer: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    tokenizer_cfg: TokenizerConfig,
    epoch: int,
    step: int,
    loss: float,
    data_dir: str | None = None,
    label_smooth_sigma: float = 0.0,
) -> None:
    """Save checkpoint in standardized format.

    Keys:
        model_state_dict   — model weights
        optimizer_state_dict — optimizer state (for resuming)
        model_config       — full ModelConfig as dict
        tokenizer_config   — full TokenizerConfig as dict (for reproducibility)
        epoch, step, loss  — training progress
        data_dir           — provenance: which data produced this checkpoint
    """
    tok_dict = asdict(tokenizer_cfg)
    from GenTRX.src.version import checkpoint_stamp

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_cfg),
            "tokenizer_config": tok_dict,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "data_dir": data_dir,
            **checkpoint_stamp(label_smooth_sigma),
        },
        path,
    )
    logger.info("Saved checkpoint: %s (step=%d loss=%.4f)", path, step, loss)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> dict:
    """Load checkpoint and return the full dict."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "model_config" not in ckpt and "config" in ckpt:
        ckpt["model_config"] = ckpt.pop("config")
    return ckpt


def build_model_from_checkpoint(
    ckpt: dict,
    device: str = "cpu",
) -> tuple[OrderModel, "OrderTokenizer", ModelConfig, "TokenizerConfig"]:
    """Reconstruct (model, tokenizer, model_cfg, tokenizer_cfg) from a
    checkpoint dict. Used by `train.train()` for the resume path and
    by gentrx-serve's ModelHolder."""
    from GenTRX.src.tokenizer import OrderTokenizer, TokenizerConfig

    saved_model_cfg = ckpt.get("model_config", {}) or {}
    model_cfg = ModelConfig(
        **{
            k: v
            for k, v in saved_model_cfg.items()
            if k in ModelConfig.__dataclass_fields__
        }
    )

    saved_tok_cfg = ckpt.get("tokenizer_config", {}) or {}
    if saved_tok_cfg:
        tokenizer_cfg = TokenizerConfig.from_dict(saved_tok_cfg)
    else:
        tokenizer_cfg = TokenizerConfig()
    tokenizer = OrderTokenizer(tokenizer_cfg)

    model = OrderModel(model_cfg).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, model_cfg, tokenizer_cfg


def train(
    train_cfg: TrainConfig | None = None,
    model_cfg: ModelConfig | None = None,
    tokenizer_cfg: TokenizerConfig | None = None,
) -> tuple[OrderModel, list[float], list[float]]:
    train_cfg = train_cfg or TrainConfig()
    model_cfg = model_cfg or ModelConfig()
    tokenizer_cfg = tokenizer_cfg or TokenizerConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(train_cfg.ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, tokenizer = create_dataloaders(
        data_dir=train_cfg.data_dir,
        seq_len=train_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        tokenizer_config=tokenizer_cfg,
        max_books=train_cfg.max_books,
        num_workers=train_cfg.num_workers,
    )
    logger.info(
        "Train: %d batches, Val: %d batches", len(train_loader), len(val_loader)
    )

    model = OrderModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    if train_cfg.resume:
        ckpt = load_checkpoint(train_cfg.resume, device)
        if model_cfg is None or model_cfg == ModelConfig():
            saved_cfg = ckpt.get("model_config", {})
            model_cfg = ModelConfig(
                **{
                    k: v
                    for k, v in saved_cfg.items()
                    if k in ModelConfig.__dataclass_fields__
                }
            )
            model = OrderModel(model_cfg).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg.lr,
                weight_decay=train_cfg.weight_decay,
            )
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        best_val = ckpt.get("loss", float("inf"))
        logger.info(
            "Resumed from %s (epoch=%d step=%d loss=%.4f)",
            train_cfg.resume,
            start_epoch,
            global_step,
            best_val,
        )

    params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params on %s", params, device)

    train_losses: list[float] = []
    val_losses: list[float] = []
    n_batches = len(train_loader)

    # LR schedule: linear warmup → cosine decay to min_lr
    total_steps = train_cfg.max_steps or (n_batches * train_cfg.epochs)

    def lr_lambda(step: int) -> float:
        if step < train_cfg.warmup_steps:
            return step / max(train_cfg.warmup_steps, 1)
        progress = (step - train_cfg.warmup_steps) / max(
            total_steps - train_cfg.warmup_steps, 1
        )
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (
            train_cfg.min_lr / train_cfg.lr
            + (1.0 - train_cfg.min_lr / train_cfg.lr) * cosine
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=global_step - 1 if global_step > 0 else -1
    )

    # Early stopping state
    patience_counter = 0
    early_stopped = False

    for epoch in range(start_epoch, start_epoch + train_cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        field_loss_accum: dict[str, float] = {}
        train_metrics = StepMetrics()
        t0 = time.perf_counter()

        logger.info("Epoch %d/%d — %d batches", epoch, start_epoch + train_cfg.epochs - 1, n_batches)

        for batch in train_loader:
            logits, labels = _forward_batch(
                model, batch, device, train_cfg.use_conditioning,
                train_cfg.predict_interval, train_cfg.predict_vol_dec,
            )
            loss, field_losses = compute_loss(
                logits, labels,
                label_smooth_sigma=train_cfg.label_smooth_sigma,
                order_type_weights=train_cfg.order_type_weights,
                exclude_fields=_excluded(train_cfg.predict_interval, train_cfg.predict_vol_dec),
            )

            # NaN guard
            if not math.isfinite(loss.item()):
                logger.error("NaN/Inf loss at step %d — stopping", global_step)
                raise RuntimeError(
                    f"Non-finite loss ({loss.item()}) at step {global_step}"
                )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.grad_clip
            )
            optimizer.step()
            scheduler.step()

            # Track accuracy metrics (detached, no grad)
            with torch.no_grad():
                train_metrics.update(logits, labels)

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            # Accumulate per-field losses
            for k, v in field_losses.items():
                field_loss_accum[k] = field_loss_accum.get(k, 0.0) + v

            elapsed = time.perf_counter() - t0
            tps = epoch_steps * train_cfg.batch_size * train_cfg.seq_len / elapsed
            cur_lr = scheduler.get_last_lr()[0]
            postfix = dict(
                loss=f"{loss_val:.3f}",
                avg=f"{epoch_loss / epoch_steps:.3f}",
                lr=f"{cur_lr:.1e}",
                tps=f"{tps:.0f}",
                gnorm=f"{grad_norm:.2f}",
            )

            if global_step % train_cfg.log_interval == 0:
                avg = epoch_loss / epoch_steps
                for k, v in field_loss_accum.items():
                    postfix[k] = f"{v / epoch_steps:.2f}"
                acc_str = train_metrics.format()
                field_avg = " ".join(
                    f"{k}={v / epoch_steps:.3f}" for k, v in field_loss_accum.items()
                )
                logger.info(
                    "step=%d loss=%.4f [%s] %s tok/s=%.0f gnorm=%.2f",
                    global_step,
                    avg,
                    field_avg,
                    acc_str,
                    tps,
                    grad_norm,
                )

            if global_step % train_cfg.val_interval == 0:
                vl, vl_fields, vm = _validate(model, val_loader, device, label_smooth_sigma=train_cfg.label_smooth_sigma, order_type_weights=train_cfg.order_type_weights, use_conditioning=train_cfg.use_conditioning, predict_interval=train_cfg.predict_interval, predict_vol_dec=train_cfg.predict_vol_dec)
                val_losses.append(vl)
                field_str = " ".join(f"{k}={v:.3f}" for k, v in vl_fields.items())
                logger.info(
                    "  val_loss=%.4f [%s] %s (best=%.4f)",
                    vl,
                    field_str,
                    vm.format(),
                    best_val,
                )
                if vl < best_val:
                    best_val = vl
                    patience_counter = 0
                    save_checkpoint(
                        ckpt_path / "best.pt",
                        model,
                        optimizer,
                        model_cfg,
                        tokenizer_cfg,
                        epoch,
                        global_step,
                        vl,
                        data_dir=train_cfg.data_dir,
                    label_smooth_sigma=train_cfg.label_smooth_sigma,
                    )
                else:
                    patience_counter += 1
                    if train_cfg.patience and patience_counter >= train_cfg.patience:
                        logger.info(
                            "  Early stopping: val loss hasn't improved for "
                            "%d checks (best=%.4f)", patience_counter, best_val
                        )
                        logger.info(
                            "Early stopping at step %d (patience=%d)",
                            global_step,
                            train_cfg.patience,
                        )
                        early_stopped = True
                save_checkpoint(
                    ckpt_path / "latest.pt",
                    model,
                    optimizer,
                    model_cfg,
                    tokenizer_cfg,
                    epoch,
                    global_step,
                    vl,
                    data_dir=train_cfg.data_dir,
                    label_smooth_sigma=train_cfg.label_smooth_sigma,
                )
                model.train()

            if train_cfg.max_steps is not None and global_step >= train_cfg.max_steps:
                logger.info("Reached max_steps=%d, stopping early", train_cfg.max_steps)
                break

            if early_stopped:
                break



        # Break outer epoch loop too
        if train_cfg.max_steps is not None and global_step >= train_cfg.max_steps:
            break
        if early_stopped:
            break

        avg_train = epoch_loss / max(epoch_steps, 1)
        train_losses.append(avg_train)
        vl, vl_fields, val_metrics = _validate(model, val_loader, device, label_smooth_sigma=train_cfg.label_smooth_sigma, order_type_weights=train_cfg.order_type_weights, use_conditioning=train_cfg.use_conditioning, predict_interval=train_cfg.predict_interval, predict_vol_dec=train_cfg.predict_vol_dec)
        val_losses.append(vl)
        if vl < best_val:
            best_val = vl
            save_checkpoint(
                ckpt_path / "best.pt",
                model,
                optimizer,
                model_cfg,
                tokenizer_cfg,
                epoch,
                global_step,
                vl,
                data_dir=train_cfg.data_dir,
                    label_smooth_sigma=train_cfg.label_smooth_sigma,
            )

        field_str = " ".join(f"{k}={v:.3f}" for k, v in vl_fields.items())
        cache_info = ""
        if hasattr(train_loader.dataset, "cache_hit_rate"):
            cache_info = f" cache_hit={train_loader.dataset.cache_hit_rate:.1%}"
        epoch_summary = (
            f"Epoch {epoch} done: train={avg_train:.4f} val={vl:.4f} "
            f"[{field_str}] time={time.perf_counter() - t0:.0f}s{cache_info}"
        )
        logger.info(epoch_summary)
        logger.info("  train: %s", train_metrics.format())
        logger.info("  val:   %s", val_metrics.format())

    save_checkpoint(
        ckpt_path / "latest.pt",
        model,
        optimizer,
        model_cfg,
        tokenizer_cfg,
        epoch,
        global_step,
        best_val,
        data_dir=train_cfg.data_dir,
                    label_smooth_sigma=train_cfg.label_smooth_sigma,
    )

    return model, train_losses, val_losses


def _validate(
    model: OrderModel,
    loader: DataLoader,
    device: str,
    max_batches: int = 200,
    label_smooth_sigma: float = 0.0,
    order_type_weights: list[float] | None = None,
    use_conditioning: bool = True,
    predict_interval: bool = True,
    predict_vol_dec: bool = True,
) -> tuple[float, dict[str, float], StepMetrics]:
    """Returns (total_loss, per_field_avg_losses, accuracy_metrics)."""
    model.eval()
    total = 0.0
    field_totals: dict[str, float] = {}
    metrics = StepMetrics()
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            logits, labels = _forward_batch(model, batch, device, use_conditioning,
                                            predict_interval, predict_vol_dec)
            loss, field_losses = compute_loss(
                logits, labels, label_smooth_sigma=label_smooth_sigma,
                order_type_weights=order_type_weights,
                exclude_fields=_excluded(predict_interval, predict_vol_dec),
            )
            total += loss.item()
            for k, v in field_losses.items():
                field_totals[k] = field_totals.get(k, 0.0) + v
            metrics.update(logits, labels)
            n += 1
    avg = total / max(n, 1)
    field_avgs = {k: v / max(n, 1) for k, v in field_totals.items()}
    return avg, field_avgs, metrics

