# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Closed-loop inference: generate → matching engine → LOB feedback → next step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from GenTRX.src.model import OrderModel
from GenTRX.src.orderbook import MatchingEngine, LobSnapshot
from GenTRX.src.tokenizer import OrderTokenizer
from GenTRX.src.util.schema import ASK, BID, CANCEL, EXEC_BUY, EXEC_SELL


def _sample_last(logits: dict, temperature: float) -> dict:
    """Sample (B, 1) tokens per field from last-position logits."""
    out = {}
    for name, fl in logits.items():
        probs = F.softmax(fl[:, -1, :] / temperature, dim=-1)
        out[name] = torch.multinomial(probs, 1)
    return out


# Field name (model head / logits key) → prompt tensor key.
_FIELD_TO_PROMPT_KEY = {
    "order_type": "order_types",
    "price": "price_bins",
    "vol_int": "vol_int_bins",
    "vol_dec": "vol_dec_bins",
    "interval": "interval_bins",
}


def score_sequence(
    model: OrderModel,
    tokenizer: OrderTokenizer,
    prompt: dict[str, torch.Tensor],
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Per-field, per-step log-probability of the observed next token.

    A density readout, not a generator. One forward over `prompt` (the
    same 8-tensor dict the generators take, but length T>1 carrying real
    observed tokens) returns `log p(token[t+1] | tokens[:t+1])` for each
    predicted field. Low values flag orders the model found surprising.
    Returns a dict keyed by field (order_type, price, vol_int, vol_dec,
    interval), each a length-(T-1) array for batch row 0.
    """
    model.eval()
    seqs = {k: v.to(device) for k, v in prompt.items()}
    with torch.no_grad():
        logits = model(
            seqs["order_types"], seqs["price_bins"], seqs["vol_int_bins"],
            seqs["vol_dec_bins"], seqs["interval_bins"],
            seqs["lob_volumes"], seqs["time_of_day"], seqs["mid_deltas"],
        )
    out: dict[str, np.ndarray] = {}
    for field, key in _FIELD_TO_PROMPT_KEY.items():
        lp = F.log_softmax(logits[field][:, :-1, :], dim=-1)
        targets = seqs[key][:, 1:].unsqueeze(-1)
        gathered = lp.gather(-1, targets).squeeze(-1)
        out[field] = gathered[0].detach().cpu().numpy()
    return out


@dataclass
class GeneratedOrder:
    order_type: int
    price_bin: int
    vol_int_bin: int
    vol_dec_bin: int
    interval_bin: int
    mid_price: int
    lob_snapshot: LobSnapshot
    is_buy: bool = True
    price: int = 0


def _evict_oldest(cache, block: int) -> None:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys = layer.keys[:, :, block:, :]
                layer.values = layer.values[:, :, block:, :]
        return
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, block:, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, block:, :]


def generate_with_engine(
    model: OrderModel,
    tokenizer: OrderTokenizer,
    engine: MatchingEngine,
    prompt: dict[str, torch.Tensor],
    n_orders: int = 100,
    temperature: float = 1.0,
    device: str = "cuda",
    vol_scale: float = 1.0,
    horizon_ns: int | None = None,
    slide_block: int = 0,
) -> list[GeneratedOrder]:
    """
    Generate orders autoregressively with matching engine feedback.

    slide_block > 0 enables a rolling KV-cache window: when the cache fills to
    max_seq_len, the oldest slide_block entries are evicted and generation
    continues past the context limit (absolute positions keep growing so RoPE
    relative offsets stay within the trained range). slide_block == 0 keeps the
    hard stop at max_seq_len.

    After each sampled order the matching engine processes it and provides an
    updated LOB snapshot, which is fed back into the next model step.

    Args:
        model (OrderModel): Trained order generation model.
        tokenizer (OrderTokenizer): Tokenizer that defines vocabulary and config.
        engine (MatchingEngine): Matching engine for processing generated orders.
        prompt (dict[str, torch.Tensor]): Conditioning context with keys
            'order_types', 'price_bins', 'vol_int_bins', 'vol_dec_bins',
            'interval_bins', 'lob_volumes', 'time_of_day', 'mid_deltas' —
            each shaped (1, T).
        n_orders (int): Number of orders to generate. Defaults to 100.
        temperature (float): Sampling temperature. Defaults to 1.0.
        device (str): Torch device string. Defaults to 'cuda'.

    Returns:
        list[GeneratedOrder]: Generated order objects with their matching-engine
            LOB snapshots.
    """
    model.eval()
    cfg = tokenizer.config
    mcfg = model.config
    max_ctx = mcfg.max_seq_len

    seqs = {k: v.to(device) for k, v in prompt.items()}
    T_prompt = seqs["order_types"].shape[1]
    last_tod = int(seqs["time_of_day"][0, -1].item())
    snap = engine.snapshot()
    session_open_mid = snap.mid_price if snap.mid_price > 0 else None

    generated: list[GeneratedOrder] = []
    cum_ns: int = 0

    cache = DynamicCache()
    with torch.no_grad():
        logits = model.forward_cached(
            seqs["order_types"], seqs["price_bins"], seqs["vol_int_bins"],
            seqs["vol_dec_bins"], seqs["interval_bins"],
            seqs["lob_volumes"], seqs["time_of_day"], seqs["mid_deltas"],
            past_key_values=cache, position_offset=0,
        )
        position = T_prompt

        for _ in range(n_orders):
            if horizon_ns is not None and cum_ns >= horizon_ns:
                break
            if cache.get_seq_length() >= max_ctx:
                if slide_block <= 0:
                    break
                _evict_oldest(cache, min(slide_block, max_ctx - 1))

            sampled = _sample_last(logits, temperature)
            otype = sampled["order_type"].item()
            p_bin = sampled["price"].item()
            vi_bin = sampled["vol_int"].item()
            vd_bin = sampled["vol_dec"].item()
            i_bin = sampled["interval"].item()

            snap = engine.snapshot()
            mid = snap.mid_price
            price = bin_to_price(p_bin, cfg.price, mid, asset_ref=_ar(cfg))
            volume = bins_to_volume(
                vi_bin, vd_bin, cfg.vol_int, cfg.vol_dec, scale=vol_scale, asset_ref=_ar(cfg),
            )
            eng_otype, price, is_buy, apply_engine = _decode_for_engine(otype, price, snap, engine)
            if volume > 0 and apply_engine:
                engine.process_order(eng_otype, price, volume, is_buy)

            new_snap = engine.snapshot()
            new_lob = _snap_to_tensor(new_snap, cfg.lob_depth, device).unsqueeze(0)

            interval_ns = bin_to_interval_ns(i_bin, cfg.interval)
            cum_ns += interval_ns
            last_tod = (last_tod + int(interval_ns / 1e9)) % 86400
            new_tod = torch.tensor([[last_tod // cfg.time_bin_seconds]], device=device)

            new_mid = new_snap.mid_price
            if session_open_mid is None and new_mid > 0:
                session_open_mid = new_mid
            delta = (new_mid - session_open_mid) if session_open_mid else 0
            delta_clipped = max(-cfg.max_mid_delta, min(cfg.max_mid_delta, delta))
            new_md = torch.tensor([[delta_clipped + cfg.max_mid_delta]], device=device)

            if apply_engine:
                generated.append(
                    GeneratedOrder(
                        order_type=otype, price_bin=p_bin, vol_int_bin=vi_bin,
                        vol_dec_bin=vd_bin, interval_bin=i_bin,
                        mid_price=new_snap.mid_price, lob_snapshot=new_snap,
                        is_buy=is_buy, price=price,
                    )
                )

            logits = model.forward_cached(
                sampled["order_type"], sampled["price"],
                sampled["vol_int"], sampled["vol_dec"], sampled["interval"],
                new_lob, new_tod, new_md,
                past_key_values=cache, position_offset=position,
            )
            position += 1

    return generated


def _decode_for_engine(otype, price, snap, engine):
    """Map a sampled order_type to (engine_type, price, is_buy, apply). Limits pass
    through; cancels snap to a resting level; executions become a marketable order
    consuming the opposite best (skipped if that side is empty)."""
    if otype == CANCEL:
        side, snapped, ok = _resolve_cancel_target(engine, price)
        return CANCEL, snapped, side, ok
    if otype in (EXEC_BUY, EXEC_SELL):
        is_buy = otype == EXEC_BUY
        opp = snap.ask_prices if is_buy else snap.bid_prices
        if opp and opp[0] > 0:
            return (BID if is_buy else ASK), opp[0], is_buy, True
        return BID, price, is_buy, False
    return otype, price, (otype == BID), True


def _resolve_cancel_target(engine, price_ticks: int) -> tuple[bool, int, bool]:
    """Infer cancel side from price vs touch, snap price to nearest level. ok=False if no levels exist."""
    snap = engine.snapshot()
    best_bid = snap.bid_prices[0] if snap.bid_prices else None
    best_ask = snap.ask_prices[0] if snap.ask_prices else None
    if best_bid is None and best_ask is None:
        return False, price_ticks, False
    if best_bid is None:
        is_buy = False
    elif best_ask is None:
        is_buy = True
    elif price_ticks <= best_bid:
        is_buy = True
    elif price_ticks >= best_ask:
        is_buy = False
    else:
        bid_dist = abs(price_ticks - best_bid)
        ask_dist = abs(price_ticks - best_ask)
        is_buy = bid_dist <= ask_dist
    levels = engine.bids if is_buy else engine.asks
    if not levels:
        return is_buy, price_ticks, False
    nearest = min(levels, key=lambda lvl: abs(lvl.price - price_ticks))
    return is_buy, nearest.price, True


def _ar(cfg):
    """The tokenizer config's asset_ref iff normalized (else None), for decode branching.
    Pass the FULL tokenizer config (tokenizer.config), not a per-field BinConfig."""
    return cfg.asset_ref if getattr(cfg, "normalize", False) else None


def bin_to_price(bin_idx: int, cfg, mid_price: int, asset_ref=None) -> int:
    """Inverse of digitize for price bins. cfg.center respects symmetric_log spacing.
    With asset_ref (normalized model) cfg.center is p_norm -> invert via price_from_norm
    to rel_price ticks; else cfg.center is already rel_price ticks."""
    if asset_ref is not None:
        from GenTRX.src.asset_norm import price_from_norm
        return mid_price + int(price_from_norm(cfg.center(bin_idx), mid_price, asset_ref))
    return mid_price + int(round(cfg.center(bin_idx)))


def bins_to_volume(
    vi_bin: int,
    vd_bin: int,
    vi_cfg,
    vd_cfg,
    scale: float = 1.0,
    asset_ref=None,
) -> int:
    """Reconstruct volume from int + dec bins. scale=1.0 for natural-
    unit engines (offline path), 10**volumeDecimals when the engine
    is in tick units (live state-stream path). With asset_ref (normalized
    model), vi+vd reconstruct qty/median_qty -> multiply by median_qty."""
    natural = vi_cfg.center(vi_bin) + vd_cfg.center(vd_bin)
    if asset_ref is not None:
        natural *= float(asset_ref["median_qty"])
    return max(1, int(round(natural * scale)))


def bin_to_interval_ns(i_bin: int, interval_cfg) -> int:
    """Reconstruct an interval in ns from a sampled bin id. Interval
    bins are log_scale so cfg.center returns the correct magnitude
    (linear-spaced midpoints mis-read by orders of magnitude)."""
    return max(0, int(round(interval_cfg.center(i_bin))))


# Legacy private aliases kept for in-tree callers.
_bin_to_price = bin_to_price
_bins_to_volume = bins_to_volume


def generate_trajectory(
    model: OrderModel,
    tokenizer: OrderTokenizer,
    engine: MatchingEngine,
    prompt: dict[str, torch.Tensor],
    n_orders: int = 100,
    temperature: float = 1.0,
    device: str = "cpu",
    seed: int | None = None,
    vol_scale: float = 1.0,
    mode: str = "closed",
    horizon_ns: int | None = None,
    block_size: int = 8,
    slide_block: int = 0,
) -> list[dict]:
    """Generate one trajectory as list of dicts. mode: closed | open | hybrid (block_size for hybrid)."""
    if seed is not None:
        torch.manual_seed(int(seed))

    if mode == "open":
        raw = generate_with_engine_open_loop(
            model=model, tokenizer=tokenizer, engine=engine, prompt=prompt,
            n_orders=n_orders, temperature=temperature, device=device,
            vol_scale=vol_scale, horizon_ns=horizon_ns,
        )
    elif mode == "hybrid":
        raw = generate_with_engine_hybrid(
            model=model, tokenizer=tokenizer, engine=engine, prompt=prompt,
            n_orders=n_orders, temperature=temperature, device=device,
            vol_scale=vol_scale, horizon_ns=horizon_ns, block_size=block_size,
        )
    else:
        raw = generate_with_engine(
            model=model, tokenizer=tokenizer, engine=engine, prompt=prompt,
            n_orders=n_orders, temperature=temperature, device=device,
            vol_scale=vol_scale, horizon_ns=horizon_ns, slide_block=slide_block,
        )

    cfg = tokenizer.config
    out: list[dict] = []
    for o in raw:
        volume = bins_to_volume(
            o.vol_int_bin, o.vol_dec_bin, cfg.vol_int, cfg.vol_dec,
            scale=vol_scale, asset_ref=_ar(cfg),
        )
        interval_ns = bin_to_interval_ns(o.interval_bin, cfg.interval)
        out.append({
            "order_type": int(o.order_type),
            "is_buy": bool(o.is_buy),
            "price": int(o.price),
            "volume": int(volume),
            "interval_ns": interval_ns,
            "mid_price": int(o.mid_price),
        })
    return out


def generate_with_engine_open_loop(
    model: OrderModel,
    tokenizer: OrderTokenizer,
    engine: MatchingEngine,
    prompt: dict[str, torch.Tensor],
    n_orders: int = 100,
    temperature: float = 1.0,
    device: str = "cpu",
    vol_scale: float = 1.0,
    horizon_ns: int | None = None,
) -> list[GeneratedOrder]:
    """Sample N orders with frozen initial conditioning, then apply to engine at the end."""
    model.eval()
    cfg = tokenizer.config
    mcfg = model.config
    max_ctx = mcfg.max_seq_len

    seqs = {k: v.to(device) for k, v in prompt.items()}
    T_prompt = seqs["order_types"].shape[1]
    fixed_lob = seqs["lob_volumes"][:, -1:, :]
    fixed_tod = seqs["time_of_day"][:, -1:]
    fixed_md = seqs["mid_deltas"][:, -1:]

    sampled_bins: list[tuple[int, int, int, int, int]] = []
    cum_ns_sample: int = 0

    cache = DynamicCache()
    with torch.no_grad():
        logits = model.forward_cached(
            seqs["order_types"], seqs["price_bins"], seqs["vol_int_bins"],
            seqs["vol_dec_bins"], seqs["interval_bins"],
            seqs["lob_volumes"], seqs["time_of_day"], seqs["mid_deltas"],
            past_key_values=cache, position_offset=0,
        )
        position = T_prompt

        for _ in range(n_orders):
            if horizon_ns is not None and cum_ns_sample >= horizon_ns:
                break
            if position >= max_ctx:
                break

            sampled = _sample_last(logits, temperature)
            otype = sampled["order_type"].item()
            p_bin = sampled["price"].item()
            vi_bin = sampled["vol_int"].item()
            vd_bin = sampled["vol_dec"].item()
            i_bin = sampled["interval"].item()
            sampled_bins.append((otype, p_bin, vi_bin, vd_bin, i_bin))
            cum_ns_sample += bin_to_interval_ns(i_bin, cfg.interval)

            logits = model.forward_cached(
                sampled["order_type"], sampled["price"],
                sampled["vol_int"], sampled["vol_dec"], sampled["interval"],
                fixed_lob, fixed_tod, fixed_md,
                past_key_values=cache, position_offset=position,
            )
            position += 1

    # Apply in order to the engine so the final state is consistent.
    generated: list[GeneratedOrder] = []
    for (otype, p_bin, vi_bin, vd_bin, i_bin) in sampled_bins:
        snap = engine.snapshot()
        mid = snap.mid_price
        price = bin_to_price(p_bin, cfg.price, mid, asset_ref=_ar(cfg))
        volume = bins_to_volume(vi_bin, vd_bin, cfg.vol_int, cfg.vol_dec, scale=vol_scale, asset_ref=_ar(cfg))
        eng_otype, price, is_buy, apply_engine = _decode_for_engine(otype, price, snap, engine)
        if volume > 0 and apply_engine:
            engine.process_order(eng_otype, price, volume, is_buy)
        new_snap = engine.snapshot()
        if apply_engine:
            generated.append(GeneratedOrder(
                order_type=otype, price_bin=p_bin, vol_int_bin=vi_bin,
                vol_dec_bin=vd_bin, interval_bin=i_bin,
                mid_price=new_snap.mid_price, lob_snapshot=new_snap,
                is_buy=is_buy, price=price,
            ))
    return generated


def generate_with_engine_hybrid(
    model: OrderModel,
    tokenizer: OrderTokenizer,
    engine: MatchingEngine,
    prompt: dict[str, torch.Tensor],
    n_orders: int = 100,
    temperature: float = 1.0,
    device: str = "cpu",
    vol_scale: float = 1.0,
    horizon_ns: int | None = None,
    block_size: int = 8,
) -> list[GeneratedOrder]:
    """Sample `block_size` orders with frozen conditioning, apply to engine, refresh. block_size=1 = closed-loop."""
    model.eval()
    cfg = tokenizer.config
    mcfg = model.config
    max_ctx = mcfg.max_seq_len

    seqs = {k: v.to(device) for k, v in prompt.items()}
    last_tod = int(seqs["time_of_day"][0, -1].item())
    init_snap = engine.snapshot()
    session_open_mid = init_snap.mid_price if init_snap.mid_price > 0 else None

    generated: list[GeneratedOrder] = []
    cum_ns: int = 0
    order_idx = 0

    cache: DynamicCache | None = None
    position: int = 0

    def _prefill_from_seqs() -> tuple[DynamicCache, int, dict]:
        c = DynamicCache()
        T = seqs["order_types"].shape[1]
        out = model.forward_cached(
            seqs["order_types"], seqs["price_bins"], seqs["vol_int_bins"],
            seqs["vol_dec_bins"], seqs["interval_bins"],
            seqs["lob_volumes"], seqs["time_of_day"], seqs["mid_deltas"],
            past_key_values=c, position_offset=0,
        )
        return c, T, out

    with torch.no_grad():
        cache, position, logits = _prefill_from_seqs()

        while order_idx < n_orders:
            if horizon_ns is not None and cum_ns >= horizon_ns:
                break
            if position >= max_ctx:
                break
            this_block = min(block_size, n_orders - order_idx)

            fixed_lob = seqs["lob_volumes"][:, -1:, :]
            fixed_tod = seqs["time_of_day"][:, -1:]
            fixed_md = seqs["mid_deltas"][:, -1:]
            block_samples: list[tuple[int, int, int, int, int]] = []

            for _ in range(this_block):
                if position >= max_ctx:
                    break
                sampled = _sample_last(logits, temperature)
                otype = sampled["order_type"].item()
                p_bin = sampled["price"].item()
                vi_bin = sampled["vol_int"].item()
                vd_bin = sampled["vol_dec"].item()
                i_bin = sampled["interval"].item()
                block_samples.append((otype, p_bin, vi_bin, vd_bin, i_bin))

                seqs["order_types"] = torch.cat([seqs["order_types"], sampled["order_type"]], dim=1)
                seqs["price_bins"] = torch.cat([seqs["price_bins"], sampled["price"]], dim=1)
                seqs["vol_int_bins"] = torch.cat([seqs["vol_int_bins"], sampled["vol_int"]], dim=1)
                seqs["vol_dec_bins"] = torch.cat([seqs["vol_dec_bins"], sampled["vol_dec"]], dim=1)
                seqs["interval_bins"] = torch.cat([seqs["interval_bins"], sampled["interval"]], dim=1)
                seqs["lob_volumes"] = torch.cat([seqs["lob_volumes"], fixed_lob], dim=1)
                seqs["time_of_day"] = torch.cat([seqs["time_of_day"], fixed_tod], dim=1)
                seqs["mid_deltas"] = torch.cat([seqs["mid_deltas"], fixed_md], dim=1)

                logits = model.forward_cached(
                    sampled["order_type"], sampled["price"],
                    sampled["vol_int"], sampled["vol_dec"], sampled["interval"],
                    fixed_lob, fixed_tod, fixed_md,
                    past_key_values=cache, position_offset=position,
                )
                position += 1

            stop_after_block = False
            for (otype, p_bin, vi_bin, vd_bin, i_bin) in block_samples:
                snap = engine.snapshot()
                mid = snap.mid_price
                price = bin_to_price(p_bin, cfg.price, mid, asset_ref=_ar(cfg))
                volume = bins_to_volume(vi_bin, vd_bin, cfg.vol_int, cfg.vol_dec, scale=vol_scale, asset_ref=_ar(cfg))
                eng_otype, price, is_buy, apply_engine = _decode_for_engine(otype, price, snap, engine)
                if volume > 0 and apply_engine:
                    engine.process_order(eng_otype, price, volume, is_buy)
                new_snap = engine.snapshot()
                interval_ns = bin_to_interval_ns(i_bin, cfg.interval)
                cum_ns += interval_ns
                last_tod = (last_tod + int(interval_ns / 1e9)) % 86400
                if apply_engine:
                    generated.append(GeneratedOrder(
                        order_type=otype, price_bin=p_bin, vol_int_bin=vi_bin,
                        vol_dec_bin=vd_bin, interval_bin=i_bin,
                        mid_price=new_snap.mid_price, lob_snapshot=new_snap,
                        is_buy=is_buy, price=price,
                    ))
                order_idx += 1
                if horizon_ns is not None and cum_ns >= horizon_ns:
                    stop_after_block = True
                    break

            if stop_after_block or order_idx >= n_orders:
                break

            # Overwrite the latest seq position so the next block sees post-block state.
            new_snap = engine.snapshot()
            new_lob_2d = _snap_to_tensor(new_snap, cfg.lob_depth, device)
            new_lob_3d = new_lob_2d.unsqueeze(0)
            new_tod = torch.tensor([[last_tod // cfg.time_bin_seconds]], device=device)
            new_mid = new_snap.mid_price
            if session_open_mid is None and new_mid > 0:
                session_open_mid = new_mid
            delta = (new_mid - session_open_mid) if session_open_mid else 0
            delta_clipped = max(-cfg.max_mid_delta, min(cfg.max_mid_delta, delta))
            new_md = torch.tensor([[delta_clipped + cfg.max_mid_delta]], device=device)

            seqs["lob_volumes"][:, -1:, :] = new_lob_3d
            seqs["time_of_day"][:, -1:] = new_tod
            seqs["mid_deltas"][:, -1:] = new_md

            # Tail K/V is stale after the overwrite: rebuild cache.
            cache, position, logits = _prefill_from_seqs()

    return generated


def generate_trajectories_batched(
    model: OrderModel,
    tokenizer: OrderTokenizer,
    engines: list[MatchingEngine],
    prompts: list[dict[str, torch.Tensor]],
    n_orders: int = 100,
    temperature: float = 1.0,
    device: str = "cpu",
    seed: int | None = None,
    vol_scale: float = 1.0,
    horizon_ns: int | None = None,
) -> list[list[dict]]:
    """K-rollout closed-loop generation. One model forward at
    batch_dim=K per autoregressive step instead of K sequential calls.
    Returns K trajectory lists with the same dict shape as
    generate_trajectory."""
    K = len(engines)
    if K == 0:
        return []
    if K != len(prompts):
        raise ValueError("engines and prompts must have the same length")
    if seed is not None:
        torch.manual_seed(int(seed))

    model.eval()
    cfg = tokenizer.config
    mcfg = model.config
    max_ctx = mcfg.max_seq_len

    # All K prompts must share T_prompt for the batch stack.
    seqs: dict[str, torch.Tensor] = {
        key: torch.cat([p[key].to(device) for p in prompts], dim=0)
        for key in (
            "order_types", "price_bins", "vol_int_bins", "vol_dec_bins",
            "interval_bins", "lob_volumes", "time_of_day", "mid_deltas",
        )
    }
    T_prompt = seqs["order_types"].shape[1]

    last_tod = [int(prompts[k]["time_of_day"][0, -1].item()) for k in range(K)]
    initial_snaps = [e.snapshot() for e in engines]
    session_open_mid: list[int | None] = [
        s.mid_price if s.mid_price > 0 else None for s in initial_snaps
    ]
    generated: list[list[dict]] = [[] for _ in range(K)]
    cum_ns: list[int] = [0] * K

    cache = DynamicCache()
    with torch.no_grad():
        logits = model.forward_cached(
            seqs["order_types"], seqs["price_bins"], seqs["vol_int_bins"],
            seqs["vol_dec_bins"], seqs["interval_bins"],
            seqs["lob_volumes"], seqs["time_of_day"], seqs["mid_deltas"],
            past_key_values=cache, position_offset=0,
        )
        position = T_prompt

        for _ in range(n_orders):
            if horizon_ns is not None and all(c >= horizon_ns for c in cum_ns):
                break
            if position >= max_ctx:
                break

            sampled = _sample_last(logits, temperature)
            otype_b = sampled["order_type"].view(-1).tolist()
            p_bin_b = sampled["price"].view(-1).tolist()
            vi_bin_b = sampled["vol_int"].view(-1).tolist()
            vd_bin_b = sampled["vol_dec"].view(-1).tolist()
            i_bin_b = sampled["interval"].view(-1).tolist()

            new_lob_rows: list[torch.Tensor] = []
            new_tod_rows: list[torch.Tensor] = []
            new_md_rows: list[torch.Tensor] = []
            for k in range(K):
                otype = int(otype_b[k])
                p_bin = int(p_bin_b[k])
                vi_bin = int(vi_bin_b[k])
                vd_bin = int(vd_bin_b[k])
                i_bin = int(i_bin_b[k])

                snap = engines[k].snapshot()
                mid = snap.mid_price
                price = bin_to_price(p_bin, cfg.price, mid, asset_ref=_ar(cfg)) if mid > 0 else 0
                volume = bins_to_volume(
                    vi_bin, vd_bin, cfg.vol_int, cfg.vol_dec, scale=vol_scale, asset_ref=_ar(cfg),
                )
                eng_otype, price, is_buy, apply_engine = _decode_for_engine(otype, price, snap, engines[k])
                if volume > 0 and apply_engine:
                    engines[k].process_order(eng_otype, price, volume, is_buy)

                new_snap = engines[k].snapshot()
                new_lob_rows.append(_snap_to_tensor(new_snap, cfg.lob_depth, device))

                interval_ns = bin_to_interval_ns(i_bin, cfg.interval)
                last_tod[k] = (last_tod[k] + int(interval_ns / 1e9)) % 86400
                new_tod_rows.append(
                    torch.tensor([[last_tod[k] // cfg.time_bin_seconds]], device=device)
                )

                new_mid = new_snap.mid_price
                if session_open_mid[k] is None and new_mid > 0:
                    session_open_mid[k] = new_mid
                delta = (new_mid - session_open_mid[k]) if session_open_mid[k] else 0
                delta_clipped = max(-cfg.max_mid_delta, min(cfg.max_mid_delta, delta))
                new_md_rows.append(
                    torch.tensor([[delta_clipped + cfg.max_mid_delta]], device=device)
                )

                if apply_engine and (horizon_ns is None or cum_ns[k] < horizon_ns):
                    generated[k].append({
                        "order_type": int(otype),
                        "is_buy": bool(is_buy),
                        "price": int(price),
                        "volume": int(volume),
                        "interval_ns": int(interval_ns),
                        "mid_price": int(new_snap.mid_price),
                    })
                cum_ns[k] += interval_ns

            new_lob_K = torch.cat(new_lob_rows, dim=0).unsqueeze(1)
            new_tod_K = torch.cat(new_tod_rows, dim=0)
            new_md_K = torch.cat(new_md_rows, dim=0)

            logits = model.forward_cached(
                sampled["order_type"], sampled["price"],
                sampled["vol_int"], sampled["vol_dec"], sampled["interval"],
                new_lob_K, new_tod_K, new_md_K,
                past_key_values=cache, position_offset=position,
            )
            position += 1

    return generated


def _snap_to_tensor(snap: LobSnapshot, depth: int, device: str) -> torch.Tensor:
    ask_vols = list(snap.ask_volumes[:depth])
    ask_vols += [0] * (depth - len(ask_vols))
    bid_vols = list(snap.bid_volumes[:depth])
    bid_vols += [0] * (depth - len(bid_vols))
    return torch.tensor([ask_vols + bid_vols], device=device, dtype=torch.float32)


def init_engine_from_data(
    encoded: dict[str, np.ndarray],
    tokenizer: OrderTokenizer,
    n_warmup: int = 100,
) -> MatchingEngine:
    """Initialize matching engine by replaying seed orders."""
    engine = MatchingEngine()
    cfg = tokenizer.config

    n = min(n_warmup, len(encoded["order_types"]))
    for i in range(n):
        otype = int(encoded["order_types"][i])
        p_bin = int(encoded["price_bins"][i])
        vi_bin = int(encoded["vol_int_bins"][i])
        vd_bin = int(encoded["vol_dec_bins"][i])

        snap = engine.snapshot()
        mid = snap.mid_price
        price = _bin_to_price(p_bin, cfg.price, mid, asset_ref=_ar(cfg))
        volume = _bins_to_volume(vi_bin, vd_bin, cfg.vol_int, cfg.vol_dec, asset_ref=_ar(cfg))
        is_buy = otype == 0
        eng_otype = otype
        if otype in (EXEC_BUY, EXEC_SELL):  # execution -> marketable order consuming opp best
            is_buy = otype == EXEC_BUY
            opp = snap.ask_prices if is_buy else snap.bid_prices
            eng_otype = (BID if is_buy else ASK) if (opp and opp[0] > 0) else None
            if eng_otype is not None:
                price = opp[0]
        if volume > 0 and eng_otype is not None:
            engine.process_order(eng_otype, price, volume, is_buy)

    return engine
