# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The per-agent scoring maps of the data-service ingest push, built in one place.

Three pushers used to build these maps by hand: the exchange path (`_build_agent_scoring_maps`), the
simulation path (an inlined copy of the same twenty lines) and the reporting child (a thinner copy that
sent a DIFFERENT quantity under `agent_kappa`: the kappa `score` where the other two sent `total`). The
copies had already drifted, and none of them carried the de-beta decomposition, so the UI could not show
what emissions follow. Every pusher now composes its scoring keys from the functions here.

Conventions the sinks rely on: uids are strings, values are floats, and a uid whose value is unknown is
OMITTED rather than sent as null. The sinks coerce with `float(... or 0)`, so a null would render an
unscored miner as ranked last; absence leaves the column empty, which is the truth.
"""
from typing import Any, Dict, Mapping, Optional


def _f(value) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def kappa_maps(kappa_values: Mapping) -> Dict[str, dict]:
    """The legacy kappa keys: raw total, normalized score, penalty, per-book maps and the median."""
    raw, score, penalty, books, books_w, median = {}, {}, {}, {}, {}, {}
    for uid, kv in dict(kappa_values or {}).items():
        if not kv or not isinstance(kv, Mapping):
            continue
        key = str(uid)
        total = _f(kv.get('total'))
        if total is not None:
            raw[key] = total
        normalized = _f(kv.get('normalized_total'))
        if normalized is not None:
            score[key] = normalized
        pen = _f(kv.get('penalty'))
        if pen is not None:
            penalty[key] = pen
        bks = {str(bid): float(v) for bid, v in (kv.get('books') or {}).items() if v is not None}
        if bks:
            books[key] = bks
        bw = {str(bid): float(v) for bid, v in (kv.get('books_weighted') or {}).items() if v is not None}
        if bw:
            books_w[key] = bw
        median[key] = round(float(kv.get('activity_weighted_normalized_median') or 0), 6)
    return {
        "agent_kappa": raw,
        "agent_kappa_score": score,
        "agent_kappa_penalty": penalty,
        "agent_kappa_books": books,
        "agent_kappa_books_w": books_w,
        "agent_median_kappa": median,
    }


# kappa_values field -> payload key. Written per uid by reward.score_uid; None on a cycle where de-beta
# did not apply to that uid (warming, fallback, never traded), which is why absence is the encoding.
_DEBETA_FIELDS = (
    ('debeta_score', 'agent_debeta_score'),
    ('making_rank', 'agent_making_rank'),
    ('skill_rank', 'agent_skill_rank'),
    ('making_raw', 'agent_making_raw'),
    ('skill_raw', 'agent_skill_raw'),
    ('p11_factor', 'agent_p11_factor'),
)


def debeta_maps(kappa_values: Mapping) -> Dict[str, dict]:
    """The de-beta decomposition per uid: score, leg ranks and raws, CP factor, coverage, eligibility."""
    out: Dict[str, dict] = {payload_key: {} for _, payload_key in _DEBETA_FIELDS}
    out['agent_num_scored_books'] = {}
    out['agent_scorable'] = {}
    for uid, kv in dict(kappa_values or {}).items():
        if not kv or not isinstance(kv, Mapping):
            continue
        key = str(uid)
        for field, payload_key in _DEBETA_FIELDS:
            value = _f(kv.get(field))
            if value is not None:
                out[payload_key][key] = value
        if kv.get('num_scored_books') is not None:
            try:
                out['agent_num_scored_books'][key] = int(kv['num_scored_books'])
            except (TypeError, ValueError):
                pass
        if kv.get('scorable') is not None:
            out['agent_scorable'][key] = bool(kv['scorable'])
    return out


def scoring_params(config: Any, kappa_values: Mapping = None) -> dict:
    """The scalar dials the UI shows beside the per-uid maps: the three trading weights, w_make and
    the floor. Weights come from the config (the dial is meaningful every cycle, warming included);
    w_make and the floor are whatever this cycle's carrier recorded, None while de-beta is not applied."""
    scoring = getattr(config, 'scoring', None)
    debeta = getattr(scoring, 'debeta', None)
    kappa_weight = _f(getattr(getattr(scoring, 'kappa', None), 'weight', None))
    pnl_weight = _f(getattr(getattr(scoring, 'pnl', None), 'weight', None))
    debeta_weight = _f(getattr(debeta, 'weight', None))
    w_make = _f(getattr(debeta, 'w_make', None))
    floor = None
    for kv in dict(kappa_values or {}).values():
        if isinstance(kv, Mapping):
            if floor is None:
                floor = _f(kv.get('debeta_floor'))
            if w_make is None:
                w_make = _f(kv.get('debeta_w_make'))
            if debeta_weight is None:
                debeta_weight = _f(kv.get('debeta_weight'))
        if floor is not None and w_make is not None and debeta_weight is not None:
            break
    return {
        "kappa_weight": kappa_weight,
        "pnl_weight": pnl_weight,
        "debeta_weight": debeta_weight,
        "debeta_w_make": w_make,
        "debeta_floor": floor,
        "debeta_enabled": bool(debeta_weight and debeta_weight > 0.0),
    }


def _sum_books(by_uid: Mapping) -> Dict[str, float]:
    return {
        str(uid): sum(float(v) for v in list(bks.values()))
        for uid, bks in list(by_uid.items())
        if bks
    }


def agent_scoring_maps(v: Any) -> dict:
    """Every per-agent scoring key of an ingest push, from the validator's accumulators.

    Both modes carry the same keys so the agent_snapshots row (top-level score/volume/pnl and per_book
    vol/fee/pnl/kappa) is populated identically; only what feeds the accumulators differs (the exchange
    folds the reconciled state, so its maps reflect settled activity only).

    Snapshots are taken in two atomic steps (outer dict copy first, then the inner copies over the
    stable snapshot) so a concurrent trade.py mutation from the reward thread cannot raise
    "dictionary changed size during iteration".
    """
    vs_outer = dict(getattr(v, "volume_sums", {}) or {})
    mvs_outer = dict(getattr(v, "maker_volume_sums", {}) or {})
    tvs_outer = dict(getattr(v, "taker_volume_sums", {}) or {})
    fs_outer = dict(getattr(v, "fee_sums", {}) or {})
    rt_outer = dict(getattr(v, "roundtrip_volume_sums", {}) or {})
    af_outer = dict(getattr(v, "activity_factors", {}) or {})
    snap_kv = dict(getattr(v, "kappa_values", {}) or {})
    scores = getattr(v, "scores", None)
    snap_scores = list(scores) if scores is not None else None
    snap_pnl_book = {u: dict(b) for u, b in dict(getattr(v, "agent_pnl_by_book", {}) or {}).items()}
    snap_pnl_total = dict(getattr(v, "agent_pnl_total", {}) or {})
    snap_vs = {u: dict(b) for u, b in vs_outer.items()}
    snap_mvs = {u: dict(b) for u, b in mvs_outer.items()}
    snap_tvs = {u: dict(b) for u, b in tvs_outer.items()}
    snap_fs = {u: dict(b) for u, b in fs_outer.items()}
    snap_rt = {u: dict(b) for u, b in rt_outer.items()}
    snap_af = {u: dict(b) for u, b in af_outer.items()}

    maps = {
        "agent_scores": ({str(i): float(snap_scores[i]) for i in range(len(snap_scores))}
                         if snap_scores is not None else {}),
        **kappa_maps(snap_kv),
        "agent_volume": _sum_books(snap_vs),
        "agent_maker_volume": _sum_books(snap_mvs),
        "agent_taker_volume": _sum_books(snap_tvs),
        "agent_pnl": {str(uid): round(float(pnl), 6) for uid, pnl in snap_pnl_total.items() if pnl != 0.0},
        "agent_pnl_book": {
            str(uid): {str(bid): round(float(val), 6) for bid, val in bks.items() if val != 0}
            for uid, bks in snap_pnl_book.items()
            if bks
        },
        "agent_volume_book": {
            str(uid): {str(bid): round(float(val), 4) for bid, val in bks.items() if val}
            for uid, bks in snap_vs.items()
            if bks
        },
        "agent_fee_book": {
            str(uid): {str(bid): round(float(val), 6) for bid, val in bks.items() if val != 0}
            for uid, bks in snap_fs.items()
            if bks
        },
        "agent_roundtrip_volume": {
            str(uid): round(float(sum(bks.values())), 4) for uid, bks in snap_rt.items() if bks
        },
        "agent_activity_factor": {
            str(uid): round(float(sum(bks.values()) / len(bks)), 4) for uid, bks in snap_af.items() if bks
        },
        **debeta_maps(snap_kv),
        "scoring_params": scoring_params(getattr(v, "config", None), snap_kv),
    }
    return maps
