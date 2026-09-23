"""The 0.6.2 skill-leg dials are wired into the scorer and default to the 0.6.1 leg.

With every new dial at its default the de-beta map is the one the previous scorer produced, byte for
byte, and the new quantities ride along in the detail for the gauges. Each dial then changes the map
the way its help text says.
"""
from collections import defaultdict
from types import SimpleNamespace

from taos.im.validator.debeta import accumulate_book_mtm
from taos.im.validator.reward import compute_debeta_scores


def _trade(p, q, s, maker, taker):
    return {"p": p, "q": q, "s": s, "Ma": maker, "Ta": taker}


def _dd2():
    return defaultdict(lambda: defaultdict(float))


def _validator(**dials):
    cfg = dict(enabled=True, weight=0.5, w_make=0.30, centered_window=15, floor_scale=0.5, min_books=1,
               presence_gate=1, presence_window=10)
    cfg.update(dials)
    self = SimpleNamespace(config=SimpleNamespace(scoring=SimpleNamespace(
        debeta=SimpleNamespace(**cfg), kappa=SimpleNamespace(lookback=30))))
    mtm, invsum, invn, inv, pf, pl, drift = _dd2(), _dd2(), {}, _dd2(), {}, {}, {}
    heldn, heldinv, helddrift, notional = _dd2(), _dd2(), _dd2(), _dd2()
    hists = {k: {} for k in ("mtm", "invsum", "invn", "drift", "heldn", "heldinv", "helddrift", "notional")}
    bg = -9
    for b in range(6):
        for ts in (10, 20, 30):
            # uid 1: a round trip that earns inside every sub-window; uid 2: buys once and rides the drift
            batch = [_trade(100.0, 1.0, 0, bg, 1), _trade(101.0 + 0.1 * b, 1.0, 0, 1, bg)]
            if ts == 10:
                batch.append(_trade(101.0 + 0.1 * b, 3.0, 0, bg, 2))
            batch.append(_trade(102.0 + 0.1 * b + ts / 100.0, 1.0, 0, bg, bg))
            accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, b, batch,
                                mtm_hist=hists["mtm"], invsum_hist=hists["invsum"], invn_hist=hists["invn"],
                                drift=drift, drift_hist=hists["drift"], ts=ts,
                                heldn=heldn, heldinv=heldinv, helddrift=helddrift, notional=notional,
                                heldn_hist=hists["heldn"], heldinv_hist=hists["heldinv"],
                                helddrift_hist=hists["helddrift"], notional_hist=hists["notional"])
    self.debeta_mtm, self.debeta_invsum, self.debeta_invn, self.debeta_drift = mtm, invsum, invn, drift
    self.debeta_heldn, self.debeta_heldinv, self.debeta_helddrift, self.debeta_notional = heldn, heldinv, helddrift, notional
    self.debeta_mtm_hist, self.debeta_invsum_hist = hists["mtm"], hists["invsum"]
    self.debeta_invn_hist, self.debeta_drift_hist = hists["invn"], hists["drift"]
    cb, cs = _dd2(), _dd2()
    for b in range(6):
        cb[1][b] = 1.0
        cs[1][b] = 1.0
        cb[2][b] = 1.0
    self.capture_buy_sums, self.capture_sell_sums = cb, cs
    self.miner_presence = {1: [True] * 10, 2: [True] * 4 + [False] * 6}
    return self


def test_defaults_reproduce_the_previous_map_and_publish_the_new_quantities():
    v = _validator()
    scores = compute_debeta_scores(v)
    assert scores and scores[1] > 0
    d = v.debeta_detail[1]
    for key in ("skill_other", "skill_weakest3", "skill_books_kept", "skill_pool_factor", "notional", "presence_share"):
        assert key in d, key
    assert d["skill_pool_factor"] == 1.0, "no hurdle, whole pool"
    assert d["notional"] > 0
    assert d["presence_share"] == 1.0 and v.debeta_detail[2]["presence_share"] == 0.4
    # a validator without the new accumulators (a snapshot from before them) still scores
    bare = _validator()
    for name in ("debeta_heldn", "debeta_heldinv", "debeta_helddrift", "debeta_notional",
                 "debeta_mtm_hist", "debeta_invsum_hist", "debeta_invn_hist", "debeta_drift_hist"):
        delattr(bare, name)
    assert compute_debeta_scores(bare) == scores


def test_held_variant_switches_the_skill_pool():
    drift_v, held_v = _validator(skill_variant="drift"), _validator(skill_variant="held")
    a, b = compute_debeta_scores(drift_v), compute_debeta_scores(held_v)
    assert drift_v.debeta_detail[2]["skill_raw"] == held_v.debeta_detail[2]["skill_other"]
    assert held_v.debeta_detail[2]["skill_raw"] == drift_v.debeta_detail[2]["skill_other"]
    assert a != b or drift_v.debeta_detail[1]["skill_raw"] != held_v.debeta_detail[1]["skill_raw"]


def test_a_prohibitive_hurdle_empties_the_pool_and_scaling_zeroes_the_skill_rank():
    v = _validator(skill_hurdle_bps=1_000_000.0, skill_pool_scaling=1)
    scores = compute_debeta_scores(v)
    d = v.debeta_detail[1]
    assert d["skill_pool_factor"] == 0.0 and d["skill_books_kept"] == 0
    assert d["skill_rank"] == 0.0
    assert abs(scores[1] - 0.30 * d["making_rank"]) < 1e-9, "only the making leg is left"


def test_subwindow_gating_keeps_the_consistent_trader_and_drops_the_rider():
    v = _validator(skill_subwindows=3, skill_subwindow_form="sign_gated", skill_subwindow_min_agree=2)
    compute_debeta_scores(v)
    assert v.debeta_detail[1]["skill_books_kept"] == 6, "positive in every sub-window on every book"
    assert v.debeta_detail[2]["skill_books_kept"] < 6, "one excursion does not agree across the window"


def test_presence_share_dials_grade_and_cut():
    weighted = _validator(presence_share_weighting=1)
    plain = _validator()
    s_w, s_p = compute_debeta_scores(weighted), compute_debeta_scores(plain)
    assert abs(s_w[2] - 0.4 * s_p[2]) < 1e-9 and s_w[1] == s_p[1]
    cut = _validator(presence_min_share=0.5)
    s_c = compute_debeta_scores(cut)
    assert 2 not in s_c and cut.debeta_detail[2]["present"] is False and 1 in s_c
