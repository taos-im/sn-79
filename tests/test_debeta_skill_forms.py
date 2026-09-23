"""The 0.6.2 skill-leg forms, each behind a dial that defaults to today's behaviour.

Held-period drift strip: alpha is stripped against the drift over the prints the uid held inventory
on, so a closed round trip is scored once and a single constant position scores exactly zero.
Sub-window alphas: the window's per-batch histories are re-summed over k equal sub-windows; with
k = 1 the result is the windowed alpha, and the two skill forms pay temporal consistency.
Hurdle: an alpha counts only above hurdle_bps of the uid's filled notional on that book, and the
skill rank is scaled by the share of the pool that clears it.
Presence share: the graded companion of the absent set.
"""
from collections import defaultdict

from taos.im.validator.debeta import (accumulate_book_mtm, book_alphas_by_book, book_alphas_by_subwindow,
                                      book_alphas_held_by_book, debeta_scores, hurdle_filter, kappa_floored,
                                      presence_shares, skill_pool_factor, subwindow_skill)


def _dd2():
    return defaultdict(lambda: defaultdict(float))


def _state():
    return dict(mtm=_dd2(), invsum=_dd2(), invn={}, inv=_dd2(), pf={}, pl={}, drift={},
                heldn=_dd2(), heldinv=_dd2(), helddrift=_dd2(), notional=_dd2(),
                mtm_hist={}, invsum_hist={}, invn_hist={}, drift_hist={},
                heldn_hist={}, heldinv_hist={}, helddrift_hist={}, notional_hist={})


def _feed(st, book, batch, ts):
    accumulate_book_mtm(st["mtm"], st["invsum"], st["invn"], st["inv"], st["pf"], st["pl"], book, batch,
                        mtm_hist=st["mtm_hist"], invsum_hist=st["invsum_hist"], invn_hist=st["invn_hist"],
                        drift=st["drift"], drift_hist=st["drift_hist"], ts=ts,
                        heldn=st["heldn"], heldinv=st["heldinv"], helddrift=st["helddrift"],
                        notional=st["notional"], heldn_hist=st["heldn_hist"], heldinv_hist=st["heldinv_hist"],
                        helddrift_hist=st["helddrift_hist"], notional_hist=st["notional_hist"])


def _trade(p, q, side, maker, taker):
    # s == 0: taker buys from maker; s == 1: maker buys, taker sells
    return {"p": p, "q": q, "s": side, "Ma": maker, "Ta": taker}


BG = -5   # a background agent: never tracked


def test_held_variant_scores_a_closed_round_trip_once_and_ignores_later_prints():
    st = _state()
    # uid 1 buys at 100, price rises to 110, uid 1 sells at 110; then the book falls to 90 with uid 1 flat
    _feed(st, 0, [_trade(100.0, 1.0, 0, BG, 1), _trade(110.0, 1.0, 0, 1, BG), _trade(110.0, 1.0, 0, BG, BG)], ts=1)
    held_after_close = book_alphas_held_by_book(st["mtm"], st["heldinv"], st["heldn"], st["helddrift"])[1][0]
    win_after_close = book_alphas_by_book(st["mtm"], st["invsum"], st["invn"], st["drift"])[1][0]
    _feed(st, 0, [_trade(100.0, 1.0, 0, BG, BG), _trade(90.0, 1.0, 0, BG, BG)], ts=2)
    held_later = book_alphas_held_by_book(st["mtm"], st["heldinv"], st["heldn"], st["helddrift"])[1][0]
    win_later = book_alphas_by_book(st["mtm"], st["invsum"], st["invn"], st["drift"])[1][0]
    assert held_later == held_after_close, "prints after the flatten do not re-mark a closed trip"
    assert win_later != win_after_close, "the windowed alpha does re-mark it: that is the property the variant removes"


def test_held_variant_gives_exactly_zero_for_a_single_constant_position():
    st = _state()
    _feed(st, 0, [_trade(100.0, 2.0, 0, BG, 7), _trade(104.0, 1.0, 0, BG, BG), _trade(97.0, 1.0, 0, BG, BG),
                  _trade(120.0, 1.0, 0, BG, BG)], ts=1)
    held = book_alphas_held_by_book(st["mtm"], st["heldinv"], st["heldn"], st["helddrift"])[7][0]
    assert abs(held) < 1e-9
    assert st["notional"][7][0] == 200.0, "filled notional at the fill price"


def test_held_variant_credits_inventory_variation_inside_the_exposure():
    st = _state()
    # uid 3 buys 1 at 100, adds 1 at 102, the book goes to 110: more inventory on the way up
    _feed(st, 0, [_trade(100.0, 1.0, 0, BG, 3), _trade(102.0, 1.0, 0, BG, 3), _trade(110.0, 1.0, 0, BG, BG)], ts=1)
    held = book_alphas_held_by_book(st["mtm"], st["heldinv"], st["heldn"], st["helddrift"])[3][0]
    assert held > 0


def test_subwindows_with_k_one_reproduce_the_windowed_alpha_and_a_one_sided_excursion_fails_both_forms():
    st = _state()
    # book 0..4: uid 9 earns all its alpha in the first third of the window on every book
    for b in range(5):
        _feed(st, b, [_trade(100.0, 1.0, 0, BG, 9), _trade(103.0, 1.0, 0, 9, BG)], ts=10)
        _feed(st, b, [_trade(103.0, 1.0, 0, BG, BG)], ts=20)
        _feed(st, b, [_trade(103.0, 1.0, 0, BG, BG)], ts=30)
    full = book_alphas_by_book(st["mtm"], st["invsum"], st["invn"], st["drift"])
    one = book_alphas_by_subwindow(st["mtm_hist"], st["invsum_hist"], st["invn_hist"], st["drift_hist"], 5, 35, 1)
    assert one is None, "k = 1 is the windowed leg itself"
    three = book_alphas_by_subwindow(st["mtm_hist"], st["invsum_hist"], st["invn_hist"], st["drift_hist"], 5, 35, 3)
    for b in range(5):
        # the first sub-window holds the whole round trip and strips only its own drift
        assert three[9][b][0] is not None and three[9][b][0] > 0
        assert three[9][b][1] is None or abs(three[9][b][1]) < 1e-9
        assert three[9][b][2] is None or abs(three[9][b][2]) < 1e-9
    weakest, info_w = subwindow_skill(full, three, "weakest", 2, 0.0)
    gated, info_g = subwindow_skill(full, three, "sign_gated", 2, 0.0)
    assert weakest[9] == 0.0, "no alpha in the later sub-windows"
    assert gated[9] == 0.0 and info_g[9]["books"] == 0, "one agreeing sub-window of three is below the gate"
    assert kappa_floored(list(full[9].values()), 0.0) > 0, "the full-window leg would have paid it"


def test_a_uid_positive_in_every_subwindow_keeps_its_full_window_skill():
    st = _state()
    for b in range(5):
        for ts in (10, 20, 30):
            _feed(st, b, [_trade(100.0, 1.0, 0, BG, 4), _trade(101.0 + b * 0.1, 1.0, 0, 4, BG)], ts=ts)
    full = book_alphas_by_book(st["mtm"], st["invsum"], st["invn"], st["drift"])
    three = book_alphas_by_subwindow(st["mtm_hist"], st["invsum_hist"], st["invn_hist"], st["drift_hist"], 5, 35, 3)
    gated, info = subwindow_skill(full, three, "sign_gated", 2, 0.0)
    assert info[4]["books"] == 5
    assert abs(gated[4] - kappa_floored(list(full[4].values()), 0.0)) < 1e-9
    weakest, _ = subwindow_skill(full, three, "weakest", 2, 0.0)
    assert weakest[4] > 0


def test_hurdle_drops_negligible_alpha_and_zero_disables_it():
    alphas = {1: {0: 5.0, 1: 5.0, 2: 5.0, 3: 5.0}, 2: {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01}}
    notional = {1: {b: 1000.0 for b in range(4)}, 2: {b: 1000.0 for b in range(4)}}
    assert hurdle_filter(alphas, notional, 0) is alphas
    kept = hurdle_filter(alphas, notional, 10)          # 10 bps of 1000 = 1.0
    assert len(kept[1]) == 4 and len(kept[2]) == 0
    assert skill_pool_factor(kept, alphas, 4) == 0.5
    assert skill_pool_factor(alphas, alphas, 4) == 1.0
    assert skill_pool_factor({}, {}, 4) == 1.0, "no pool, nothing to scale"


def test_debeta_scores_takes_an_external_skill_and_scales_its_rank():
    cb = {1: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}, 2: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}}
    cs = {1: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}, 2: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}}
    # kappa is magnitude-blind: uid 1 is consistent, uid 2 has one losing book
    alphas = {1: [3.0, 3.0, 3.0, 3.0], 2: [1.0, -1.0, 1.0, 1.0]}
    base = debeta_scores(cb, cs, alphas, floor=0.0, w_make=0.0)
    swapped = debeta_scores(cb, cs, alphas, floor=0.0, w_make=0.0, skill_values={1: 0.0, 2: 5.0})
    assert base[1] > base[2] and swapped[2] > swapped[1]
    half = debeta_scores(cb, cs, alphas, floor=0.0, w_make=0.0, skill_rank_scale=0.5)
    assert abs(half[1] - 0.5 * base[1]) < 1e-9


def test_presence_shares_grade_what_the_absent_set_cuts():
    presence = {1: [True] * 10, 2: [False] * 10, 3: [True, False] * 5, 4: []}
    shares = presence_shares(presence, 10)
    assert shares[1] == 1.0 and shares[2] == 0.0 and abs(shares[3] - 0.5) < 1e-9
    assert 4 not in shares
    assert presence_shares({5: [False] * 8 + [True] * 2}, 2)[5] == 1.0, "only the last window counts"
