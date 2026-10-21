# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A load rebuilds the alternative mark's and the maker-liquidity legs' running sums from their saved histories.

The paid leg's running sums are rebuilt from the restored histories at load. The legs added in 0.6.5 restored their
histories and left their sums to be created empty, and the prune and the boundary shift subtract the expired mass from
a sum to keep it equal to its kept history, so after a resume those sums went negative and the published alternative
skill and liquidity read wrong for a whole window. The load now rebuilds them as it does the paid leg's.
"""
import test_scoring_shadow as tss
from taos.im.validator import persistence as p
from taos.im.validator import trade
from taos.im.validator.debeta import prune_hist_2level, sum_hist_1level, sum_hist_2level

TWO_LEVEL = (("debeta_mtm_alt_hist", "debeta_mtm_alt"), ("debeta_invsum_alt_hist", "debeta_invsum_alt"),
             ("debeta_maker_buy_notional_hist", "debeta_maker_buy_notional"),
             ("debeta_maker_sell_notional_hist", "debeta_maker_sell_notional"),
             ("debeta_at_best_hist", "debeta_at_best"), ("debeta_best_size_hist", "debeta_best_size"))
ONE_LEVEL = (("debeta_invn_alt_hist", "debeta_invn_alt"), ("debeta_drift_alt_hist", "debeta_drift_alt"))


def _t(p_, q, side, ma, ta):
    return {"y": "t", "p": p_, "q": q, "s": side, "Ma": ma, "Ta": ta}


def _restored():
    main = tss._fresh_main()
    trade.init_alt_mark_state(main)
    trade.init_liquidity_state(main)
    for b in range(tss._BOOKS):
        for k in range(1, 4):
            ts = k * tss._S
            book = {"b": [{"p": 99.9 + 0.5 * k, "q": 1.0}], "a": [{"p": 100.1 + 0.5 * k, "q": 1.0}]}
            fills = [_t(100.0 + 0.5 * k, 1.0, 0, 1, 2), _t(100.0 + 0.5 * k, 2.0, 1, 2, 1)]
            trade.accumulate_alt_mark(main, b, fills, book, "mid", 0, ts=ts)
            trade.accumulate_maker_notional(main, b, fills, ts=ts)
            trade.accumulate_at_best(main, 1, {b: book}, {b: {"o": [{"p": book["b"][0]["p"], "q": 3.0, "s": 0}]}}, ts=ts)
    state = p.build_debeta_state(main)
    assert any(state[h] for h, _ in TWO_LEVEL) and any(state[h] for h, _ in ONE_LEVEL), "the fixture fills every leg"
    v = tss._fresh_main()
    if not hasattr(v, "effective_max_uids"):
        v.effective_max_uids = tss._UIDS
    book_ids = list(range(tss._BOOKS))
    p._restore_trade_volumes(v, state, book_ids, set(book_ids))
    return v


def _plain2(d):
    return {k1: {k2: round(x, 9) for k2, x in d2.items()} for k1, d2 in d.items()}


def test_the_two_level_sums_equal_their_restored_histories():
    v = _restored()
    for hist_name, sum_name in TWO_LEVEL:
        assert hasattr(v, sum_name), sum_name
        assert _plain2(getattr(v, sum_name)) == _plain2(sum_hist_2level(getattr(v, hist_name))), sum_name
    assert any(getattr(v, s) for _, s in TWO_LEVEL), "the rounds left mass in at least one leg"


def test_the_one_level_sums_and_the_mark_state_exist_after_a_load():
    v = _restored()
    for hist_name, sum_name in ONE_LEVEL:
        assert hasattr(v, sum_name), sum_name
        got = {k: round(x, 9) for k, x in getattr(v, sum_name).items()}
        assert got == {k: round(x, 9) for k, x in sum_hist_1level(getattr(v, hist_name)).items()}, sum_name
    assert v.debeta_pfirst_alt == {} and v.debeta_mark_state_alt == {}


def test_the_first_prune_after_a_load_keeps_every_sum_equal_to_its_kept_history():
    v = _restored()
    for hist_name, sum_name in TWO_LEVEL:
        h, s = getattr(v, hist_name), getattr(v, sum_name)
        latest = max((ts for d2 in h.values() for tsd in d2.values() for ts in tsd), default=0)
        prune_hist_2level(h, s, latest)
        assert _plain2(s) == _plain2(sum_hist_2level(h)), sum_name
        if "mark" not in sum_name and "mtm" not in sum_name and "invsum" not in sum_name:
            assert all(x >= 0.0 for d2 in s.values() for x in d2.values()), f"{sum_name} is a volume, a count or a size"
