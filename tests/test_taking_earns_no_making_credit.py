# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Taking never earns making credit. The captured pair books both legs of every fill against the centered mid, and
the skill leg reads it for the books a uid filled on, so it keeps every leg; a taker-leg pair beside it receives each
taker leg, and the making pool pays on the captured pair with the taker legs taken back out, in the quality factor's
denominator as in the paid credit. Before, a taker that traded ahead of a move booked positive capture and was paid
it as making. A state saved before the taker pair existed carries none, so the old taker legs age out of the lookback
while the new ones are taken out."""
import inspect
from collections import defaultdict
from types import SimpleNamespace

from taos.im.validator import debeta, persistence, reward, scoring_shadow, trade


def _pair():
    return defaultdict(dict), defaultdict(dict)


def test_a_taker_leg_is_booked_into_the_taker_pair_and_a_maker_leg_is_not():
    cb, cs = _pair()
    tb, ts = _pair()
    # side 1: the maker (Ma 7) buys at 99, the taker (Ta 9) sells; marked against a mid of 100
    t = {"p": 99.0, "q": 2.0, "s": 1, "Ma": 7, "Ta": 9}
    debeta._attribute_capture(cb, cs, 0, t, 100.0, None, None, None, taker=(tb, ts, None, None))
    assert cb[7][0] == 2.0 and cs[9][0] == -2.0, "the captured pair keeps both legs"
    assert ts[9][0] == -2.0 and 7 not in tb and 7 not in ts, "only the taker's leg enters the taker pair"


def test_a_pure_takers_captured_credit_is_all_taker_legs():
    cb, cs = _pair()
    tb, ts = _pair()
    trades = [{"p": 100.0 + 0.1 * i, "q": 1.0, "s": i % 2, "Ma": 7, "Ta": 9} for i in range(12)]
    debeta.accumulate_book_capture(cb, cs, 0, trades, 3, taker=(tb, ts, None, None))
    assert cb.get(9) == tb.get(9) and cs.get(9) == ts.get(9), "uid 9 only takes: its captured pair is its taker pair"
    assert 7 not in tb and 7 not in ts


def test_the_live_path_books_the_taker_pair_when_the_fill_settles():
    cb, cs = _pair()
    tb, ts = _pair()
    state = {}
    trades = [{"p": 100.0 + 0.1 * i, "q": 1.0, "s": i % 2, "Ma": 7, "Ta": 9, "ts": i} for i in range(12)]
    hists = (defaultdict(dict), defaultdict(dict))
    debeta.accumulate_book_capture(cb, cs, 0, trades, 3, mid_state=state, ts=1, now_ns=1,
                                   taker=(tb, ts, hists[0], hists[1]))
    debeta.flush_capture_state(state, cb, cs, 3, ts=2, now_ns=2, force=True, taker=(tb, ts, hists[0], hists[1]))
    assert cb.get(9) == tb.get(9) and cs.get(9) == ts.get(9)
    assert sum(v for books in hists[1].values() for by_ts in books.values() for v in by_ts.values()) == \
        sum(v for books in ts.values() for v in books.values()), "the taker legs are windowed like the rest"


def _validator(basis, cb, cs, tb=None, ts=None, rb=None, rs=None):
    v = SimpleNamespace(capture_buy_sums=cb, capture_sell_sums=cs, realized_buy_sums=rb or {}, realized_sell_sums=rs or {})
    if tb is not None:
        v.capture_taker_buy_sums, v.capture_taker_sell_sums = tb, ts
    v.config = SimpleNamespace(scoring=SimpleNamespace(debeta=SimpleNamespace(making_basis=basis)))
    return v


def test_e7b1s_case_a_taker_leg_no_longer_raises_a_makers_weaker_side():
    # uid 7 makes: buy side 100, sell side 20 on book 0; it also takes, adding +80 on the sell side
    v = _validator("captured", {7: {0: 100.0}}, {7: {0: 100.0}}, {7: {}}, {7: {0: 80.0}})
    mb, ms = reward.making_basis_sums(v)
    assert debeta.balanced_reward_per_book(mb, ms, [7]) == {7: 40.0}, "2 x min(100, 20), not 2 x min(100, 100)"


def test_the_quality_factor_reads_maker_fills_in_its_denominator_too():
    v = _validator("realized", {7: {0: 100.0}}, {7: {0: 100.0}}, {7: {}}, {7: {0: 80.0}},
                   rb={7: {0: 50.0}}, rs={7: {0: 10.0}})
    mb, ms = reward.making_basis_sums(v)
    # q = R / C over maker fills: 60 / 120 = 0.5, so the paid credit is 0.5 x 2 x min(100, 20)
    assert debeta.balanced_reward_per_book(mb, ms, [7]) == {7: 20.0}


def test_a_state_without_the_taker_pair_pays_as_before():
    v = _validator("captured", {7: {0: 100.0}}, {7: {0: 100.0}})
    assert reward.maker_capture_sums(v) == ({7: {0: 100.0}}, {7: {0: 100.0}})


def test_the_skill_leg_still_reads_every_fill():
    src = inspect.getsource(reward.compute_debeta_scores)
    assert "getattr(self, 'capture_buy_sums', {}) or {}, getattr(self, 'capture_sell_sums', {}) or {}," in src, \
        "traded_book_alphas keeps the all-fills pair for the books a uid filled on"
    assert "_cb, _cs = maker_capture_sums(self)" in src, "the published making gauges read the maker-only pair"


def test_the_pair_is_accumulated_windowed_saved_shadowed_and_reset():
    upd = inspect.getsource(trade.update_trade_volumes)
    assert upd.count("taker=(self.capture_taker_buy_sums, self.capture_taker_sell_sums,") == 2, "accumulate and flush"
    assert "prune_hist_2level(self.debeta_captkbuy_hist, self.capture_taker_buy_sums" in upd
    assert ("debeta_captkbuy_hist", "capture_taker_buy_sums") in trade._TWO_LEVEL_HISTS
    assert "debeta_captkbuy_hist" in persistence._DEBETA_HIST3 and "debeta_captksell_hist" in persistence._DEBETA_HIST3
    assert "capture_taker_buy_sums" in scoring_shadow._DEBETA_DD2 and "debeta_captkbuy_hist" in scoring_shadow._DEBETA_PLAIN
    assert "'capture_taker_buy_sums', 'capture_taker_sell_sums'" in inspect.getsource(trade.reset_agent_histories)
    assert "taker=_taker_pair(self)" in inspect.getsource(trade.shift_simulation_histories)
