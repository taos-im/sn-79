# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The liquidity a maker provides is accumulated per uid and published, so a making rule paying for it (two-sided
filled notional, the volume provided to takers on both sides) is judged on live data before it pays.

Three windowed quantities per uid and book, pruned, shifted, persisted and reset with the other de-beta accumulators:
the maker-side notional of its fills by side (buy, sell), the states in which it had an order resting at the best bid
or ask, and the size it had resting there. The scorer publishes per uid the two-sided notional (2 x min of the sides,
summed over books), the seconds at the best and the size at the best; pay reads none of them.
"""
import inspect
from types import SimpleNamespace

from taos.im.validator import report, reward, trade


def _t(p, q, s, ma, ta):
    return {"y": "t", "p": p, "q": q, "s": s, "Ma": ma, "Ta": ta}


def test_the_state_names_are_classified_and_handled_with_the_others():
    for n in ("debeta_maker_buy_notional", "debeta_maker_sell_notional", "debeta_at_best", "debeta_best_size",
              "debeta_maker_buy_notional_hist", "debeta_maker_sell_notional_hist", "debeta_at_best_hist", "debeta_best_size_hist"):
        assert n in trade.LIQUIDITY_STATE, n
    src = inspect.getsource(trade)
    for name in ("debeta_maker_buy_notional_hist", "debeta_at_best_hist"):
        assert src.count(name) >= 4, f"{name}: init, prune, shift and restore"
    assert "'debeta_maker_buy_notional'" in inspect.getsource(trade.reset_agent_histories)
    from taos.im.validator import persistence, scoring_shadow
    assert "debeta_at_best_hist" in persistence._DEBETA_HIST3 and "debeta_at_best" in scoring_shadow._DEBETA_DD2


def test_maker_notional_is_booked_by_side_from_the_fills():
    v = SimpleNamespace()
    trade.init_liquidity_state(v)
    trades = [_t(100.0, 2.0, 0, 5, -1), _t(101.0, 1.0, 1, 5, 7), _t(99.0, 3.0, 0, -1, 5)]
    trade.accumulate_maker_notional(v, 0, trades, ts=1)
    assert v.debeta_maker_sell_notional[5][0] == 200.0, "side 0: the taker buys, the maker sold"
    assert v.debeta_maker_buy_notional[5][0] == 101.0
    assert 5 not in v.debeta_maker_buy_notional or 0 not in v.debeta_maker_buy_notional.get(-1, {})
    assert v.debeta_maker_sell_notional_hist[5][0] == {1: 200.0}


def test_resting_at_the_best_is_read_from_the_accounts_orders_and_the_top_of_book():
    v = SimpleNamespace()
    trade.init_liquidity_state(v)
    books = {0: {"b": [{"p": 100.0, "q": 5.0}], "a": [{"p": 100.2, "q": 4.0}]},
             1: {"b": [{"p": 50.0, "q": 5.0}], "a": [{"p": 50.5, "q": 4.0}]}}
    accounts_uid = {0: {"o": [{"p": 100.0, "q": 2.0, "s": 0}, {"p": 100.2, "q": 1.0, "s": 1}, {"p": 99.0, "q": 9.0, "s": 0}]},
                    1: {"o": [{"p": 49.0, "q": 9.0, "s": 0}]}}
    trade.accumulate_at_best(v, 5, books, accounts_uid, ts=1)
    trade.accumulate_at_best(v, 5, books, accounts_uid, ts=2)
    assert v.debeta_at_best[5][0] == 2 and 1 not in v.debeta_at_best[5], "two states at the best on book 0, none on book 1"
    assert v.debeta_best_size[5][0] == 6.0, "2 at the bid plus 1 at the ask, twice"
    assert v.debeta_at_best_hist[5][0] == {1: 1, 2: 1}
    assert "accumulate_at_best(self, uid_item, books, accounts_uid, ts=sampled_timestamp)" in inspect.getsource(trade._process_uid_trade_volumes)
    assert "accumulate_maker_notional(self, bookId, de_trades, ts=sampled_timestamp)" in inspect.getsource(trade.update_trade_volumes)


def test_the_scorer_publishes_the_three_quantities_per_uid():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("pg", pathlib.Path(__file__).with_name("test_debeta_presence_gate.py"))
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    v = pg._fake_validator()
    trade.init_liquidity_state(v)
    for b in range(6):
        trade.accumulate_maker_notional(v, b, [_t(100.0, 2.0, 0, 1, -1), _t(100.0, 1.0, 1, 1, -1)], ts=1)
        trade.accumulate_at_best(v, 1, {b: {"b": [{"p": 100.0, "q": 1}], "a": [{"p": 100.2, "q": 1}]}},
                                 {b: {"o": [{"p": 100.0, "q": 3.0, "s": 0}]}}, ts=1)
    scores = reward.compute_debeta_scores(v)
    assert scores
    d = v.debeta_detail
    assert abs(d[1]["maker_notional_two_sided"] - 6 * 2 * 100.0) < 1e-9, "2 x min(200, 100) per book over six books"
    assert d[1]["at_best_seconds"] == 6 and d[1]["best_size"] == 18.0
    assert d[2]["maker_notional_two_sided"] == 0.0 and d[2]["at_best_seconds"] == 0
    rsrc = inspect.getsource(report)
    for g in ("('maker_notional_two_sided', 'debeta_maker_notional_two_sided')", "('at_best_seconds', 'debeta_at_best_seconds')",
              "('best_size', 'debeta_best_size')"):
        assert g in rsrc, g
