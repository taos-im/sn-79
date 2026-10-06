# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Under a multi-asset layout the counterparty rows P11 reads are kept per asset class as well as pooled, and each
class's scoring run judges a maker's counterparty mix against its own class's taker mix. The tether keeps reading
the pooled rows: liquidity supplied to the market in any class is service.

Why: P11's excess concentration compares a maker's top takers with those takers' share of every other maker's
flow. Pooled across classes, a small class's honest maker is read as concentrated, because its takers trade only
that class and are a sliver of the whole field's taker mix. Judged within the class the same maker is ordinary.
Decided 1 October 2026 for the 96+32 production layout (simulation_0 0 to 95, regime B 96 to 127)."""
import inspect
import types

from taos.im.validator.debeta import MARKET_FLOW, debeta_scores, p11_discount, tether_miner_sourced


def _field():
    # class A: twenty makers served by the market and twenty shared takers; class B: two makers served by the
    # market and two takers who trade only class B
    cp = {}
    for m in range(20):
        cp[m] = {MARKET_FLOW: 900.0, **{100 + t: 5.0 for t in range(20)}}
    cp[50] = {MARKET_FLOW: 100.0, 200: 50.0, 201: 50.0}
    cp[51] = {MARKET_FLOW: 100.0, 200: 50.0, 201: 50.0}
    cls = {m: 0 for m in range(20)} | {50: 1, 51: 1}
    by_class = {0: {m: cp[m] for m in range(20)}, 1: {50: cp[50], 51: cp[51]}}
    return cp, cls, by_class


def test_a_small_class_maker_is_concentrated_against_the_field_but_ordinary_within_its_class():
    cp, cls, by_class = _field()
    pooled = p11_discount({u: 1.0 for u in cp}, cp, list(cp), 1.0, topk=10)
    within = p11_discount({u: 1.0 for u in by_class[1]}, by_class[1], [50, 51], 1.0, topk=10)
    assert pooled[50] < 0.6, pooled[50]
    assert within[50] > 0.99, within[50]
    assert all(pooled[m] > 0.99 for m in range(20))


def test_debeta_scores_reads_p11_rows_and_tether_rows_separately():
    cp, cls, by_class = _field()
    caps = {u: {b: 10.0 for b in range(4)} for u in (50, 51)}
    alphas = {u: [0.1, -0.2, 0.3, 0.1] for u in (50, 51)}
    d_pooled, d_class = {}, {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, p11_strength=1.0, p11_topk=10, s3_k=4.0, detail=d_pooled)
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=by_class[1], cp_tether=cp, p11_strength=1.0, p11_topk=10,
                  s3_k=4.0, detail=d_class)
    assert d_pooled[50]["p11_factor"] < 0.6 and d_class[50]["p11_factor"] > 0.99
    # the tether read the pooled rows in both runs: 100 market, 100 miner, k 4 keeps everything
    assert d_class[50]["making_s3_factor"] == 1.0 and d_pooled[50]["making_s3_factor"] == 1.0
    assert abs(d_class[50]["bg_share"] - 0.5) < 1e-9, "the maker's background share of fill volume travels in the detail"


def test_the_tether_on_pooled_rows_still_removes_a_class_private_ring():
    ring = {1: {20: 300.0, 21: 300.0}, 2: {20: 300.0, 21: 300.0}}
    assert tether_miner_sourced({1: 1.0, 2: 1.0}, ring, [1, 2], 4.0) == {1: 0.0, 2: 0.0}


def _stub(class_of_book):
    v = types.SimpleNamespace()
    v.simulation_config = {"class_of_book": {str(b): c for b, c in class_of_book.items()}}
    v.debeta_cp, v.debeta_cp_hist, v.debeta_cpc, v.debeta_cpc_hist = {}, {}, {}, {}
    return v


def test_fills_accumulate_into_the_pooled_rows_and_into_their_class_rows():
    from taos.im.validator.trade import accumulate_counterparty_rows

    v = _stub({0: 0, 1: 0, 2: 1})
    trades = [{"p": 1.0, "q": 2.0, "s": 1, "Ma": 7, "Ta": 9}, {"p": 1.0, "q": 1.0, "s": 1, "Ma": 7, "Ta": -1}]
    accumulate_counterparty_rows(v, 2, trades, ts=10)
    accumulate_counterparty_rows(v, 0, [{"p": 1.0, "q": 5.0, "s": 1, "Ma": 7, "Ta": 9}], ts=10)
    assert v.debeta_cp[7] == {9: 7.0, MARKET_FLOW: 1.0}
    assert v.debeta_cpc[1][7] == {9: 2.0, MARKET_FLOW: 1.0} and v.debeta_cpc[0][7] == {9: 5.0}
    assert v.debeta_cpc_hist[1][7][9] == {10: 2.0}


def test_a_single_market_keeps_no_class_rows():
    from taos.im.validator.trade import accumulate_counterparty_rows

    v = _stub({})
    accumulate_counterparty_rows(v, 0, [{"p": 1.0, "q": 2.0, "s": 1, "Ma": 7, "Ta": 9}], ts=10)
    assert v.debeta_cp[7] == {9: 2.0} and v.debeta_cpc == {}


def test_a_reset_clears_the_uid_from_every_class_as_maker_and_as_counterparty():
    from taos.im.validator.trade import clear_counterparty_rows

    v = _stub({})
    v.debeta_cp = {7: {9: 1.0}, 8: {7: 2.0, 9: 1.0}}
    v.debeta_cp_hist = {7: {9: {10: 1.0}}, 8: {7: {10: 2.0}, 9: {10: 1.0}}}
    v.debeta_cpc = {0: {7: {9: 1.0}, 8: {7: 2.0}}, 1: {8: {7: 3.0}}}
    v.debeta_cpc_hist = {0: {7: {9: {10: 1.0}}, 8: {7: {10: 2.0}}}, 1: {8: {7: {10: 3.0}}}}
    clear_counterparty_rows(v, 7)
    assert v.debeta_cp == {8: {9: 1.0}} and v.debeta_cpc == {0: {8: {}}, 1: {8: {}}}
    assert v.debeta_cpc_hist == {0: {8: {}}, 1: {8: {}}}


def test_the_class_run_passes_class_rows_to_p11_and_pooled_rows_to_the_tether():
    from taos.im.validator import reward

    src = inspect.getsource(reward)
    i = src.index("for _c, _part in _by_class.items():")
    block = src[i:i + 2500]
    assert "cp=((getattr(self, 'debeta_cpc', None) or {}).get(_c) or cp), cp_tether=cp" in block


def test_persistence_saves_and_restores_the_class_rows():
    from taos.im.validator.persistence import debeta_hist_snapshot, load_class_counterparty_rows

    v = _stub({})
    v.debeta_cpc_hist = {1: {7: {9: {10: 2.0}}}}
    for name in ("debeta_capbuy_hist", "debeta_capsell_hist", "debeta_mtm_hist", "debeta_invsum_hist", "debeta_cp_hist",
                 "debeta_heldn_hist", "debeta_heldinv_hist", "debeta_helddrift_hist", "debeta_notional_hist",
                 "debeta_invn_hist", "debeta_drift_hist"):
        setattr(v, name, {})
    snap = debeta_hist_snapshot(v)
    assert snap["debeta_cpc_hist"] == {1: {7: {9: {10: 2.0}}}}
    hist, rows = load_class_counterparty_rows({"debeta_cpc_hist": {"1": {"7": {"9": {"10": 2.0}}}}})
    assert hist == {1: {7: {9: {10: 2.0}}}} and rows == {1: {7: {9: 2.0}}}
    assert load_class_counterparty_rows({}) == ({}, {})


def test_the_shadow_tees_the_class_rows():
    from taos.im.validator import scoring_shadow

    src = inspect.getsource(scoring_shadow)
    assert '"debeta_cpc"' in src and '"debeta_cpc_hist"' in src
