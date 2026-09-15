# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""P11 measures a maker's counterparty concentration against its WHOLE flow.

Until now `accumulate_counterparties` dropped every taker with a negative id, so the concentration
measure saw miner takers only. A maker can take nearly all of its flow from non-miner counterparties
and only a sliver from miners, and the miner-only measure then read that sliver as concentration and
discounted it heavily. In exchange mode external takers and pool fills carry no miner id and were
dropped the same way, so a maker mostly hit by real external flow would have been judged on its
miner slice too.

Now non-miner takers are kept under the MARKET_FLOW bucket: it counts in the denominator, it is never
a top counterparty, and the leave-one-out market share is over the whole flow as well. A maker served
by the diverse market gets no discount; a maker fed 90 per cent by one miner still loses 90 per cent.
"""
import pytest

from taos.im.validator.debeta import (MARKET_FLOW, accumulate_counterparties, counterparty_ec,
                                      p11_discount)


def _trade(maker, taker, q):
    return {"Ma": maker, "Ta": taker, "q": q, "p": 1.0, "s": 0}


def test_background_and_external_takers_land_in_the_market_bucket():
    cp = {}
    accumulate_counterparties(cp, 0, [_trade(129, -72, 5.0), _trade(129, -1304, 3.0), _trade(129, None, 2.0),
                                      _trade(129, 134, 1.0)])
    assert cp[129][MARKET_FLOW] == pytest.approx(10.0)
    assert cp[129][134] == pytest.approx(1.0)


def test_self_trades_and_non_miner_makers_are_still_skipped():
    cp = {}
    accumulate_counterparties(cp, 0, [_trade(129, 129, 5.0), _trade(-72, 129, 5.0), _trade(-72, -80, 5.0)])
    assert cp == {}


def test_a_maker_served_by_the_market_takes_no_discount():
    # 99 of 100 units from the market, half a unit each from two miners: the old measure saw only the
    # two miners and called the maker fully concentrated.
    cp = {129: {MARKET_FLOW: 99.0, 134: 0.5, 100: 0.5},
          134: {MARKET_FLOW: 50.0, 133: 5.0, 132: 5.0, 2: 3.0}}
    assert counterparty_ec(cp, 129) < 0.02
    out = p11_discount({129: 1000.0, 134: 500.0}, cp, [129, 134], strength=1.0)
    assert out[129] == pytest.approx(1000.0, rel=0.02)


def test_a_feeder_fed_maker_still_loses_its_making():
    cp = {7: {MARKET_FLOW: 5.0, 8: 90.0, 9: 5.0},          # 90 per cent from one miner
          134: {MARKET_FLOW: 50.0, 133: 5.0, 132: 5.0, 2: 3.0}}
    ec = counterparty_ec(cp, 7)
    assert 0.85 < ec <= 1.0
    out = p11_discount({7: 1000.0, 134: 500.0}, cp, [7, 134], strength=1.0)
    assert out[7] == pytest.approx(1000.0 * (1.0 - ec), rel=1e-6)
    # 134 takes 79 per cent of its flow from the market and 16 per cent from two miners nobody else
    # trades with: the discount is proportional to that miner share, about 16 per cent, where the
    # miner-only measure would have called it 77 per cent concentrated.
    ec134 = counterparty_ec(cp, 134)
    assert 0.10 < ec134 < 0.20
    assert out[134] == pytest.approx(500.0 * (1.0 - ec134), rel=1e-6)


def test_the_market_bucket_is_never_a_top_counterparty():
    # With topk=1 the single top counterparty must be a miner even when the market bucket is largest.
    cp = {7: {MARKET_FLOW: 80.0, 8: 20.0}, 9: {MARKET_FLOW: 100.0, 8: 1.0}}
    ec = counterparty_ec(cp, 7, topk=1)
    # my_share of miner 8 is 0.20, its share of the other makers' flow is 1/101
    assert ec == pytest.approx(0.20 - 1.0 / 101.0, abs=1e-9)


def test_miner_only_maps_keep_the_previous_arithmetic():
    """States saved before this change carry no market bucket; the measure must read them as before."""
    cp = {7: {8: 9.0, 9: 1.0}, 10: {8: 1.0, 11: 9.0}, 12: {13: 5.0, 14: 5.0}}
    # top-2 of maker 7 is everything it has: my_share 1.0; those takers' share of the others' flow
    # is (1 + 0) / 20
    assert counterparty_ec(cp, 7) == pytest.approx(1.0 - 1.0 / 20.0)
    out = p11_discount({7: 100.0, 10: 100.0, 12: 100.0}, cp, [7, 10, 12], strength=1.0)
    assert out[7] == pytest.approx(100.0 * (1.0 - (1.0 - 1.0 / 20.0)))


def test_the_fast_path_matches_the_reference_with_market_flow():
    cp = {129: {MARKET_FLOW: 99.0, 134: 0.5, 100: 0.5},
          7: {MARKET_FLOW: 5.0, 8: 90.0, 9: 5.0},
          134: {MARKET_FLOW: 50.0, 133: 5.0, 132: 5.0, 2: 3.0}}
    own = {129: 1000.0, 7: 1000.0, 134: 500.0}
    fast = p11_discount(dict(own), cp, [129, 7, 134], strength=1.0)
    for u in own:
        ref = own[u] * max(0.0, 1.0 - max(0.0, counterparty_ec(cp, u)))
        assert fast[u] == pytest.approx(ref, rel=1e-9)
