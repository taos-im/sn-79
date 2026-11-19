"""The making leg's order-to-trade discount: a maker whose messages per fill over the making window
exceed the threshold keeps min(1, (threshold / messages_per_fill) ** power) of its making credit before ranking; the
real-market mirror is the order-to-trade ratio limit every venue sets under MiFID II RTS 9. Off at threshold 0."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taos.im.validator.debeta import debeta_scores, otr_discount  # noqa: E402


def test_the_discount_leaves_quiet_makers_alone_and_scales_noisy_ones_by_the_ratio():
    own = {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0}
    otr = {1: 50.0, 2: 100.0, 3: 400.0, 4: None}
    out = otr_discount(own, otr, [1, 2, 3, 4], threshold=100.0, power=1.0)
    assert out[1] == 10.0 and out[2] == 10.0, "at or under the threshold nothing moves"
    assert abs(out[3] - 2.5) < 1e-9, "four times the threshold keeps a quarter"
    assert out[4] == 10.0, "a uid without a reading keeps its credit"


def test_the_power_softens_the_discount():
    out = otr_discount({3: 10.0}, {3: 400.0}, [3], threshold=100.0, power=0.5)
    assert abs(out[3] - 5.0) < 1e-9


def test_a_maker_with_messages_and_no_fills_is_discounted_fully():
    out = otr_discount({5: 10.0}, {5: float("inf")}, [5], threshold=100.0, power=1.0)
    assert out[5] == 0.0


def _legs():
    buy = {1: {0: 5.0, 1: 5.0}, 2: {0: 5.0, 1: 5.0}, 3: {0: 1.0}}
    sell = {1: {0: 5.0, 1: 5.0}, 2: {0: 5.0, 1: 5.0}, 3: {0: 1.0}}
    alphas = {1: [0.0, 0.0], 2: [0.0, 0.0], 3: [0.0]}
    return buy, sell, alphas


def test_debeta_scores_applies_the_discount_before_the_making_rank_and_publishes_it():
    buy, sell, alphas = _legs()
    detail = {}
    base = debeta_scores(buy, sell, alphas, detail=detail, w_make=1.0)
    assert base[1] == base[2], "two equal makers rank equal"
    assert detail[1].get("making_otr_factor") == 1.0 and detail[1].get("messages_per_fill") is None
    detail = {}
    scored = debeta_scores(buy, sell, alphas, detail=detail, w_make=1.0,
                           otr={1: 50.0, 2: 1000.0, 3: 50.0}, otr_threshold=100.0, otr_power=1.0)
    assert scored[1] > scored[2], "the noisy maker ranks below its quiet twin"
    assert abs(detail[2]["making_otr_factor"] - 0.1) < 1e-9 and detail[2]["messages_per_fill"] == 1000.0
    assert detail[1]["making_otr_factor"] == 1.0
    assert detail[2]["making_pre_otr"] == detail[1]["making_pre_otr"], "the credit before the discount stays what was measured"


def test_the_discount_is_off_at_threshold_zero():
    buy, sell, alphas = _legs()
    plain = debeta_scores(buy, sell, alphas, w_make=1.0)
    off = debeta_scores(buy, sell, alphas, w_make=1.0, otr={1: 50.0, 2: 1000.0, 3: 50.0}, otr_threshold=0.0)
    assert plain == off
