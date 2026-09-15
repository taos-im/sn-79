# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The capture mid must not depend on batch shape. Live batches are one state update per book and
median 4 prints (measured over a live board), so a batch-local centered mid degrades to the
plain batch mean 91% of the time and to the fill's own price on single-print batches: the
drift-stripping the centered design exists for barely operates, and splitting the same prints into
more batches changes scores. Reported by a miner on the 0.6.0 tape and confirmed here.

The fix carries a rolling per-book print window across batches (mid_state): a fill waits for W
forward prints (or flush_ns of sim time) and is then booked against its full centered window, so
only the print stream itself, never the batching, decides capture.
"""
from collections import defaultdict

import pytest

from taos.im.validator.debeta import (
    accumulate_book_capture, flush_capture_state, centered_mid, CAPTURE_FLUSH_NS,
)

W = 15
BOOK = 7


def _mk(price, q=1.0, s=0, ma=1, ta=2):
    return {"p": price, "q": q, "s": s, "Ma": ma, "Ta": ta}


def _run(batches, *, state=None, flush=True, ts0=0):
    buy = defaultdict(dict)
    sell = defaultdict(dict)
    for k, batch in enumerate(batches):
        accumulate_book_capture(buy, sell, BOOK, batch, W,
                                ts=ts0 + k * 1_000_000_000, mid_state=state)
    if state is not None and flush:
        flush_capture_state(state, buy, sell, W, ts=ts0 + len(batches) * 1_000_000_000, force=True)
    return buy, sell


def test_batch_shape_invariance():
    """The same print stream split 1-per-batch, 4-per-batch, or whole must attribute identically."""
    trades = [_mk(100.0 + (i % 7) - 3 + 0.1 * i, q=1.0 + (i % 3), s=i % 2) for i in range(80)]
    whole = _run([trades], state={})
    ones = _run([[t] for t in trades], state={})
    fours = _run([trades[i:i + 4] for i in range(0, 80, 4)], state={})
    for (buy, sell) in (ones, fours):
        for uid in (1, 2):
            assert buy[uid].get(BOOK, 0.0) == pytest.approx(whole[0][uid].get(BOOK, 0.0), abs=1e-9)
            assert sell[uid].get(BOOK, 0.0) == pytest.approx(whole[1][uid].get(BOOK, 0.0), abs=1e-9)


def test_rolling_matches_full_stream_centered_mid():
    """With the carry, per-print mids equal centered_mid over the WHOLE stream, regardless of batching."""
    prices = [100.0 + 0.5 * i + (1 if i % 3 == 0 else -1) for i in range(60)]
    trades = [_mk(p, s=0) for p in prices]
    buy, _ = _run([trades[i:i + 3] for i in range(0, 60, 3)], state={})
    mids = centered_mid(prices, W)
    expected = sum((m - p) * 1.0 for m, p in zip(mids, prices))
    assert buy[2][BOOK] == pytest.approx(expected, abs=1e-9)


def test_trend_leakage_to_the_early_buyer_is_stripped():
    """The exploit shape from the miner report: in a trend, whoever buys EARLY within each batch
    harvests capture against the batch-mean mid (the same 90 prints score more when split). With
    the rolling window the mid tracks the trend across batch edges, so interior capture is ~0 and
    the early buyer keeps only the genuine stream-edge truncation, a fraction of the leakage."""
    trades = []
    for i in range(90):
        trades.append(_mk(100.0 + 1.0 * i, s=0, ma=1, ta=(8 if i % 10 < 5 else 9)))
    batches = [trades[i:i + 10] for i in range(0, 90, 10)]
    lbuy, _ = _run(batches, state=None)   # legacy: per-batch mean mids
    rbuy, _ = _run(batches, state={})     # rolling: full-stream centered mids
    prices = [float(t["p"]) for t in trades]
    mids = centered_mid(prices, W)
    full_a = sum((m - p) for m, p, t in zip(mids, prices, trades) if t["Ta"] == 8)
    legacy_a = lbuy[8][BOOK]
    assert legacy_a > 50.0, "early buyer must harvest trend under batch-local mids"
    assert rbuy[8][BOOK] == pytest.approx(full_a, abs=1e-9)
    assert abs(rbuy[8][BOOK]) < 0.4 * legacy_a


def test_zero_sum_per_fill_under_any_batching():
    trades = [_mk(100.0 + (i * 37 % 11) - 5, q=2.0, s=i % 2, ma=i % 5, ta=(i + 1) % 5)
              for i in range(50)]
    buy, sell = _run([trades[i:i + 2] for i in range(0, 50, 2)], state={})
    total = sum(v.get(BOOK, 0.0) for v in buy.values()) + sum(v.get(BOOK, 0.0) for v in sell.values())
    assert total == pytest.approx(0.0, abs=1e-9)


def test_stale_pending_flushes_by_sim_time():
    """A quiet book cannot hold capture hostage: with no forward prints, the fill finalizes with a
    truncated window once flush_ns of sim time passes."""
    state = {}
    buy = defaultdict(dict)
    sell = defaultdict(dict)
    accumulate_book_capture(buy, sell, BOOK, [_mk(100.0), _mk(104.0)], W, ts=0, mid_state=state)
    assert buy[2].get(BOOK) is None  # pending: only 2 prints, no forward window yet
    flush_capture_state(state, buy, sell, W, ts=CAPTURE_FLUSH_NS)  # not force: age does it
    # window is both prints: mid 102, buyer(taker 2) gets (102-100)+(102-104) = 0 net across fills
    assert buy[2][BOOK] == pytest.approx(0.0, abs=1e-9)
    assert not state[BOOK]["pend"]


def test_wash_prints_shape_mid_but_earn_nothing():
    """Ma==Ta fills stay excluded from attribution, but their prints still enter the window."""
    trades = [_mk(100.0), _mk(200.0, ma=3, ta=3), _mk(100.0)]
    buy, sell = _run([trades], state={})
    assert 3 not in buy and 3 not in sell
    # mid over [100, 200, 100] = 133.33: the wash print moved the honest fills' mid
    assert buy[2][BOOK] == pytest.approx(2 * (400.0 / 3 - 100.0), abs=1e-6)


def test_state_stays_bounded():
    state = {}
    buy = defaultdict(dict)
    sell = defaultdict(dict)
    for k in range(500):
        accumulate_book_capture(buy, sell, BOOK, [_mk(100.0 + k % 9)], W,
                                ts=k * 1_000_000_000, mid_state=state)
    st = state[BOOK]
    assert len(st["pend"]) <= W + 1
    assert len(st["prices"]) <= 2 * W + 1


def test_wash_print_mid_shaping_dilutes_per_fill_and_stays_bounded_in_total():
    """A wash print (Ma==Ta) shapes the mid without earning, and the rolling window bounds what that
    shaping can be worth to whoever placed it. Its influence on that party's OWN fill is diluted at
    live print density, the influence it sheds lands on the neighbouring fills instead, and the
    TOTAL is conserved: bounded by the wash's own price deviation and zero-sum against the
    counterparties, so collecting it would require owning the same side of every nearby fill."""
    others = [_mk(100.0, ta=7) for _ in range(14)]
    attacker_fill = _mk(100.0, ta=2)
    wash = _mk(160.0, ma=9, ta=9)
    # legacy at median density: attacker's fill shares a 4-print batch with the wash
    lbuy, _ = _run([[attacker_fill, wash, _mk(100.0, ta=7), _mk(100.0, ta=7)]], state=None)
    legacy_gain = lbuy[2][BOOK]
    # rolling: same wash, attacker's single fill inside a dense honest stream owned by others
    stream = others + [attacker_fill, wash] + [_mk(100.0, ta=7) for _ in range(16)]
    rbuy, _ = _run([stream[i:i + 4] for i in range(0, len(stream), 4)], state={})
    attacker_gain = rbuy[2][BOOK]
    assert legacy_gain > 0
    assert 0 < attacker_gain < legacy_gain / 5, (
        f"per-fill wash shaping must dilute: legacy {legacy_gain}, rolling {attacker_gain}")
    total_shaping = sum(v.get(BOOK, 0.0) for u, v in rbuy.items() if u != 9)
    assert total_shaping <= 2 * (160.0 - 100.0), "total influence bounded by the wash deviation"


def test_cross_second_wash_reach_adds_no_magnitude():
    """The rolling window lets a wash print in the NEXT batch reach an earlier fill's mid, but the
    magnitude is identical to placing it in the SAME batch (already possible): time freedom, no new
    power."""
    fill = _mk(100.0)
    wash = _mk(160.0, ma=9, ta=9)
    pad = [_mk(100.0) for _ in range(29)]
    same, _ = _run([[fill, wash] + pad], state={})
    split, _ = _run([[fill], [wash] + pad], state={})
    assert same[2][BOOK] == pytest.approx(split[2][BOOK], abs=1e-9)


def test_flush_knob_is_bounded_by_the_trailing_window():
    """Adversarial residual: on a thin book an attacker can withhold forward prints so their fill
    finalizes against the trailing window only. The gain is exactly the truncated-window
    attribution (deterministic, bounded by trailing-mid minus price times quantity); the legacy
    path scored such fills 0, so this hands honest thin-book makers fair credit at the cost of a
    bounded, monitorable knob."""
    state = {}
    buy = defaultdict(dict)
    sell = defaultdict(dict)
    trail = [_mk(110.0) for _ in range(15)]
    accumulate_book_capture(buy, sell, BOOK, trail + [_mk(100.0)], W, ts=0, mid_state=state)
    flush_capture_state(state, buy, sell, W, ts=CAPTURE_FLUSH_NS)  # silence: age-flush, no forward prints
    prices = [110.0] * 15 + [100.0]
    mids = centered_mid(prices, W)
    expected = sum((m - p) for m, p in zip(mids, prices))
    assert buy[2][BOOK] == pytest.approx(expected, abs=1e-9)
    per_fill_bound = (sum(prices) / len(prices)) - 100.0
    assert buy[2][BOOK] <= per_fill_bound * len(prices) + 1e-9


def test_legacy_path_unchanged_without_state():
    """Offline callers (whole run as one batch) keep byte-identical behavior."""
    trades = [_mk(100.0 + i, s=i % 2) for i in range(40)]
    old_buy, old_sell = _run([trades], state=None)
    prices = [float(t["p"]) for t in trades]
    mids = centered_mid(prices, W)
    exp = sum((m - p) * (1 if int(t["s"]) == 0 else 1) * 0 + (m - p) for m, p, t in zip(mids, prices, trades) if int(t["s"]) == 0)
    got = old_buy[2].get(BOOK, 0.0)
    assert got == pytest.approx(exp, abs=1e-9)
