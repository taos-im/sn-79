# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The realized making basis marks a fill against the centered mark making_horizon_s of simulation time after it.
On the live path the accumulator carried the print window across state updates and stamped every print and every
pending fill with the timestamp it was given for the volume histories, which is the SAMPLED timestamp (floored to
scoring.activity.trade_volume_sampling_interval, 600 s in production). With every print in a bucket stamped alike, a
fill could only mature when the next bucket began, so the "20 s" mark was taken anywhere from 0 to 600 s after the
fill, and nothing matured inside the first bucket of a run (the 0.6.3 testnet warmed for 46 minutes on 6 October;
the replay harness reproduced it on a 45 s stream). The offline path, which the battery used, read each trade's own
time, so the battery measured the rule and the validator ran another. These tests hold the live path to the rule:
the print clock is the trade's own time, the history key stays the sampled timestamp."""
from collections import defaultdict

from taos.im.validator.debeta import accumulate_book_capture, flush_capture_state

S = 1_000_000_000
SAMPLE = 600 * S
H = 20 * S
W = 3


def _dd():
    return defaultdict(lambda: defaultdict(float))


def _trade(ts, maker, taker, price, side, q=1.0):
    return {"y": "t", "i": 0, "s": side, "t": ts, "q": q, "p": price, "Ma": maker, "Ta": taker, "Ti": 0, "Mi": 0}


def _run(seconds, fill_at=1, quiet_after=None):
    """Maker 1 buys from a background taker at 99.98 in second `fill_at`; the background market then prints at 100.5
    every second (or goes quiet after `quiet_after`). All inside one 600 s sampling bucket."""
    cb, cs, rb, rs = _dd(), _dd(), _dd(), _dd()
    hb, hs, hrb, hrs = {}, {}, {}, {}
    mid_state = {}
    booked_at = None
    for k in range(1, seconds + 1):
        ts = k * S
        sampled = (ts // SAMPLE) * SAMPLE
        trades = []
        if k == fill_at:
            trades.append(_trade(ts, 1, -1, 99.98, 1))
        if quiet_after is None or k <= quiet_after:
            trades.append(_trade(ts, -2, -3, 100.0 if k <= fill_at else 100.5, 0))
        if trades:
            accumulate_book_capture(cb, cs, 0, trades, W, buy_hist=hb, sell_hist=hs, ts=sampled, mid_state=mid_state,
                                    real_buy_sums=rb, real_sell_sums=rs, real_buy_hist=hrb, real_sell_hist=hrs,
                                    horizon_ns=H, now_ns=ts)
        else:
            flush_capture_state(mid_state, cb, cs, W, buy_hist=hb, sell_hist=hs, ts=sampled, real_buy_sums=rb,
                                real_sell_sums=rs, real_buy_hist=hrb, real_sell_hist=hrs, horizon_ns=H, now_ns=ts)
        if booked_at is None and rb.get(1, {}).get(0):
            booked_at = k
    return cb, rb, hrb, booked_at


def test_a_fill_matures_at_the_horizon_inside_one_sampling_bucket():
    cb, rb, hrb, booked_at = _run(60)
    assert cb[1][0] > 0.0, "the captured spread books at the fill as before"
    assert booked_at is not None, "the realized spread never matured inside the bucket: the print clock is the sampled timestamp"
    assert 1 + 20 <= booked_at <= 1 + 20 + W + 1, f"booked {booked_at - 1} s after the fill; the rule is the horizon plus W forward prints"
    # marked against the background prints at the horizon; since 7 October 2026 a fill's realized spread is clamped to
    # its captured spread (debeta._capped_mark), and this fill captured less than the 0.52 move
    assert abs(rb[1][0] - min(100.5 - 99.98, cb[1][0])) < 1e-9, (rb[1][0], cb[1][0])


def test_the_history_key_stays_the_sampled_timestamp_so_pruning_and_re_basing_are_unchanged():
    _cb, rb, hrb, booked_at = _run(60)
    assert booked_at is not None
    assert set(hrb[1][0]) == {0}, f"the realized history is keyed by the sampled timestamp of the booking update, got {set(hrb[1][0])}"
    assert abs(sum(hrb[1][0].values()) - rb[1][0]) < 1e-12, "running sum equals the history within the window"


def test_a_quiet_book_releases_the_fill_on_the_true_clock():
    _cb, rb, _hrb, booked_at = _run(120, quiet_after=5)
    assert booked_at is not None, "a fill on a book that went quiet is released after horizon plus the flush grace of true sim time"
    assert booked_at <= 1 + (20 + 60) + 1, f"released at {booked_at} s; the grace is 60 s past the horizon"
