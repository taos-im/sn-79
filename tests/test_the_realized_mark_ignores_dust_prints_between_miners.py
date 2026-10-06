"""The realized making basis marks each fill against the centered window around the print 60 simulation seconds
later. That window is weighted by print size and built from prints with the background market on at least one
side, so two cooperating miners cannot move the mark with dust prints to each other at the horizon. A window with
no background print at all falls back to the plain average, so a quiet book still settles."""
import collections

from taos.im.validator.debeta import accumulate_book_capture

S = 1_000_000_000
H = 60 * S
W = 15


def _sums():
    return (collections.defaultdict(lambda: collections.defaultdict(float)),
            collections.defaultdict(lambda: collections.defaultdict(float)))


def _t(p, q, s, ma, ta, ts):
    return {"p": p, "q": q, "s": s, "Ma": ma, "Ta": ta, "ts": ts}


def _tape(pump):
    """Miner 1 buys 1.0 at 99.9 from the background at t=0 (maker side, s=1). Background prints at 100.0 fill the
    tape every second. With pump, miners 2 and 3 print 0.01 to each other at 120.0 around the horizon."""
    tr = [_t(99.9, 1.0, 1, 1, -5, 0)]
    for k in range(1, 120):
        tr.append(_t(100.0, 1.0, k % 2, -3, -4, k * S))
        if pump and 55 <= k <= 65:
            tr.append(_t(120.0, 0.01, 0, 2, 3, k * S + 1))
    return tr


def _realized_offline(trades):
    cb, cs = _sums(); rb, rs = _sums()
    accumulate_book_capture(cb, cs, 0, trades, W, real_buy_sums=rb, real_sell_sums=rs, horizon_ns=H)
    return rb[1].get(0, 0.0)


def _realized_live(trades):
    cb, cs = _sums(); rb, rs = _sums(); st = {}
    for t in trades:
        accumulate_book_capture(cb, cs, 0, [t], W, ts=t["ts"], mid_state=st,
                                real_buy_sums=rb, real_sell_sums=rs, horizon_ns=H)
    return rb[1].get(0, 0.0)


def test_a_dust_pump_between_miners_at_the_horizon_does_not_move_the_mark_offline():
    clean, pumped = _realized_offline(_tape(False)), _realized_offline(_tape(True))
    assert abs(clean - 0.1) < 1e-9
    assert abs(pumped - clean) < 1e-9, (clean, pumped)


def test_a_dust_pump_between_miners_at_the_horizon_does_not_move_the_mark_live():
    clean, pumped = _realized_live(_tape(False)), _realized_live(_tape(True))
    assert abs(clean - 0.1) < 1e-6
    assert abs(pumped - clean) < 1e-6, (clean, pumped)


def test_the_mark_is_weighted_by_print_size():
    # one large background print at 101 among small ones at 100 pulls the mark toward 101 by its size share
    tr = [_t(99.9, 1.0, 1, 1, -5, 0)]
    for k in range(1, 120):
        big = k == 60
        tr.append(_t(101.0 if big else 100.0, 30.0 if big else 1.0, k % 2, -3, -4, k * S))
    got = _realized_offline(tr)
    window = [(101.0, 30.0)] + [(100.0, 1.0)] * (2 * W)
    mark = sum(p * q for p, q in window) / sum(q for _, q in window)
    assert abs(got - (mark - 99.9)) < 1e-9, (got, mark - 99.9)


def test_a_window_without_background_prints_falls_back_to_the_plain_average():
    tr = [_t(99.9, 1.0, 1, 1, 2, 0)] + [_t(100.0 + (k % 3) * 0.1, 0.5, k % 2, 4, 5, k * S) for k in range(1, 120)]
    assert _realized_offline(tr) != 0.0
