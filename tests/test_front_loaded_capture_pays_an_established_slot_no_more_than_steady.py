"""The per-book volume cap is a 24-sim-hour budget and the score reads a 3-sim-hour window, so a miner could spend
the budget in the window and coast. Pay is the share of that rolling window smoothed by the track-record EMA, one
schedule for the trading score and the pool's share vector. Both are linear in the capture history for an
established slot, so the same total capture integrates to the same pay whatever its timing. A fresh slot's standing
seeds at its live value, which carries a burst above the same capture spread over the day; that premium protects a
genuine newcomer inside its immunity window and must stay bounded."""
from taos.im.validator.reward import apply_track_record_ema

HL = 3 * 3600
W = 3 * 3600
DELTA = 60


def integrated_pay(burst, newcomer, hours=48, v=1.0):
    n = int(hours * 3600 / DELTA)
    h = 3 * 3600 if burst else 24 * 3600
    rate = v / (h / DELTA)
    fills = [rate if t * DELTA < h else 0.0 for t in range(n)]
    ema, ema_n, last = {}, {}, None
    base = 0
    if not newcomer:
        for t in range(int(10 * HL / DELTA)):
            _, last = apply_track_record_ema({1: v * W / (24 * 3600)}, [1], [], t * DELTA, HL, ema, ema_n, last)
        base = last + DELTA
    window, pay, wlen = 0.0, 0.0, int(W / DELTA)
    for t in range(n):
        window += fills[t]
        if t >= wlen:
            window -= fills[t - wlen]
        smoothed, last = apply_track_record_ema({1: window}, [1], [], base + t * DELTA, HL, ema, ema_n, last)
        pay += smoothed[1]
    return pay


def test_an_established_slot_is_paid_the_same_for_a_burst_and_a_steady_day():
    steady = integrated_pay(burst=False, newcomer=False)
    burst = integrated_pay(burst=True, newcomer=False)
    assert abs(burst / steady - 1.0) <= 0.01


def test_a_fresh_slot_carries_a_burst_but_within_half_again_of_steady():
    steady = integrated_pay(burst=False, newcomer=True)
    burst = integrated_pay(burst=True, newcomer=True)
    assert burst > steady
    assert burst / steady <= 1.5


def test_a_fresh_slot_spreading_its_day_earns_less_than_an_established_one():
    assert integrated_pay(burst=False, newcomer=True) < integrated_pay(burst=False, newcomer=False)
