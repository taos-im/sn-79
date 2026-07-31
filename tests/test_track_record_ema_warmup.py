"""Track-record EMA newcomer-warmup protection (annealing counter guard).

A fresh miner sits in its min_lookback window for ~1000 scoring rounds with no valid
Kappa (scorable=False, trading_score=0). Before the fix those zero rounds advanced the
annealing counter k, so by the first real score k was ~1000, the 1/(k+1) newcomer
protection was spent, and the standing had to crawl up from 0 across the whole immunity
window ("kappa is reached fast but the overall score is not"). The fix keeps k=0 through
warmup so the first real score seeds the standing at full value.
"""

from taos.im.validator.reward import apply_track_record_ema

HL = 10_800_000_000_000  # 3 sim-h in ns


def test_warmup_does_not_advance_counter():
    ema, ema_n = {}, {}
    ts = 1_000_000_000
    # 1000 min_lookback rounds: uid not scorable, score 0.
    for _ in range(1000):
        apply_track_record_ema({0: 0.0}, [0], [], ts, HL, ema, ema_n, None, scorable=set())
        ts += 5_560_000_000  # ~one round (5.56 sim-s)
    assert ema_n.get(0, 0) == 0, "warmup rounds must not advance the annealing counter"
    assert 0 not in ema, "warmup miner must not accrue an EMA value"


def test_first_real_score_seeds_full_value():
    ema, ema_n = {}, {}
    ts = 1_000_000_000
    for _ in range(1000):  # warmup
        apply_track_record_ema({0: 0.0}, [0], [], ts, HL, ema, ema_n, None, scorable=set())
        ts += 5_560_000_000
    # First scorable round with a genuine kappa-driven score.
    apply_track_record_ema({0: 0.45}, [0], [], ts, HL, ema, ema_n, None, scorable={0})
    assert ema_n[0] == 1
    assert abs(ema[0] - 0.45) < 1e-9, "first real score must seed at full value (k=0 -> alpha=1)"


def test_established_miner_going_silent_still_decays():
    # An already-scored miner (ema_n>0) that loses all valid kappa must NOT freeze —
    # it decays toward 0 (the trader's-option protection the EMA exists for).
    ema, ema_n = {0: 0.5}, {0: 100}
    last = 1_000_000_000
    ts = last + HL  # a full half-life later
    apply_track_record_ema({0: 0.0}, [0], [], ts, HL, ema, ema_n, last, scorable=set())
    assert ema_n[0] == 101, "established miner must keep updating even when unscorable"
    assert ema[0] < 0.5, "silent established miner must decay, not retain standing"


def test_scorable_none_preserves_legacy_behaviour():
    # Back-compat: no scorable set passed -> every uid updates (old behaviour).
    ema, ema_n = {}, {}
    apply_track_record_ema({0: 0.0}, [0], [], 1_000_000_000, HL, ema, ema_n, None)
    assert ema_n.get(0, 0) == 1
