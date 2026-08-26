# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""In-scorer de-beta wiring: MTM alpha reconstruction (drift-strip), E5 floor, and the combined
score. The core invariant is drift-INVARIANCE of alpha: a buy-and-hold drift-rider strips to ~0
while a genuine two-sided maker keeps its (positive) alpha when a uniform drift is added to the book.
Reconstruction faithfulness is validated end-to-end by the sim A/B; these are the unit gates."""
from collections import defaultdict

from taos.im.validator.debeta import (
    accumulate_book_mtm,
    book_alphas_from_mtm,
    median_abs_floor,
    kappa_floored,
    debeta_scores,
    accumulate_counterparties,
    counterparty_ec,
    p11_discount,
)


def _fresh():
    return (
        defaultdict(lambda: defaultdict(float)),  # mtm
        defaultdict(lambda: defaultdict(float)),  # invsum
        {},  # invn
        defaultdict(lambda: defaultdict(float)),  # inv
        {},  # p_first
        {},  # p_last
    )


def _trade(p, q, s, ma, ta):
    return {"p": p, "q": q, "s": s, "Ma": ma, "Ta": ta}


def _run(book_trades_by_batch):
    """Feed a list of per-batch trade lists for ONE book through the cross-batch accumulator."""
    mtm, invsum, invn, inv, pf, pl = _fresh()
    for trades in book_trades_by_batch:
        accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 1, trades)
    return book_alphas_from_mtm(mtm, invsum, invn, pf, pl)


def test_buy_and_hold_drift_rider_strips_to_zero():
    # uid 1 buys q=10 at the first (low) price and holds while price drifts 100 -> 110.
    # Raw MTM is large-positive (rode the drift) but alpha must be ~0 (pure beta, no skill).
    trades = [_trade(100.0 + i, 10.0, 0, 2, 1) if i == 0 else _trade(100.0 + i, 1.0, 0, 3, 4) for i in range(11)]
    alphas = _run([trades])
    assert abs(alphas[1][0]) < 1e-6, alphas[1]


def test_two_sided_maker_keeps_positive_alpha_under_added_drift():
    # uid 3 makes a two-sided market around 100, ends flat, captures spread (buys at 99, sells at 101).
    base = []
    for _ in range(6):
        base.append(_trade(99.0, 5.0, 1, 3, 7))  # s==1: maker(3) buys at 99 from taker 7
        base.append(_trade(101.0, 5.0, 0, 3, 8))  # s==0: taker 8 buys -> maker(3) sells at 101
    a0 = _run([base])[3]
    # Add a uniform up-drift to every price; a genuine (flat-inventory) maker's alpha is ~invariant.
    drifted = [_trade(t["p"] + 0.5 * i, t["q"], t["s"], t["Ma"], t["Ta"]) for i, t in enumerate(base)]
    a1 = _run([drifted])[3]
    assert sum(a0) > 0.0
    assert abs(sum(a1) - sum(a0)) < 0.15 * abs(sum(a0)) + 1e-6, (sum(a0), sum(a1))


def test_cross_batch_carry_matches_single_batch():
    # Splitting the same stream into two publish batches must give the same alpha (inv + prev price carry).
    trades = [_trade(100.0 + i, 10.0, 0, 2, 1) if i == 0 else _trade(100.0 + i, 3.0, 1, 1, 9) for i in range(12)]
    whole = _run([trades])
    split = _run([trades[:5], trades[5:]])
    assert abs(whole[1][0] - split[1][0]) < 1e-6, (whole[1], split[1])


def test_e5_floor_and_kappa_floored_kill_tiny_spam():
    # Tiny-consistent alphas (spam) are below the floor -> < 4 qualifying books -> kappa 0.
    tiny = {u: [1e-4, 1e-4, 1e-4, 1e-4, 1e-4] for u in range(3)}
    big = {u: [5.0, 6.0, 4.0, 5.5, 5.0] for u in range(3, 6)}
    alla = {**tiny, **big}
    floor = median_abs_floor(alla, scale=0.5)
    assert floor > 1e-4
    assert kappa_floored(tiny[0], floor) == 0.0
    assert kappa_floored(big[3], floor) > 0.0


def test_debeta_scores_ranks_maker_over_drift_rider():
    # End-to-end: a two-sided maker (making + skill) outranks a one-sided drift-rider (making ~0, skill ~0).
    buy = defaultdict(lambda: defaultdict(float))
    sell = defaultdict(lambda: defaultdict(float))
    alphas = {}
    for u in range(8):
        # makers 0..3: balanced two-sided capture + consistent positive alpha
        if u < 4:
            for b in range(6):
                buy[u][b] = 3.0
                sell[u][b] = 3.0
            alphas[u] = [1.0, 1.2, 0.9, 1.1, 1.0, 1.05]
        else:
            # drift-riders 4..7: one-sided capture (buy only), inconsistent/zero alpha
            for b in range(6):
                buy[u][b] = 4.0
            alphas[u] = [2.0, -2.0, 1.5, -1.8, 0.1, -0.2]
    floor = median_abs_floor(alphas, scale=0.5)
    scores = debeta_scores(buy, sell, alphas, floor=floor, w_make=0.30)
    maker_mean = sum(scores[u] for u in range(4)) / 4
    rider_mean = sum(scores[u] for u in range(4, 8)) / 4
    assert maker_mean > rider_mean, scores


def test_p11_flags_feeder_leaves_genuine_untouched():
    # 10 genuine makers share a taker pool; 1 maker (uid 1) is fed only by a dedicated feeder (uid 2).
    cp = {}
    for mk in range(10, 20):
        tr = []
        for i in range(24):
            tk = 100 + ((mk * 7 + i) % 20)
            tr.append(_trade(100.0, 5.0, 1, mk, tk))
        accumulate_counterparties(cp, mk, tr)
    accumulate_counterparties(cp, 1, [_trade(100.0, 5.0, 1, 1, 2) for _ in range(24)])
    ec_fed = counterparty_ec(cp, 1)
    ec_gen = sum(counterparty_ec(cp, mk) for mk in range(10, 20)) / 10
    assert ec_fed > 0.8, ec_fed
    assert ec_gen < 0.2, ec_gen
    # discount removes the fed maker's making, leaves genuine ~intact
    own = {u: 100.0 for u in list(range(10, 20)) + [1]}
    disc = p11_discount(own, cp, list(own), strength=1.0)
    assert disc[1] < 20.0, disc[1]
    assert min(disc[mk] for mk in range(10, 20)) > 80.0, disc


def test_p11_discount_off_by_default_is_identity():
    cp = {1: {2: 100.0}}
    own = {1: 50.0, 3: 40.0}
    assert p11_discount(own, cp, [1, 3], strength=0.0) == own


def test_debeta_scores_p11_demotes_feeder_maker():
    # Feeder-fed maker (uid 1) fabricates the HIGHEST raw making; genuine makers have distinct lower
    # making. w_make=1.0 (making only, no skill-tie noise). Without P11 the feeder ranks top; with
    # P11 its (dedicated-counterparty) making is discounted to ~0 and it ranks bottom.
    buy = defaultdict(lambda: defaultdict(float))
    sell = defaultdict(lambda: defaultdict(float))
    cp = {}
    buy[1][0] = sell[1][0] = 10.0  # feeder fabricates making 20 (highest)
    for j, mk in enumerate(range(10, 16)):
        buy[mk][0] = sell[mk][0] = 5.0 + 0.5 * j  # distinct making 10..12.5
        accumulate_counterparties(cp, mk, [_trade(100.0, 5.0, 1, mk, 100 + ((mk + i) % 12)) for i in range(24)])
    accumulate_counterparties(cp, 1, [_trade(100.0, 5.0, 1, 1, 2) for _ in range(24)])
    alphas = {mk: [] for mk in list(range(10, 16)) + [1]}
    without = debeta_scores(buy, sell, alphas, floor=0.0, w_make=1.0)
    with_p11 = debeta_scores(buy, sell, alphas, floor=0.0, w_make=1.0, cp=cp, p11_strength=1.0)
    assert without[1] > max(without[mk] for mk in range(10, 16)), without  # top without P11
    assert with_p11[1] < min(with_p11[mk] for mk in range(10, 16)), with_p11  # bottom with P11


def _fake_validator(enabled, w_make=0.30, min_books=4):
    from types import SimpleNamespace

    debeta_cfg = SimpleNamespace(
        enabled=enabled, w_make=w_make, centered_window=15, floor_scale=0.5, min_books=min_books
    )
    self = SimpleNamespace(config=SimpleNamespace(scoring=SimpleNamespace(debeta=debeta_cfg)))
    # Warm accumulators: 6 books, a clean two-sided maker (uid 1) + a one-sided rider (uid 2).
    mtm, invsum, invn, inv, pf, pl = _fresh()
    drift = {}
    for b in range(6):
        trades = []
        for _ in range(4):
            trades.append(_trade(99.0, 5.0, 1, 1, 7))
            trades.append(_trade(101.0, 5.0, 0, 1, 8))
        # rider buys and holds through an up-drift
        trades += [_trade(100.0 + i, 6.0, 0, 3, 2) if i == 0 else _trade(100.0 + i, 1.0, 0, 4, 5) for i in range(8)]
        accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, b, trades, drift=drift)
    self.debeta_mtm, self.debeta_invsum, self.debeta_invn = mtm, invsum, invn
    self.debeta_drift = drift  # production finalizer (book_alphas_from_drift) reads this
    self.debeta_pfirst, self.debeta_plast = pf, pl
    cb = defaultdict(lambda: defaultdict(float))
    cs = defaultdict(lambda: defaultdict(float))
    for b in range(6):
        cb[1][b] = 3.0
        cs[1][b] = 3.0
        cb[2][b] = 4.0
    self.capture_buy_sums, self.capture_sell_sums = cb, cs
    return self


def test_debeta_histories_survive_msgpack_and_rebuild_sums():
    # Persistence now serialises the TIMESTAMPED HISTORIES (making + skill windows) + reconstruction
    # STATE; the running sums are rebuilt from the histories on load. msgpack must preserve the nested
    # int keys (incl. large ns timestamps), and sum_hist_* must reproduce the running sums exactly.
    import msgpack
    from taos.im.validator.debeta import sum_hist_2level, sum_hist_1level

    saved = {
        "debeta_capbuy_hist": {1: {0: {100: 3.0, 200: 1.0}}, 2: {0: {100: 1.5}}},
        "debeta_mtm_hist": {1: {0: {100: 12.5}}},
        "debeta_invn_hist": {0: {100: 60.0, 200: 40.0}},
        "debeta_drift_hist": {0: {100: 2.0, 200: -1.0}},
        "debeta_cp_hist": {1: {900: {100: 50.0}}},
        "debeta_plast": {0: 101.0},
        "debeta_inv": {0: {1: 5.0, 2: -3.0}},
    }
    back = msgpack.unpackb(msgpack.packb(saved), raw=False, strict_map_key=False)
    assert sum_hist_2level(back["debeta_capbuy_hist"])[1][0] == 4.0     # 3.0 + 1.0
    assert sum_hist_2level(back["debeta_cp_hist"])[1][900] == 50.0
    assert sum_hist_1level(back["debeta_invn_hist"])[0] == 100.0        # 60 + 40
    assert sum_hist_1level(back["debeta_drift_hist"])[0] == 1.0         # 2 - 1
    assert back["debeta_plast"][0] == 101.0 and back["debeta_inv"][0][2] == -3.0


def test_debeta_dereg_reset_clears_reused_uid():
    # A reused UID must not inherit the deregistered miner's de-beta accumulation.
    from taos.im.validator.trade import reset_agent_histories

    class _Fake:
        def __getattr__(self, n):  # auto-vivify the many histories reset_agent_histories touches
            d = defaultdict(lambda: defaultdict(float))
            object.__setattr__(self, n, d)
            return d

    self = _Fake()
    self.capture_buy_sums = defaultdict(lambda: defaultdict(float), {5: {0: 9.0}, 6: {0: 1.0}})
    self.capture_sell_sums = defaultdict(lambda: defaultdict(float), {5: {0: 9.0}})
    self.debeta_mtm = defaultdict(lambda: defaultdict(float), {5: {0: 20.0}})
    self.debeta_invsum = defaultdict(lambda: defaultdict(float), {5: {0: 50.0}})
    self.debeta_inv = defaultdict(lambda: defaultdict(float), {0: defaultdict(float, {5: 7.0, 6: 2.0})})
    self.debeta_cp = {5: {900: 40.0}, 6: {5: 15.0}}  # uid 5 as maker AND as a counterparty of maker 6
    # the timestamped histories must be cleared for the reused uid too (else running != sum(hist))
    self.debeta_capbuy_hist = {5: {0: {100: 9.0}}, 6: {0: {100: 1.0}}}
    self.debeta_capsell_hist = {5: {0: {100: 9.0}}}
    self.debeta_mtm_hist = {5: {0: {100: 20.0}}}
    self.debeta_invsum_hist = {5: {0: {100: 50.0}}}
    self.debeta_cp_hist = {5: {900: {100: 40.0}}, 6: {5: {100: 15.0}}}
    reset_agent_histories(self, 5, [0, 1])
    assert 5 not in self.capture_buy_sums and 5 not in self.capture_sell_sums
    assert 5 not in self.debeta_mtm and 5 not in self.debeta_invsum
    assert 5 not in self.debeta_inv[0]                       # inventory carry cleared
    assert 5 not in self.debeta_cp and 5 not in self.debeta_cp.get(6, {})  # maker + counterparty refs cleared
    assert 5 not in self.debeta_capbuy_hist and 5 not in self.debeta_mtm_hist  # histories cleared
    assert 5 not in self.debeta_cp_hist and 5 not in self.debeta_cp_hist.get(6, {})
    assert 6 in self.capture_buy_sums and 6 in self.debeta_capbuy_hist  # other UIDs untouched


def test_compute_debeta_scores_gate():
    from taos.im.validator.reward import compute_debeta_scores

    # Disabled -> empty (legacy path).
    assert compute_debeta_scores(_fake_validator(enabled=False)) == {}
    # Enabled + warm -> map that ranks the two-sided maker above the one-sided rider.
    scores = compute_debeta_scores(_fake_validator(enabled=True, min_books=1))
    assert scores, scores
    assert scores.get(1, 0.0) > scores.get(2, 0.0), scores
    # Enabled but coverage guard trips (min_books huge) -> empty (warmup safety).
    assert compute_debeta_scores(_fake_validator(enabled=True, min_books=9999)) == {}


# --- windowing / continuity (batch-timestamped histories, prune, sim-boundary re-base) ---------------

def test_windowed_drift_finalizer_equals_fullrun_over_nonpruned():
    # book_alphas_from_drift (production, uses the telescoped dp-sum) must equal book_alphas_from_mtm
    # (offline full-run, uses p_last-p_first) over a NON-pruned run. This is the equivalence that lets the
    # windowed metric inherit the offline validation.
    import random
    from taos.im.validator.debeta import book_alphas_from_drift
    mtm, invsum, invn, inv, pf, pl = _fresh()
    drift = {}
    rng = random.Random(7)
    for step in range(4):
        for b in range(5):
            tr = []
            p = 100 + rng.uniform(-3, 3)
            for _ in range(15):
                p += rng.uniform(-1, 1)
                tr.append(_trade(p, rng.uniform(1, 4), rng.randint(0, 1), rng.randint(0, 6), rng.randint(0, 6)))
            accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, b, tr, drift=drift, ts=step)
    a_full = book_alphas_from_mtm(mtm, invsum, invn, pf, pl)
    a_win = book_alphas_from_drift(mtm, invsum, invn, drift)
    assert set(a_full) == set(a_win)
    for u in a_full:
        assert all(abs(x - y) < 1e-6 for x, y in zip(sorted(a_full[u]), sorted(a_win[u]))), u


def test_live_prune_windows_and_preserves_invariant():
    # Pruning drops out-of-window buckets and subtracts exactly their mass, keeping running == sum(kept).
    from taos.im.validator.debeta import accumulate_book_capture, prune_hist_2level
    buy = defaultdict(lambda: defaultdict(float))
    sell = defaultdict(lambda: defaultdict(float))
    bh, sh = {}, {}
    trades = [_trade(100.0, 5.0, 1, 1, 7), _trade(101.0, 5.0, 0, 1, 8)]
    for ts in (100, 200, 300):
        accumulate_book_capture(buy, sell, 0, trades, 15, buy_hist=bh, sell_hist=sh, ts=ts)
    before = buy[1][0]
    prune_hist_2level(bh, buy, threshold=250)          # drop ts 100, 200
    assert set(bh[1][0].keys()) == {300}
    assert abs(buy[1][0] - sum(bh[1][0].values())) < 1e-9   # invariant held
    assert buy[1][0] < before                                # mass was actually pruned


def test_sim_boundary_rebase_excludes_price_jump():
    # The de-beta MTM is path-dependent; a naive carry across a sim restart would inject a phantom
    # dp = (openB - closeA). Re-basing the reconstructed inventory + last price (as the shift does) means
    # the first new-sim trade emits no dp, so the boundary jump never enters the drift accumulator.
    mtm, invsum, invn, inv, pf, pl = _fresh()
    drift = {}
    simA = [_trade(100.0, 5.0, 0, 1, 7), _trade(101.0, 1.0, 0, 2, 8), _trade(102.0, 1.0, 0, 2, 8)]
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, simA, drift=drift)
    drift_a = drift[0]                                  # internal sim-A drift (102-100 = 2)
    # BOUNDARY re-base (what shift_simulation_histories does): fresh flat inventory + cleared price refs.
    inv = defaultdict(lambda: defaultdict(float))
    pl.clear()
    pf.clear()
    simB = [_trade(1000.0, 5.0, 0, 1, 7), _trade(1001.0, 1.0, 0, 2, 8)]   # opens 900 above sim-A close
    accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, simB, drift=drift)
    # drift picked up only sim-B's internal dp (1001-1000 = 1), NOT the 900 boundary jump
    assert abs(drift[0] - (drift_a + 1.0)) < 1e-6, drift[0]
    assert drift[0] < 10.0


def test_shift_hist_spans_boundary_and_prunes_old():
    # shift_hist_* remaps timestamps onto the new clock so in-window mass carries (continuity) and
    # out-of-window mass prunes, subtracting from the running sum.
    from taos.im.validator.debeta import shift_hist_2level
    hist = {1: {0: {800: 1.0, 900: 2.0, 999: 3.0}}}    # new-clock ts: -200, -100, -1
    running = {1: {0: 6.0}}
    shift_hist_2level(hist, running, old_ts=1000, new_ts=0, threshold=-150)  # keep >= -150
    assert set(hist[1][0].keys()) == {-100, -1}
    assert abs(running[1][0] - 5.0) < 1e-9              # 2 + 3 kept; 1 pruned
    assert abs(running[1][0] - sum(hist[1][0].values())) < 1e-9
