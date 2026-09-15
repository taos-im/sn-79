"""Per-book two-sided making: a miner one-sided on every book must not earn two-sided credit.

The old rule computed making as ``2*min(sum_b buy_b, sum_b sell_b)``. That aggregates first, so a
miner buying only on book A and selling only on book B presents a balanced GLOBAL pair and is
indistinguishable from a maker quoting both sides of one book. Fill-level capture is two-sided; the
aggregation is not. On one mainnet window, 229 of 238 miners are one-sided on at least one
book, so the two definitions are not interchangeable.

The shipped making leg sums ``2*min(buy_b, sell_b)`` over books and clamps the TOTAL. There is no
flag: rollback lives at ``scoring.debeta.enabled``, and the only extra state a flag could reach was
de-beta enabled WITH the global rule, which measured worse (+0.292 vs +0.365 against market-maker
ground truth) and is the configuration carrying the defect above.
"""

from taos.im.validator.debeta import (
    balanced_reward,
    balanced_reward_per_book,
    debeta_scores,
    median_abs_floor,
)

# uid 1: genuinely two-sided on book 0.
# uid 2: buys only on book 0, sells only on book 1 -> globally balanced, never two-sided.
# uid 3: two-sided across two books, half the size of uid 1.
CB = {1: {0: 5.0}, 2: {0: 5.0, 1: 0.0}, 3: {0: 1.0, 1: 1.5}, 4: {0: 8.0, 1: -6.0}}
CS = {1: {0: 5.0}, 2: {0: 0.0, 1: 5.0}, 3: {0: 1.0, 1: 1.5}, 4: {0: 8.0, 1: -6.0}}
AL = {u: [1.0, 1.1, 0.9, 1.2] for u in (1, 2, 3, 4)}


def test_global_min_cannot_distinguish_a_per_book_one_sided_miner():
    caps = {u: [sum(CB[u].values()), sum(CS[u].values())] for u in CB}
    own = balanced_reward(caps)
    # This is the defect: the sham maker scores exactly like the genuine one.
    assert own[1] == own[2] == 10.0


def test_per_book_min_gives_the_one_sided_miner_nothing():
    own = balanced_reward_per_book(CB, CS, [1, 2, 3, 4])
    assert own[1] == 10.0
    assert own[2] == 0.0
    assert own[3] == 5.0


def test_negative_capture_books_cannot_be_discarded():
    """A loss-making book must count against the miner, not be dropped.

    Capture is (mid - p) * q and is negative on 43% of per-book values on real mainnet data. An
    earlier version clamped at 0 INSIDE the per-book loop, which let a miner keep its profitable
    books and discard the rest - scoring HIGHER than the global rule and rewarding the selective
    one-sidedness this function exists to stop. The clamp belongs on the total.
    """
    cb = {9: {0: +10.0, 1: -30.0}}
    cs = {9: {0: +10.0, 1: -30.0}}
    # global: 2*min(-20,-20) = -40 -> 0.  per-book must also be 0, not +20.
    assert balanced_reward_per_book(cb, cs, [9])[9] == 0.0


def test_per_book_is_never_greater_than_global():
    """sum_b min(a_b, b_b) <= min(sum a, sum b), so with the clamp on the TOTAL the per-book rule
    can only lower a score. Holds for negative capture too, which the per-book clamp did not."""
    caps = {u: [sum(CB[u].values()), sum(CS[u].values())] for u in CB}
    glob = balanced_reward(caps)
    per = balanced_reward_per_book(CB, CS, list(CB))
    for u in CB:
        assert per[u] <= glob[u] + 1e-12


def test_empty_and_missing_uids_do_not_raise():
    assert balanced_reward_per_book({}, {}, []) == {}
    out = balanced_reward_per_book({1: {0: 2.0}}, {}, [1, 99])
    assert out[1] == 0.0 and out[99] == 0.0


def test_making_uses_the_per_book_rule_unconditionally():
    """No flag: the per-book form IS the making leg.

    It shipped briefly behind --scoring.debeta.per_book_making. That was wrong: the only state the
    flag added was de-beta enabled WITH the global rule, which is the configuration measured worse
    (+0.292 vs +0.365 against market-maker ground truth) and the one carrying the defect above.
    Rollback already exists at scoring.debeta.enabled. So a score computed through debeta_scores must
    match the per-book rule, and a globally-balanced per-book-one-sided miner must earn no making
    without anyone having to set anything.
    """
    floor = median_abs_floor(AL)
    sc = debeta_scores(CB, CS, AL, floor=floor, w_make=1.0)   # w_make 1.0 -> score IS the making rank
    # uid 2 is one-sided on every book: it must rank at the bottom of the making leg.
    assert sc[2] == min(sc.values())
    assert sc[1] > sc[2]



def test_p11_discount_fast_path_matches_naive():
    """p11_discount was O(makers x total_cp_entries): counterparty_ec rebuilt the leave-one-out
    market aggregate from scratch per maker (measured 609ms of a 629ms scoring cycle at mainnet
    scale). The fast path precomputes the global aggregate once and subtracts; it must be
    numerically identical to the naive per-maker rebuild."""
    import random
    from taos.im.validator.debeta import counterparty_ec, p11_discount
    random.seed(3)
    cp = {m: {t: random.random() * 100 for t in random.sample(range(300), random.randint(1, 40))}
          for m in range(250)}
    own = {m: random.random() * 50 for m in range(250)}
    uids = list(range(250))
    fast = p11_discount(dict(own), cp, uids, strength=1.0)
    naive = {u: own.get(u, 0.0) * max(0.0, 1.0 - 1.0 * max(0.0, counterparty_ec(cp, u, 2)))
             for u in uids}
    for u in uids:
        assert fast[u] == __import__('pytest').approx(naive[u], abs=1e-12), u
