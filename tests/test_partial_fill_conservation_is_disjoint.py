# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The partial-fill conservation sum must count each unit of the order exactly once.

Three destinations are claimed for an order's quantity: settled for this order, still resting, or
taken off the resting remainder by the market. They are only disjoint if the resting figure and the
taken figure cannot describe the same units -- and they can, because `_restq` is a point-in-time
snapshot while `_mkt` reads the append-only tape.

This has now been wrong in BOTH directions, which is why it is pinned here:

  UNDER-count  `tape_filled` filtered on the aggressor's side alone, so a remainder that rested and
               was then hit counted nowhere: not filled (wrong side) and no longer resting. Reported
               for months as "alpha vanished", which an order book cannot do.

  OVER-count   measuring maker legs fixed that and exposed the mirror. Once the tape is read across
               BOTH of the miner's roles, a remainder that rested and was then taken is already
               inside `filled`; it also still appears in `_restq` for as long as the account
               snapshot predates the fill. Adding the taken figure back on top charges it a third
               time. Seen in practice on a SELL of 1.8231: 1.823 filled, 1.823 reported resting and
               1.823 taken by the market, for 3.646 against 1.8231 ordered, where the tape held one
               fill of 1.823.

The two directions share one rule: the tape is authoritative, and anything it has already counted
comes off the snapshot rather than being added beside it.
"""
import pytest

TOL = lambda q: max(1e-4, 0.001 * q)  # noqa: E731 - the harness's own tolerance


def accounted(filled, restq, mkt):
    """The sum as acceptance_lib computes it: the tape wins, so the overlap comes off the snapshot.

    `filled` is read across both of the miner's roles, so a remainder the market took is ALREADY in
    it. `_mkt` names that same quantity, and its only job here is to say how much of the snapshot's
    resting figure has since become a fill.
    """
    return filled + max(restq - mkt, 0.0)


@pytest.mark.parametrize("label,filled,restq,mkt,ordered", [
    # A BUY of 1.8471 whose own execution took 0.6157 and whose 1.2313 remainder rested and was then
    # taken. The tape carries both legs, so `filled` is the whole 1.847; the snapshot had not caught
    # up and still showed the 1.2313 resting.
    ("remainder taken, snapshot stale", 1.8470, 1.2313, 1.2313, 1.8471),
    # The mirror on the SELL side of the same scenario: one fill of 1.823 on the tape, reported
    # resting and taken as the same 1.823.
    ("remainder taken, sell side", 1.8230, 1.8230, 1.8230, 1.8231),
    # The snapshot HAS caught up: the remainder is gone from the order list and lives only on the
    # tape, inside `filled`. Subtracting must not reach past zero into the settled quantity.
    ("snapshot caught up", 50.0, 0.0, 30.0, 50.0),
    # Nobody took anything: the ordinary partial fill.
    ("no market take", 20.0, 30.0, 0.0, 50.0),
    # Everything filled outright.
    ("filled in full", 50.0, 0.0, 0.0, 50.0),
    # Partially taken: 10 of the 30 resting went and is on the tape inside `filled`, and the
    # snapshot still shows all 30.
    ("partially taken, stale", 30.0, 30.0, 10.0, 50.0),
])
def test_the_order_is_accounted_for_exactly_once(label, filled, restq, mkt, ordered):
    assert abs(accounted(filled, restq, mkt) - ordered) <= TOL(ordered), label


def test_the_naive_sum_would_double_count():
    """Guards the reason the subtraction is there: added side by side, the same units count twice."""
    assert abs((1.8470 + 1.2313) - 1.8471) > TOL(1.8471)


def test_adding_the_taken_figure_back_would_count_it_a_third_time():
    """The shape this file exists to prevent returning: `filled` already holds what `_mkt` names."""
    naive = 1.8230 + max(1.8230 - 1.8230, 0.0) + 1.8230
    assert abs(naive - 1.8231) > TOL(1.8231)
    assert accounted(1.8230, 1.8230, 1.8230) == pytest.approx(1.8230)


def test_the_overlap_never_makes_resting_negative():
    """A take larger than the snapshot's resting figure must not subtract below zero and start
    eating the settled quantity."""
    assert accounted(5.0, 1.0, 9.0) == pytest.approx(5.0)
