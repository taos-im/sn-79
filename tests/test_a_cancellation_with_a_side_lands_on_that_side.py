# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The engine publishes a cancellation without a side, so the book rebuild guesses the side from the best ask; a
wrong guess moves a whole level to the other side (RECON 27.2469@297.49 against TARGET 0 on the 6 October testnet
log). A cancellation may now carry a side, the same values as an order's (0 buy, 1 sell); when it does the rebuild
uses it, and when it does not the old guess runs unchanged, so an engine that does not publish the side behaves as
before."""
import copy

from taos.im.protocol.orders import Cancellation
from taos.im.utils import history as H

SNAP = {'timestamp': 0,
        'bids': {99.0: {'p': 99.0, 'q': 5.0, 'o': None}},
        'asks': {100.5: {'p': 100.5, 'q': 7.0, 'o': None}, 99.0: {'p': 99.0, 'q': 3.0, 'o': None}}}  # a crossed-looking ask at 99.0, as the engine can publish between matches


def test_the_cancellation_model_takes_an_optional_side():
    plain = Cancellation.model_validate({'y': 'c', 'orderId': 7, 'timestamp': 1, 'price': 99.0, 'quantity': 2.0})
    assert plain.s is None
    sided = Cancellation.model_validate({'y': 'c', 'orderId': 7, 'timestamp': 1, 'price': 99.0, 'quantity': 2.0, 'side': 1})
    assert sided.s == 1
    assert Cancellation.model_validate(sided.model_dump(by_alias=True)).s == 1


def test_a_sell_cancellation_below_the_best_ask_reduces_the_ask_level_when_it_says_so():
    hist, _ = H.history(copy.deepcopy(SNAP), [{'y': 'c', 'i': 7, 't': 1, 'p': 99.0, 'q': 2.0, 's': 1}], 4)
    after = hist[1]
    assert after['asks'][99.0]['q'] == 1.0, "the sell side was named, so the ask level took the cancel"
    assert after['bids'][99.0]['q'] == 5.0


def test_without_a_side_the_guess_runs_exactly_as_before():
    hist, _ = H.history(copy.deepcopy(SNAP), [{'y': 'c', 'i': 7, 't': 1, 'p': 99.0, 'q': 2.0}], 4)
    after = hist[1]
    # 99.0 is below the best ask (99.0 >= 99.0 is the guess's test: price at or above the best ask is an ask)
    assert after['asks'][99.0]['q'] == 1.0 and after['bids'][99.0]['q'] == 5.0
    hist2, _ = H.history(copy.deepcopy(SNAP), [{'y': 'c', 'i': 7, 't': 1, 'p': 98.0, 'q': 2.0}], 4)
    assert hist2[1]['bids'] == SNAP['bids'], "a price below the best ask with no such bid level changes nothing"


def test_a_buy_cancellation_at_or_above_the_best_ask_reduces_the_bid_level_when_it_says_so():
    hist, _ = H.history(copy.deepcopy(SNAP), [{'y': 'c', 'i': 8, 't': 1, 'p': 99.0, 'q': 2.0, 's': 0}], 4)
    assert hist[1]['bids'][99.0]['q'] == 3.0 and hist[1]['asks'][99.0]['q'] == 3.0


def test_an_empty_ask_side_does_not_break_the_guess():
    snap = {'timestamp': 0, 'bids': {99.0: {'p': 99.0, 'q': 5.0, 'o': None}}, 'asks': {}}
    hist, _ = H.history(snap, [{'y': 'c', 'i': 9, 't': 1, 'p': 99.0, 'q': 2.0}], 4)
    assert hist[1]['bids'][99.0]['q'] == 3.0
