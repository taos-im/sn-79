# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The report publishes each miner's resting orders and open loans, summarised.

The state carries every account's open orders and loans, and the reporting child summarises them per uid and book
(orders and resting volume by side, the miner's own best bid and ask, the resting notional, the open loans) and per
uid, from the lists the payload already carries. Per-book series exist only where the miner rests orders or holds
loans, and a book the miner has left is cleared at the next cycle, so the agent gauges grow by the books in use, not
by every book."""
import inspect

from taos.im.validator import order_stats as ost


def _acct(orders=(), loans=None):
    return {'bb': {'t': 1.0, 'f': 1.0, 'r': 0.0}, 'qb': {'t': 100.0, 'f': 100.0, 'r': 0.0},
            'bl': 0.0, 'ql': 0.0, 'bc': 0.0, 'qc': 0.0, 'o': list(orders), 'l': dict(loans or {}),
            'f': {'v': None, 'm': 0.0, 't': 0.0002}, 'v': 0.0}


def _order(side, price, qty):
    return {'y': 'o', 'i': 1, 'c': None, 't': 0, 'q': qty, 's': side, 'p': price, 'l': 0.0}


def test_each_book_with_orders_or_loans_is_summarised_and_the_others_are_left_out():
    accounts = {
        3: {0: _acct([_order(0, 10.0, 2.0), _order(0, 11.0, 1.0), _order(1, 12.5, 4.0)]), 1: _acct()},
        4: {0: _acct(loans={7: {'i': 7, 'a': 50.0, 'c': 1, 'bc': 0.0, 'qc': 60.0}}),
            2: _acct([_order(1, 3.0, 1.5)])},
        5: {0: _acct()},
    }
    stats = ost.book_stats(accounts)
    assert stats == {
        3: {0: ost.BookOrders(2, 1, 3.0, 4.0, 11.0, 12.5, 31.0, 50.0, 0)},
        4: {0: ost.BookOrders(0, 0, 0.0, 0.0, None, None, 0.0, 0.0, 1),
            2: ost.BookOrders(0, 1, 0.0, 1.5, None, 3.0, 0.0, 4.5, 0)},
    }


def test_an_order_without_a_price_is_counted_but_does_not_move_the_prices_or_the_notional():
    stats = ost.book_stats({1: {0: _acct([_order(0, None, 2.0), _order(0, 9.0, 1.0)])}})
    assert stats == {1: {0: ost.BookOrders(2, 0, 3.0, 0.0, 9.0, None, 9.0, 0.0, 0)}}


def test_the_uid_totals_sum_the_books():
    stats = ost.book_stats({
        3: {0: _acct([_order(0, 10.0, 2.0), _order(1, 12.5, 4.0)]), 1: _acct([_order(1, 2.0, 1.0)])},
        4: {0: _acct(loans={1: {}, 2: {}})},
    })
    assert ost.uid_totals(stats) == {
        3: {'open_bids': 1, 'open_asks': 2, 'resting_bid_notional': 20.0, 'resting_ask_notional': 52.0,
            'order_books': 2, 'open_loans': 0},
        4: {'open_bids': 0, 'open_asks': 0, 'resting_bid_notional': 0.0, 'resting_ask_notional': 0.0,
            'order_books': 0, 'open_loans': 2},
    }


def test_the_book_gauges_publish_counts_and_loans_but_not_a_makers_quotes():
    row = ost.BookOrders(2, 0, 3.0, 0.0, 9.0, None, 9.0, 0.0, 1)
    assert dict(ost.book_gauges(row)) == {'open_bids': 2, 'open_asks': 0, 'open_loans': 1}
    assert set(ost.BOOK_GAUGES) == {'open_bids', 'open_asks', 'open_loans'}, (
        "a book's own best bid and ask and its resting volume per side give a maker's quoting away; the public "
        "series carry the counts per book and the totals per uid")


def test_the_reporter_publishes_them_and_evicts_the_series_it_published_last_cycle_and_not_this_one():
    from taos.im.validator import report
    src = inspect.getsource(report.report)
    assert "order_stats.book_stats(" in src and "order_stats.uid_totals(" in src
    at = src.index("getattr(self, '_order_book_keys', set()) - _order_keys")
    assert "_remove_and_evict(" in src[at:at + 300] and "self._order_book_keys = _order_keys" in src[at:at + 400], (
        "the series published last cycle are remembered, and the ones not published again are removed")
    for label in ('open_bids', 'open_asks', 'resting_notional', 'open_loans'):
        assert f"'{label}'" in inspect.getsource(report.ReportingService._init_prometheus), label
