# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The report carries each miner's 50 newest trades across its books, and the newest maker and taker fee rate of
every book.

The dashboard's table of a miner's recent trades shows the newest: the payload carries the 50 newest of each miner,
each book's in the order kept. The fee-rate gauges stay per book: main sends the newest maker and taker fee rate of
every book, read from the newest trade of each role."""
import inspect

from taos.im.protocol.events import TradeEvent
from taos.im.validator import recent_trades


def _trade(book, t, fee=0.0001, maker_fee=0.00005, price=100.0, qty=2.0):
    return TradeEvent.model_construct(y="ET", i=t, c=None, t=t, b=book, s=0, p=price, q=qty, Ti=1, Toi=None, Ta=5,
                                      Tf=fee, Mi=2, Ma=7, Mf=maker_fee, a=None, cr=None)


def test_each_uid_keeps_its_newest_trades_across_books_each_book_in_its_order():
    books = {
        0: [(_trade(0, 10), 'maker'), (_trade(0, 40), 'taker'), (_trade(0, 70), 'maker')],
        1: [(_trade(1, 20), 'taker'), (_trade(1, 50), 'maker')],
        2: [(_trade(2, 30), 'maker')],
        3: [],
    }
    kept, _ = recent_trades.newest_per_uid({7: books}, keep=4)
    assert {b: [x['trade']['t'] for x in tl] for b, tl in kept[7].items()} == {0: [40, 70], 1: [50], 2: [30]}
    assert [x['role'] for x in kept[7][0]] == ['taker', 'maker']


def test_the_fee_rates_are_the_newest_of_each_role_on_every_book_kept_or_not():
    books = {
        0: [(_trade(0, 10, maker_fee=0.02), 'maker'), (_trade(0, 40, fee=0.06), 'taker'),
            (_trade(0, 70, maker_fee=0.04), 'maker')],
        5: [(_trade(5, 1, fee=0.02), 'taker')],
    }
    _, rates = recent_trades.newest_per_uid({7: books}, keep=1)
    assert rates == {7: {0: [0.04 / 200.0, 0.06 / 200.0], 5: [None, 0.02 / 200.0]}}


def test_a_trade_without_its_fee_field_is_passed_over_for_the_rate():
    old = TradeEvent.model_construct(y="ET", i=9, c=None, t=90, b=0, s=0, p=100.0, q=2.0, Ti=1, Toi=None, Ta=5,
                                     Mi=2, Ma=7, a=None, cr=None)
    books = {0: [(_trade(0, 10, maker_fee=0.02), 'maker'), (old, 'maker')]}
    _, rates = recent_trades.newest_per_uid({7: books}, keep=50)
    assert rates == {7: {0: [0.02 / 200.0, None]}}


def test_main_sends_them_and_the_reporter_reads_the_rates_it_is_sent():
    from taos.im.neurons import validator
    from taos.im.validator import report
    assert recent_trades.RECENT_TRADES_PER_UID == 50
    assert "newest_per_uid(self.recent_miner_trades)" in inspect.getsource(validator)
    assert "'recent_fee_rates': recent_fee_rates" in inspect.getsource(validator)
    src = inspect.getsource(report.report)
    assert "self.recent_fee_rates" in src and "last_maker_trade = miner_trade" not in src
