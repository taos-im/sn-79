# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""On the exchange one quote balance is shared by every book, and the report counts it once.

A miner's TAO is one balance whatever book it trades, so each book's account shows all of it. The report summed the
quote over books, and every book's inventory value (its wealth less its wealth at first appearance) carried every move
of that quote, so the account's total counted each move once per book: spending 100 TAO on alpha worth 100 read as a
loss of 100 on a two-book account.
"""
import collections
import types

from taos.im.validator import report, trade

S = 1_000_000_000


def test_a_purchase_at_fair_value_is_not_a_loss():
    inv = {0: {2: 0.0, 5: 0.0}, 1: {2: 0.0, 5: -100.0}}
    quote = {0: 1000.0, 1: 900.0}
    totals, per_book = report.exchange_inventory(inv, quote, {2: 1000.0, 5: 1000.0})
    assert totals == [0.0, 0.0]
    assert per_book[1] == {2: 50.0, 5: -50.0}


def test_a_price_gain_counts_once():
    inv = {0: {2: 0.0, 5: 0.0}, 1: {2: 0.0, 5: -100.0}, 2: {2: 50.0, 5: -100.0}}
    quote = {0: 1000.0, 1: 900.0, 2: 900.0}
    totals, per_book = report.exchange_inventory(inv, quote, {2: 1000.0, 5: 1000.0})
    assert totals == [0.0, 0.0, 50.0]
    assert abs(sum(per_book[2].values()) - totals[-1]) < 1e-9


def test_a_book_that_appears_later_starts_from_its_own_baseline():
    inv = {0: {2: 0.0}, 1: {2: 50.0, 7: 0.0}}
    quote = {0: 1000.0, 1: 900.0}
    # book 2 bought 100 of alpha now worth 150; book 7 appeared after the purchase, with the quote at 900
    totals, _ = report.exchange_inventory(inv, quote, {2: 1000.0, 7: 900.0})
    assert totals == [0.0, 50.0]


def test_a_timestamp_without_a_recorded_quote_keeps_its_plain_sum():
    inv = {0: {2: 0.0, 5: 0.0}, 1: {2: 0.0, 5: -100.0}, 2: {2: 50.0, 5: -100.0}}
    totals, _ = report.exchange_inventory(inv, {2: 900.0}, {2: 1000.0, 5: 1000.0})
    assert totals == [0.0, -100.0, 50.0]


def test_the_quote_totals_count_the_shared_balance_once():
    acct = {"qb": {"t": 900.0}, "ql": 0.0, "qc": 0.0}
    accounts = {2: acct, 5: dict(acct)}
    assert report.account_quote_totals(accounts, [2, 5], True) == (900.0, 0.0, 0.0)
    assert report.account_quote_totals(accounts, [2, 5], False) == (1800.0, 0.0, 0.0)


def _validator(exchange):
    dd = lambda: collections.defaultdict(lambda: collections.defaultdict(float))  # noqa: E731
    sim = types.SimpleNamespace() if exchange else types.SimpleNamespace(block_count=1)
    return types.SimpleNamespace(
        simulation=sim, trade_volumes={}, volume_sums=dd(), maker_volume_sums=dd(), taker_volume_sums=dd(),
        self_volume_sums=dd(), realized_pnl_history={3: {}}, inventory_history={3: {}}, initial_balances={3: {}},
        open_positions={3: {}}, roundtrip_volumes={3: {}}, roundtrip_volume_sums=dd())


BOOKS = {b: {"b": [], "a": []} for b in (2, 5)}


def _account(q):
    return {"WEALTH": q, "qb": {"t": q}, "ql": 0.0, "qc": 0.0, "bb": {"t": 0.0}}


def _run(v, quotes):
    for i, q in enumerate(quotes):
        ts = (i + 1) * S
        accounts = {3: {b: _account(q) for b in BOOKS}}
        trade._process_uid_trade_volumes(v, 3, BOOKS, accounts, {}, ts, ts, False, 0, 4, {}, {}, {}, set())


def test_the_exchange_records_the_quote_at_each_kept_timestamp():
    v = _validator(True)
    _run(v, [1000.0, 950.0, 900.0, 880.0])
    assert sorted(v.inventory_quote[3]) == sorted(v.inventory_history[3]) == [1 * S, 3 * S, 4 * S]
    assert v.inventory_quote[3] == {1 * S: 1000.0, 3 * S: 900.0, 4 * S: 880.0}


def test_the_simulation_records_nothing_extra():
    v = _validator(False)
    _run(v, [1000.0, 950.0])
    assert not getattr(v, "inventory_quote", None)
