# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A uid absent from a state's accounts keeps the same bounded inventory history as one that has an account.

The per-uid update keeps three entries of inventory history for a uid with an account (the first, the previous and
the newest), but appended a zero row for every book on every update for a uid without one, and nothing prunes that
history. A uid the engine does not report, or one with no account on the exchange, grew it for the whole run. A
state can omit a uid's account, so that history must stay bounded too.
"""
import collections
import types

from taos.im.validator import trade

S = 1_000_000_000


def _validator():
    dd = lambda: collections.defaultdict(lambda: collections.defaultdict(float))  # noqa: E731
    return types.SimpleNamespace(
        trade_volumes={}, volume_sums=dd(), maker_volume_sums=dd(), taker_volume_sums=dd(), self_volume_sums=dd(),
        realized_pnl_history={3: {}}, inventory_history={3: {}}, initial_balances={3: {}}, open_positions={3: {}},
        roundtrip_volumes={3: {}}, roundtrip_volume_sums=dd())


BOOKS = {b: {"b": [], "a": []} for b in range(4)}


def _run(v, accounts, n):
    for i in range(n):
        ts = (i + 1) * S
        trade._process_uid_trade_volumes(v, 3, BOOKS, accounts, {}, ts, ts, False, 0, 4, {}, {}, {}, set())


def test_a_uid_without_an_account_keeps_three_entries():
    v = _validator()
    _run(v, {}, 50)
    hist = v.inventory_history[3]
    assert sorted(hist) == [1 * S, 49 * S, 50 * S]
    assert all(row == {b: 0.0 for b in BOOKS} for row in hist.values())


def test_the_two_paths_keep_the_same_shape():
    a, b = _validator(), _validator()
    _run(a, {}, 7)
    _run(b, {3: {bk: {"WEALTH": 100.0} for bk in BOOKS}}, 7)
    assert sorted(a.inventory_history[3]) == sorted(b.inventory_history[3]) == [1 * S, 6 * S, 7 * S]


def test_a_uid_first_seen_without_an_account_gets_a_history():
    v = _validator()
    del v.inventory_history[3]
    _run(v, {}, 2)
    assert sorted(v.inventory_history[3]) == [1 * S, 2 * S]
