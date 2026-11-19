# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The at-best and the conjunction legs read each account's resting orders in one pass, with the results of two.

Both legs walk every resting order of every (uid, book) entry each state: the at-best leg sums the size resting at the
book's best bid and ask, the conjunction the buy size at or above the band's low edge and the sell size at or below its
high edge. At 256 books that was two passes over every order, each with a type check per field, and the larger part of
the state's accounting. accumulate_order_legs reads each order once and feeds both sums, in the order the orders come,
so every structure the two legs maintain is left as the two legs apart leave it, value for value and key for key.

Main and the scoring service run the same function, so the test is equality, to the last bit and the key order.
"""
import copy
import os
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_the_per_book_touch_is_computed_once_a_state as tpb  # noqa: E402

import taos.im.validator.trade as trade  # noqa: E402

LEGS = ("debeta_at_best", "debeta_best_size", "debeta_at_best_hist", "debeta_best_size_hist", "debeta_conj",
        "debeta_conj_hist")


def _legs_apart(self, uid_item, books, accounts_uid, dt, ts=None, best=None, bands=None, at_best=True, conj=True,
               weight=1):
    assert weight == 1, "the two legs apart count one state a run"
    if at_best:
        trade.accumulate_at_best(self, uid_item, books, accounts_uid, ts=ts, best=best)
    if conj:
        trade.accumulate_conjunction(self, uid_item, books, accounts_uid, dt=dt, ts=ts, bands=bands)


def _run(apart, states, per_uid=False):
    main = tpb._every_state_main()
    legs, touch = trade.accumulate_order_legs, trade._per_book_touch
    if apart:
        trade.accumulate_order_legs = _legs_apart
    if per_uid:
        trade._per_book_touch = lambda self, books, debeta_on: (None, None, None)
    try:
        for st in states:
            trade.update_trade_volumes(main, copy.deepcopy(st))
    finally:
        trade.accumulate_order_legs, trade._per_book_touch = legs, touch
    return main


def _same(a, b):
    sa, sb = tpb._structures(a), tpb._structures(b)
    assert sa.keys() == sb.keys()
    for leg in ("debeta_at_best", "debeta_conj"):
        assert any(v for books in sa[leg].values() for v in books.values()), f"{leg} was not exercised"
    for name in sa:
        assert repr(sa[name]) == repr(sb[name]), f"{name} differs between one pass and the two legs apart"


def test_one_pass_leaves_every_structure_as_the_two_legs_do():
    states = tpb._states(16, seed=21)
    _same(_run(False, states), _run(True, states))


def test_one_pass_matches_on_the_per_uid_path_too():
    # a book whose touch cannot be read sends the whole state down the per-uid path: no best, no bands
    states = tpb._states(10, seed=8)
    _same(_run(False, states, per_uid=True), _run(True, states, per_uid=True))


def test_orders_that_are_not_dicts_are_read_as_the_legs_read_them():
    rng = random.Random(4)
    books = {b: {"b": [{"p": 99.9, "q": 1.0}], "a": [{"p": 100.1, "q": 1.0}]} for b in range(3)}
    accounts_uid = {b: SimpleNamespace(o=[SimpleNamespace(p=rng.choice([99.9, 100.1, 99.5, 100.5, None]),
                                                          q=round(rng.uniform(0.5, 4), 2), s=rng.randrange(2))
                                          for _ in range(8)]) for b in range(3)}
    best = {b: (99.9, 100.1) for b in range(3)}
    bands = {b: (100.0, 99.6, 100.4, 1.0) for b in range(3)}
    one, two = SimpleNamespace(), SimpleNamespace()
    for v in (one, two):
        trade.init_liquidity_state(v)
        trade.init_conjunction_state(v)
        trade.accumulate_at_best(v, 0, books, {}, ts=1)  # the liquidity legs' structures, as main has them
    for ts in (1, 2):
        trade.accumulate_order_legs(one, 7, books, accounts_uid, 1.0, ts=ts, best=best, bands=bands)
        _legs_apart(two, 7, books, accounts_uid, 1.0, ts=ts, best=best, bands=bands)
    for name in LEGS:
        assert repr(tpb.tsc._plain(getattr(one, name))) == repr(tpb.tsc._plain(getattr(two, name))), name
    assert one.debeta_at_best[7] and one.debeta_conj[7], "both legs were exercised"


def test_a_leg_switched_off_is_left_alone():
    books = {0: {"b": [{"p": 99.9, "q": 1.0}], "a": [{"p": 100.1, "q": 1.0}]}}
    accounts_uid = {0: {"o": [{"p": 99.9, "q": 2.0, "s": 0}, {"p": 100.1, "q": 2.0, "s": 1}]}}
    v = SimpleNamespace()
    trade.init_liquidity_state(v)
    trade.init_conjunction_state(v)
    trade.accumulate_order_legs(v, 3, books, accounts_uid, 1.0, ts=1, best={0: (99.9, 100.1)},
                                bands={0: (100.0, 99.6, 100.4, 1.0)}, at_best=False)
    assert not v.debeta_at_best.get(3) and v.debeta_conj[3][0] == 2.0
    trade.accumulate_order_legs(v, 4, books, accounts_uid, 0.0, ts=1, best={0: (99.9, 100.1)},
                                bands={0: (100.0, 99.6, 100.4, 1.0)})
    assert v.debeta_at_best[4][0] == 1 and not v.debeta_conj.get(4), "no interval, no presence: the first state"


def test_the_uid_loop_calls_the_one_pass():
    import inspect
    src = inspect.getsource(trade._process_uid_trade_volumes)
    assert "accumulate_order_legs(" in src
    assert "accumulate_at_best(self" not in src and "accumulate_conjunction(self" not in src
