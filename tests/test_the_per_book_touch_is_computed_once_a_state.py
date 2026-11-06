# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The per-state accounting reads each book's touch once a state, not once a miner, with the per-uid results.

update_trade_volumes walks every uid and, for each uid, every book it holds an account on: the at-best leg reads the
book's best bid and ask, the conjunction the book's band (its touch, its references, the class's dials and the class
name), and the inventory the book's mid. None of these depends on the uid, so the touch, the band and the mid are
computed once a book before the uid loop, after the books loop has updated the references they read, and handed to the
legs.

Main and the scoring service run the same function, so the test is equality: the same states through the per-uid
path and through the cached one leave every structure the accounting maintains identical, to the last bit.
"""
import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_scoring_shadow as tss  # noqa: E402
import test_shadow_snapshot_completeness as tsc  # noqa: E402

import taos.im.validator.trade as trade  # noqa: E402

UIDS, BOOKS = 12, 6


def _states(n, seed):
    rng = random.Random(seed)
    out = []
    for k in range(1, n + 1):
        fills = [(rng.randrange(UIDS), rng.randrange(UIDS), rng.randrange(BOOKS), round(rng.uniform(0.25, 5), 2),
                  round(300 + rng.uniform(-3, 3), 2), j % 2) for j in range(20)]
        st = tsc._fill_state(k * tss._S, fills)
        for b in range(BOOKS):
            mid = 300 + rng.uniform(-3, 3)
            bk = st.books.setdefault(b, {"e": []})
            if b == BOOKS - 1 and k % 3 == 0:
                bk["b"], bk["a"] = [], [{"p": round(mid, 2), "q": 1.0}]  # a one-sided book: no touch, no band
                continue
            bk["b"] = [{"p": round(mid - 0.05 * (i + 1), 2), "q": round(rng.uniform(0.25, 20), 2)} for i in range(5)]
            bk["a"] = [{"p": round(mid + 0.05 * (i + 1), 2), "q": round(rng.uniform(0.25, 20), 2)} for i in range(5)]
        st.accounts = {
            u: {b: {"bb": {"t": 100.0 + u, "f": 90.0, "r": 10.0}, "qb": {"t": 30000.0 - b, "f": 29000.0, "r": 1000.0},
                    "bl": 0.0, "ql": 0.0, "bc": 0.0, "qc": 0.0,
                    "o": [{"p": round(st.books[b]["b"][0]["p"] if (st.books[b]["b"] and i == 0)
                                      else 300 + rng.uniform(-1, 1), 2),
                           "q": round(rng.uniform(0.5, 4), 2), "s": i % 2} for i in range(rng.randrange(5))]}
                for b in range(BOOKS) if rng.random() < 0.9}
            for u in range(UIDS)}
        out.append(st)
    return out


def _structures(main):
    """Every structure the scoring service mirrors from main: what the accounting maintains."""
    from taos.im.validator.scoring_shadow import _STRUCT_NAMES
    return {n: tsc._plain(getattr(main, n)) for n in sorted(set(_STRUCT_NAMES)) if hasattr(main, n)}


def _run(per_uid, states):
    main = tss._fresh_main(uids=UIDS, books=BOOKS)
    original = trade._per_book_touch
    if per_uid:
        trade._per_book_touch = lambda self, books, debeta_on: (None, None, None)
    try:
        for st in states:
            trade.update_trade_volumes(main, copy.deepcopy(st))
    finally:
        trade._per_book_touch = original
    return main


def test_the_cached_touch_leaves_every_structure_as_the_per_uid_one_does():
    states = _states(14, seed=11)
    a, b = _run(True, states), _run(False, states)
    sa, sb = _structures(a), _structures(b)
    assert sa.keys() == sb.keys()
    for leg in ("debeta_at_best", "debeta_conj"):
        assert any(v for books in sa[leg].values() for v in books.values()), f"{leg} was not exercised"
    for name in sa:
        assert sa[name] == sb[name], f"{name} differs between the per-uid and the cached touch"


def test_the_cache_holds_one_entry_a_book():
    states = _states(4, seed=3)
    main = _run(False, states[:3])
    st = states[3]
    best, bands, mids = trade._per_book_touch(main, st.books, True)
    assert bands is not None, "after a few states the conjunction has an interval and references"
    assert set(best) == set(st.books) and set(mids) == set(st.books)
    for b, book in st.books.items():
        assert best[b] == (trade._level_price(book["b"]), trade._level_price(book["a"]))
        assert bands[b] == trade._conj_band(main, b, book)


def test_a_malformed_book_fails_only_the_uids_that_hold_it():
    # the touch is read before the uid loop, outside its per-uid try: if any book's touch cannot be read, the state uses
    # the per-uid path, where only the uids that hold that book fail
    states = _states(6, seed=5)
    for st in states[3:]:
        del st.books[0]["a"]  # a book the books loop reads (by .get) and get_inventory_value cannot (by ['a'])
        for u in range(UIDS // 2):
            st.accounts[u].pop(0, None)  # half the uids hold no account on the malformed book
    main = tss._fresh_main(uids=UIDS, books=BOOKS)
    main.pagerduty_alert = lambda *a, **k: None
    for st in states:
        trade.update_trade_volumes(main, copy.deepcopy(st))
    last = states[-1].timestamp
    for u in range(UIDS // 2):
        assert last in main.inventory_history[u], f"uid {u} holds no account on the malformed book and is accounted"
    ref = tss._fresh_main(uids=UIDS, books=BOOKS)
    ref.pagerduty_alert = lambda *a, **k: None
    original = trade._per_book_touch
    trade._per_book_touch = lambda self, books, debeta_on: (None, None, None)
    try:
        for st in states:
            trade.update_trade_volumes(ref, copy.deepcopy(st))
    finally:
        trade._per_book_touch = original
    assert _structures(main) == _structures(ref), "the fallback is the per-uid path, uid for uid"
