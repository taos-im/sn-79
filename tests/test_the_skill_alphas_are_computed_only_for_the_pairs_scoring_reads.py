# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The skill leg's per-book alphas are computed only for the (uid, book) pairs scoring reads, with the same values in
the same order.

The mark-to-market and inventory accumulators carry every book a uid holds a position on, while the skill pool keeps
only the books it filled inside the window (traded_book_alphas), and subwindow_skill reads only the pool's.
book_alphas_by_book, book_alphas_held_by_book and the sub-window alphas each take `only`, the pairs to compute, and
iterate the same sets in the same order, so each output is the filtered output entry for entry, and every sum taken
over it after is the same float."""
import inspect
import random
import re

from taos.im.validator import debeta, reward


def _world(seed=1, uids=40, books=24, stamps=30):
    rnd = random.Random(seed)
    mtm, invsum, heldinv, helddrift = {}, {}, {}, {}
    invn, drift, heldn = {}, {}, {}
    mtm_h, invsum_h, invn_h, drift_h = {}, {}, {}, {}
    caps_b, caps_s = {}, {}
    ts_list = [1000 + 60 * i for i in range(stamps)]
    for b in range(books):
        invn[b] = rnd.randint(1, 50)
        drift[b] = rnd.uniform(-3, 3)
        invn_h[b] = {ts: rnd.randint(0, 3) for ts in ts_list}
        drift_h[b] = {ts: rnd.uniform(-0.2, 0.2) for ts in ts_list}
    for u in range(uids):
        held = rnd.sample(range(books), rnd.randint(3, books))
        mtm[u] = {b: rnd.uniform(-5, 5) for b in held}
        invsum[u] = {b: rnd.uniform(-20, 20) for b in held}
        heldinv[u] = {b: rnd.uniform(-20, 20) for b in held}
        helddrift[u] = {b: rnd.uniform(-3, 3) for b in held}
        heldn[u] = {b: rnd.randint(0, 50) for b in held}
        mtm_h[u] = {b: {ts: rnd.uniform(-0.5, 0.5) for ts in ts_list if rnd.random() < 0.7} for b in held}
        invsum_h[u] = {b: {ts: rnd.uniform(-2, 2) for ts in ts_list if rnd.random() < 0.7} for b in held}
        traded = [b for b in held if rnd.random() < 0.5]
        caps_b[u] = {b: 1.0 for b in traded[::2]}
        caps_s[u] = {b: 1.0 for b in traded[1::2]}
    return dict(mtm=mtm, invsum=invsum, invn=invn, drift=drift, heldinv=heldinv, heldn=heldn, helddrift=helddrift,
                mtm_h=mtm_h, invsum_h=invsum_h, invn_h=invn_h, drift_h=drift_h, caps_b=caps_b, caps_s=caps_s,
                t0=ts_list[0], t1=ts_list[-1])


def _traded(w):
    return {u: set(w["caps_b"].get(u) or {}) | set(w["caps_s"].get(u) or {}) for u in set(w["mtm"]) | set(w["invsum"])}


def _same(a, b):
    assert list(a) == list(b)
    for u in a:
        assert list(a[u].items()) == list(b[u].items()), u


def test_the_alphas_by_book_of_the_traded_pairs_are_the_filtered_ones_entry_for_entry():
    w = _world()
    full = debeta.traded_book_alphas(debeta.book_alphas_by_book(w["mtm"], w["invsum"], w["invn"], w["drift"]),
                                     w["caps_b"], w["caps_s"])
    only = debeta.book_alphas_by_book(w["mtm"], w["invsum"], w["invn"], w["drift"], only=_traded(w))
    _same(full, only)


def test_the_held_alphas_of_the_traded_pairs_are_the_filtered_ones_entry_for_entry():
    w = _world(2)
    full = debeta.traded_book_alphas(
        debeta.book_alphas_held_by_book(w["mtm"], w["heldinv"], w["heldn"], w["helddrift"]), w["caps_b"], w["caps_s"])
    only = debeta.book_alphas_held_by_book(w["mtm"], w["heldinv"], w["heldn"], w["helddrift"], only=_traded(w))
    _same(full, only)


def test_the_sub_window_alphas_of_the_pool_are_the_full_ones_and_the_skill_is_the_same():
    w = _world(3)
    pool = debeta.traded_book_alphas(debeta.book_alphas_by_book(w["mtm"], w["invsum"], w["invn"], w["drift"]),
                                     w["caps_b"], w["caps_s"])
    args = (w["mtm_h"], w["invsum_h"], w["invn_h"], w["drift_h"], w["t0"], w["t1"])
    for k in (3, 4):
        full = debeta.book_alphas_by_subwindow(*args, k)
        only = debeta.book_alphas_by_subwindow(*args, k, only=pool)
        for u, by_book in only.items():
            for b, row in by_book.items():
                assert b in pool.get(u, {}) and row == full[u][b]
        for u, by_book in pool.items():
            for b in by_book:
                if b in full.get(u, {}):
                    assert only[u][b] == full[u][b]
        floor = 0.05
        for form, agree in (("weakest", 2), ("sign_gated", 2)):
            assert debeta.subwindow_skill(pool, full, form, agree, floor, 4) == \
                debeta.subwindow_skill(pool, only, form, agree, floor, 4), (k, form)


def test_scoring_passes_the_pairs_it_reads():
    src = inspect.getsource(reward.compute_debeta_scores)
    assert "only=_traded" in src and src.count("only=_traded") >= 2
    assert src.count("book_alphas_by_subwindow(") == 1, "sub-windows only when skill_subwindows pays them"
    assert "book_alphas_by_subwindow(*_hists, _t_start, _t_end, _k, only=skill_pool)" in src


def test_the_alternative_marks_skill_is_read_on_the_same_traded_pairs_as_the_paid_leg():
    """The alternative mark's skill (published, never paid) reads the same traded pairs as the paid leg, so its floor
    rule and bar are the paid leg's."""
    src = inspect.getsource(reward.compute_debeta_scores)
    assert re.search(r"getattr\(self, 'debeta_invn_alt', \{\}\) or \{\}, "
                     r"getattr\(self, 'debeta_drift_alt', \{\}\) or \{\},\s*only=_traded\)", src)
