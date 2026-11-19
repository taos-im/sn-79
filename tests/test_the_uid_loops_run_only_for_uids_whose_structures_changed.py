# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The volume slots and the starting balances are set up only for uids whose structures changed since the last pass.

Two passes in the per-uid body walked every (uid, book) entry each state whatever the fills: the sampled volume slot
(created once a sampling interval) and a new account's starting balances (taken once, when first seen). At 256 books
they were about 40% of main's remaining accounting once the per-entry legs moved to the scoring service.
Every writer of a uid's volumes or balances replaces them whole or moves them, and the one that edits in place (the
book strip at a layout change) shrinks them, so a uid whose objects are the same, of the same size, at the same
sampling interval, needs neither pass. The test is equality: the same states and the same writers between them, with
the passes forced every state and with them skipped, leave every structure the same, value for value and key for key.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_scoring_shadow as tss  # noqa: E402
import test_the_per_book_touch_is_computed_once_a_state as tpb  # noqa: E402

import taos.im.validator.trade as trade  # noqa: E402

S = tss._S


def _writers(main, k):
    """The writers between states, applied alike to both mains."""
    if k == 5:
        trade.reset_agent_histories(main, 3, list(range(tpb.BOOKS)))
    if k == 7:
        del main.trade_volumes[2][0]  # the book strip at a layout change edits a uid's volumes in place
    if k == 9:
        for d in (main.trade_volumes, main.initial_balances):  # a benchmark's uid remap moves the objects
            d[10], d[11] = d[11], d[10]


def _run(forced, states):
    main = tss._fresh_main(uids=tpb.UIDS, books=tpb.BOOKS)
    for k, st in enumerate(states):
        _writers(main, k)
        if forced:
            main._slot_cache, main._ib_cache = {}, {}
        trade.update_trade_volumes(main, copy.deepcopy(st))
    return main


def _states():
    states = tpb._states(12, seed=13)  # each uid holds a random nine books in ten each state
    later = copy.deepcopy(states[-1])
    later.timestamp += 700 * S  # past the next sampling interval's boundary: every uid's slots again
    return states + [later]


def test_skipping_the_passes_leaves_every_structure_as_forcing_them():
    states = _states()
    forced, skipped = _run(True, states), _run(False, states)
    sf, ss_ = tpb._structures(forced), tpb._structures(skipped)
    assert sf.keys() == ss_.keys()
    for name in sf:
        assert repr(sf[name]) == repr(ss_[name]), f"{name} differs when the passes are skipped"
    assert 0 in skipped.trade_volumes[2], "the stripped book's slot came back"
    last = states[-1].timestamp // (600 * S) * (600 * S)
    assert all(last in skipped.trade_volumes[u][0]["total"] for u in range(tpb.UIDS)), "the new interval's slots"


def test_the_passes_are_skipped_for_a_uid_that_did_not_change():
    states = tpb._states(3, seed=2)
    main = tss._fresh_main(uids=tpb.UIDS, books=tpb.BOOKS)
    trade.update_trade_volumes(main, copy.deepcopy(states[0]))
    tv, ib = main.trade_volumes[4], main.initial_balances[4]
    assert main._slot_cache[4] == (0, tv) and main._ib_cache[4][0] is ib
    assert main._ib_cache[4][1] == frozenset(states[0].accounts[4]), "the books whose balances it filled"
    trade.update_trade_volumes(main, copy.deepcopy(states[1]))
    assert main._slot_cache[4][1] is tv and main._ib_cache[4][0] is ib, "the same objects, nothing rebuilt"


def test_the_caches_are_each_process_own():
    from test_shadow_snapshot_completeness import KNOBS
    assert {"_slot_cache", "_ib_cache"} <= KNOBS
