# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Each state's accounting leaves a profile of where its time went, and the validator logs it as one line.

The time logged for a state's accounting ("Agent Scores Data Updated") is wall time, and on a live validator the
report's read and the publish run beside it in the same process: at 256 books testnet logged 0.62 s a state where the
function alone, replayed offline on the bed's states, takes 0.08 to 0.23 s. The profile splits the function's wall
time into its parts (the books loop, the de-beta prune, the uid loop with its order legs and inventory, the closing
prune and rounding) and sets the thread's CPU time beside it, so a log tells the accounting's own cost from the time it
waited for the interpreter; with the account entries and resting orders it read, the cost per entry and per order.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_the_per_book_touch_is_computed_once_a_state as tpb  # noqa: E402

import taos.im.validator.trade as trade  # noqa: E402

KEYS = {"wall", "cpu", "books", "prune", "uids", "legs", "inventory", "post", "entries", "orders", "due"}


def _accounted(n, seed):
    states = tpb._states(n, seed=seed)
    main = tpb._every_state_main()
    for st in states:
        trade.update_trade_volumes(main, copy.deepcopy(st))
    return main, states[-1]


def test_each_state_leaves_its_profile():
    main, last = _accounted(6, seed=2)
    p = main.accounting_profile
    assert set(p) == KEYS
    assert p["entries"] == sum(len(a) for a in last.accounts.values())
    assert p["orders"] == sum(len(e.get("o") or []) for a in last.accounts.values() for e in a.values())
    assert p["orders"] > 0, "the states carry resting orders"
    assert p["due"] == 1, "a scoring interval of one state runs the legs every state"
    for k in KEYS - {"entries", "orders", "due"}:
        assert p[k] >= 0.0, k
    assert p["uids"] >= p["legs"] + p["inventory"] - 1e-9
    assert p["wall"] >= p["books"] + p["prune"] + p["uids"] + p["post"] - 1e-9
    assert p["cpu"] > 0.0


def test_the_profile_is_main_s_own_and_never_shipped():
    from taos.im.validator.scoring_shadow import _STRUCT_NAMES
    assert "accounting_profile" not in _STRUCT_NAMES


def test_the_line_names_every_part():
    line = trade.accounting_profile_line({"wall": 0.4123, "cpu": 0.1804, "books": 0.0201, "prune": 0.0, "uids": 0.35,
                                          "legs": 0.2, "inventory": 0.06, "post": 0.0101, "entries": 67584,
                                          "orders": 20281})
    assert line == ("[ACCOUNTING-PROFILE] wall=0.412s cpu=0.180s books=0.020s prune=0.000s uids=0.350s legs=0.200s "
                    "inventory=0.060s post=0.010s entries=67584 orders=20281 due=1")


def test_the_validator_logs_the_line_after_each_state_s_accounting():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "taos", "im", "neurons",
                            "validator.py")).read()
    i = src.index("self._update_trade_volumes(state)")
    assert "accounting_profile_line(" in src[i:i + 600], "logged right after the state's accounting"
