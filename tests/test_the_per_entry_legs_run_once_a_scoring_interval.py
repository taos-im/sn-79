# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The per-entry legs run once a scoring interval, weighted by the states and the time since their last run.

Three parts of a state's accounting walk every (uid, book) account entry whatever the fills: the inventory's value, the
size resting at the best bid and ask, and the conjunction (two-sided resting presence near the touch). At 256 books
they were about 88% of the accounting's time, and none of them is paid: the report reads the inventory history, and
the published detail and class summary read at-best and the conjunction. They run at the first state at or past
each scoring interval's boundary in the simulation:
at-best counts the states the run stands for, the conjunction and its farm reference the time since the last run, and
the inventory history's newest entry is the run's. The exchange's states are a block apart, so there they run every
state, as before. A new account's starting balances are still taken at the state it is first seen.

Pay is the test's first concern: every structure the per-fill accounting keeps is the same whatever the cadence.
"""
import copy
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_scoring_shadow as tss  # noqa: E402
import test_shadow_snapshot_completeness as tsc  # noqa: E402
import test_the_per_book_touch_is_computed_once_a_state as tpb  # noqa: E402

import taos.im.validator.trade as trade  # noqa: E402

S = tss._S
LEGS_OWN = {"inventory_history", "inventory_quote", "debeta_at_best", "debeta_best_size", "debeta_at_best_hist",
            "debeta_best_size_hist", "debeta_conj", "debeta_conj_hist", "debeta_conj_farm", "debeta_conj_farm_hist",
            "debeta_conj_clock", "legs_clock"}


def _main(interval_s, uids=tpb.UIDS, books=tpb.BOOKS):
    main = tss._fresh_main(uids=uids, books=books)
    main.config.scoring.interval = interval_s * S
    return main


def _account(bid_q, ask_q, bid=99.9, ask=100.1):
    return {"bb": {"t": 100.0, "f": 90.0, "r": 10.0}, "qb": {"t": 30000.0, "f": 29000.0, "r": 1000.0},
            "bl": 0.0, "ql": 0.0, "bc": 0.0, "qc": 0.0,
            "o": [{"p": bid, "q": bid_q, "s": 0}, {"p": ask, "q": ask_q, "s": 1}]}


def _steady(n, first_seen=None):
    """n states a second apart on one book whose touch stays at 99.9 / 100.1, uid 1 resting 2.0 at both, uids 2 and
    3 trading 1.0 at 100.0 so the book has a fill-size reference (the gate is a quarter of it), the others resting 0.1.
    first_seen: (uid, state) at which uid joins."""
    out = []
    for k in range(1, n + 1):
        st = tsc._fill_state(k * S, [(2, 3, 0, 1.0, 100.0, k % 2)])
        st.books[0]["b"], st.books[0]["a"] = [{"p": 99.9, "q": 5.0}], [{"p": 100.1, "q": 5.0}]
        st.accounts = {u: {0: _account(2.0 if u == 1 else 0.1, 2.0 if u == 1 else 0.1)} for u in range(4)}
        if first_seen and k < first_seen[1]:
            st.accounts.pop(first_seen[0])
        out.append(st)
    return out


def _accounted(main, states):
    for st in states:
        trade.update_trade_volumes(main, copy.deepcopy(st))
    return main


def test_the_legs_run_at_the_first_state_of_each_scoring_interval():
    main = _accounted(_main(5, uids=4, books=1), _steady(12))
    assert sorted(main.inventory_history[1]) == [1 * S, 5 * S, 10 * S], "the first, the previous and the newest run"
    assert main.legs_clock == {"n": 2, "last": 10 * S}, "states 11 and 12 wait for the next run"
    assert main.accounting_profile["due"] == 0


def test_the_cadence_answers_per_state():
    v = SimpleNamespace(config=SimpleNamespace(scoring=SimpleNamespace(interval=5 * S)))
    got = [trade.per_entry_legs_due(v, k * S) for k in range(1, 13)]
    assert [d for d, _ in got] == [k in (1, 5, 10) for k in range(1, 13)]
    assert [w for d, w in got if d] == [1, 4, 5]
    assert trade.per_entry_legs_due(v, 8 * S) == (False, 0), "a state behind the last run was counted by it"
    v.legs_clock["last"] = 1000 * S
    assert trade.per_entry_legs_due(v, 2 * S) == (True, 1), "a new run's clock runs the legs at once, for itself"
    v.engine = SimpleNamespace(mode="exchange")
    assert all(trade.per_entry_legs_due(v, k * S) == (True, 1) for k in range(3, 9)), "the exchange, every state"


def test_pay_does_not_depend_on_the_cadence():
    states = tpb._states(14, seed=6)
    every, fifth = _accounted(_main(1), states), _accounted(_main(5), states)
    se, sf = tpb._structures(every), tpb._structures(fifth)
    assert se.keys() == sf.keys()
    checked = [n for n in se if n not in LEGS_OWN]
    assert {"volume_sums", "realized_pnl_history", "capture_buy_sums", "debeta_mtm", "initial_balances"} <= set(checked)
    for name in checked:
        assert repr(se[name]) == repr(sf[name]), f"{name} depends on the legs' cadence"
    assert repr(se["debeta_at_best"]) != repr(sf["debeta_at_best"]), "the legs ran on fewer states"


def test_at_best_and_the_conjunction_add_up_to_the_time_they_cover():
    states = _steady(10)
    every, fifth = _accounted(_main(1, uids=4, books=1), states), _accounted(_main(5, uids=4, books=1), states)
    for v in (every, fifth):
        assert v.debeta_at_best[1][0] == 10, "ten states at the best: 1 + 4 + 5 on the fifth's runs"
        assert v.debeta_best_size[1][0] == pytest.approx(40.0), "2.0 at the bid and 2.0 at the ask, ten states"
        assert v.debeta_conj[1][0] == pytest.approx(18.0), "2.0 for the nine seconds after the first state"
    assert fifth.debeta_conj_farm[0] == pytest.approx(every.debeta_conj_farm[0])
    assert 0 not in fifth.debeta_conj.get(0, {}), "a uid resting 0.1 a side stays under the gate"


def test_a_new_account_takes_its_starting_balances_when_first_seen():
    main = _main(5, uids=4, books=1)
    main.initial_balances[3] = {}
    _accounted(main, _steady(3, first_seen=(3, 3)))
    assert main.legs_clock["last"] == 1 * S, "state 3 runs no legs"
    assert main.initial_balances[3][0]["WEALTH"] == pytest.approx(30000.0 + 100.0 * 100.0), "taken at state 3"


def test_the_scoring_service_runs_the_legs_on_the_same_states():
    from taos.im.validator.scoring_shadow import _STRUCT_NAMES, _rebuild_structs
    assert "legs_clock" in _STRUCT_NAMES
    assert _rebuild_structs({"legs_clock": {"n": 3, "last": 5 * S}})["legs_clock"] == {"n": 3, "last": 5 * S}
