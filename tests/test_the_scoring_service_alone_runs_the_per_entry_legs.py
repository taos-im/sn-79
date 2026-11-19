# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""With the cutover scoring service up, it alone runs the per-entry legs; main keeps everything paid.

The per-entry legs (the inventory's value, at-best and the conjunction with its farm reference) walk every (uid, book)
account entry, and none of them is paid. Main ran them on the round's process although the scoring service runs the same
accounting on the same states, so the service alone runs them: main keeps the per-fill accounting, the requests' volumes
and its in-process fallback scoring, and takes what the report reads of the legs from the service's scores. The service
keeps its legs across a re-INIT, since main's copies are no longer current; a fresh service takes main's (the last
saved) with its clocks restarted. A save main writes itself pulls the service's legs first. While the service is down
the published figures pause, pay unaffected, and a gauge gives the age of the legs' last run for a monitor."""
import copy
import inspect
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_scoring_shadow as tss  # noqa: E402
import test_shadow_snapshot_completeness as tsc  # noqa: E402
import test_the_per_book_touch_is_computed_once_a_state as tpb  # noqa: E402

import taos.im.validator.scoring_shadow as ss  # noqa: E402
import taos.im.validator.trade as trade  # noqa: E402

S = tss._S


def _main(legs_elsewhere):
    main = tss._fresh_main(uids=tpb.UIDS, books=tpb.BOOKS)
    main.legs_elsewhere = legs_elsewhere
    return main


def test_main_runs_no_legs_while_the_service_runs_them_and_pays_the_same():
    states = tpb._states(12, seed=9)
    alone, both = _main(True), _main(False)
    for st in states:
        trade.update_trade_volumes(alone, copy.deepcopy(st))
        trade.update_trade_volumes(both, copy.deepcopy(st))
        assert alone.accounting_profile["due"] == 0 and alone.accounting_profile["orders"] == 0
    assert all(not h for h in alone.inventory_history.values()), "main values no inventory"
    assert not any(alone.debeta_at_best.values()) and not any(alone.debeta_conj.values())
    assert any(both.debeta_at_best.values()), "the legs run where they are not elsewhere"
    sa, sb = tpb._structures(alone), tpb._structures(both)
    for name in sa:
        if name not in trade.LEGS_OWN:
            assert repr(sa[name]) == repr(sb[name]), f"{name}: pay must not depend on where the legs run"


def test_a_state_behind_the_last_run_was_counted_and_a_new_runs_clock_runs_them():
    v = SimpleNamespace(config=SimpleNamespace(scoring=SimpleNamespace(interval=5 * S)))
    for k in range(1, 11):
        trade.per_entry_legs_due(v, k * S)
    assert v.legs_clock == {"n": 0, "last": 10 * S}
    assert trade.per_entry_legs_due(v, 8 * S) == (False, 0), "a replay after a re-INIT"
    assert trade.per_entry_legs_due(v, 10 * S) == (False, 0)
    assert v.legs_clock == {"n": 0, "last": 10 * S}, "a replay counts no state"
    v.legs_clock = {"n": 3, "last": 1000 * S}
    assert trade.per_entry_legs_due(v, 2 * S) == (True, 1), "a new run's clock, for itself"
    assert "self.legs_clock = {}" in inspect.getsource(trade.shift_simulation_histories), "a run change restarts it"


def test_the_service_keeps_its_legs_across_a_reinit_and_a_fresh_one_restarts_the_clocks():
    own = {"inventory_history": {1: {5 * S: {0: 1.0}}}, "legs_clock": {"n": 1, "last": 5 * S},
           "debeta_conj_clock": {"prev": 5 * S}}
    shadow = SimpleNamespace(legs_here=True, **copy.deepcopy(own))
    kept = ss.set_aside_legs(shadow)
    assert kept == own
    fresh = SimpleNamespace(legs_here=True, inventory_history={1: {1 * S: {0: 9.0}}}, legs_clock={"n": 4, "last": S},
                            debeta_conj_clock={"prev": S})
    ss.install_legs(fresh, kept)
    assert fresh.inventory_history == own["inventory_history"] and fresh.legs_clock == own["legs_clock"]
    respawned = SimpleNamespace(legs_here=True, inventory_history={1: {S: {0: 9.0}}}, legs_clock={"n": 4, "last": S},
                                debeta_conj_clock={"prev": S})
    ss.install_legs(respawned, None)
    assert respawned.inventory_history == {1: {S: {0: 9.0}}}, "main's last saved copy"
    assert respawned.legs_clock == {} and respawned.debeta_conj_clock == {}, "the first run counts only itself"
    plain = SimpleNamespace(legs_here=False, legs_clock={"n": 4, "last": S}, debeta_conj_clock={"prev": S})
    assert ss.set_aside_legs(plain) is None
    ss.install_legs(plain, None)
    assert plain.legs_clock == {"n": 4, "last": S}, "a replica beside main's legs takes main's as before"


def test_the_report_reads_the_services_inventory_and_the_gauges_its_age():
    hist = {1: {S: {0: 1.0}, 5 * S: {0: 2.0}, 10 * S: {0: 3.0}, 15 * S: {0: 4.0}}, 2: {}}
    assert trade.minimal_inventory(hist) == {1: {S: {0: 1.0}, 10 * S: {0: 3.0}, 15 * S: {0: 4.0}}}
    v = SimpleNamespace(legs_elsewhere=False, inventory_history=hist, simulation_timestamp=20 * S)
    assert trade.report_inventory(v) == trade.minimal_inventory(hist)
    assert trade.legs_health(v) == {"legs_in_service": 0, "legs_age_s": 0.0}
    v.legs_elsewhere = True
    assert trade.report_inventory(v) == trade.minimal_inventory(hist), "main's last loaded copy until one is shipped"
    assert trade.legs_health(v) == {"legs_in_service": 1, "legs_age_s": -1.0}
    v.service_legs = {"inventory": {1: {15 * S: {0: 7.0}}}, "ts": 15 * S}
    assert trade.report_inventory(v) == {1: {15 * S: {0: 7.0}}}
    assert trade.legs_health(v) == {"legs_in_service": 1, "legs_age_s": 5.0}


def test_main_saves_pull_the_services_legs_and_the_gauges_publish_them():
    import taos.im.validator.persistence as persistence
    stop = inspect.getsource(persistence.save_state_on_stop)
    assert stop.index("await pull_service_legs(self)") < stop.index("shadow.stop"), "pulled before it stops"
    src = inspect.getsource(persistence)
    assert src.count("await pull_service_legs(self)") == 3, "the stop save, the in-process save, the fallback"
    report = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "taos", "im", "validator",
                               "report.py")).read()
    assert '("scoring_shadow_legs_age_s", "legs_age_s")' in report
    main = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "taos", "im", "neurons",
                             "validator.py")).read()
    assert "self.legs_elsewhere = bool(self._scoring_proc_cutover)" in main
    assert "self.service_legs = adopted['legs']" in main
    assert "minimal_inventory = report_inventory(self)" in main


def _round(ts, fills):
    """A state with its books (the fills' books, a touch around each), as main reads it, and its bytes, as teed."""
    import msgpack
    st = tsc._fill_state(ts, fills) if fills else tsc._fill_state(ts, [(1, 2, 0, 0.0, 100.0, 0)])
    raw = msgpack.packb({"timestamp": ts, "books": st.books, "accounts": {}, "notices": st.notices},
                        use_bin_type=True)
    return st, raw


def test_the_child_reads_the_legs_cadence_from_its_config():
    duck = ss._config_duck(tss._fresh_main().config)
    assert duck.scoring.interval == 5 * S, "without it the child ran the legs every state"


def test_the_real_service_runs_the_legs_ships_them_and_keeps_them_across_a_reinit():
    main = tss._fresh_main()
    main.legs_elsewhere = True
    st, _raw = _round(1 * S, [(1, 2, 0, 3.0, 100.0, 0)])
    trade.update_trade_volumes(main, st)
    main._shadow_applied_ts = 1 * S
    shadow = ss.ScoringShadow(parity_ns=5 * S)
    shadow.start()
    try:
        shadow.send_init(main).result(timeout=60)
        for ts, fills in [(2 * S, [(2, 1, 0, 1.5, 101.0, 1)]), (3 * S, []), (4 * S, []),
                          (5 * S, [(1, 3, 1, 2.0, 56.0, 1)])]:
            st, raw = _round(ts, fills)
            trade.update_trade_volumes(main, st)
            shadow.tee(raw, ts)
        result = shadow.request_scores(5 * S, 5 * S, [], {}, {}, timeout=60)
        assert result is not None and result["legs"]["ts"] == 5 * S, "the service's legs ran at the boundary"
        shipped = result["legs"]["inventory"]
        assert shipped and all(sorted(h) == [2 * S, 5 * S] for h in shipped.values()), shipped
        assert all(not h for h in main.inventory_history.values()), "main ran none"
        pulled = shadow.request_legs(timeout=60)
        assert pulled["legs_clock"]["last"] == 5 * S
        first = {u: min(h) for u, h in pulled["inventory_history"].items() if h}
        assert first and all(t == 2 * S for t in first.values()), "a fresh service's first run is its first state"

        shadow.request_reinit(storm=False)
        deadline = time.monotonic() + 30
        while not shadow.initialized and time.monotonic() < deadline:
            time.sleep(0.1)
            main._shadow_applied_ts = 5 * S
            shadow.send_init(main).result(timeout=60)
        for ts in (6 * S, 7 * S):
            st, raw = _round(ts, [])
            trade.update_trade_volumes(main, st)
            shadow.tee(raw, ts)
        again = shadow.request_legs(timeout=60)
        assert again["legs_clock"]["last"] == 5 * S, "kept across the re-INIT, not main's empty copy"
        assert {u: min(h) for u, h in again["inventory_history"].items() if h} == first
        assert ss.install_pulled_legs(main, again) and main.legs_clock["last"] == 5 * S
    finally:
        shadow.stop()
