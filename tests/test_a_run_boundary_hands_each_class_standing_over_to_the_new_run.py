# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A run boundary that changes the layout or a class's market hands each class's standing over to the new run.

The books' records follow the change map: a book keeps its history only when its market is unchanged, judged on the
class's whole configuration, the price process included, so no record is scored in another market's pool. The
standing follows the class: each class's making and skill shares as paid at the end of the old run are carried by the
class's name, wherever its books now sit and however many there are, and blended into the shares it earns in the new
run, the carried ones weighing 1 at the boundary and 0 one lookback later. A class new to the layout ramps its weight
in over the same span, its weight handed to the other classes meanwhile.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_the_validator_scores_the_layout_of_the_run_it_is_on import (  # noqa: E402
    OLD_RUN, _run_dir, tree as _tree_fixture,
)

from taos.im.validator import persistence, reward, trade  # noqa: E402

tree = _tree_fixture  # that module's fixture, collected here under its own name
_H = 3 * 3600 * 10 ** 9  # one lookback, simulated nanoseconds


def _layout(tree, name, comp):
    d = _run_dir(tree.root, name, comp, tree.config_dir)
    found = tree.engine._run_layout(d / "simulation_0-0")
    assert found is not None
    return found[1]


# ── the change map: the market and the class names ───────────────────────────────────────────────────────────────

def test_the_change_map_names_the_classes_of_both_layouts(tree):
    old = _layout(tree, OLD_RUN, [("simulation_0", 6), ("simulation_1", 2)])
    new = _layout(tree, "20261024_031003", [("simulation_0", 4), ("simulation_1", 2), ("simulation_2", 2)])
    ch = trade.layout_book_changes(old, new)
    assert ch["names_old"] == ["simulation_0", "simulation_1"]
    assert ch["names_new"] == ["simulation_0", "simulation_1", "simulation_2"]


def test_the_next_growth_keeps_class_0s_first_books_and_starts_the_rest(tree):
    old = _layout(tree, OLD_RUN, [("simulation_0", 4), ("simulation_1", 2), ("simulation_2", 2)])
    new = _layout(tree, "20261102_124000", [("simulation_0", 8), ("simulation_1", 4), ("simulation_2", 4)])
    books = trade.layout_book_changes(old, new)["books"]
    assert all(books[b] == "keep" for b in range(64)), "class 0's books keep their records"
    assert all(books[b] == "clear" for b in range(64, 128)), "books that were classes 1 and 2 become class 0"
    assert all(books[b] == "new" for b in range(128, 256)), "classes 1 and 2 on new books"


def test_any_change_of_a_classs_market_clears_its_records_the_price_process_included():
    cfg = SimpleNamespace(model_dump=lambda: {"fp_sigma": 0.01, "init_price": 300.0, "duration": 1, "logDir": "a"})
    same = SimpleNamespace(model_dump=lambda: {"fp_sigma": 0.01, "init_price": 300.0, "duration": 2, "logDir": "b"})
    vol = SimpleNamespace(model_dump=lambda: {"fp_sigma": 0.02, "init_price": 300.0, "duration": 1, "logDir": "a"})
    assert trade._market_digest(cfg) == trade._market_digest(same), "the run's own bookkeeping is not the market"
    assert trade._market_digest(cfg) != trade._market_digest(vol), "a volatility refinement is another market"


# ── the handover's start, weight and blend ───────────────────────────────────────────────────────────────────────

def _vectors():
    return {"simulation_0": {"making": {1: 0.6, 2: 0.4}, "skill": {1: 0.5, 2: 0.5}},
            "simulation_1": {"making": {3: 1.0}, "skill": {3: 0.7, 4: 0.3}}}


def test_the_boundary_carries_each_persisting_class_by_name_and_lists_the_new_ones():
    v = SimpleNamespace(_debeta_class_vectors=_vectors())
    ch = {"books": {0: "clear"}, "classes": [1], "names_old": ["simulation_0", "simulation_1"],
          "names_new": ["simulation_0", "simulation_1", "simulation_2"]}
    trade.begin_class_handover(v, ch, start_ts=100, span=_H)
    assert v._debeta_handover["start"] == 100 and v._debeta_handover["span"] == _H
    assert set(v._debeta_handover["carried"]) == {"simulation_0", "simulation_1"}
    assert v._debeta_handover["new"] == ["simulation_2"]


def _scorer(now, names, handover):
    return SimpleNamespace(simulation_timestamp=now, simulation=None, _debeta_handover=handover,
                           simulation_config={"class_names": names})


def test_the_carried_weight_falls_linearly_and_the_handover_ends_after_one_lookback():
    h = {"start": 0, "span": _H, "carried": _vectors(), "new": ["simulation_2"]}
    names = ["simulation_1", "simulation_0", "simulation_2"]  # another order: the carried shares follow the names
    at0 = reward.class_handover_now(_scorer(0, names, dict(h)))
    assert at0["w"] == 1.0 and set(at0["carried"]) == {0, 1} and at0["new"] == [2]
    assert at0["carried"][1]["making"] == {1: 0.6, 2: 0.4}, "simulation_0 is index 1 in the new layout"
    half = reward.class_handover_now(_scorer(_H // 2, names, dict(h)))
    assert abs(half["w"] - 0.5) < 1e-9
    s = _scorer(_H, names, dict(h))
    assert reward.class_handover_now(s) is None and not s._debeta_handover, "after one lookback it is gone"


def test_a_class_pays_its_carried_shares_at_the_boundary_and_its_own_after_the_lookback():
    uids = [1, 2, 3]
    fresh = {0: {1: {"making_raw": 0.0, "skill_raw": 0.0}, 2: {"making_raw": 1.0, "skill_raw": 0.0},
                 3: {"making_raw": 3.0, "skill_raw": 0.0}}}
    carried = {0: {"making": {1: 1.0}, "skill": {}}}
    for w, want in ((1.0, {1: 1.0, 2: 0.0, 3: 0.0}), (0.5, {1: 0.5, 2: 0.125, 3: 0.375}), (0.0, {1: 0.0, 2: 0.25, 3: 0.75})):
        out = {}
        making, _skill = reward.class_pool_shares(fresh, [1.0], uids, {0: 1}, handover={"w": w, "carried": carried, "new": []},
                                                  vectors_out=out)
        assert all(abs(making[u] - want[u]) < 1e-9 for u in uids), (w, making)
        assert out[0]["making"] == making, "the vectors kept for the next boundary are the ones paid"


def test_a_class_with_nothing_fresh_yet_pays_its_carried_shares_and_keeps_its_weight():
    uids = [1, 2]
    details = {0: {1: {"making_raw": 1.0}, 2: {"making_raw": 1.0}}, 1: {1: {"making_raw": 0.0}, 2: {"making_raw": 0.0}}}
    carried = {1: {"making": {2: 1.0}, "skill": {}}}
    making, _ = reward.class_pool_shares(details, [0.5, 0.5], uids, {0: 1, 1: 1},
                                         handover={"w": 0.9, "carried": carried, "new": []})
    assert abs(making[1] - 0.25) < 1e-9 and abs(making[2] - 0.75) < 1e-9, "class 1 keeps its half, paid as carried"


def test_a_new_class_ramps_its_weight_in_over_the_lookback():
    uids = [1, 2]
    details = {0: {1: {"making_raw": 1.0}, 2: {"making_raw": 0.0}}, 1: {1: {"making_raw": 0.0}, 2: {"making_raw": 1.0}}}
    for w, want2 in ((1.0, 0.0), (0.5, 0.5 * 0.5 / (0.5 + 0.25)), (0.0, 0.5)):
        making, _ = reward.class_pool_shares(details, [0.5, 0.5], uids, {0: 1, 1: 1},
                                             handover={"w": w, "carried": {}, "new": [1]})
        assert abs(making[2] - want2) < 1e-9, (w, making)


def test_with_no_handover_the_mixing_is_unchanged():
    uids = [1, 2]
    details = {0: {1: {"making_raw": 1.0}, 2: {"making_raw": 3.0}}}
    assert reward.class_pool_shares(details, [1.0], uids, {0: 1}) == \
        reward.class_pool_shares(details, [1.0], uids, {0: 1}, handover=None)


# ── wiring, persistence and the reset ────────────────────────────────────────────────────────────────────────────

def test_the_shift_starts_the_handover_whenever_the_layout_or_a_market_changed():
    import inspect
    src = inspect.getsource(trade.shift_simulation_histories)
    assert "begin_class_handover(self, book_changes, new_ts, lookback)" in src
    rsrc = inspect.getsource(reward.compute_debeta_scores)
    assert "_handover = class_handover_now(self)" in rsrc and "handover=_handover, vectors_out=_vectors" in rsrc


def test_a_restart_inside_the_lookback_resumes_the_handover():
    v = SimpleNamespace(_debeta_class_vectors=_vectors(),
                        _debeta_handover={"start": 5, "span": _H, "carried": _vectors(), "new": ["simulation_2"]})
    out = persistence.build_debeta_state(v)
    assert out["debeta_class_vectors"]["simulation_1"]["skill"] == {3: 0.7, 4: 0.3}
    loaded = persistence._load_class_vectors(out["debeta_handover"]["carried"])
    assert loaded == _vectors() and out["debeta_handover"]["new"] == ["simulation_2"]


def test_a_reset_miner_leaves_the_carried_shares():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "rc", pathlib.Path(__file__).with_name("test_a_reset_clears_every_per_uid_state.py"))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    v = rc._validator()
    v._debeta_class_vectors = _vectors()
    v._debeta_handover = {"start": 0, "span": _H, "carried": _vectors(), "new": []}
    trade.reset_agent_histories(v, 3, rc.BOOKS)
    assert 3 not in v._debeta_class_vectors["simulation_1"]["skill"]
    assert 3 not in v._debeta_handover["carried"]["simulation_1"]["making"]


# ── every per-book set follows the new layout ────────────────────────────────────────────────

def test_the_shift_rebuilds_the_legacy_factor_sets_and_republishes_the_initial_balances():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("tss", pathlib.Path(__file__).with_name("test_scoring_shadow.py"))
    tss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tss)
    main = tss._main_with_history() if hasattr(tss, "_main_with_history") else None
    if main is None:
        import test_shadow_snapshot_completeness as tsc
        main = tsc._debeta_main()
    if not hasattr(main.config.scoring, "debeta"):
        main.config.scoring.debeta = SimpleNamespace(making_fee_weight=0.0, centered_window=15)
    uids = tss._UIDS
    main.activity_factors = {u: {0: 0.7, 1: 0.2} for u in range(uids)}
    main.pnl_factors = {u: {0: 1.3, 1: 0.9} for u in range(uids)}
    main.kappa_values = {u: {"books": {0: 0.5, 1: 0.1}, "books_weighted": {0: 0.5, 1: 0.1}} for u in range(uids)}
    main.initial_balances_published = {u: True for u in range(uids)}
    trade.shift_simulation_histories(
        main, old_ts=6 * tss._S, new_ts=0, book_count=3, volume_decimals=4, lookback=10800 * tss._S,
        volume_assessment_period=3600 * tss._S, miner_wealth=50000.0, effective_max_uids=uids, book_ids=[0, 2, 3],
        book_changes={"books": {0: "keep", 1: "drop", 2: "new", 3: "new"}, "classes": [], "names_old": [], "names_new": []})
    assert main.activity_factors[0] == {0: 0.7, 2: 0.0, 3: 0.0}, "kept book keeps its factor, new books start at 0"
    assert main.pnl_factors[0] == {0: 1.3, 2: 1.0, 3: 1.0}
    assert main.kappa_values[0]["books"] == {0: 0.5, 2: None, 3: None}
    assert not any(main.initial_balances_published.values()), "the new run's initial balances are published again"


def test_the_run_change_frame_carries_the_new_layout_to_the_scoring_child():
    import inspect
    from taos.im.validator import scoring_shadow
    from taos.im.validator.engines import simulation as sim_engine
    assert '_book_changes["layout"] = {' in inspect.getsource(sim_engine)
    src = inspect.getsource(scoring_shadow)
    assert 'shadow.simulation_config.update(' in src and '"class_of_book", "class_names"' in src



def test_each_scoring_inside_the_handover_logs_its_weight_and_the_class_weights():
    """Each scoring inside the handover logs the blend weight and each class's weight, a new class's ramp included,
    so they are read from the validator's log rather than recomputed."""
    h = {"w": 0.25, "carried": {0: {}, 1: {}}, "new": [2]}
    line = reward.handover_log_line(h, [0.5, 0.25, 0.25], ["simulation_0", "simulation_1", "simulation_2"])
    assert line == ("[DEBETA-HANDOVER] w=0.2500 weights=simulation_0:0.5000,simulation_1:0.2500,simulation_2:0.1875 "
                    "new=simulation_2")
    plain = reward.handover_log_line({"w": 1.0, "carried": {0: {}}, "new": []}, [1.0], ["simulation_0"])
    assert plain.endswith("weights=simulation_0:1.0000 new=-")
    import inspect
    src = inspect.getsource(reward.compute_debeta_scores)
    assert "bt.logging.info(handover_log_line(_handover, _cw, _names))" in src


def test_the_handover_logs_once_when_it_runs_out(monkeypatch):
    lines = []
    monkeypatch.setattr(reward.bt.logging, "info", lambda msg, *a, **k: lines.append(str(msg)))
    h = {"start": 0, "span": _H, "carried": _vectors(), "new": ["simulation_2"]}
    s = _scorer(_H, ["simulation_0", "simulation_1", "simulation_2"], dict(h))
    assert reward.class_handover_now(s) is None and reward.class_handover_now(s) is None
    assert [x for x in lines if x.startswith("[DEBETA-HANDOVER] done")] == [
        "[DEBETA-HANDOVER] done: the carried shares and the new classes' ramp ran out"], lines
