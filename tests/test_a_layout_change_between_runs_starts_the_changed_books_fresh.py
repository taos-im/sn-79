"""A layout change between runs is handled per book.

A book's previous-run histories are kept only where its market is unchanged: a book that changed market must not
carry the old market's records into its new class's skill pool, and a layout on other book ids than the loaded ones
is adopted rather than refused, so a book count that grows or shrinks between runs is scored on the new layout.

The two runs' layouts are compared class by class at the open: a class whose market is unchanged keeps its books'
histories in order, its k-th book taking the old k-th's wherever the ids moved, and its added books start empty; a
class whose market changed starts fresh and restarts its class pool; a class new to the layout starts empty and ramps
in; an old book whose records no new book takes is dropped from every structure, its sums with it. The map travels to
the shadow scoring service in the same run-change frame, so both sides make the identical transition, and a layout on
other books is adopted, not refused.
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_scoring_shadow as tss  # noqa: E402
import test_shadow_snapshot_completeness as tsc  # noqa: E402
from test_the_validator_scores_the_layout_of_the_run_it_is_on import (  # noqa: E402
    NEW_RUN, OLD_RUN, _composition, _run_dir, tree as _tree_fixture,
)

tree = _tree_fixture  # that module's fixture, collected here under its own name

import taos.im.validator.engines.simulation as sim_engine  # noqa: E402
import taos.im.validator.scoring_shadow as shadow_mod  # noqa: E402
import taos.im.validator.trade as trade  # noqa: E402

_S = tss._S


def _layout(tree, name, comp):
    d = _run_dir(tree.root, name, comp, tree.config_dir)
    found = tree.engine._run_layout(d / "simulation_0-0")
    assert found is not None
    return found[1]


# ── the change map between two layouts ───────────────────────────────────────────────────────────────────────────

def test_a_class_whose_ids_shift_takes_its_records_and_a_new_class_starts_empty(tree):
    old = _layout(tree, OLD_RUN, [("simulation_0", 6), ("simulation_1", 2)])
    new = _layout(tree, NEW_RUN, [("simulation_0", 4), ("simulation_1", 2), ("simulation_2", 2)])
    ch = trade.layout_book_changes(old, new)
    books = ch["books"]
    assert all(books[b] == "keep" for b in range(64))
    assert all(books[b] == "move" for b in range(64, 96)), "simulation_1 sits on 64-95 and its records follow it"
    assert all(ch["dest"][96 + k] == 64 + k for k in range(32))
    assert all(ch["dest"][b] is None for b in range(64, 96)), "simulation_0's dropped realizations go nowhere"
    assert all(books[b] == "new" for b in range(96, 128)), "simulation_2 is new to the layout"
    assert set(books) == set(range(128))
    assert ch["classes"] == [], "no class's market changed: simulation_2 is new and ramps in, the others keep their pools"


def test_a_count_decrease_drops_the_vanished_books(tree):
    old = _layout(tree, OLD_RUN, [("simulation_0", 6), ("simulation_1", 2)])
    new = _layout(tree, NEW_RUN, [("simulation_0", 4), ("simulation_1", 2)])
    ch = trade.layout_book_changes(old, new)
    assert all(ch["books"][b] == "keep" for b in range(64))
    assert all(ch["books"][b] == "move" for b in range(64, 96)), "simulation_1 moves down onto 64-95"
    assert all(ch["books"][b] == "drop" for b in range(96, 128))
    assert all(ch["dest"][b] is None for b in range(64, 96)), "simulation_0's vanished realizations are dropped"
    assert ch["classes"] == []


def test_a_count_increase_adds_fresh_books(tree):
    old = _layout(tree, OLD_RUN, [("simulation_0", 4), ("simulation_1", 2)])
    new = _layout(tree, NEW_RUN, [("simulation_0", 4), ("simulation_1", 2), ("simulation_2", 2)])
    ch = trade.layout_book_changes(old, new)
    assert all(ch["books"][b] == "keep" for b in range(96))
    assert all(ch["books"][b] == "new" for b in range(96, 128))
    assert ch["classes"] == [], "a class new to the layout has no pool to restart"


def test_the_same_layout_changes_nothing(tree):
    old = _layout(tree, OLD_RUN, [("simulation_0", 6), ("simulation_1", 2)])
    new = _layout(tree, NEW_RUN, [("simulation_0", 6), ("simulation_1", 2)])
    ch = trade.layout_book_changes(old, new)
    assert set(ch["books"].values()) == {"keep"} and ch["classes"] == []
    assert trade.layout_book_changes(None, new) == {"books": {}, "classes": []}, "no old layout: nothing to compare"


# ── the shift on the histories ───────────────────────────────────────────────────────────────────────────────────

def _books_in(main):
    """The books that carry history entries (a book in the layout keeps an empty shell after the rebuild)."""
    seen = set()
    for attr in ("debeta_capbuy_hist", "debeta_capsell_hist", "debeta_mtm_hist", "debeta_invsum_hist",
                 "debeta_heldn_hist", "debeta_notional_hist"):
        for books in (getattr(main, attr, None) or {}).values():
            seen |= {int(b) for b, tsd in books.items() if tsd}
    for attr in ("debeta_invn_hist", "debeta_drift_hist"):
        seen |= {int(b) for b, tsd in (getattr(main, attr, None) or {}).items() if tsd}
    for books in main.trade_volumes.values():
        seen |= {int(b) for b, roles in books.items() if any(roles.get(r) for r in ("total", "maker", "taker", "self"))}
    for books in main.roundtrip_volumes.values():
        seen |= {int(b) for b, tsd in books.items() if tsd}
    for ts_books in main.realized_pnl_history.values():
        for books in ts_books.values():
            seen |= {int(b) for b in books}
    return seen


def _shift(main, changes, book_ids):
    # the shift's capture flush reads the de-beta dials; the fixture's config carries none (production configs do)
    if not hasattr(main.config.scoring, "debeta"):
        main.config.scoring.debeta = SimpleNamespace(making_fee_weight=0.0, centered_window=15)
    trade.shift_simulation_histories(
        main, old_ts=6 * _S, new_ts=0, book_count=len(book_ids), volume_decimals=4, lookback=10800 * _S,
        volume_assessment_period=3600 * _S, miner_wealth=50000.0, effective_max_uids=tss._UIDS,
        book_ids=book_ids, book_changes=changes)


def test_a_cleared_book_starts_fresh_and_a_kept_one_is_shifted():
    main = tsc._debeta_main()
    assert _books_in(main) >= {0, 1}, "the fixture trades books 0 and 1"
    kept_before = {u: dict(b.get(0, {})) for u, b in main.debeta_mtm_hist.items()}
    _shift(main, {"books": {0: "keep", 1: "clear"}, "classes": []}, [0, 1])
    assert 1 not in _books_in(main), "book 1's histories are gone from every structure"
    for u, before in kept_before.items():
        after = main.debeta_mtm_hist.get(u, {}).get(0, {})
        assert len(after) == len(before), "book 0's entries are shifted, not dropped"
    for sums in (main.volume_sums, main.maker_volume_sums, main.taker_volume_sums, main.self_volume_sums,
                 main.roundtrip_volume_sums, main.capture_buy_sums, main.capture_sell_sums, main.debeta_mtm):
        for books in sums.values():
            assert float(books.get(1, 0.0) or 0.0) == 0.0, "a cleared book's running sums are zero"
    assert 1 in main.recent_trades, "book 1 still exists in the new layout, as an empty shell"


def test_a_dropped_book_leaves_every_structure():
    main = tsc._debeta_main()
    _shift(main, {"books": {0: "keep", 1: "drop"}, "classes": []}, [0])
    assert _books_in(main) == {0}
    for u in main.open_positions:
        assert set(main.open_positions[u]) == {0}
    assert set(main.recent_trades) == {0}


def test_a_changed_class_restarts_its_pool_and_an_unchanged_one_keeps_it():
    main = tsc._debeta_main()
    main.debeta_cpc_hist = {0: {7: {8: {5 * _S: 1.0}}}, 1: {7: {9: {5 * _S: 2.0}}}}
    main.debeta_cpc = {0: {7: {8: 1.0}}, 1: {7: {9: 2.0}}}
    _shift(main, {"books": {0: "keep", 1: "clear"}, "classes": [1]}, [0, 1])
    assert main.debeta_cpc_hist[0] and main.debeta_cpc[0] == {7: {8: 1.0}}
    assert not main.debeta_cpc_hist.get(1) and not main.debeta_cpc.get(1)


def test_no_change_map_shifts_every_book_as_a_keep_map_does():
    a, b = tsc._debeta_main(), tsc._debeta_main()
    _shift(a, None, [0, 1])
    _shift(b, {"books": {0: "keep", 1: "keep"}, "classes": []}, [0, 1])
    assert _books_in(a) == _books_in(b) == {0, 1}


# ── the frame to the shadow and the engine's open ────────────────────────────────────────────────────────────────

def test_the_run_change_frame_carries_the_change_map(monkeypatch):
    sent = []
    monkeypatch.setattr(shadow_mod, "_send_frame", lambda sock, frame: sent.append(frame))
    sh = shadow_mod.ScoringShadow.__new__(shadow_mod.ScoringShadow)
    import threading
    sh._submit_lock = threading.Lock()
    sh._teed_recent = set()
    sh._init_floor_ts = 1
    sh._sock = object()
    sh._executor = SimpleNamespace(submit=lambda fn: fn())
    sh.is_alive = lambda: True
    changes = {"books": {0: "keep", 1: "clear"}, "classes": [1]}
    sh.emit_sim_start(5 * _S, 0, changes)
    assert sent == [("sim_start", (5 * _S, 0, changes))]
    sh.emit_sim_start(5 * _S, 0)
    assert sent[-1] == ("sim_start", (5 * _S, 0, None))


def test_a_layout_on_other_books_is_adopted_not_refused(tree):
    _run_dir(tree.root, OLD_RUN, [("simulation_0", 6), ("simulation_1", 2)], tree.config_dir)
    assert tree.engine._follow_resumed_run_layout() is True
    assert tree.engine._book_ids == list(range(128))
    new = _run_dir(tree.root, NEW_RUN, [("simulation_0", 4), ("simulation_1", 2)], tree.config_dir)
    before = tree.v.simulation.label()
    tree.v._scoring_shadow = SimpleNamespace(request_reinit=lambda: None)
    tree.engine._load_config()
    assert tree.engine._adopt_run_layout(new / "simulation_0-0", "the run opened", previous_label=before) is True
    assert _composition(tree.v.simulation) == [("simulation_0", 4), ("simulation_1", 2)]
    assert tree.engine._book_ids == list(range(96)), "the book ids follow the adopted layout"


def test_the_open_computes_the_changes_and_passes_them_on(tree, monkeypatch):
    _run_dir(tree.root, OLD_RUN, [("simulation_0", 6), ("simulation_1", 2)], tree.config_dir)
    assert tree.engine._follow_resumed_run_layout() is True
    new = _run_dir(tree.root, NEW_RUN, [("simulation_0", 4), ("simulation_1", 2), ("simulation_2", 2)],
                   tree.config_dir)
    v = tree.v
    v.config.scoring = SimpleNamespace(kappa=SimpleNamespace(lookback=10800 * _S),
                                       activity=SimpleNamespace(trade_volume_assessment_period=3600 * _S))
    v.effective_max_uids = 4
    frames, shifts = [], []

    class _Stop(Exception):
        """The open's later steps (clocks, step rates, the realizations' dirs) are not this test's."""

    def _record(*a, **k):
        shifts.append(k)
        raise _Stop()
    v._scoring_shadow = SimpleNamespace(request_reinit=lambda: None,
                                        emit_sim_start=lambda old, new, changes=None: frames.append((old, new, changes)))
    monkeypatch.setattr(trade, "shift_simulation_histories", _record)
    monkeypatch.setattr(sim_engine, "_reward_lock_held", lambda v: __import__("contextlib").nullcontext())
    with pytest.raises(_Stop):
        tree.engine._open_run(7 * _S, 0, str(new / "simulation_0-0"), reason="the test")
    assert len(shifts) == 1 and len(frames) == 1
    changes = shifts[0]["book_changes"]
    assert changes["books"][0] == "keep" and changes["books"][64] == "move" and changes["books"][127] == "new"
    assert changes["dest"][96] == 64 and changes["dest"][64] is None, "simulation_1's records follow it"
    assert frames[0][2] == changes, "the shadow gets the same map in its run-change frame"
    assert shifts[0]["book_ids"] == list(range(128))
