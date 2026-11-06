# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A class keeps its records when the book count grows and its book ids shift.

The published default runs 128/64/64 books (eight, four and four realizations of sixteen); the 64/32/32 layout is kept
under its own name. Book ids stay contiguous per class, so from 64/32/32 to 128/64/64 simulation_1 moves from books
64-95 to 128-191 and simulation_2 from 96-127 to 192-255.

The boundary compares class by class: a class whose market is unchanged keeps its records in order, its k-th book
taking the old layout's k-th book of that class wherever the ids went, and its added books start empty, in every
book-keyed structure of main and of the scoring service alike. A validator down across the boundary loads the old
run's records onto the new layout; the state says which layout they were scored on (a state that does not: the
config.xml of the run it was saved in), and the open moves them by that.
"""
import copy
import os
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_scoring_shadow as tss  # noqa: E402
import test_shadow_snapshot_completeness as tsc  # noqa: E402
from test_the_validator_scores_the_layout_of_the_run_it_is_on import (  # noqa: E402
    CONFIGS, NEEDED, NEW_RUN, OLD_RUN, _composition, _run_dir, tree as _tree_fixture,
)

import taos.im.validator.engines.simulation as sim_engine  # noqa: E402
import taos.im.validator.persistence as persistence  # noqa: E402
import taos.im.validator.scoring_shadow as shadow_mod  # noqa: E402
import taos.im.validator.trade as trade  # noqa: E402
from taos.im.protocol.config import MultiAssetSimulationConfig  # noqa: E402
from taos.im.validator.book_layout import (  # noqa: E402
    BOOK_KEYED, CLASS_KEYED, NOT_BOOK_KEYED, book_changes, layout_of, rekey_books,
)

tree = _tree_fixture
pytestmark = pytest.mark.skipif(not all((CONFIGS / n).exists() for n in [*NEEDED, "multiasset_simulation_0.xml"]),
                                reason="the class files are not here")

_S = tss._S
OLD = [("simulation_0", 4), ("simulation_1", 2), ("simulation_2", 2)]
NEW = [("simulation_0", 8), ("simulation_1", 4), ("simulation_2", 4)]


def _parse(name):
    import xml.etree.ElementTree as ET
    return MultiAssetSimulationConfig.from_multiasset_xml(ET.parse(CONFIGS / name).getroot(), CONFIGS)


def _ranges(sim):
    return [(c.name, c.books[0], c.books[-1]) for c in sim.asset_classes()]


# ── the published layouts ─────────────────────────────────────────────────────────────────────────────────────────

def test_the_published_default_is_128_64_64_and_the_old_layout_is_kept_by_name():
    new, old = _parse("multiasset_simulation_0.xml"), _parse("multiasset_simulation_0_64_32_32.xml")
    assert _composition(new) == NEW and len(new.book_ids) == 256
    assert _ranges(new) == [("simulation_0", 0, 127), ("simulation_1", 128, 191), ("simulation_2", 192, 255)]
    assert _composition(old) == OLD and len(old.book_ids) == 128
    assert _ranges(old) == [("simulation_0", 0, 63), ("simulation_1", 64, 95), ("simulation_2", 96, 127)]
    assert old.label() != new.label(), "the layouts are told apart by their labels"


def test_the_change_map_from_64_32_32_to_128_64_64():
    ch = trade.layout_book_changes(_parse("multiasset_simulation_0_64_32_32.xml"), _parse("multiasset_simulation_0.xml"))
    books, dest = ch["books"], ch["dest"]
    assert {b: books[b] for b in range(64)} == {b: "keep" for b in range(64)}
    assert all(dest[b] == b for b in range(64)), "simulation_0's books keep their ids"
    assert all(dest[64 + k] == 128 + k for k in range(32)), "simulation_1's records go to 128-159"
    assert all(dest[96 + k] == 192 + k for k in range(32)), "simulation_2's records go to 192-223"
    assert set(dest) == set(range(128)), "every old book has a destination"
    assert all(books[b] == "move" for b in [*range(128, 160), *range(192, 224)])
    assert all(books[b] == "new" for b in [*range(64, 128), *range(160, 192), *range(224, 256)]), "added books"
    assert ch["classes"] == [], "a class that only gains books keeps its pool"
    assert ch["class_sources"] == {0: 0, 1: 1, 2: 2}


def test_a_class_whose_market_changes_while_it_grows_starts_fresh():
    old = layout_of(_parse("multiasset_simulation_0_64_32_32.xml"))
    new = copy.deepcopy(layout_of(_parse("multiasset_simulation_0.xml")))
    new["classes"][1]["market"] = "another market"
    ch = book_changes(old, new)
    assert all(ch["books"][b] == "clear" for b in range(128, 192)), "no record is scored in another market's pool"
    assert all(ch["dest"][b] is None for b in range(64, 96))
    assert all(ch["dest"][96 + k] == 192 + k for k in range(32)), "the other classes still move"
    assert ch["classes"] == [1]


def test_the_same_layout_and_no_old_layout_change_nothing():
    lay = layout_of(_parse("multiasset_simulation_0.xml"))
    ch = book_changes(lay, copy.deepcopy(lay))
    assert set(ch["books"].values()) == {"keep"} and ch["classes"] == []
    assert book_changes(None, lay) == {"books": {}, "classes": []}


# ── every book-keyed structure follows the map ───────────────────────────────────────────────────────────────────

def test_every_scoring_structure_is_classified_as_book_keyed_or_not():
    known = {n for names in BOOK_KEYED.values() for n in names} | set(CLASS_KEYED) | set(NOT_BOOK_KEYED)
    every = set(shadow_mod._STRUCT_NAMES) | set(persistence._DEBETA_HIST3) | set(persistence._DEBETA_HIST2)
    assert every - known == set(), (
        "a structure the scoring state holds is not in book_layout's BOOK_KEYED or NOT_BOOK_KEYED: a layout change "
        "would leave its records on the old book ids")


def _at(holder, name, kind, book):
    d = getattr(holder, name, None)
    if not isinstance(d, dict):
        return None
    if kind == "book":
        return d.get(book)
    if kind == "uid":
        return {u: x.get(book) for u, x in d.items() if isinstance(x, dict) and book in x}
    if kind == "uid_ts":
        return {u: {ts: bk[book] for ts, bk in x.items() if book in bk} for u, x in d.items() if isinstance(x, dict)}
    return {u: {k: (kv.get(k) or {}).get(book) for k in ("books", "books_weighted")}
            for u, kv in d.items() if isinstance(kv, dict)}


def _boundary(main, book_ids, changes):
    if not hasattr(main.config.scoring, "debeta"):
        main.config.scoring.debeta = SimpleNamespace(making_fee_weight=0.0, centered_window=15)
    trade.shift_simulation_histories(
        main, old_ts=6 * _S, new_ts=0, book_count=len(book_ids), volume_decimals=4, lookback=10800 * _S,
        volume_assessment_period=3600 * _S, miner_wealth=50000.0, effective_max_uids=tss._UIDS,
        book_ids=book_ids, book_changes=changes)


# the fixture trades two books; as two classes of one book each that grow to two books each, class B moves from 1 to 2
_TWO = {"classes": [{"name": "A", "books": [0], "market": "a"}, {"name": "B", "books": [1], "market": "b"}]}
_FOUR = {"classes": [{"name": "A", "books": [0, 1], "market": "a"}, {"name": "B", "books": [2, 3], "market": "b"}]}


def test_a_moved_book_holds_what_it_held_before_and_an_added_book_what_any_added_book_holds():
    ref, moved = tsc._debeta_main(), tsc._debeta_main()
    _boundary(ref, [0, 1], None)
    ch = book_changes(_TWO, _FOUR)
    assert ch["dest"] == {0: 0, 1: 2}
    _boundary(moved, [0, 1, 2, 3], ch)
    checked = 0
    for kind, names in BOOK_KEYED.items():
        for name in names:
            if getattr(ref, name, None) is None:
                continue
            assert _at(moved, name, kind, 2) == _at(ref, name, kind, 1), f"{name}: class B's records on its book"
            assert _at(moved, name, kind, 0) == _at(ref, name, kind, 0), f"{name}: class A's first book unchanged"
            assert _at(moved, name, kind, 1) == _at(moved, name, kind, 3), f"{name}: an added book starts as any"
            checked += 1
    assert checked > 50, "the fixture holds the scoring state's structures"


def test_the_running_sums_stay_the_sums_of_their_moved_histories():
    moved = tsc._debeta_main()
    _boundary(moved, [0, 1, 2, 3], book_changes(_TWO, _FOUR))
    for uid, books in moved.debeta_capbuy_hist.items():
        for b, hist in books.items():
            assert sum(hist.values()) == pytest.approx(moved.capture_buy_sums.get(uid, {}).get(b, 0.0))
    for uid, books in moved.trade_volumes.items():
        for b, roles in books.items():
            assert sum(roles.get("total", {}).values()) == pytest.approx(moved.volume_sums.get(uid, {}).get(b, 0.0))


def test_main_and_the_scoring_service_make_the_same_move():
    main = tss._fresh_main()
    for st in tss._ROUNDS:
        tss.update_trade_volumes(main, st)
    shadow = tss._snapshot_to_shadow(main)
    ch = book_changes(_TWO, _FOUR)
    for s in (main, shadow):
        trade.shift_simulation_histories(
            s, old_ts=4 * _S, new_ts=0, book_count=4, volume_decimals=4, lookback=10800 * _S,
            volume_assessment_period=3600 * _S, miner_wealth=50000.0, effective_max_uids=tss._UIDS,
            book_ids=[0, 1, 2, 3], book_changes=ch)
    assert tss.compute_parity_digest(shadow) == tss.compute_parity_digest(main)


def test_a_strip_moves_nothing_else():
    main = tsc._debeta_main()
    before = copy.deepcopy(main.volume_sums)
    rekey_books(main, {1: None}, keep_unmapped=True)
    assert all(1 not in books for books in main.volume_sums.values())
    assert {u: b.get(0) for u, b in main.volume_sums.items()} == {u: b.get(0) for u, b in before.items()}


# ── the open, after a restart in the old run and after a validator was down across the boundary ──────────────────

def _start_on_the_default(tree):
    shutil.copy(CONFIGS / "multiasset_simulation_0.xml", tree.config_dir / "multiasset_simulation_0.xml")
    tree.v.config.simulation.xml_config = str(tree.config_dir / "multiasset_simulation_0.xml")
    tree.engine._load_config()
    assert _composition(tree.v.simulation) == NEW


class _Stop(Exception):
    """The open's later steps (clocks, step rates, the realizations' dirs) are not these tests'."""


def _open(tree, monkeypatch, run):
    v = tree.v
    v.config.scoring = SimpleNamespace(kappa=SimpleNamespace(lookback=10800 * _S),
                                       activity=SimpleNamespace(trade_volume_assessment_period=3600 * _S))
    v.effective_max_uids = 4
    shifts = []

    def _record(*a, **k):
        shifts.append(k)
        raise _Stop()
    v._scoring_shadow = SimpleNamespace(request_reinit=lambda: None, emit_sim_start=lambda *a: None)
    monkeypatch.setattr(trade, "shift_simulation_histories", _record)
    monkeypatch.setattr(sim_engine, "_reward_lock_held", lambda v: __import__("contextlib").nullcontext())
    with pytest.raises(_Stop):
        tree.engine._open_run(7 * _S, 0, str(run / "simulation_0-0"), reason="the test")
    return shifts[0].get("book_changes")


def test_a_validator_restarted_in_the_old_run_on_the_new_default_moves_the_records_at_the_boundary(tree, monkeypatch):
    _start_on_the_default(tree)
    _run_dir(tree.root, OLD_RUN, OLD, tree.config_dir)
    assert tree.engine._follow_resumed_run_layout() is True, "the old run is scored on its own 128 books"
    assert tree.engine._book_ids == list(range(128))
    tree.engine._settle_scored_layout()
    assert tree.v._scored_layout == layout_of(tree.v.simulation)
    new = _run_dir(tree.root, NEW_RUN, NEW, tree.config_dir)
    ch = _open(tree, monkeypatch, new)
    assert ch is not None and all(ch["dest"][64 + k] == 128 + k for k in range(32))
    assert tree.engine._book_ids == list(range(256))


def test_a_validator_down_across_the_boundary_puts_each_classs_records_on_its_books(tree, monkeypatch):
    _run_dir(tree.root, OLD_RUN, OLD, tree.config_dir)
    new = _run_dir(tree.root, NEW_RUN, NEW, tree.config_dir)
    _start_on_the_default(tree)
    assert tree.engine._follow_resumed_run_layout() is False, "the resumed run is the new one, not the state's"
    assert _composition(tree.v.simulation) == NEW
    tree.engine._settle_scored_layout()
    assert [c["books"][0] for c in tree.v._scored_layout["classes"]] == [0, 64, 96], (
        "a state saved without a layout takes it from the run it was saved in")
    ch = _open(tree, monkeypatch, new)
    assert ch is not None, "without the saved layout this open would compare 128/64/64 with itself and move nothing"
    assert all(ch["dest"][96 + k] == 192 + k for k in range(32))


def test_a_saved_layout_is_the_one_the_open_compares(tree, monkeypatch):
    new = _run_dir(tree.root, NEW_RUN, NEW, tree.config_dir)
    _start_on_the_default(tree)
    tree.v._scored_layout = layout_of(_parse("multiasset_simulation_0_64_32_32.xml"))
    tree.engine._settle_scored_layout()
    assert [c["books"][0] for c in tree.v._scored_layout["classes"]] == [0, 64, 96], "a saved layout is kept"
    ch = _open(tree, monkeypatch, new)
    assert all(ch["dest"][64 + k] == 128 + k for k in range(32))


def test_the_saved_state_names_the_layout_its_records_are_on():
    lay = layout_of(_parse("multiasset_simulation_0_64_32_32.xml"))
    v = SimpleNamespace(_scored_layout=lay, simulation=_parse("multiasset_simulation_0.xml"))
    assert persistence.scored_layout(v) == lay, "the records' layout, not the loaded config's"
    v._scored_layout = None
    assert persistence.scored_layout(v) == layout_of(v.simulation)


# ── which classes restart, and the handover a count change runs ──────────────────────────────────────────────────

def _vectors(names):
    return {n: {"making": {1: 0.5, 2: 0.5}, "skill": {1: 1.0}} for n in names}


def test_a_count_change_restarts_no_class_and_runs_no_handover():
    ch = trade.layout_book_changes(_parse("multiasset_simulation_0_64_32_32.xml"), _parse("multiasset_simulation_0.xml"))
    assert ch["classes"] == []
    holder = SimpleNamespace(_debeta_class_vectors=_vectors(["simulation_0", "simulation_1", "simulation_2"]))
    trade.begin_class_handover(holder, ch, 0, 3600 * _S)
    assert holder._debeta_handover == {}, "a change that only moves and adds books hands nothing over"


def test_a_count_change_with_one_market_changed_restarts_and_carries_that_class_alone():
    old = layout_of(_parse("multiasset_simulation_0_64_32_32.xml"))
    new = copy.deepcopy(layout_of(_parse("multiasset_simulation_0.xml")))
    new["classes"][2]["market"] = "simulation_2 with its sigma changed"
    ch = book_changes(old, new)
    assert ch["classes"] == [2]
    assert all(ch["books"][b] == "clear" for b in range(192, 256))
    assert all(ch["dest"][64 + k] == 128 + k for k in range(32)), "simulation_1 still moves"
    holder = SimpleNamespace(_debeta_class_vectors=_vectors(["simulation_0", "simulation_1", "simulation_2"]))
    trade.begin_class_handover(holder, ch, 0, 3600 * _S)
    assert set(holder._debeta_handover["carried"]) == {"simulation_2"} and holder._debeta_handover["new"] == []


# ── a fourth class: a 64/64/64/64 layout ─────────────────────────────────────────────────────────────────────────

def _four_class_tree(tmp_path, counts):
    """A config directory with the three shipped classes, a simulation_3 on simulation_2's file with its own market,
    and a wrapper of the given instance counts. Returns the parsed layout."""
    import re
    import xml.etree.ElementTree as ET
    d = tmp_path / "config"
    d.mkdir(parents=True, exist_ok=True)
    for n in ("simulation_0", "simulation_1", "simulation_2"):
        shutil.copy(CONFIGS / f"{n}.xml", d / f"{n}.xml")
    text = (CONFIGS / "simulation_2.xml").read_text()
    text, n = re.subn(r'(<FundamentalPrice name="fundamental"[^>]*?sigma=")0\.0125(")', r"\g<1>0.0150\2", text, count=1)
    assert n == 1, "simulation_2.xml's fundamental sigma moved; update the fourth class's edit"
    (d / "simulation_3.xml").write_text(text)
    names = ["simulation_0", "simulation_1", "simulation_2", "simulation_3"][:len(counts)]
    bgs = "\n".join(f'  <Background name="{n}" path="{n}.xml" instanceCount="{c}"/>' for n, c in zip(names, counts))
    wrapper = (CONFIGS / "multiasset_simulation_0.xml").read_text()
    wrapper = re.sub(r"(?s)\s*<Background .*?/>\s*(?=</MultiAssetSimulation>)", "\n" + bgs + "\n", wrapper)
    (d / "wrapper.xml").write_text(wrapper)
    return MultiAssetSimulationConfig.from_multiasset_xml(ET.parse(d / "wrapper.xml").getroot(), d)


def test_the_four_class_layout_loads_with_contiguous_classes_and_equal_book_weights(tmp_path):
    from taos.im.validator.reward import class_weights_for_layout
    sim = _four_class_tree(tmp_path, [4, 4, 4, 4])
    assert _ranges(sim) == [("simulation_0", 0, 63), ("simulation_1", 64, 127), ("simulation_2", 128, 191),
                            ("simulation_3", 192, 255)]
    cmap = {b: i for i, c in enumerate(sim.asset_classes()) for b in c.books}
    assert class_weights_for_layout("books", cmap) == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_from_64_32_32_to_64_64_64_64_each_class_keeps_its_records_and_the_fourth_ramps_in(tmp_path):
    old, new = _four_class_tree(tmp_path / "a", [4, 2, 2]), _four_class_tree(tmp_path / "b", [4, 4, 4, 4])
    ch = trade.layout_book_changes(old, new)
    books, dest = ch["books"], ch["dest"]
    assert all(books[b] == "keep" for b in range(96)), "simulation_0 and simulation_1's first books stay where they are"
    assert all(books[b] == "new" for b in range(96, 128)), "simulation_1's added books"
    assert all(dest[96 + k] == 128 + k for k in range(32)), "simulation_2's records move to 128-159"
    assert all(books[b] == "new" for b in [*range(160, 192), *range(192, 256)])
    assert ch["classes"] == [] and ch["class_sources"] == {0: 0, 1: 1, 2: 2}
    assert ch["names_new"][3] == "simulation_3" and "simulation_3" not in ch["names_old"]
    holder = SimpleNamespace(_debeta_class_vectors=_vectors(ch["names_old"]))
    trade.begin_class_handover(holder, ch, 0, 3600 * _S)
    assert holder._debeta_handover["carried"] == {} and holder._debeta_handover["new"] == ["simulation_3"], (
        "the new class ramps its weight in; no other class hands anything over")


def test_from_128_64_64_to_64_64_64_64_the_shrunk_class_drops_its_tail_and_the_others_move_down(tmp_path):
    old, new = _four_class_tree(tmp_path / "a", [8, 4, 4]), _four_class_tree(tmp_path / "b", [4, 4, 4, 4])
    ch = trade.layout_book_changes(old, new)
    dest = ch["dest"]
    assert all(dest[b] == b for b in range(64))
    assert all(dest[b] is None for b in range(64, 128)), "simulation_0's last four realizations are dropped"
    assert all(dest[128 + k] == 64 + k for k in range(64)), "simulation_1 moves down to 64-127"
    assert all(dest[192 + k] == 128 + k for k in range(64)), "simulation_2 moves down to 128-191"
    assert all(ch["books"][b] == "new" for b in range(192, 256)) and ch["classes"] == []


def test_a_moved_grown_and_new_class_layout_keeps_every_structure_consistent():
    # two one-book classes growing to four books: A gains book 1, B moves from 1 to 2, a new class C sits on 3
    two = {"classes": [{"name": "A", "books": [0], "market": "a"}, {"name": "B", "books": [1], "market": "b"}]}
    four = {"classes": [{"name": "A", "books": [0, 1], "market": "a"}, {"name": "B", "books": [2], "market": "b"},
                        {"name": "C", "books": [3], "market": "c"}]}
    ref, moved = tsc._debeta_main(), tsc._debeta_main()
    _boundary(ref, [0, 1], None)
    _boundary(moved, [0, 1, 2, 3], book_changes(two, four))
    for kind, names in BOOK_KEYED.items():
        for name in names:
            if getattr(ref, name, None) is None:
                continue
            assert _at(moved, name, kind, 2) == _at(ref, name, kind, 1), name
            assert _at(moved, name, kind, 1) == _at(moved, name, kind, 3), name
