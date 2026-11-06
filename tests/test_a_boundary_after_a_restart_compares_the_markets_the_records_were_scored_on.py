# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A run boundary after a restart compares the markets the records were scored on, not the files the validator loaded.

Each class's market is a digest of its whole class file, and the validator saves the layout its records are on,
digests included (persistence.scored_layout), and takes it as the old side of the comparison at every run open. Here a
run's records were scored on frozen class files (simulation_2 at sigma 0.0200), the validator restarts on the start
config's files (sigma 0.0125) without that run's directory to follow, and the next run opens on the start config's
files: the saved digests say simulation_2's market changed under its records, so its pool restarts and its books start
fresh, while the other two classes keep theirs. Compared with the loaded files instead, the new run's files would equal
the loaded ones and every book would keep records scored on another market.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_a_class_keeps_its_records_when_the_book_count_grows import _open  # noqa: E402
from test_the_validator_scores_the_market_of_the_run_it_is_on import (  # noqa: E402
    CLASSES, CONFIGS, NEEDED, NEW_RUN, _class_copies, _layout, _run_dir, tree,
)

from taos.im.validator.book_layout import layout_of  # noqa: E402

__all__ = ["tree"]
pytestmark = pytest.mark.skipif(not all((CONFIGS / n).exists() for n in NEEDED), reason="the class files are not here")
FROZEN = {"simulation_2": (r'( sigma=")0\.0125', r"\g<1>0.0200")}


def _saved_on_frozen_files(tree):
    """The layout a state saved in the frozen run carries, as the load restores it."""
    frozen = _class_copies(tree.root / "frozen", FROZEN)
    return layout_of(_layout(tree.root / "frozen", frozen))


def test_the_open_after_a_restart_restarts_the_class_whose_market_changed_under_its_records(tree, monkeypatch):
    tree.v._scored_layout = _saved_on_frozen_files(tree)
    tree.engine._settle_scored_layout()
    assert tree.v._scored_layout["classes"][2]["market"] != layout_of(tree.v.simulation)["classes"][2]["market"], (
        "the validator runs the start config's files, the records were scored on the frozen ones")
    new = _run_dir(tree, NEW_RUN, {c: tree.config_dir / f"{c}.xml" for c in CLASSES})
    ch = _open(tree, monkeypatch, new)
    assert ch["classes"] == [2], "simulation_2's pool restarts"
    assert {b for b, d in ch["books"].items() if d == "clear"} == set(range(96, 128))
    assert {b for b, d in ch["books"].items() if d == "keep"} == set(range(96))


def test_without_the_saved_layout_the_same_open_would_keep_every_book(tree, monkeypatch):
    tree.v._scored_layout = None
    new = _run_dir(tree, NEW_RUN, {c: tree.config_dir / f"{c}.xml" for c in CLASSES})
    ch = _open(tree, monkeypatch, new)
    assert not ch or (not ch["classes"] and set(ch["books"].values()) <= {"keep"}), (
        "the loaded files compared with themselves: the record of what the books were scored on is the saved layout")
