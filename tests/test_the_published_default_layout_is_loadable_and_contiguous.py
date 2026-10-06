# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The layout the published launcher runs by default must load, and look the way the release says.

run_validator.sh defaults SIMULATION_CONFIG to multiasset_simulation_0, so this is the configuration a
miner actually trades unless they pass -g. Until 0.6.3 the carve shipped neither it nor the per-class
config it references, and the published suite covered multi-asset nowhere: the tests that read a layout
all read an INTERNAL bench (multiasset_mp.xml, multiasset_96_32.xml, multiasset_acceptance.xml) and were
excluded from the carve for raising FileNotFoundError on files that do not ship. The release's headline
capability therefore had no published test at all.

This one reads the layout that ships, so it fails in a miner's tree if the carve ever stops shipping a
piece of it, and it states the properties an agent is entitled to rely on: contiguous ids from zero, two
named classes, and a grid per book rather than one top-level priceDecimals.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "simulate" / "trading" / "run" / "config"
DEFAULT_LAYOUT = CONFIG_DIR / "multiasset_simulation_0.xml"

pytestmark = pytest.mark.skipif(not DEFAULT_LAYOUT.exists(),
                                reason="the default layout is not present in this tree")


def _layout():
    from taos.im.protocol.models import MultiAssetSimulationConfig

    return MultiAssetSimulationConfig.from_multiasset_xml(
        ET.parse(DEFAULT_LAYOUT).getroot(), CONFIG_DIR)


def test_every_class_the_default_layout_names_is_present():
    """A wrapper whose per-class config is missing is the defect this file exists for."""
    named = {bg.get("path") for bg in ET.parse(DEFAULT_LAYOUT).getroot().iter("Background")}
    missing = sorted(p for p in named if p and not (CONFIG_DIR / p).exists())
    assert not missing, f"the default layout references {missing}, which this tree does not contain"


def test_the_default_layout_loads_as_two_named_classes():
    classes = _layout().asset_classes()
    assert [c.name for c in classes] == ["simulation_0", "simulation_1"], (
        "the published layout's classes are the names the migration note and the announcement use")


def test_book_ids_run_contiguously_from_zero_across_the_whole_layout():
    """An agent written on the 0.6.2 base class iterates range(book_count); that must still work."""
    cfg = _layout()
    assert cfg.book_ids == list(range(len(cfg.book_ids)))
    covered = [b for c in cfg.asset_classes() for b in c.books]
    assert sorted(covered) == cfg.book_ids, "a book belongs to exactly one class, and none is orphaned"


def test_each_book_carries_its_own_grid():
    cfg = _layout()
    first, second = cfg.asset_classes()
    for book in (first.books[0], first.books[-1]):
        assert cfg.config_for_book(book).priceDecimals == first.priceDecimals
    for book in (second.books[0], second.books[-1]):
        assert cfg.config_for_book(book).priceDecimals == second.priceDecimals
    with pytest.raises(KeyError):
        cfg.config_for_book(cfg.book_ids[-1] + 1)
