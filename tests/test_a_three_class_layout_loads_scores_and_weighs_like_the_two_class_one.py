# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A layout of three asset classes loads, numbers its books contiguously in document order with the coarsest grid
first, and is weighed by the scorer as the two-class layout is: "books" gives each class its share of the books, an
explicit dial pays only with one entry per class.

The shipped default, multiasset_simulation_0.xml, is that layout (64/32/32, simulation_2 on a finer price grid).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from taos.im.protocol.models import MultiAssetSimulationConfig
from taos.im.validator.reward import class_weights_for_layout

CONFIG_DIR = Path(__file__).resolve().parent.parent / "simulate" / "trading" / "run" / "config"
SHIPPED = "multiasset_simulation_0.xml"
# name: (class names, first and last book id per class, priceDecimals per class)
LAYOUTS = {
    SHIPPED: (["simulation_0", "simulation_1", "simulation_2"], [(0, 63), (64, 95), (96, 127)], [2, 2, 4]),
}
CASES = sorted(LAYOUTS)


def _load(name):
    return MultiAssetSimulationConfig.from_multiasset_xml(ET.parse(CONFIG_DIR / name).getroot(), CONFIG_DIR)


def _cmap(cfg):
    return {b: i for i, c in enumerate(cfg.asset_classes()) for b in c.books}


@pytest.mark.parametrize("name", CASES)
def test_the_classes_are_named_numbered_and_gridded_in_document_order(name):
    names, ranges, decimals = LAYOUTS[name]
    classes = _load(name).asset_classes()
    assert [c.name for c in classes] == names
    assert [(min(c.books), max(c.books)) for c in classes] == ranges
    assert [len(c.books) for c in classes] == [z - a + 1 for a, z in ranges], "every id in the range, none twice"
    assert [c.priceDecimals for c in classes] == decimals
    assert decimals[0] == min(decimals), "the coarsest grid comes first, so an older agent's price is valid on every class"


@pytest.mark.parametrize("name", CASES)
def test_books_weighs_each_class_by_its_share_of_the_books(name):
    cfg = _load(name)
    assert class_weights_for_layout("books", _cmap(cfg)) == pytest.approx([0.5, 0.25, 0.25])


def test_an_explicit_dial_pays_only_with_one_entry_per_class():
    cmap = _cmap(_load(SHIPPED))
    assert class_weights_for_layout("0.6,0.2,0.2", cmap) == pytest.approx([0.6, 0.2, 0.2])
    assert class_weights_for_layout("0.75,0.25", cmap) == [], "the two-class dial on a three-class layout is the flat rule for the round"
