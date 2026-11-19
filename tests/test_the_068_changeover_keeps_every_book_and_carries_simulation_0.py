# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The layout half of the 0.6.7 to 0.6.8 changeover, read from the tree's own layout files through the validator's
functions: the kept 128/64/64 layout (multiasset_simulation_0_128_64_64.xml) against the published four classes of 64,
both naming the live class files, so the two sides differ in their books alone and what is read is the shrink and the
id moves, with every market the same. Every existing book carries over and only the genuinely new books start
blank. simulation_0 keeps books 0 to 63 with their records and carries the standing it earned
on all 128 books over one lookback; simulation_1 and simulation_2 move down with their records; simulation_3 alone
starts empty. The changeover with simulation_2's market changed as well is read
from 0.6.7's own files in tests/test_the_068_changeover_from_0_6_7s_own_files.py. Skipped until both layout files are
in the tree.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

import taos.im.validator.trade as trade
from taos.im.protocol.config import MultiAssetSimulationConfig

CONFIGS = Path(__file__).resolve().parents[1] / "simulate" / "trading" / "run" / "config"
OLD, NEW = "multiasset_simulation_0_128_64_64.xml", "multiasset_simulation_0.xml"
NAMES = ["simulation_0", "simulation_1", "simulation_2", "simulation_3"]
pytestmark = pytest.mark.skipif(
    not all((CONFIGS / n).exists() for n in (OLD, NEW, *(f"{c}.xml" for c in NAMES))),
    reason="0.6.7's kept layout or 0.6.8's four class files are not in the tree")


def _parse(name):
    return MultiAssetSimulationConfig.from_multiasset_xml(ET.parse(CONFIGS / name).getroot(), CONFIGS)


def _change():
    return trade.layout_book_changes(_parse(OLD), _parse(NEW))


def test_every_book_of_the_0_6_7_layout_keeps_or_moves_its_records_and_only_simulation_3_starts_empty():
    ch = _change()
    assert ch["names_old"] == NAMES[:3] and ch["names_new"] == NAMES
    assert ch["classes"] == [], "no class's market changes, so no pool restarts"
    assert ch["shrunk"] == [0], "simulation_0 alone loses books"
    dest = ch["dest"]
    assert {b: dest[b] for b in range(64)} == {b: b for b in range(64)}, "simulation_0 keeps its first 64 in place"
    assert all(dest[b] is None for b in range(64, 128)), "simulation_0's last 64 have no book of their market"
    assert {b: dest[b] for b in range(128, 256)} == {b: b - 64 for b in range(128, 256)}, "the others move down"
    assert sorted(b for b, a in ch["books"].items() if a == "new") == list(range(192, 256)), "simulation_3 alone"
    assert not [b for b, a in ch["books"].items() if a in ("clear", "drop")], "no book starts fresh"


def test_simulation_0_carries_its_standing_and_simulation_3_ramps_in():
    vectors = {n: {"making": {1: 0.25, 2: 0.75}, "skill": {3: 1.0}} for n in NAMES[:3]}
    holder = SimpleNamespace(_debeta_class_vectors=vectors)
    trade.begin_class_handover(holder, _change(), 0, 3600 * 10**9)
    h = holder._debeta_handover
    assert set(h["carried"]) == {"simulation_0"}
    assert h["carried"]["simulation_0"] == {"making": {1: 0.25, 2: 0.75}, "skill": {3: 1.0}}
    assert h["new"] == ["simulation_3"]
