# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The 0.6.7 to 0.6.8 changeover read from 0.6.7's own published files, against the shipped 0.6.8 layout, through
the validator's own functions.

tests/fixtures/classes_0_6_7 holds 0.6.7's layout and its three class files byte for byte. The kept 0.6.7 layout in
the config folder names the live class files, so a change to a live class does not show from it; these files read the
changeover a validator restarted on a 0.6.7 state meets. simulation_0 keeps its first 64 books and drops the rest,
simulation_1 moves down with its records, simulation_2's market changes in 0.6.8 so its books start fresh while the
class carries its standing over one lookback, and simulation_3 is new. Every 0.6.8 class sends only the accounts a step
changed (dirtyAccountsOnly), which 0.6.7's do not: that is how the state is sent, not the market, so it changes no
class's records.
"""
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import taos.im.validator.trade as trade
from taos.im.protocol.config import MultiAssetSimulationConfig, market_source_digest

ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = ROOT / "tests" / "fixtures" / "classes_0_6_7"
NEW_DIR = ROOT / "simulate" / "trading" / "run" / "config"
LAYOUT = "multiasset_simulation_0.xml"
MBE = "Agents/MultiBookExchangeAgent"
OLD_SHA256 = {
    "multiasset_simulation_0.xml": "d1dece4abb555b03c764667c93540ec97fbde4b0c47c8ba62c4809537a8131f1",
    "simulation_0.xml": "72f4500f92dc67364e35e9447d80f432ce898d9452cc94b7c04f6e7b6beeb1f4",
    "simulation_1.xml": "15f59e4005fd69a2a09eac06f7b40834b9db17e5722de845dfa1086626077c9c",
    "simulation_2.xml": "49ab715606e0a7f484887c85470010abf5d745cc2d0b5c19289c75a9b85c000b",
}


def _parse(folder):
    return MultiAssetSimulationConfig.from_multiasset_xml(ET.parse(folder / LAYOUT).getroot(), folder)


def _change():
    return trade.layout_book_changes(_parse(OLD_DIR), _parse(NEW_DIR))


def test_the_fixtures_are_0_6_7s_files_unchanged():
    for name, sha in OLD_SHA256.items():
        assert hashlib.sha256((OLD_DIR / name).read_bytes()).hexdigest() == sha, name


def test_simulation_0_and_1_keep_their_markets_although_0_6_8_sends_only_changed_accounts():
    for name in ("simulation_0.xml", "simulation_1.xml"):
        old, new = ET.parse(OLD_DIR / name).getroot(), ET.parse(NEW_DIR / name).getroot()
        assert "dirtyAccountsOnly" not in old.find(MBE).attrib and new.find(MBE).get("dirtyAccountsOnly") == "1", name
        assert market_source_digest(old) == market_source_digest(new), name


def test_only_simulation_2s_market_changes():
    ch = _change()
    assert ch["names_old"] == ["simulation_0", "simulation_1", "simulation_2"]
    assert ch["names_new"] == ["simulation_0", "simulation_1", "simulation_2", "simulation_3"]
    assert [ch["names_old"][c] for c in ch["classes"]] == ["simulation_2"]
    assert ch["shrunk"] == [0], "simulation_0 alone loses books"


def test_each_book_of_0_6_7_keeps_moves_or_starts_fresh_by_its_class():
    ch = _change()
    by_action = {}
    for book, action in ch["books"].items():
        by_action.setdefault(action, []).append(book)
    assert {a: sorted(b) for a, b in by_action.items()} == {
        "keep": list(range(0, 64)),
        "move": list(range(64, 128)),
        "clear": list(range(128, 192)),
        "new": list(range(192, 256)),
    }
    dest = ch["dest"]
    assert {b: dest[b] for b in range(64)} == {b: b for b in range(64)}, "simulation_0 keeps its first 64 in place"
    assert all(dest[b] is None for b in range(64, 128)), "simulation_0's last 64 have no book of their market"
    assert {b: dest[b] for b in range(128, 192)} == {b: b - 64 for b in range(128, 192)}, "simulation_1 moves down"
    assert all(dest[b] is None for b in range(192, 256)), "simulation_2's books carry no records into a new market"


def test_simulation_0_and_2_carry_their_standing_and_simulation_3_ramps_in():
    vectors = {n: {"making": {1: 0.25, 2: 0.75}, "skill": {3: 1.0}} for n in ("simulation_0", "simulation_1",
                                                                               "simulation_2")}
    holder = SimpleNamespace(_debeta_class_vectors=vectors)
    trade.begin_class_handover(holder, _change(), 0, 3600 * 10**9)
    h = holder._debeta_handover
    assert set(h["carried"]) == {"simulation_0", "simulation_2"}
    assert h["carried"]["simulation_2"] == {"making": {1: 0.25, 2: 0.75}, "skill": {3: 1.0}}
    assert h["new"] == ["simulation_3"]
