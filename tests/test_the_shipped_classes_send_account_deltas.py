# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Every shipped class sends account deltas.

The engine leaves an account entry unchanged since the last answered state out of each state only when its exchange
carries dirtyAccountsOnly="1"; the attribute defaults to off in the engine, so a shipped configuration must set it for
the validator's account_deltas lever to have anything to act on. The validator fills every left-out entry back in (kept_accounts.fill_accounts), so what miners receive is
unchanged. Every class of a layout must agree: the state's format follows the layout's exchanges.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parent.parent / "simulate" / "trading" / "run" / "config"
CLASSES = [f"simulation_{i}.xml" for i in range(4)]


@pytest.mark.parametrize("name", CLASSES)
def test_the_class_exchange_sends_account_deltas(name):
    exchange = ET.parse(CONFIG / name).getroot().find("Agents/MultiBookExchangeAgent")
    assert exchange is not None, f"{name} has no exchange"
    assert exchange.get("dirtyAccountsOnly") == "1", f"{name} sends every account in full"


@pytest.mark.parametrize("layout", ["multiasset_simulation_0.xml"])
def test_every_class_a_layout_loads_sends_account_deltas(layout):
    paths = [el.get("path") for el in ET.parse(CONFIG / layout).getroot().iter() if el.get("path")]
    assert paths, f"{layout} names no class files"
    for path in paths:
        exchange = ET.parse(CONFIG / path).getroot().find("Agents/MultiBookExchangeAgent")
        assert exchange is not None and exchange.get("dirtyAccountsOnly") == "1", (
            f"{layout} loads {path}, whose exchange sends every account in full"
        )
