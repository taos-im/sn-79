# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A class's market is its class file less the run's own bookkeeping, read the same way from the parsed config and
from the file's source.

The validator clears a class's records and restarts its pool when its market changes between runs. The market is
every field of the parsed configuration but the run's bookkeeping, plus a digest of the class file's whole source,
so a parameter the protocol does not parse counts too. Both leave out the same fields, the source digest the
attributes those fields are read from, so a change of uid slots, of the grace period or of the books a block
carries does not read as a new market, while any market parameter still does."""
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from taos.im.protocol import config as pconfig
from taos.im.protocol.config import MarketSimulationConfig, market_source_digest
from taos.im.validator import trade

CLASS_FILE = Path(__file__).resolve().parent.parent / "simulate" / "trading" / "run" / "config" / "simulation_0.xml"
MBE = "Agents/MultiBookExchangeAgent"
BOOKS = "Agents/MultiBookExchangeAgent/Books"


def _root():
    return ET.parse(CLASS_FILE).getroot()


def _bookkeeping_changed():
    root = _root()
    mbe, books = root.find(MBE), root.find(BOOKS)
    mbe.set("remoteAgentCount", str(int(mbe.get("remoteAgentCount")) + 8))
    mbe.set("gracePeriod", str(int(mbe.get("gracePeriod")) * 2))
    books.set("instanceCount", str(int(books.get("instanceCount")) * 2))
    books.set("maxDepth", str(int(books.get("maxDepth")) + 4))
    books.set("detailedDepth", str(int(books.get("detailedDepth")) + 1))
    root.set("duration", str(int(root.get("duration")) * 2))
    return root


def _market_changed():
    root = _root()
    fp = root.find(BOOKS + "/Processes/FundamentalPrice")
    fp.set("sigma", str(float(fp.get("sigma")) * 2))
    return root


def test_a_class_file_differing_only_in_the_runs_bookkeeping_digests_the_same():
    assert market_source_digest(_bookkeeping_changed()) == market_source_digest(_root())


def test_a_market_attribute_changes_the_digest():
    assert market_source_digest(_market_changed()) != market_source_digest(_root())


def test_the_digest_leaves_the_class_file_as_it_was():
    root = _root()
    before = ET.tostring(root)
    market_source_digest(root)
    assert ET.tostring(root) == before


def test_the_validator_reads_the_same_market_from_the_parsed_class_file():
    base = trade._market_digest(MarketSimulationConfig.from_xml(_root()))
    assert trade._market_digest(MarketSimulationConfig.from_xml(_bookkeeping_changed())) == base
    assert trade._market_digest(MarketSimulationConfig.from_xml(_market_changed())) != base


def test_one_list_names_the_runs_bookkeeping_for_both():
    assert trade._MARKET_EXCLUDE == frozenset(pconfig.RUN_BOOKKEEPING_FIELDS)
    fields = set(MarketSimulationConfig.model_fields)
    assert set(pconfig.RUN_BOOKKEEPING_FIELDS) <= fields, set(pconfig.RUN_BOOKKEEPING_FIELDS) - fields
    root = _root()
    for field, where in pconfig.RUN_BOOKKEEPING_FIELDS.items():
        if where is not None:
            path, attribute = where
            assert root.find(path) is not None and attribute in root.find(path).attrib, (field, where)


def test_a_copy_is_digested_like_its_original():
    root = _root()
    assert market_source_digest(copy.deepcopy(root)) == market_source_digest(root)
