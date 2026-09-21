"""The run's book id set, not a dense range, is what the validator's query validation and report loops
iterate, in every engine mode.

A layout whose books are not 0..book_count-1 (exchange netuids, or a multi-book layout with gaps) has two
failure modes under a dense range: real books above book_count are refused or skipped, and phantom ids
inside the gaps are accepted and forwarded. The single-market simulation is unaffected because its id set
equals the range, which is why the two configs expose the same dense `book_ids` surface.
"""
import re
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _request(book_ids, book_count, mode):
    return {
        "book_count": book_count,
        "book_ids": book_ids,
        "engine_mode": mode,
        "miner_wealth": 0,
        "capital_turnover_cap": 0,
        "volume_decimals": 4,
        "books": {},
        "volume_sums": {},
    }


def _synapse(uid, book_ids):
    def instruction(book_id):
        return types.SimpleNamespace(agentId=uid, bookId=book_id, type="PLACE_ORDER_MARKET")
    return types.SimpleNamespace(
        is_timeout=False, is_failure=False, is_success=True, compressed=False,
        dendrite=types.SimpleNamespace(hotkey="5FakeHotkey", status_message=""),
        response=types.SimpleNamespace(agent_id=uid, instructions=[instruction(b) for b in book_ids]),
    )


def _surviving_books(request_data, uid, submitted):
    from taos.im.validator.query import QueryService
    synapse = _synapse(uid, submitted)
    QueryService.validate_responses(None, {uid: synapse}, request_data, set())
    return sorted(i.bookId for i in synapse.response.instructions)


def test_simulation_mode_validates_against_the_supplied_id_set():
    sparse = list(range(20)) + [32, 33, 34, 35, 48, 49, 50, 51]
    kept = _surviving_books(_request(sparse, 28, "simulation"), uid=7, submitted=[32, 20])
    assert kept == [32], "a real book above book_count is kept, a gap id inside it is refused"


def test_exchange_mode_keeps_validating_against_the_id_set():
    netuids = list(range(1, 129))
    kept = _surviving_books(_request(netuids, 128, "exchange"), uid=7, submitted=[128, 0])
    assert kept == [128]


def test_without_an_id_set_the_dense_range_still_applies():
    kept = _surviving_books(_request(None, 4, "simulation"), uid=7, submitted=[3, 4])
    assert kept == [3]


def test_report_loops_iterate_the_id_set_and_fall_back_to_the_range():
    from taos.im.validator.report import report_book_ids
    assert list(report_book_ids({"book_count": 28, "book_ids": [0, 1, 32, 33]})) == [0, 1, 32, 33]
    assert list(report_book_ids({"book_count": 4})) == [0, 1, 2, 3]
    assert list(report_book_ids({"book_count": 4, "book_ids": None})) == [0, 1, 2, 3]


def test_report_source_carries_no_dense_book_range():
    src = (REPO_ROOT / "taos" / "im" / "validator" / "report.py").read_text(encoding="utf-8")
    assert "range(validator_data['book_count'])" not in src
    assert "range(self.simulation.book_count)" not in src
    assert re.search(r"'book_ids':\s*list\(self\.simulation\.book_ids\)", src), "validator_data carries the id set"


def test_both_configs_expose_a_dense_book_ids_surface():
    from taos.im.protocol.exchange_config import ExchangeConfig
    from taos.im.protocol.models import MarketSimulationConfig
    assert ExchangeConfig(book_count=3).book_ids == [0, 1, 2]
    sim = MarketSimulationConfig.model_construct(book_count=5)
    assert sim.book_ids == [0, 1, 2, 3, 4]
