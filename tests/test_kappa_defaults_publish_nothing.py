"""A uid whose kappa has not been computed must publish nothing per book.

The default kappa record used for a fresh, reset or newly loaded uid used to fill the per-book weighted
values with 0.0 while the raw values were None. Every consumer treats None as absent and 0.0 as a
measurement, so an uncomputed kappa reached the dashboards as a flat zero line on every book. The default
now leaves both per-book maps None-filled, and the payload builder drops None, so an uncomputed kappa is
absent downstream rather than zero.
"""
import re
from collections import defaultdict
from pathlib import Path

from taos.im.validator.ingest_payload import kappa_maps
from taos.im.validator.trade import reset_agent_histories


class _Validator:
    """Attribute bag: anything the reset touches that is not set here becomes a nested float map."""

    def __init__(self):
        self.kappa_values = {}
        self.kappa_cache = {}

    def __getattr__(self, name):
        value = defaultdict(lambda: defaultdict(float))
        object.__setattr__(self, name, value)
        return value


def test_a_reset_uid_carries_no_per_book_weighted_kappa():
    v = _Validator()
    reset_agent_histories(v, 7, [1, 2, 3])
    record = v.kappa_values[7]
    assert set(record["books"]) == {1, 2, 3} and all(x is None for x in record["books"].values())
    assert set(record["books_weighted"]) == {1, 2, 3} and all(x is None for x in record["books_weighted"].values())
    assert record["total"] is None and record["median"] is None


def test_the_payload_drops_an_uncomputed_kappa_instead_of_publishing_zero():
    v = _Validator()
    reset_agent_histories(v, 7, [1, 2, 3])
    maps = kappa_maps(v.kappa_values)
    assert "7" not in maps["agent_kappa_books_w"]
    assert "7" not in maps["agent_kappa_books"]
    assert "7" not in maps["agent_kappa"]


def test_no_default_record_fills_the_weighted_map_with_zero():
    root = Path(__file__).resolve().parents[1] / "taos" / "im" / "validator"
    sources = [root / "persistence.py", root / "trade.py", root / "engines" / "simulation.py", root / "engines" / "exchange.py",
               root.parent / "neurons" / "validator.py"]
    offenders = []
    for path in sources:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "books_weighted" in line and re.search(r":\s*0\.0\s+for\s", line):
                offenders.append(f"{path.name}:{lineno}")
    assert offenders == [], offenders
