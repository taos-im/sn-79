# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The dust floor's live line. A scoring rule ships with an acceptance check and a monitor line that read the same
quantity with the same threshold. The quantity is the share of prints in a book's live capture window that fall
below their floor (what the floor excludes), beside the floor's own level. The validator computes both per book
from the live window the scorer already keeps, ships them with the reporting payload, and the report publishes
them as book_gauges capture_dust_share and capture_floor; the board monitor flags a book over its share threshold
and the injected acceptance rows assert the same share under the same prints."""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from taos.im.validator.debeta import accumulate_book_capture, capture_floor_stats
from taos.im.validator.report import publish_capture_floor_gauges

W = 15
BOOK = 7
MIN_SIZE = 0.25
ROOT = Path(__file__).resolve().parents[1]


def _mk(price, q=10.0, s=0, ma=1, ta=-2):
    return {"p": price, "q": q, "s": s, "Ma": ma, "Ta": ta}


def _stream(n=90):
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(_mk(100.0 + 0.01 * ((i * 7) % 5 - 2), q=10.0 + (i % 4), ma=-3, ta=-2))
        elif (i // 2) % 2 == 0:
            out.append(_mk(99.6, q=10.0 + (i % 4), s=1))
        else:
            out.append(_mk(100.4, q=10.0 + (i % 4), s=0))
    return out


def _with_dust(trades, at, price=105.0, q=0.001):
    out = list(trades)
    for k, i in enumerate(sorted(at)):
        out.insert(i + k, _mk(price, q=q, s=k % 2, ma=7, ta=8))
    return out


def _live(trades):
    state, buy, sell = {}, defaultdict(dict), defaultdict(dict)
    for k, t in enumerate(trades):
        accumulate_book_capture(buy, sell, BOOK, [t], W, ts=k * 1_000_000_000, mid_state=state, min_size=MIN_SIZE)
    return state


def test_a_clean_window_reports_no_dust_and_the_floor_the_market_sets():
    s = capture_floor_stats(_live(_stream()))
    assert set(s) == {BOOK}
    assert s[BOOK]["dust_share"] == 0.0 and s[BOOK]["n"] > 0
    assert 0.9 <= s[BOOK]["floor"] <= 1.4, "a tenth of the typical background print, sizes 10 to 13"


def test_dust_in_the_window_shows_as_the_share_the_floor_excludes():
    # thirty dust prints among the last thirty real ones: about half of the live window
    s = capture_floor_stats(_live(_with_dust(_stream(), at=tuple(range(60, 90)))))[BOOK]
    assert 0.3 <= s["dust_share"] <= 0.7, s
    assert 0.9 <= s["floor"] <= 1.4, "the floor is the market's, a flood of residual dust cannot lower it"


def test_an_empty_or_absent_state_reports_nothing():
    assert capture_floor_stats({}) == {} and capture_floor_stats(None) == {}
    assert capture_floor_stats({3: {}}) == {} and capture_floor_stats({3: {"sizes": [], "floors": []}}) == {}


def test_the_report_publishes_the_floor_and_the_dust_share_per_book():
    me = SimpleNamespace(prometheus_book_gauges="BOOK",
                         capture_floor_stats={7: {"floor": 1.1, "dust_share": 0.25, "n": 31}, 9: {"floor": 0.0, "dust_share": 0.0, "n": 0}})
    updates = []
    publish_capture_floor_gauges(me, updates, "wallet", 79, "sim-1")
    assert ("BOOK", 1.1, "wallet", 79, "sim-1", 7, 0, "capture_floor") in updates
    assert ("BOOK", 0.25, "wallet", 79, "sim-1", 7, 0, "capture_dust_share") in updates
    assert len(updates) == 4, "every book the validator reports, two series each"
    updates = []
    publish_capture_floor_gauges(SimpleNamespace(prometheus_book_gauges="BOOK"), updates, "wallet", 79, "sim-1")
    assert updates == []


def test_the_validator_ships_the_stats_and_the_report_reads_them():
    validator_src = (ROOT / "taos/im/neurons/validator.py").read_text()
    assert "'capture_floor_stats': capture_floor_stats(" in validator_src, "the reporting payload carries the per-book stats"
    report_src = (ROOT / "taos/im/validator/report.py").read_text()
    assert "data.get('capture_floor_stats')" in report_src, "the report deserialises them"
    assert "publish_capture_floor_gauges(self, updates, wallet_addr, netuid, simid)" in report_src, "and publishes them with the book metrics"
