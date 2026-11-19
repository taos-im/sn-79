# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator counts, per uid, the fills it takes and makes under the book's dust floor.

A class's prints under their book's floor can come mostly from one coldkey's takers, at just above the minimum order;
the floor's footprint published per book could say only which books, never who took them. Now each fill side under its
print's floor is counted for its taker and its maker as timestamped histories over the making window, like the
order-to-trade reading (block_stats.note_otr): pruned and shifted with the de-beta histories, saved with the state,
handed to the scoring service child, and published per uid as debeta_dust_takes, debeta_dust_makes and
debeta_dust_take_share (takes over the uid's fills), so the board's line and the acceptance's row name the taker on
every read. Nothing is paid or discounted by it."""
import inspect
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taos.im.validator import (  # noqa: E402
    block_stats, book_layout, debeta, persistence, report, reward, scoring_shadow, trade,
)

S = 10 ** 9


def test_dust_takes_and_makes_land_in_timestamped_histories_beside_the_fills():
    v = SimpleNamespace()
    block_stats.note_otr(v, 7, "dust_take", 100, 3)
    block_stats.note_otr(v, 7, "dust_take", 160)
    block_stats.note_otr(v, 9, "dust_make", 100, 2)
    block_stats.note_otr(v, 7, "fill", 100, 8)
    assert v.debeta_dusttake_hist[7] == {100: 3, 160: 1} and v.debeta_dusttakes[7] == 4
    assert v.debeta_dustmake_hist[9] == {100: 2} and v.debeta_dustmakes[9] == 2
    assert block_stats.dust_fill_shares(v) == {7: {"takes": 4, "makes": 0, "take_share": 0.5},
                                               9: {"takes": 0, "makes": 2, "take_share": 0.0}}


def _batch(ts, prints):
    """prints: (size, taker uid, maker uid); background agents are negative, as the engine stamps them."""
    return [{"p": 100.0, "q": q, "s": 0, "Ta": ta, "Ma": ma, "j": ts + i} for i, (q, ta, ma) in enumerate(prints)]


def test_the_capture_reports_each_print_under_its_floor_by_taker_and_maker():
    """The floor is DUST_FLOOR_FRAC of the median of the trailing background-sided prints at or over the minimum
    order, fixed per print on arrival; a print under it is reported for its miner sides, a print at or over it is not,
    and a background side is never counted."""
    buy, sell, mid = {}, {}, {}
    out = {}
    bg = [(10.0, -1, -2)] * 6                       # the reference: background prints of size 10, floor 10 * frac
    small = debeta.DUST_FLOOR_FRAC * 10.0 * 0.5     # under the floor
    big = debeta.DUST_FLOOR_FRAC * 10.0 * 2.0       # over it
    prints = bg + [(small, 7, -1), (small, 7, 9), (big, 7, 9), (small, -3, 9)]
    debeta.accumulate_book_capture(buy, sell, 0, _batch(1000 * S, prints), 15, mid_state=mid, ts=1000 * S,
                                   min_size=1.0, dust_out=out)
    assert out == {7: {"takes": 2, "makes": 0}, 9: {"takes": 0, "makes": 2}}, out
    debeta.accumulate_book_capture(buy, sell, 0, _batch(1100 * S, [(big, 7, 9)]), 15, mid_state=mid, ts=1100 * S,
                                   min_size=1.0, dust_out=out)
    assert out == {7: {"takes": 2, "makes": 0}, 9: {"takes": 0, "makes": 2}}, "a print over the floor adds nothing"


def test_the_live_path_counts_the_report_into_the_histories():
    src = inspect.getsource(trade)
    i = src.index("dust_out=_dust_fills")
    j = src.index('note_otr(self, _u, "dust_take", sampled_timestamp, _d["takes"])', i)
    assert 0 < j - i < 1500, "the capture's dust report is counted right after the capture"
    assert 'note_otr(self, _u, "dust_make", sampled_timestamp, _d["makes"])' in src[i:j + 400]


def test_the_histories_are_pruned_shifted_saved_restored_handed_to_the_child_and_not_book_keyed():
    src = inspect.getsource(trade)
    for name in ("debeta_dusttake_hist", "debeta_dustmake_hist"):
        assert src.count(name) >= 4, f"{name}: init, prune, shift and restore"
    for name in ("debeta_dusttake_hist", "debeta_dustmake_hist"):
        assert name in persistence._DEBETA_HIST2
    for name in ("debeta_dusttakes", "debeta_dustmakes", "debeta_dusttake_hist", "debeta_dustmake_hist"):
        assert name in scoring_shadow._DEBETA_PLAIN and name in book_layout.NOT_BOOK_KEYED
    assert "debeta_dusttake_hist" in inspect.getsource(persistence._restore_trade_volumes)
    assert ("debeta_dusttake_hist", "debeta_dusttakes") in trade._ONE_LEVEL_HISTS
    assert ("debeta_dustmake_hist", "debeta_dustmakes") in trade._ONE_LEVEL_HISTS


def test_a_reset_uid_loses_its_dust_counts():
    v = SimpleNamespace()
    block_stats.note_otr(v, 7, "dust_take", 100, 3)
    block_stats.note_otr(v, 8, "dust_make", 100, 1)
    src = inspect.getsource(trade)
    i = src.index("for _name in ('debeta_msg_hist', 'debeta_fill_hist', 'debeta_msgs', 'debeta_fills',")
    assert "'debeta_dusttake_hist'" in src[i:i + 400] and "'debeta_dustmakes'" in src[i:i + 400]


def test_the_scorer_notes_the_reading_on_every_uid_and_the_report_publishes_it():
    src = inspect.getsource(reward)
    assert "dust_fill_shares(self)" in src
    for key in ('"dust_takes"', '"dust_makes"', '"dust_take_share"'):
        assert key in src, key
    rsrc = inspect.getsource(report)
    for pair in ("('dust_takes', 'debeta_dust_takes')", "('dust_makes', 'debeta_dust_makes')",
                 "('dust_take_share', 'debeta_dust_take_share')"):
        assert pair in rsrc, pair
    assert "'dust_takes': kappa_values.get('dust_takes')" in rsrc
