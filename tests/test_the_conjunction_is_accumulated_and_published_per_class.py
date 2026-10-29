# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The conjunction, a miner's two-sided resting presence near the touch, is accumulated per uid and book and published
per miner and per asset class, so it is watched on mainnet before any rule pays for it.

At each state, on each book with a bid and an ask that do not cross, a uid's resting buy volume within the band below
the mid and its resting sell volume within the band above it are summed; when both reach the size gate the smaller of
the two, times the interval since the previous state, is its presence for that interval. The band is the class's
bound (`--scoring.debeta.conjunction_bounds`, one value per class in the layout's order) times the book's spread
reference, a running mean of the quoted spread in basis points of the mid; the gate is
`--scoring.debeta.conjunction_min_size_frac` of the book's size reference, a running mean of its fill sizes. The farm
reference per book is what a quote at the gate's size on both sides would score: the gate times every counted
interval. All of it is windowed on the kappa lookback, pruned, shifted, persisted and reset with the other de-beta
accumulators, and pay reads none of it.
"""
import inspect
from types import SimpleNamespace

from taos.im.validator import report, reward, trade


def _book(bid, ask):
    return {"b": [{"p": bid, "q": 1.0}], "a": [{"p": ask, "q": 1.0}]}


def _v(bounds="simulation_0:5,simulation_1:2,simulation_2:5", frac=0.25, classes=None):
    v = SimpleNamespace()
    names = [SimpleNamespace(name=f"simulation_{i}") for i in range(3)]
    v.simulation = SimpleNamespace(asset_classes=lambda: names)
    v.config = SimpleNamespace(scoring=SimpleNamespace(debeta=SimpleNamespace(conjunction_bounds=bounds,
                                                                              conjunction_min_size_frac=frac)))
    v._debeta_class_map_cache = ((id(v.simulation), id(None)), dict(classes or {}))
    trade.init_conjunction_state(v)
    return v


def test_the_state_names_are_classified_and_handled_with_the_others():
    for n in ("debeta_conj", "debeta_conj_hist", "debeta_conj_farm", "debeta_conj_farm_hist"):
        assert n in trade.CONJUNCTION_STATE, n
    src = inspect.getsource(trade)
    for name in ("debeta_conj_hist", "debeta_conj_farm_hist"):
        assert src.count(name) >= 4, f"{name}: init, prune, shift and restore"
    assert "'debeta_conj'" in inspect.getsource(trade.reset_agent_histories)
    from taos.im.validator import persistence, scoring_shadow
    assert "debeta_conj_hist" in persistence._DEBETA_HIST3 and "debeta_conj_farm_hist" in persistence._DEBETA_HIST2
    assert "debeta_conj" in scoring_shadow._DEBETA_DD2


def test_the_book_references_follow_the_quoted_spread_and_the_fill_sizes():
    v = _v()
    trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), [{"y": "t", "q": 2.0}])
    spread, size = v.debeta_conj_ref[0]
    assert abs(spread - 20.0) < 1e-9 and size == 2.0, "the first sample sets each reference"
    trade.conjunction_book_refs(v, 0, _book(99.8, 100.2), [{"y": "t", "q": 4.0}])
    spread2, size2 = v.debeta_conj_ref[0]
    assert 20.0 < spread2 < 40.0 and 2.0 < size2 < 4.0, "then a running mean moves toward each new sample"
    trade.conjunction_book_refs(v, 1, _book(100.2, 100.1), [])
    assert 1 not in v.debeta_conj_ref, "a crossed book sets no spread reference"


def test_two_sided_presence_inside_the_band_counts_the_smaller_side_times_the_interval():
    v = _v(classes={0: 0})
    trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), [{"y": "t", "q": 1.0}])  # 20 bps, size 1
    books = {0: _book(99.9, 100.1)}
    accounts = {0: {"o": [{"p": 99.5, "q": 2.0, "s": 0}, {"p": 100.5, "q": 3.0, "s": 1}]}}
    trade.accumulate_conjunction(v, 7, books, accounts, dt=2.0, ts=10)
    assert v.debeta_conj[7][0] == 4.0, "class 0's bound 5 x 20 bps = 1.0 either side of 100: min(2, 3) x 2 s"
    assert v.debeta_conj_hist[7][0] == {10: 4.0}


def test_one_side_a_side_outside_the_band_or_under_the_gate_counts_nothing():
    v = _v(classes={0: 0, 1: 1})
    for b in (0, 1):
        trade.conjunction_book_refs(v, b, _book(99.9, 100.1), [{"y": "t", "q": 1.0}])
    books = {0: _book(99.9, 100.1), 1: _book(99.9, 100.1)}
    one_side = {0: {"o": [{"p": 99.5, "q": 2.0, "s": 0}]}}
    outside = {0: {"o": [{"p": 98.0, "q": 2.0, "s": 0}, {"p": 100.5, "q": 3.0, "s": 1}]}}
    under_gate = {0: {"o": [{"p": 99.5, "q": 0.1, "s": 0}, {"p": 100.5, "q": 3.0, "s": 1}]}}
    narrower = {1: {"o": [{"p": 99.5, "q": 2.0, "s": 0}, {"p": 100.5, "q": 3.0, "s": 1}]}}
    for uid, acc in ((1, one_side), (2, outside), (3, under_gate), (4, narrower)):
        trade.accumulate_conjunction(v, uid, books, acc, dt=1.0, ts=1)
        assert not v.debeta_conj.get(uid), uid
    trade.accumulate_conjunction(v, 5, {0: _book(100.2, 100.1)}, {0: {"o": [{"p": 100.0, "q": 2.0, "s": 0},
                                                                       {"p": 100.3, "q": 2.0, "s": 1}]}}, dt=1.0, ts=1)
    assert not v.debeta_conj.get(5), "a crossed book is no sample"


def test_the_farm_reference_counts_the_gate_size_for_every_counted_interval():
    v = _v(classes={0: 0})
    trade.conjunction_book_refs(v, 0, _book(99.9, 100.1), [{"y": "t", "q": 2.0}])
    trade.accumulate_conjunction_farm(v, {0: _book(99.9, 100.1)}, dt=3.0, ts=5)
    assert v.debeta_conj_farm[0] == 0.25 * 2.0 * 3.0 and v.debeta_conj_farm_hist[0] == {5: 1.5}


def test_the_interval_is_the_time_since_the_previous_state_and_is_capped():
    v = _v()
    assert trade.conjunction_interval(v, 10 * 10 ** 9) == 0.0, "the first state after a start counts no interval"
    assert trade.conjunction_interval(v, 12 * 10 ** 9) == 2.0
    assert trade.conjunction_interval(v, 500 * 10 ** 9) == trade.CONJ_MAX_INTERVAL_S, "a gap counts at most the cap"


def test_the_accumulators_are_wired_into_the_state_update():
    src = inspect.getsource(trade.update_trade_volumes)
    assert "conjunction_book_refs(self, bookId, book, trades)" in src
    assert "accumulate_conjunction_farm(self, books, dt=_conj_dt, ts=sampled_timestamp)" in src
    assert "_conj_dt = conjunction_interval(self, timestamp)" in src
    assert "accumulate_conjunction(self, uid_item, books, accounts_uid, dt=self._conj_dt, ts=sampled_timestamp)" \
        in inspect.getsource(trade._process_uid_trade_volumes)


def test_presence_and_the_farm_are_gathered_per_class():
    v = _v(classes={0: 0, 1: 0, 2: 1})
    v.debeta_conj = {7: {0: 2.0, 1: 3.0, 2: 4.0}, 8: {2: 1.0}}
    v.debeta_conj_farm = {0: 1.0, 1: 1.0, 2: 5.0}
    by_class, farm = reward.conjunction_by_class(v, {0: 0, 1: 0, 2: 1})
    assert by_class == {0: {7: 5.0}, 1: {7: 4.0, 8: 1.0}} and farm == {0: 2.0, 1: 5.0}
    rsrc = inspect.getsource(reward.compute_debeta_scores)
    assert "conjunction_by_class(self, _cmap)" in rsrc, "the class details carry each uid's presence in the class"
    assert "_dd['conjunction'] = " in rsrc, "the flat detail carries each uid's presence over all books"


def test_the_report_publishes_the_conjunction_per_miner_and_per_class():
    assert ("conjunction", "conjunction") in report.CLASS_DETAIL_GAUGES
    assert "('conjunction', 'debeta_conjunction')" in inspect.getsource(report)
    coll = object()
    v = SimpleNamespace(prometheus_class_gauges=coll, simulation=None,
                        debeta_class_summary={0: {"conjunction_farm": 2.0}})
    updates = []
    report.publish_class_gauges(v, updates, "w", 79, "sim")
    assert (coll, 2.0, "w", 79, "sim", "0", "conjunction_farm") in updates


def test_the_dials_have_defaults_and_are_published():
    from taos.im.config import debeta_defaults
    d = debeta_defaults()
    assert d["conjunction_bounds"] == "simulation_0:5,simulation_1:2,simulation_2:5"
    assert d["conjunction_min_size_frac"] == 0.25
    from taos.im.neurons import validator as nv
    src = inspect.getsource(nv)
    assert "'debeta_conjunction_bounds'" in src and "'debeta_conjunction_min_size_frac'" in src


def test_the_bound_follows_the_class_name_not_its_position():
    v = _v(bounds="simulation_1:2,simulation_0:5", classes={0: 0, 1: 1})
    v.simulation = SimpleNamespace(asset_classes=lambda: [SimpleNamespace(name="simulation_1"),
                                                          SimpleNamespace(name="simulation_0")])
    v._debeta_class_map_cache = ((id(v.simulation), id(None)), {0: 0, 1: 1})
    for b in (0, 1):
        trade.conjunction_book_refs(v, b, _book(99.9, 100.1), [{"y": "t", "q": 1.0}])
    acc = {"o": [{"p": 99.5, "q": 2.0, "s": 0}, {"p": 100.5, "q": 3.0, "s": 1}]}
    trade.accumulate_conjunction(v, 9, {0: _book(99.9, 100.1), 1: _book(99.9, 100.1)}, {0: acc, 1: acc}, dt=1.0, ts=1)
    assert 0 not in v.debeta_conj[9], "class index 0 is simulation_1 here: its bound 2 leaves both orders outside"
    assert v.debeta_conj[9][1] == 2.0, "class index 1 is simulation_0, bound 5: both orders inside"
