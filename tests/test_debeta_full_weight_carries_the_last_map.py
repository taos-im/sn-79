# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""At the (0, 0, 1) rung a cycle with no de-beta map carries the previous one instead of zeroing the board.

Below weight 1.0 an empty map renormalises the de-beta share onto the legacy components and the
cycle scores normally. At weight 1.0 kappa and pnl carry weight 0 and are not computed, so there is
nothing to renormalise onto: the warming guard (fewer positive scores than min_books) or any exception
inside the computation scored every miner 0 for that cycle (miner report on the 0.6.1 testnet
ratchet). The previous cycle's map is carried for up to ten minutes, with its
decomposition, and every carry is logged; with nothing to carry the zero cycle is logged as an error.
"""
import time
from types import SimpleNamespace

import taos.im.validator.reward as reward


def _validator(weight, min_books=5):
    v = SimpleNamespace()
    v.config = SimpleNamespace(scoring=SimpleNamespace(debeta=SimpleNamespace(
        weight=weight, w_make=0.30, floor_scale=1.0, min_books=min_books, p11_strength=0.0)))
    # Empty accumulators: the computation runs and produces no positive score, so the warming guard trips.
    v.capture_buy_sums = {}
    v.capture_sell_sums = {}
    v.debeta_mtm = {}
    v.debeta_invsum = {}
    v.debeta_invn = {}
    v.debeta_drift = {}
    return v


def _logged(monkeypatch):
    lines = {"warning": [], "error": [], "info": []}
    for level in lines:
        monkeypatch.setattr(reward.bt.logging, level, lambda msg, *a, _l=level, **k: lines[_l].append(str(msg)))
    return lines


def test_warming_below_full_weight_stays_the_empty_map(monkeypatch):
    lines = _logged(monkeypatch)
    v = _validator(0.5)
    v._debeta_last = {"scores": {1: 0.8}, "detail": {}, "floor": 0.1, "w_make": 0.3, "ts": time.time()}
    assert reward.compute_debeta_scores(v) == {}, "a blend rung renormalises onto legacy; nothing to carry"
    assert not lines["warning"] and not lines["error"]


def test_full_weight_carries_the_previous_cycles_map(monkeypatch):
    lines = _logged(monkeypatch)
    v = _validator(1.0)
    v._debeta_last = {"scores": {1: 0.8, 2: 0.2}, "detail": {1: {"making": 0.5}}, "floor": 0.1,
                      "w_make": 0.3, "ts": time.time() - 30}
    scores = reward.compute_debeta_scores(v)
    assert scores == {1: 0.8, 2: 0.2}
    assert scores is not v._debeta_last["scores"], "a copy, so the caller cannot mutate the carried map"
    assert v.debeta_detail == {1: {"making": 0.5}} and v.debeta_floor == 0.1 and v.debeta_w_make == 0.3
    assert any("carrying the previous map" in ln for ln in lines["warning"]), lines


def test_a_stale_map_is_not_carried_and_the_zero_cycle_is_an_error(monkeypatch):
    lines = _logged(monkeypatch)
    v = _validator(1.0)
    v._debeta_last = {"scores": {1: 0.8}, "detail": {}, "floor": 0.1, "w_make": 0.3,
                      "ts": time.time() - reward._DEBETA_CARRY_MAX_S - 1}
    assert reward.compute_debeta_scores(v) == {}
    assert v.debeta_detail == {}
    assert any("every trading score is 0 this cycle" in ln for ln in lines["error"]), lines


def test_with_nothing_to_carry_the_zero_cycle_is_still_an_error(monkeypatch):
    lines = _logged(monkeypatch)
    assert reward.compute_debeta_scores(_validator(1.0)) == {}
    assert lines["error"]


def test_an_exception_at_full_weight_also_carries(monkeypatch):
    lines = _logged(monkeypatch)
    monkeypatch.setattr(reward.bt.logging, "exception", lambda *a, **k: None)
    monkeypatch.setattr(reward, "book_alphas_from_drift", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    v = _validator(1.0)
    v._debeta_last = {"scores": {1: 0.8}, "detail": {}, "floor": 0.1, "w_make": 0.3, "ts": time.time()}
    assert reward.compute_debeta_scores(v) == {1: 0.8}
    assert any("exception" in ln for ln in lines["warning"])


def test_a_successful_cycle_records_the_map_to_carry(monkeypatch):
    _logged(monkeypatch)
    v = _validator(1.0, min_books=0)
    monkeypatch.setattr(reward, "debeta_scores", lambda *a, **k: {1: 0.6})
    assert reward.compute_debeta_scores(v) == {1: 0.6}
    assert v._debeta_last["scores"] == {1: 0.6} and time.time() - v._debeta_last["ts"] < 5
