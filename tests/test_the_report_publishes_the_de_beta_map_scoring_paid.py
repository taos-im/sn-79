# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The report publishes the de-beta map the scoring of its boundary paid, not a recomputation of it.

The scoring service returns the map with its class summary and details, main keeps them with the boundary they belong
to, and the report reads them for its own boundary. It recomputes, without ending a handover, only when they belong to
another (a fallback cycle, a missed reply)."""
import inspect

from taos.im.validator import reward, scoring_shadow


class _V:
    pass


def _validator(stash_ts, interval=5):
    v = _V()
    v.config = type("C", (), {"scoring": type("S", (), {"interval": interval})()})()
    v._scored_debeta = {'ts': stash_ts, 'scores': {1: 0.5, 2: 0.25}, 'class_summary': {0: {'w': 1.0}},
                        'class_details': {0: {1: {'skill': 0.1}}}}
    v._debeta_class_summary = {0: {'w': 'recomputed'}}
    v._debeta_class_details = {0: {1: {'skill': 'recomputed'}}}
    return v


def test_the_map_scoring_paid_at_this_boundary_is_published_without_a_recomputation(monkeypatch):
    def _no(*a, **k):
        raise AssertionError("recomputed")
    monkeypatch.setattr(reward, "compute_debeta_scores", _no)
    v = _validator(stash_ts=20)
    assert reward.debeta_for_report(v, 22) == ({1: 0.5, 2: 0.25}, {0: {'w': 1.0}}, {0: {1: {'skill': 0.1}}})


def test_a_map_from_another_boundary_is_recomputed(monkeypatch):
    calls = []

    def _compute(self, end_handover=True, **k):
        calls.append(end_handover)
        return {1: 0.75}
    monkeypatch.setattr(reward, "compute_debeta_scores", _compute)
    v = _validator(stash_ts=15)
    assert reward.debeta_for_report(v, 22) == ({1: 0.75}, {0: {'w': 'recomputed'}}, {0: {1: {'skill': 'recomputed'}}})
    assert calls == [False]
    v._scored_debeta = None
    assert reward.debeta_for_report(v, 22)[0] == {1: 0.75}


def test_the_scoring_service_returns_the_map_and_main_keeps_it_with_its_boundary():
    src = inspect.getsource(scoring_shadow.shadow_score)
    tail = src[src.rindex("return {"):]
    for key in ("'debeta_scores'", "'debeta_class_summary'", "'debeta_class_details'"):
        assert key in tail, key
    from taos.im.neurons import validator
    vsrc = inspect.getsource(validator)
    assert "self._scored_debeta = {" in vsrc
    assert "debeta_for_report(self," in vsrc
    assert "compute_debeta_scores(self, end_handover=False)" not in vsrc
