# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A second mark is accumulated beside the paid one and its skill is published per uid, so a change of mark is judged
on live data before it pays (the captured and realized making bases' pattern of 0.6.3).

scoring.debeta.mark_mode_alt (default mid) names the other mark; the trade path keeps a second set of mark-to-market
accumulators under it (mtm, inventory time-sum, print count, drift, with their windowed histories, pruned and shifted
with the paid set), the scorer derives the alternative skill with the paid leg's floor rule and bar, and the report
publishes debeta_skill_mark_alt and debeta_skill_net_alpha_mark_alt per uid. Pay reads the paid mark alone. The leg
is skipped when the alternative names the paid mark.
"""
import inspect
from types import SimpleNamespace

from taos.im.config import debeta_defaults
from taos.im.validator import trade, report, reward


def _t(p, q, s, ma, ta):
    return {"p": p, "q": q, "s": s, "Ma": ma, "Ta": ta}


def test_the_dial_defaults_to_mid_and_the_leg_names_are_classified():
    assert debeta_defaults()["mark_mode_alt"] == "mid"
    for n in ("debeta_mtm_alt", "debeta_invsum_alt", "debeta_invn_alt", "debeta_drift_alt", "debeta_inv_alt",
              "debeta_pfirst_alt", "debeta_plast_alt", "debeta_mark_state_alt",
              "debeta_mtm_alt_hist", "debeta_invsum_alt_hist", "debeta_invn_alt_hist", "debeta_drift_alt_hist"):
        assert n in trade.ALT_MARK_STATE, n


def test_the_trade_path_accumulates_both_marks_from_one_batch():
    v = SimpleNamespace()
    trade.init_alt_mark_state(v)
    book = {"b": [{"p": 100.0, "q": 1.0}], "a": [{"p": 100.2, "q": 1.0}]}
    # uid 5 buys at the bid then the mid stands still while a print lands at the ask: the paid (last) leg marks the
    # bounce, the alternative (mid) leg books nothing
    trade.accumulate_alt_mark(v, 0, [_t(100.0, 1.0, 1, 5, -1)], book, "mid", 0, ts=1)
    trade.accumulate_alt_mark(v, 0, [_t(100.2, 1.0, 0, -1, 7)], book, "mid", 0, ts=2)
    assert v.debeta_mtm_alt[5][0] == 0.0 and v.debeta_mark_state_alt[0]["prev"] == 100.1
    assert v.debeta_invn_alt[0] == 2
    assert "accumulate_alt_mark(self, bookId, de_trades, book, _mark_alt, _mark_window, ts=sampled_timestamp)" in \
        inspect.getsource(trade.update_trade_volumes)
    assert "if _mark_alt not in ('none', _mark_mode):" in inspect.getsource(trade.update_trade_volumes)


def test_the_alt_leg_is_pruned_shifted_and_reset_with_the_paid_one():
    src = inspect.getsource(trade)
    for name in ("debeta_mtm_alt_hist", "debeta_invsum_alt_hist", "debeta_invn_alt_hist", "debeta_drift_alt_hist"):
        assert src.count(name) >= 4, f"{name}: init, prune, shift and restore"
    assert "self.debeta_mark_state_alt = {}" in src and "'debeta_mtm_alt'" in inspect.getsource(trade.reset_agent_histories)
    from taos.im.validator import persistence, scoring_shadow
    assert "debeta_mtm_alt_hist" in persistence._DEBETA_HIST3 and "debeta_drift_alt_hist" in persistence._DEBETA_HIST2
    assert "debeta_mtm_alt" in scoring_shadow._DEBETA_DD2 and "debeta_mark_state_alt" in scoring_shadow._DEBETA_PLAIN


def test_the_scorer_publishes_the_alternative_skill_per_uid():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("pg", pathlib.Path(__file__).with_name("test_debeta_presence_gate.py"))
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    _fake_validator = pg._fake_validator
    v = _fake_validator()
    trade.init_alt_mark_state(v)
    for b in range(6):
        for k in range(1, 6):
            trade.accumulate_alt_mark(v, b, [_t(100.0 + 0.5 * k, 1.0, 0, -1, 1), _t(100.0 + 0.5 * k, 1.0, 1, 2, -1)],
                                      {"b": [{"p": 99.9 + 0.5 * k, "q": 1}], "a": [{"p": 100.1 + 0.5 * k, "q": 1}]}, "mid", 0, ts=k)
    scores = reward.compute_debeta_scores(v)
    assert scores
    d = v._debeta_last["detail"] if isinstance(getattr(v, "_debeta_last", None), dict) and "detail" in v._debeta_last else v.debeta_detail
    assert any("skill_mark_alt" in dd for dd in d.values())
    assert all("skill_net_alpha_mark_alt" in dd for dd in d.values() if "skill_mark_alt" in dd)
    rsrc = inspect.getsource(report)
    assert "('skill_mark_alt', 'debeta_skill_mark_alt')" in rsrc and "('skill_net_alpha_mark_alt', 'debeta_skill_net_alpha_mark_alt')" in rsrc
