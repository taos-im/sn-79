# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator books a fill twice: the notice path, from each uid's ET notice and the fee it carries, and the de-beta
capture, from the book's own fill event through fee_complete_sides. The capture books a side only when the fee its
notice needs is there; the second half is the per-state check that the two paths see the same fee: for every side the
capture books, the uid's notice for that trade must exist and carry the fill's fee. No miner can produce a disagreeing
fill, so the count is a check on the validator itself, kept as a running total and a per-state count and published as
validator gauges, expected at 0."""
import inspect

from taos.im.validator import trade


def _fill(i=11, **over):
    t = {"y": "t", "i": i, "p": 99.0, "q": 2.0, "s": 0, "Ma": 7, "Ta": 9, "Mf": 0.05, "Tf": 0.1}
    t.update(over)
    return t


def _notice(uid, i=11, book=3, **over):
    n = {"y": "ET", "b": book, "i": i, "a": 9, "Ma": 7, "Ta": 9, "Mf": 0.05, "Tf": 0.1, "q": 2.0, "p": 99.0, "s": 0}
    n.update(over)
    return n


def test_a_fill_whose_two_notices_carry_its_fees_agrees():
    notices = {7: [_notice(7)], 9: [_notice(9)]}
    assert trade.capture_fee_disagreements(3, [_fill()], notices) == 0


def test_a_notice_with_another_fee_is_a_disagreement_for_that_side_only():
    notices = {7: [_notice(7, Mf=0.07)], 9: [_notice(9)]}
    assert trade.capture_fee_disagreements(3, [_fill()], notices) == 1
    notices = {7: [_notice(7, Mf=0.07)], 9: [_notice(9, Tf=None)]}
    assert trade.capture_fee_disagreements(3, [_fill()], notices) == 2


def test_a_booked_side_without_a_notice_is_a_disagreement_and_an_unbooked_side_is_not():
    assert trade.capture_fee_disagreements(3, [_fill()], {7: [_notice(7)]}) == 1, "the taker has no notice"
    assert trade.capture_fee_disagreements(3, [_fill()], {7: [_notice(7, i=12)], 9: [_notice(9)]}) == 1, "another trade's notice"
    assert trade.capture_fee_disagreements(3, [_fill()], {7: [_notice(7, book=4)], 9: [_notice(9)]}) == 1, "another book's"
    unbooked = trade.fee_complete_sides(_fill(Mf=None))
    assert trade.capture_fee_disagreements(3, [unbooked], {9: [_notice(9)]}) == 0, "a side the capture dropped needs no notice"
    assert trade.capture_fee_disagreements(3, [_fill(Ta=-1)], {7: [_notice(7)]}) == 0, "a background taker has no notice"


def test_the_check_tolerates_float_noise_but_not_a_fee_difference():
    notices = {7: [_notice(7, Mf=0.05 + 1e-15)], 9: [_notice(9)]}
    assert trade.capture_fee_disagreements(3, [_fill()], notices) == 0
    notices = {7: [_notice(7, Mf=0.0500001)], 9: [_notice(9)]}
    assert trade.capture_fee_disagreements(3, [_fill()], notices) == 1


def test_the_live_path_runs_the_check_per_book_in_simulation_mode_and_publishes_it():
    src = inspect.getsource(trade.update_trade_volumes)
    assert "_fee_check = self.capture_fee_check = {}" in src and "_fee_check['state'] = 0" in src
    assert "capture_fee_disagreements(bookId, de_trades, notices)" in src
    from taos.im.validator import scoring_shadow
    assert "capture_fee_check" in scoring_shadow._DEBETA_PLAIN, "the counts travel to the scoring child with the INIT"
    assert "if de_trades and not _debeta_exchange:" in src, "the exchange books the notices themselves, so there is nothing to compare"
    from taos.im.validator import report
    rsrc = inspect.getsource(report)
    assert '"capture_fee_disagreements"' in rsrc and '"capture_fee_disagreements_state"' in rsrc
