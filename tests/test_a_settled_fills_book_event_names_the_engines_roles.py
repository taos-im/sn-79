# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The book's own trade event for a settled exchange fill names the roles the engine stated.

Each block the validator appends a 't' event per settled fill to the book it publishes (GenTRX inputs, the ingest
books, the recent-trades panel). Its roles came from a trade event of the same id, which the asynchronous settlement
never supplies, so every fill read as the settling agent making against nobody: a taker's pool fill was published as a
maker fill. The executed fill already carries the roles (reconcile.resolve_trade_roles, the same answer the miner's
notice reads), and the event's side is the taker's direction, as every reader of 's' takes it.
"""
from types import SimpleNamespace

from taos.im.neurons import validator as validator_mod

_event = validator_mod._settled_fill_event
UID, BOOK, BUY, SELL = 171, 5, 0, 1


def _fill(**kw):
    base = dict(netuid=BOOK, trade_id="5:430", agent_id=UID, price=0.0136, alpha_volume=3.6772, direction=BUY)
    base.update(kw)
    return base


def test_a_taker_pool_fill_reads_as_the_agent_taking():
    ev = _event(_fill(taker_uid=UID, maker_uid=None), {}, 1)
    assert (ev["Ta"], ev["Ma"]) == (UID, None)
    assert ev["s"] == BUY, "the side is the taker's own direction"


def test_a_resting_order_filled_by_the_pool_reads_as_the_agent_making():
    ev = _event(_fill(taker_uid=None, maker_uid=UID, direction=BUY), {}, 1)
    assert (ev["Ta"], ev["Ma"]) == (None, UID)
    assert ev["s"] == SELL, "the pool sold into the agent's resting buy, so the taker's side is SELL"


def test_a_matching_trade_event_still_decides():
    te = SimpleNamespace(taker_uid=208, maker_uid=UID)
    ev = _event(_fill(taker_uid=None, maker_uid=UID), {"5:430": te}, 1)
    assert (ev["Ta"], ev["Ma"]) == (208, UID)


def test_a_fill_record_without_roles_keeps_the_old_reading():
    ev = _event(_fill(), {}, 7)
    assert (ev["Ta"], ev["Ma"], ev["s"], ev["t"]) == (None, UID, BUY, 7)
    assert (ev["y"], ev["nid"], ev["i"], ev["p"], ev["q"]) == ("t", BOOK, "5:430", 0.0136, 3.6772)


# The settled fill's ET notice, which scoring reads from state.notices, comes from the same roles. Without a matching
# trade event it named the settling agent as the taker, so a resting order filled by the pool was booked as a taker's
# fill and lost its making credit. A record without roles keeps that old reading.
_notice = validator_mod._settled_fill_notice


def test_the_scoring_notice_of_a_resting_fill_names_the_agent_as_maker():
    notice, uids = _notice(_fill(taker_uid=None, maker_uid=UID, direction=BUY), {})
    assert (notice["Ta"], notice["Ma"]) == (None, UID)
    assert notice["s"] == SELL
    assert uids == {UID}


def test_the_scoring_notice_of_a_taker_fill_names_the_agent_as_taker():
    notice, uids = _notice(_fill(taker_uid=UID, maker_uid=None), {})
    assert (notice["Ta"], notice["Ma"], notice["s"]) == (UID, None, BUY)
    assert uids == {UID}


def test_a_book_match_notifies_both_agents_once():
    notice, uids = _notice(_fill(taker_uid=208, maker_uid=UID), {})
    assert uids == {208, UID}


def test_the_scoring_notice_of_a_fill_record_without_roles_keeps_the_old_reading():
    notice, uids = _notice(_fill(), {})
    assert (notice["Ta"], notice["Ma"], notice["s"]) == (UID, None, BUY)
    assert notice["i"] == "5:430" and notice["q"] == 3.6772
