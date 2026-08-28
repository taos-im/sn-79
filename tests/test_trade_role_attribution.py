# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Order role must come from the engine, not be assumed.

The engine states which side an agent was on, via the LOB signals:

    signal_index is not None  -- derived from a new instruction submitted this batch
    signal_index is None      -- trade-feed signal: a resting LOB order that was filled
                                 by a crossing aggressor

So there are three cases, and only the first has two agents in it:

    aggressor crosses another agent's resting order   maker = resting agent, taker = aggressor
    marketable order swept against the pool           maker = POOL,          taker = agent
    resting order filled as the pool price moves      maker = agent,         taker = POOL

Settlement is always against the pool and funds never move agent to agent, so "the pool" is a
real party here, represented by an absent uid: exactly one side is absent, which is what makes
it unambiguous. Recording the agent as taker in the third case states the opposite of what
happened, and recording it as its own maker in the second states something impossible.
"""
from taos.im.validator.engines import resolve_trade_roles


def test_aggressing_fill_against_the_pool():
    """signal_index set, no distinct counterparty: the agent took, the pool made."""
    maker, taker = resolve_trade_roles(uid=171, signal_idx=3, maker_candidates=[])
    assert taker == 171
    assert maker is None


def test_resting_fill_makes_the_agent_the_maker():
    """signal_index None: the agent's resting order was filled, so it is the MAKER."""
    maker, taker = resolve_trade_roles(uid=171, signal_idx=None, maker_candidates=[])
    assert maker == 171
    assert taker is None, "the counterparty is the pool, not the resting agent itself"


def test_agent_to_agent_crossing_names_both():
    maker, taker = resolve_trade_roles(uid=171, signal_idx=3, maker_candidates=[208])
    assert taker == 171
    assert maker == 208


def test_taker_is_never_its_own_maker():
    for sidx in (None, 0, 7):
        maker, taker = resolve_trade_roles(uid=171, signal_idx=sidx, maker_candidates=[171])
        assert not (maker == taker and maker is not None)


def test_exactly_one_side_is_absent_for_a_pool_fill():
    for sidx in (None, 5):
        maker, taker = resolve_trade_roles(uid=171, signal_idx=sidx, maker_candidates=[])
        assert (maker is None) != (taker is None), "exactly one side is the pool"


def test_candidate_equal_to_the_agent_is_not_a_counterparty():
    """A self candidate means no counterparty was found, so the pool made the trade."""
    maker, taker = resolve_trade_roles(uid=171, signal_idx=1, maker_candidates=[171, 208])
    assert taker == 171
    assert maker == 208


def test_signal_index_zero_is_still_an_instruction():
    """0 is a valid signal index; only None means a resting fill."""
    maker, taker = resolve_trade_roles(uid=171, signal_idx=0, maker_candidates=[])
    assert taker == 171 and maker is None


def test_engine_counterparty_overrides_the_local_candidate_list():
    """The engine holds this fact; a locally derived candidate must not win over it."""
    maker, taker = resolve_trade_roles(
        uid=171, signal_idx=3, maker_candidates=[999], counterparty=208)
    assert taker == 171 and maker == 208


def test_engine_counterparty_on_a_resting_fill_names_the_aggressor():
    maker, taker = resolve_trade_roles(
        uid=171, signal_idx=None, maker_candidates=[], counterparty=208)
    assert maker == 171, "the resting agent is the maker"
    assert taker == 208, "the aggressor that crossed it is the taker"


def test_engine_counterparty_equal_to_self_is_ignored():
    """A counterparty equal to the agent cannot be real; fall back rather than self-trade."""
    maker, taker = resolve_trade_roles(
        uid=171, signal_idx=3, maker_candidates=[], counterparty=171)
    assert taker == 171 and maker is None
