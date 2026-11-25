# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A synchronous block is many uids of one coldkey taking on the same side within one state across the books: one
process on many slots, with the capital to move a thin class in a second. The skill half marks a position to the
later mid, so a block would earn the fall it makes. Under the block dials its members' block fills wait in a pending
list for the impact window and join their marked inventory at the mark then: the move in between is not their alpha,
the fill's own charge at the mark still is, and everything after the window is marked as before. A single uid is
never a block, the other side's fills are not deferred, the pending list travels in the saved state and the scoring
service's INIT, and with the dial off nothing changes.
"""
from collections import defaultdict

from taos.im.validator import debeta, persistence, scoring_shadow

CK = {u: ("A" if u < 20 else "B") for u in range(40)}
NS = 1_000_000_000


def _t(p, q, s, ma, ta):
    return {"p": p, "q": q, "s": s, "Ma": ma, "Ta": ta, "Mf": 0.0, "Tf": 0.0}


def test_a_block_is_many_uids_of_one_coldkey_on_one_side_in_one_state():
    batches = {b: [_t(300.0, 1.0, 1, 30, u) for u in range(17)] for b in range(3)}     # 17 of A sell in 3 books
    block = debeta.block_uids_in(batches, CK.get, 8)
    assert set(block) == set(range(17)) and set(block.values()) == {1}
    batches[0].append(_t(300.0, 50.0, 1, 31, 25))                                       # one uid of B, huge: no block
    assert 25 not in debeta.block_uids_in(batches, CK.get, 8)
    assert debeta.block_uids_in({0: [_t(300.0, 1.0, 1, 30, u) for u in range(5)]}, CK.get, 8) == {}
    assert debeta.block_uids_in(batches, CK.get, 8, min_notional=1e9) == {}, "the notional bar"
    assert debeta.block_uids_in(batches, CK.get, 0) == {} and debeta.block_uids_in(batches, None, 8) == {}
    buys = {0: [_t(300.0, 1.0, 0, 30, u) for u in range(8)]}
    assert set(debeta.block_uids_in(buys, CK.get, 8).values()) == {0}, "a buy block carries its side"


def _state():
    return (defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float)), {},
            defaultdict(lambda: defaultdict(float)), {}, {})


def _run(block, window_ns, pending):
    mtm, invsum, invn, inv, pf, pl = _state()
    ms = {}
    kw = dict(mark_state=ms, mark_mode="mid", fill_charge="none", block_uids=block, impact_window_ns=window_ns,
              pending=pending)
    # ts 0: the block sells 10 to maker 3 at 100 with the mid at 100
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(100.0, 10.0, 1, 3, 7)], ts=0, mark_mid=100.0, **kw)
    # ts 30 s: the mid falls to 99 on a background trade
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(99.0, 1.0, 0, -1, -2)], ts=30 * NS, mark_mid=99.0, **kw)
    gain_30 = mtm[7].get(0, 0.0)
    # ts 60 s: the release (the window is 60 s), the mid at 99
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(99.0, 1.0, 0, -1, -2)], ts=60 * NS, mark_mid=99.0, **kw)
    # ts 90 s: the mid falls to 98
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(98.0, 1.0, 0, -1, -2)], ts=90 * NS, mark_mid=98.0, **kw)
    return gain_30, mtm[7].get(0, 0.0), mtm[3].get(0, 0.0), inv[0].get(7, 0.0)


def test_the_blocks_fills_earn_nothing_inside_the_window_and_everything_after():
    pending = {}
    gain_30, gain_90, maker, inv7 = _run({7: 1}, 60 * NS, pending)
    assert gain_30 == 0.0, "the move inside the window is not the block's"
    assert gain_90 == 10.0, "after the release the short earns the next move (10 x 1)"
    assert maker == -20.0, "the maker's long is marked throughout"
    assert inv7 == -10.0 and pending == {}, "the fill joined the inventory at the release"
    gain_30, gain_90, maker, inv7 = _run(None, 0, None)
    assert gain_30 == 10.0 and gain_90 == 20.0, "without the rule the short earns the whole fall"


def test_the_other_side_and_other_uids_are_booked_at_once():
    mtm, invsum, invn, inv, pf, pl = _state()
    pending = {}
    kw = dict(mark_state={}, mark_mode="mid", fill_charge="none", block_uids={7: 1}, impact_window_ns=60 * NS, pending=pending)
    # uid 7 BUYS (side 0) while its block side is the sell side: booked at once; uid 9 is no member: booked at once
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(100.0, 5.0, 0, 3, 7), _t(100.0, 4.0, 1, 3, 9)],
                               ts=0, mark_mid=100.0, **kw)
    assert inv[0][7] == 5.0 and inv[0][9] == -4.0 and pending == {}


def test_the_pending_list_travels_in_the_state_and_the_init():
    v = type("V", (), {})()
    v.debeta_pending_inv = {5: [(120 * NS, 7, -10.0)]}
    v.debeta_inv = {}
    v.debeta_plast = {}
    saved = persistence.build_debeta_state(v)
    assert saved["debeta_pending_inv"] == {5: [[120 * NS, 7, -10.0]]}
    assert "debeta_pending_inv" in scoring_shadow._DEBETA_PLAIN, "the INIT ships it to the scoring service"
    src = open(persistence.__file__).read()
    assert 'validator_state.get("debeta_pending_inv")' in src


def test_the_dials_ship_on_at_the_confirmed_settings():
    from taos.im.config import debeta_defaults
    d = debeta_defaults()
    assert d["block_min_uids"] == 8 and d["block_min_notional"] == 1_000_000.0 and d["block_impact_window_s"] == 120.0


def test_the_taker_window_defers_every_takers_fill_and_the_dial_ships_off():
    mtm, invsum, invn, inv, pf, pl = _state()
    pending = {}
    kw = dict(mark_state={}, mark_mode="mid", fill_charge="none", taker_window_ns=60 * NS, pending=pending)
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(100.0, 10.0, 1, 3, 9)], ts=0, mark_mid=100.0, **kw)
    assert inv[0].get(9, 0.0) == 0.0 and inv[0][3] == 10.0 and pending[0] == [(60 * NS, 9, -10.0)]
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(99.0, 1.0, 0, -1, -2)], ts=30 * NS, mark_mid=99.0, **kw)
    assert mtm[9].get(0, 0.0) == 0.0 and mtm[3][0] == -10.0
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(99.0, 1.0, 0, -1, -2)], ts=60 * NS, mark_mid=99.0, **kw)
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(98.0, 1.0, 0, -1, -2)], ts=90 * NS, mark_mid=98.0, **kw)
    assert mtm[9][0] == 10.0 and inv[0][9] == -10.0 and pending == {}
    from taos.im.config import debeta_defaults
    assert debeta_defaults()["taker_impact_window_s"] == 0.0



def test_the_release_follows_the_states_own_clock_not_the_sampling_step():
    """`ts` is the volume-sampling stamp (the state's time floored to the sampling interval, 600 s live), so a
    window keyed on it would release at the next step whatever its length; the pending list keys on the state's own
    clock instead, and the validator passes it."""
    import inspect
    from taos.im.validator import trade
    mtm, invsum, invn, inv, pf, pl = _state()
    pending, ms = {}, {}
    kw = dict(mark_state=ms, mark_mode="mid", fill_charge="none", block_uids={7: 1}, impact_window_ns=60 * NS,
              pending=pending)
    # four states inside one sampling step (ts stays 0); the state's own clock advances
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(100.0, 10.0, 1, 3, 7)], ts=0, clock_ns=0,
                               mark_mid=100.0, **kw)
    assert pending[0][0][0] == 60 * NS, "the release time is the state's clock plus the window"
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(99.0, 1.0, 0, -1, -2)], ts=0, clock_ns=30 * NS,
                               mark_mid=99.0, **kw)
    assert mtm[7].get(0, 0.0) == 0.0 and 0 in pending, "still pending at 30 s"
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(99.0, 1.0, 0, -1, -2)], ts=0, clock_ns=60 * NS,
                               mark_mid=99.0, **kw)
    assert 0 not in pending and inv[0].get(7, 0.0) == -10.0, "released at 60 s on the state's clock, same step"
    debeta.accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, 0, [_t(98.0, 1.0, 0, -1, -2)], ts=0, clock_ns=90 * NS,
                               mark_mid=98.0, **kw)
    assert mtm[7].get(0, 0.0) > 0.0, "the move after the release is the block's"
    assert "clock_ns=timestamp" in inspect.getsource(trade.update_trade_volumes), "the validator passes its clock"
