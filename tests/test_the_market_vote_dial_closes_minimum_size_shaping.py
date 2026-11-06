# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""scoring.debeta.capture_market_votes (default off). Off, any print that clears a fill's dust floor votes in its
capture mid, so two miners trading minimum-size lots to each other off the price move an honest maker's captured
spread. On, only prints with the background market on one side vote, over the same span (the 15 nearest prints a side
that clear the floor), reaching on until CAPTURE_MARKET_MIN such prints a side vote: no print between two miners moves
anyone's mid at any size (it still takes a place in the span, so the market prints the mid averages shift a little).
Without miner-to-miner prints the two settings read the same mid. The batch and live paths agree whatever the
batching, and the dial reaches the scorer from config."""
import copy
from collections import defaultdict
from types import SimpleNamespace

from taos.im.validator.debeta import (
    CAPTURE_MARKET_MIN, accumulate_book_capture, centered_mid_floored, flush_capture_state, running_floors,
)

W = 15
BOOK = 97
MAKER = 3
M1, M2 = 9, 10
BG, BG2 = -2, -3
MIN_SIZE = 2.5364


def _mk(price, q=10.0, s=0, ma=BG2, ta=BG):
    return {"p": price, "q": q, "s": s, "Ma": ma, "Ta": ta}


def _stream(n=200):
    """The background prints around 20 and a maker is filled by it 2 bp inside on alternating sides."""
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(_mk(20.0 + 0.0004 * (((i * 7) % 5) - 2), q=10.0 + (i % 4)))
        elif (i // 2) % 2 == 0:
            out.append(_mk(19.996, q=10.0, s=1, ma=MAKER))   # the maker buys from a background seller
        else:
            out.append(_mk(20.004, q=10.0, s=0, ma=MAKER))   # the maker sells to a background buyer
    return out


def _shaped(trades, every=3, n=2, dev=0.01, q=MIN_SIZE):
    """Two miners trade `n` minimum-size lots to each other `dev` over the price after every `every`-th print."""
    out = []
    for i, t in enumerate(trades):
        out.append(t)
        if i % every == 0:
            out += [_mk(20.0 * (1.0 + dev), q=q, s=k % 2, ma=M1 if k % 2 else M2, ta=M2 if k % 2 else M1)
                    for k in range(n)]
    return out


def _run(seq, market, batch=None):
    buy, sell = defaultdict(dict), defaultdict(dict)
    if batch is None:
        accumulate_book_capture(buy, sell, BOOK, seq, W, ts=0, min_size=MIN_SIZE, market_votes=market)
        return buy, sell
    state = {}
    for k in range(0, len(seq), batch):
        accumulate_book_capture(buy, sell, BOOK, seq[k:k + batch], W, ts=k, mid_state=state, min_size=MIN_SIZE,
                                market_votes=market)
    flush_capture_state(state, buy, sell, W, ts=len(seq), force=True)
    return buy, sell


def _two_sided(buy, sell, uid=MAKER):
    return min(buy[uid].get(BOOK, 0.0), sell[uid].get(BOOK, 0.0))


def _mids(seq, market):
    sizes = [float(t["q"]) for t in seq]
    bg = [t["Ma"] < 0 or t["Ta"] < 0 for t in seq]
    return centered_mid_floored([t["p"] for t in seq], sizes, running_floors(sizes, bg, MIN_SIZE), W, bg,
                                market=market)


def test_off_minimum_size_prints_between_two_miners_take_a_makers_spread_and_on_their_price_moves_nothing():
    clean = _stream()
    base = _two_sided(*_run(clean, False))
    assert base > 0.5, "the fixture maker earns a real two-sided spread"
    assert abs(_two_sided(*_run(clean, True)) - base) < 1e-9, "without miner prints the dial changes nothing"
    for n in (1, 2, 8):
        assert _two_sided(*_run(_shaped(clean, n=n, dev=0.01), False)) < 0.0, f"off: {n} prints 1% over take it all"
        on = {dev: _two_sided(*_run(_shaped(clean, n=n, dev=dev), True)) for dev in (0.0, 0.01, 0.05)}
        # on: the miners' prints vote for no one, so where they trade cannot matter, only that they take places in
        # the span (fewer market prints in the mean): at 8 prints after every third the maker keeps over 85%
        assert max(on.values()) - min(on.values()) < 1e-9, (n, on)
        assert on[0.01] > 0.85 * base, (n, on[0.01], base)
        assert abs(_two_sided(*_run(_shaped(clean, n=n, dev=0.01), True, batch=5)) - on[0.01]) < 1e-9, n


def test_on_a_print_between_two_miners_moves_no_mid_and_a_print_with_the_market_does():
    clean = _stream(80)
    i = 41
    with_miners = clean[:i] + [_mk(21.0, q=10.0, ma=M1, ta=M2)] + clean[i:]
    with_market = clean[:i] + [_mk(21.0, q=10.0, ma=M1, ta=BG)] + clean[i:]
    base = _mids(clean, True)[i]
    assert abs(_mids(with_miners, True)[i + 1] - base) < 0.0005, "the miners' print is not a voter"
    assert _mids(with_market, True)[i + 1] - base > 0.01, "a print with the background on one side votes"
    assert _mids(with_miners, False)[i + 1] - base > 0.01, "off, the miners' print votes"


def test_on_the_span_is_kept_and_a_side_short_of_market_votes_reaches_for_them():
    clean = _stream(200)
    i = 101
    assert abs(_mids(clean, True)[i] - _mids(clean, False)[i]) < 1e-12, "no miner prints: the same mid"
    # the fill's backward neighbours are all miner prints for 20 places: off, they vote; on, the side reaches past
    # them for CAPTURE_MARKET_MIN market prints, so the mid stays the market's
    wall = [_mk(21.0, q=10.0, ma=M1, ta=M2) for _ in range(20)]
    seq = clean[:i] + wall + clean[i:]
    j = i + len(wall)
    assert abs(_mids(seq, True)[j] - 20.0) < 0.002 and _mids(seq, False)[j] > 20.3
    assert CAPTURE_MARKET_MIN >= 1


def test_on_a_fill_between_two_miners_books_against_the_markets_mid():
    seq = _stream(120)
    seq.insert(60, _mk(20.2, q=MIN_SIZE, s=1, ma=M1, ta=M2))   # M1 buys 1% over the market from M2
    buy, sell = _run(seq, True)
    assert abs(buy[M1][BOOK] - (20.0 - 20.2) * MIN_SIZE) < 0.005, "graded against the market's 20, not its own print"
    assert abs(buy[M1][BOOK] + sell[M2][BOOK]) < 1e-12, "zero-sum per fill"


def test_on_the_live_path_agrees_with_the_batch_path_with_miner_prints_present():
    seq = _shaped(_stream(300), every=4, n=3)
    whole = _run(seq, True)
    for b in (1, 7, 40):
        live = _run(seq, True, batch=b)
        for uid in (MAKER, M1, M2):
            for side in (0, 1):
                assert abs(whole[side][uid].get(BOOK, 0.0) - live[side][uid].get(BOOK, 0.0)) < 1e-9, (b, uid, side)


def test_a_window_persisted_without_the_vote_flags_continues():
    seq = _stream(120)
    state = {}
    buy, sell = defaultdict(dict), defaultdict(dict)
    accumulate_book_capture(buy, sell, BOOK, seq[:60], W, ts=0, mid_state=state, min_size=MIN_SIZE, market_votes=True)
    old = copy.deepcopy(state)
    old[BOOK].pop("bgs", None)    # a window saved before the vote flags existed
    b2, s2 = copy.deepcopy(buy), copy.deepcopy(sell)
    accumulate_book_capture(buy, sell, BOOK, seq[60:], W, ts=1, mid_state=state, min_size=MIN_SIZE, market_votes=True)
    accumulate_book_capture(b2, s2, BOOK, seq[60:], W, ts=1, mid_state=old, min_size=MIN_SIZE, market_votes=True)
    assert len(old[BOOK]["bgs"]) == len(old[BOOK]["prices"]), "the flags are padded to the carried window"
    assert abs(_two_sided(buy, sell) - _two_sided(b2, s2)) < 1e-9, "carried background prints keep their votes"


def test_the_dial_reaches_the_capture_accumulator_from_config(monkeypatch):
    import test_shadow_snapshot_completeness as tsc
    from taos.im.validator import trade

    seen = []
    real = trade.accumulate_book_capture

    def spy(*a, **kw):
        seen.append(kw.get("market_votes"))
        return real(*a, **kw)

    monkeypatch.setattr(trade, "accumulate_book_capture", spy)
    for setting, expected in ((0, False), (1, True)):
        seen.clear()
        main = tsc.tss._fresh_main()
        main.config.scoring.debeta = SimpleNamespace(capture_market_votes=setting)
        for st in tsc._debeta_rounds(count=2):
            trade.update_trade_volumes(main, st)
        assert seen and set(seen) == {expected}, (setting, seen)
