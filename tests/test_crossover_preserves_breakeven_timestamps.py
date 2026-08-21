"""A breakeven round must survive the simulation crossover.

THE DEFECT. shift_simulation_histories rebuilt the PnL history with a nested
`for book_id, pnl in books.items()`, which never creates the timestamp key when `books` is `{}`.
An empty books-dict is exactly what a scored interval with no closed position looks like -- a
breakeven round -- and process_trade_notices writes one for every processed interval.

WHY IT MATTERS AT THE CROSSOVER ONLY. Those timestamps count toward the kappa assessment span in-sim.
Dropping them at the crossover can collapse a miner's span below min_lookback, so kappa returns None and
the trading score becomes 0; for an established miner that then drags its track-record EMA down. In-sim
nothing is wrong, so the damage appears only on the first scoring round after a simulation restart.

THE CONTRACT. The crossover is a pure time-rebase of the exact in-sim history: every timestamp inside the
window survives with its books-dict intact, empty or not.
"""
from collections import defaultdict

import pytest

from taos.im.validator.trade import shift_simulation_histories


class _Container:
    """Minimal stand-in carrying only what shift_simulation_histories touches."""

    def __init__(self, pnl_history):
        self.realized_pnl_history = pnl_history
        self._last_prune_timestamp = 12345
        self.trade_volumes = {}
        self.roundtrip_volumes = {}
        self.inventory_history = {}
        self.recent_trades = {}
        self.recent_miner_trades = {}
        self.open_positions = defaultdict(lambda: defaultdict(dict))
        self.kappa_values = {}
        self.activity_factors = {}
        self.pnl_factors = {}
        self.initial_balances = {}
        self.agent_pnl_by_book = defaultdict(lambda: defaultdict(float))
        self.agent_pnl_total = defaultdict(float)
        for name in ("volume_sums", "maker_volume_sums", "taker_volume_sums",
                     "self_volume_sums", "roundtrip_volume_sums", "fee_sums"):
            setattr(self, name, defaultdict(lambda: defaultdict(float)))


def _shift(pnl_history, *, old_ts, new_ts, lookback):
    c = _Container(pnl_history)
    shift_simulation_histories(
        c, old_ts, new_ts,
        book_count=2, volume_decimals=4, lookback=lookback,
        volume_assessment_period=lookback, miner_wealth={}, effective_max_uids=3,
    )
    return c.realized_pnl_history


OLD_TS = 10_000
NEW_TS = 1_000_000
LOOKBACK = 1_000_000        # wide enough that nothing is pruned for age


def test_breakeven_timestamp_survives_the_crossover():
    # uid 1 traded at t=9000 (a real PnL) and had a BREAKEVEN interval at t=9500 ({} books)
    hist = {1: {9000: {0: 5.0}, 9500: {}}}
    out = _shift(hist, old_ts=OLD_TS, new_ts=NEW_TS, lookback=LOOKBACK)
    shifted_real = NEW_TS - (OLD_TS - 9000)
    shifted_breakeven = NEW_TS - (OLD_TS - 9500)
    assert shifted_real in out[1], "a scored interval with PnL must survive"
    assert shifted_breakeven in out[1], (
        "the BREAKEVEN interval must survive too: it counts toward the kappa span, and dropping it "
        "can collapse the span below min_lookback -> kappa None -> score 0"
    )
    assert out[1][shifted_breakeven] == {}, "it must remain an empty books-dict, not be invented"


def test_span_length_is_preserved_across_the_crossover():
    # three breakeven intervals plus one real: the SPAN is four, and must stay four
    hist = {2: {9000: {}, 9200: {}, 9400: {}, 9600: {1: -2.5}}}
    out = _shift(hist, old_ts=OLD_TS, new_ts=NEW_TS, lookback=LOOKBACK)
    assert len(out[2]) == 4, f"span collapsed from 4 to {len(out[2])} across the crossover"


def test_all_breakeven_uid_is_not_erased_entirely():
    # a miner whose whole window was breakeven still has a span; it must not vanish
    hist = {1: {9000: {}, 9500: {}}}
    out = _shift(hist, old_ts=OLD_TS, new_ts=NEW_TS, lookback=LOOKBACK)
    assert len(out[1]) == 2


def test_values_are_rebased_not_altered():
    hist = {1: {9000: {0: 5.0, 1: -1.25}}}
    out = _shift(hist, old_ts=OLD_TS, new_ts=NEW_TS, lookback=LOOKBACK)
    shifted = NEW_TS - (OLD_TS - 9000)
    assert out[1][shifted] == {0: 5.0, 1: -1.25}, "a rebase must not change the PnL values"


def test_entries_outside_the_lookback_are_still_pruned():
    # the fix must not disable pruning: an entry older than the window is dropped as before
    hist = {1: {10: {}, 9500: {0: 1.0}}}
    out = _shift(hist, old_ts=OLD_TS, new_ts=NEW_TS, lookback=5_000)
    assert (NEW_TS - (OLD_TS - 10)) not in out.get(1, {}), "aged-out entries must still prune"
    assert (NEW_TS - (OLD_TS - 9500)) in out[1]
