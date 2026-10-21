# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Making capture must read --scoring.debeta.centered_window.

A non-default window changes the live capture window. The default, and any
non-positive flag, leave capture matching CAPTURE_W.
"""
from collections import defaultdict, deque
from types import SimpleNamespace

from taos.im.validator.debeta import accumulate_book_capture, flush_capture_state
from taos.im.validator.trade import CAPTURE_W, _centered_window, update_trade_volumes

S = 1_000_000_000
BOOK = 0
TS = 1000 * S
SAMPLED = (TS // (600 * S)) * (600 * S)


def _self(centered_window=None):
    debeta = SimpleNamespace()
    if centered_window is not None:
        debeta.centered_window = centered_window
    return SimpleNamespace(config=SimpleNamespace(scoring=SimpleNamespace(debeta=debeta)))


def _prints(n=80):
    """Print time equals the update clock so age-flush does not hide W: a fill waits for W
    forward prints, so the live window (pending count and kept prices) is W itself."""
    return [
        {"y": "t", "p": 100.0, "q": 1.0, "s": 0, "Ma": 1, "Ta": 2,
         "i": i, "Ti": 0, "Mi": 0, "Tf": 0.0, "Mf": 0.0, "t": TS}
        for i in range(n)
    ]


def _window(s):
    st = s.debeta_capture_mid[BOOK]
    return (int(st["W"]), len(st["pend"]))


def _validator(centered_window=None):
    def boom(msg, **k):
        raise AssertionError(f"pagerduty_alert hit (fixture infidelity): {msg}")

    debeta = SimpleNamespace()
    if centered_window is not None:
        debeta.centered_window = centered_window
    return SimpleNamespace(
        simulation=SimpleNamespace(volumeDecimals=4),
        config=SimpleNamespace(scoring=SimpleNamespace(
            activity=SimpleNamespace(
                trade_volume_sampling_interval=600 * S,
                trade_volume_assessment_period=3600 * S,
            ),
            kappa=SimpleNamespace(lookback=10800 * S),
            debeta=debeta,
        )),
        step=1,
        effective_max_uids=3,
        engine=SimpleNamespace(mode="simulation"),
        trade_volumes={},
        volume_sums=defaultdict(lambda: defaultdict(float)),
        maker_volume_sums=defaultdict(lambda: defaultdict(float)),
        taker_volume_sums=defaultdict(lambda: defaultdict(float)),
        self_volume_sums=defaultdict(lambda: defaultdict(float)),
        roundtrip_volume_sums=defaultdict(lambda: defaultdict(float)),
        fee_sums=defaultdict(lambda: defaultdict(float)),
        recent_trades={},
        recent_miner_trades=defaultdict(dict),
        open_positions=defaultdict(lambda: defaultdict(lambda: {"longs": deque(), "shorts": deque()})),
        realized_pnl_history=defaultdict(dict),
        agent_pnl_by_book=defaultdict(lambda: defaultdict(float)),
        agent_pnl_total=defaultdict(float),
        roundtrip_volumes=defaultdict(lambda: defaultdict(lambda: defaultdict(float))),
        inventory_history=defaultdict(dict),
        initial_balances=defaultdict(dict),
        pagerduty_alert=boom,
        _last_prune_timestamp=None,
    )


def _state(trades):
    return SimpleNamespace(
        timestamp=TS,
        books={BOOK: {"e": trades}, 1: {"e": []}},
        accounts={},
        notices={},
    )


def _run_live(centered_window=None):
    s = _validator(centered_window)
    update_trade_volumes(s, _state(_prints()))
    return s


def _run_hardcoded(W):
    """The 0.6.4 path: same stream, W passed straight in. Default live capture must match this at CAPTURE_W."""
    buy = defaultdict(lambda: defaultdict(float))
    sell = defaultdict(lambda: defaultdict(float))
    mid = {}
    trades = _prints()
    accumulate_book_capture(buy, sell, BOOK, trades, W, ts=SAMPLED, mid_state=mid, now_ns=TS)
    flush_capture_state(mid, buy, sell, W, ts=SAMPLED, now_ns=TS)
    return buy, sell, mid


def _cap_map(sums):
    return {int(uid): {int(book): float(v) for book, v in books.items()} for uid, books in sums.items()}


def test_centered_window_reads_the_flag():
    assert _centered_window(_self(7)) == 7
    assert _centered_window(_self(1)) == 1


def test_centered_window_defaults_when_the_flag_is_missing():
    assert _centered_window(_self()) == CAPTURE_W
    assert _centered_window(SimpleNamespace()) == CAPTURE_W


def test_centered_window_negative_falls_back_to_the_default():
    assert _centered_window(_self(-3)) == CAPTURE_W
    assert _centered_window(_self(0)) == CAPTURE_W
    live = _run_live(-3)
    default = _run_live(None)
    assert _window(live) == _window(default)


def test_non_default_centered_window_changes_the_capture_window():
    live = _run_live(3)
    default = _run_live(None)
    assert _window(live) == (3, 3)
    assert _window(default) == (CAPTURE_W, CAPTURE_W)
    assert _window(live) != _window(default)

    expected_buy, expected_sell, expected_mid = _run_hardcoded(3)
    assert live.debeta_capture_mid[BOOK]["W"] == expected_mid[BOOK]["W"]
    assert len(live.debeta_capture_mid[BOOK]["pend"]) == len(expected_mid[BOOK]["pend"])
    assert _cap_map(live.capture_buy_sums) == _cap_map(expected_buy)
    assert _cap_map(live.capture_sell_sums) == _cap_map(expected_sell)


def test_default_leaves_capture_unchanged():
    live = _run_live(None)
    expected_buy, expected_sell, expected_mid = _run_hardcoded(CAPTURE_W)
    assert _window(live) == (CAPTURE_W, CAPTURE_W)
    assert _cap_map(live.capture_buy_sums) == _cap_map(expected_buy)
    assert _cap_map(live.capture_sell_sums) == _cap_map(expected_sell)
    explicit = _run_live(CAPTURE_W)
    assert _window(explicit) == _window(live)
    assert _cap_map(explicit.capture_buy_sums) == _cap_map(live.capture_buy_sums)
    assert _cap_map(explicit.capture_sell_sums) == _cap_map(live.capture_sell_sums)
