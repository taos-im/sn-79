# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""PYDANTIC-SER-WARN on every testnet state save (): `Expected str, got int 0`. The engine
packs a trade's close reason `cr` as 0 / 1 / 2. The miner-facing path normalises it to None / 'SL' /
'TP' in FinanceEvent.from_json, but the validator's own copies in recent_miner_trades were built with
TradeEvent.model_construct straight from the notice dict, which skips validation, so they kept the
integer, contradicted the model's `cr: str | None`, and tripped the serializer when the state was
dumped. The restore path rebuilt them the same way from files that already carried the integer.

One constructor for every unvalidated TradeEvent build from wire or stored data, normalising `cr`.
"""
import warnings
from pathlib import Path

import test_scoring_shadow as tss
from taos.im.protocol.events import FinanceEvent, TradeEvent, trade_event_from_wire
from taos.im.validator.trade import update_trade_volumes

DEV = Path(__file__).resolve().parents[1]
TRADE = (DEV / "taos/im/validator/trade.py").read_text()
PERSISTENCE = (DEV / "taos/im/validator/persistence.py").read_text()
_S = tss._S


def _notice(ts, taker, maker, book, qty, price, side, cr):
    from taos.im.validator.engines import NormalizedTradeEvent

    d = NormalizedTradeEvent(
        book_id=book, quantity=qty, price=price, side=side, maker_uid=maker, taker_uid=taker,
        maker_fee=qty * price * 0.001, taker_fee=qty * price * 0.002, timestamp=ts,
    ).to_notice_dict()
    d["cr"] = cr  # as the engine packs it: an integer, 0 for an ordinary trade
    return d


def _dumps_without_serializer_warning(event):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dumped = event.model_dump(mode="json")
    return dumped, [str(w.message) for w in caught]


def test_recent_miner_trades_carry_the_normalised_close_reason_and_dump_clean():
    main = tss._fresh_main()
    ordinary = _notice(1 * _S, 1, 2, 0, 3.0, 100.0, 0, cr=0)
    stop_loss = _notice(1 * _S, 2, 3, 1, 1.0, 55.0, 1, cr=1)
    ordinary["i"], stop_loss["i"] = 11, 12  # the builder leaves the engine trade id unset
    state = tss.SimpleNamespace(timestamp=1 * _S, notices={1: [ordinary], 2: [ordinary, stop_loss], 3: [stop_loss]},
                                books={}, accounts={})
    update_trade_volumes(main, state)

    seen = {}
    for uid, books in main.recent_miner_trades.items():
        for book_id, entries in books.items():
            for event, role in entries:
                assert isinstance(event, TradeEvent)
                dumped, warned = _dumps_without_serializer_warning(event)
                assert not warned, f"uid {uid} book {book_id} {role}: {warned}"
                seen[(book_id, event.i)] = dumped["cr"]
    assert seen[(0, ordinary["i"])] is None, "an ordinary trade's close reason is None, not 0"
    assert seen[(1, stop_loss["i"])] == "SL"


def test_wire_constructor_normalises_every_engine_encoding():
    base = {"y": "ET", "t": 1, "b": 0, "i": 7, "Ti": 1, "Mi": 2, "q": 1.0, "p": 2.0, "s": 0,
            "Ma": 3, "Ta": 4, "Mf": 0.0, "Tf": 0.0}
    assert trade_event_from_wire({**base, "cr": 0}).cr is None
    assert trade_event_from_wire({**base, "cr": 1}).cr == "SL"
    assert trade_event_from_wire({**base, "cr": 2}).cr == "TP"
    assert trade_event_from_wire({**base, "cr": "tp"}).cr == "TP"
    assert trade_event_from_wire(base).cr is None, "absent stays None"
    for cr in (0, 1, 2, None):
        _dumped, warned = _dumps_without_serializer_warning(trade_event_from_wire({**base, "cr": cr}))
        assert not warned, (cr, warned)
    raw = {**base, "cr": 1}
    trade_event_from_wire(raw)
    assert raw["cr"] == 1, "the caller's dict is not mutated"


def test_miner_facing_path_is_unchanged():
    base = {"y": "ET", "t": 1, "b": 0, "i": 7, "Ti": 1, "Mi": 2, "q": 1.0, "p": 2.0, "s": 0,
            "Ma": 3, "Ta": 4, "Mf": 0.0, "Tf": 0.0}
    assert FinanceEvent.from_json({**base, "cr": 1}).cr == "SL"
    assert FinanceEvent.from_json({**base, "cr": 0}).cr is None
    assert FinanceEvent.from_json({**base, "cr": 2}).closeReason == "TP"


def test_every_unvalidated_trade_event_build_goes_through_the_wire_constructor():
    assert "TradeEvent.model_construct(**trade)" not in TRADE, "trade.py must build recent_miner_trades via trade_event_from_wire"
    assert "trade_event_from_wire(trade)" in TRADE
    restore = PERSISTENCE[PERSISTENCE.index("loaded_recent_miner_trades = simulation_state.get"):]
    restore = restore[:restore.index("exceeds effective_max_uids")]
    assert "TradeEvent.model_construct(**t)" not in restore, "a file saved before the fix carries the integer"
    assert "trade_event_from_wire(t)" in restore
