# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A trade id restored from persisted state must be the engine's integer id.

`recent_trades` is persisted, and a restored id goes back into the engine as `ExecutedFill`, where
`tradeId` is `std::optional<uint64_t>`: a non-integer throws `std::bad_cast` and drops the whole batch.

`TradeInfo.model_construct(**t)` skips validation by design (hot path), so a bad type in the id field
passes silently on load and only fails much later at `model_dump` in another module. Restored ids are
therefore checked explicitly here.

A restored trade whose id is not an integer is dropped rather than repaired: no such id exists on the
tape, so there is nothing to join it to, and inventing one would give a single fill a different identity
on each surface.
"""
from taos.im.validator.persistence import sanitize_recent_trades


def test_a_synthetic_extrinsic_id_is_dropped():
    """A non-integer id from persisted state is dropped, not coerced."""
    trades = [
        {"i": 469, "q": 1.0},
        {"i": "x:0xdeadbeef:248:0:f", "q": 2.0},
    ]
    kept, dropped = sanitize_recent_trades(trades)
    assert [t["i"] for t in kept] == [469]
    assert dropped == 1


def test_engine_integer_ids_are_kept_unchanged():
    trades = [{"i": 1}, {"i": 469}, {"i": 18446744073709551615}]
    kept, dropped = sanitize_recent_trades(trades)
    assert [t["i"] for t in kept] == [1, 469, 18446744073709551615]
    assert dropped == 0


def test_a_numeric_string_is_coerced_rather_than_thrown_away():
    """msgpack round-trips can stringify a number; that id IS joinable, so keep it."""
    kept, dropped = sanitize_recent_trades([{"i": "469"}])
    assert kept == [{"i": 469}]
    assert dropped == 0


def test_a_missing_or_null_id_is_kept():
    """Absent is not wrong: a pre-settlement trade legitimately has no id yet.

    Dropping these would lose real trades, which is a worse failure than the warning.
    """
    kept, dropped = sanitize_recent_trades([{"q": 1.0}, {"i": None, "q": 2.0}])
    assert len(kept) == 2
    assert dropped == 0


def test_a_negative_id_is_dropped_because_the_engine_type_is_unsigned():
    kept, dropped = sanitize_recent_trades([{"i": -1}])
    assert kept == []
    assert dropped == 1


def test_a_float_id_is_dropped_rather_than_truncated():
    """Truncating would invent a different trade's id."""
    kept, dropped = sanitize_recent_trades([{"i": 469.7}])
    assert kept == []
    assert dropped == 1


def test_an_empty_buffer_is_not_an_error():
    assert sanitize_recent_trades([]) == ([], 0)


def test_the_input_list_is_not_mutated():
    """Load runs on state that other code may still hold a reference to."""
    src = [{"i": "x:0xdead:171:0:p"}, {"i": 5}]
    before = [dict(t) for t in src]
    sanitize_recent_trades(src)
    assert src == before
