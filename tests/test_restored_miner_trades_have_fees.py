# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A restored miner trade must carry the fee fields its consumers read as attributes.

Measured live at 13:11, 13:36, 13:40 and 13:47 on 2026-08-04, all AFTER the validator restart that
landed the ET-notice fix:

    ERROR | ALERT: Unable to publish metrics : 'TradeEvent' object has no attribute 'Tf'
      File "taos/im/validator/report.py", line 1674, in report
        fee = miner_trade.makerFee if role == 'maker' else miner_trade.takerFee

Fixing the producer was not enough. `takerFee` is a property returning `self.Tf`, `recent_miner_trades`
is PERSISTED, and it is restored with `TradeEvent.model_construct(**t)`, which skips validation. So an
entry written before the fix comes back as an object with no `Tf` attribute at all, and the buffer keeps
5 entries per uid per book, so the residue survives until five fresh trades displace it.

That is the same shape as the stale trade ids: the producer stops making bad records, and the bad
records already on disk keep arriving. Third time model_construct has turned an absent field into a
fault far from its cause.

Two ends again, and neither invents a fee:

  * a restored entry with no usable fee is DROPPED on load, because a fee cannot be reconstructed and
    fabricating zero would understate cost wherever fees are real.
  * report.py stops letting one malformed slot abort the whole metrics publish. That file already holds
    that principle for a missing timestamp ("must never abort the entire metrics publish — skip this one
    slot"); it simply was not applied to the fee read.
"""
from taos.im.validator.persistence import sanitize_miner_trades


def test_an_entry_without_fee_fields_is_dropped():
    """The live case: a notice persisted before the producer fix."""
    trades = [
        [{"i": 1, "q": 1.0, "p": 0.5, "s": 0, "Mf": 0.0, "Tf": 0.0}, "taker"],
        [{"i": 2, "q": 2.0, "p": 0.5, "s": 0}, "taker"],
    ]
    kept, dropped = sanitize_miner_trades(trades)
    assert [t[0]["i"] for t in kept] == [1]
    assert dropped == 1


def test_a_partial_fee_pair_is_also_dropped():
    """makerFee and takerFee are read independently, so one alone is still a crash waiting."""
    kept, dropped = sanitize_miner_trades([[{"i": 3, "Mf": 0.0}, "maker"]])
    assert kept == []
    assert dropped == 1


def test_complete_entries_survive_untouched():
    entry = [{"i": 4, "q": 1.0, "p": 2.0, "s": 1, "Mf": 0.25, "Tf": 0.75}, "maker"]
    kept, dropped = sanitize_miner_trades([entry])
    assert kept == [entry]
    assert dropped == 0


def test_a_stated_zero_fee_is_kept_because_zero_is_a_real_fee():
    """Exchange mode charges nothing, so zero is the normal value and must not look like absence."""
    kept, dropped = sanitize_miner_trades([[{"i": 5, "Mf": 0.0, "Tf": 0.0}, "taker"]])
    assert len(kept) == 1
    assert dropped == 0


def test_malformed_shapes_do_not_raise():
    """Load must never die over the shape of a persisted buffer."""
    for bad in ([None], [[]], [[{"i": 6, "Mf": 0.0, "Tf": 0.0}]], [{"not": "a pair"}]):
        kept, dropped = sanitize_miner_trades(bad)
        assert isinstance(kept, list) and isinstance(dropped, int)


def test_an_empty_buffer_is_not_an_error():
    assert sanitize_miner_trades([]) == ([], 0)


def test_a_restored_miner_trade_with_a_synthetic_id_is_also_dropped():
    """The id must be validated here too, not only in sanitize_recent_trades.

    Found by the first-occurrence stack capture at 13:56:54 on 2026-08-04, FOUR MINUTES after the
    sanitiser had already dropped 24 fee-less entries: validator.py:1894 _prepare_reporting_data was
    still serialising a `recent_miner_trades` entry whose `i` was a string.

    Cause was my own inconsistency: two sanitisers with different criteria. sanitize_recent_trades
    validated the id and ignored fees; sanitize_miner_trades validated fees and ignored the id. An entry
    with good fees and a synthetic x:0x<extrinsic>:<uid>:<dir>:<p|f> id therefore passed both. Same shape
    as the two ET notice builders drifting apart.
    """
    trades = [
        [{"i": 601, "Mf": 0.0, "Tf": 0.0}, "taker"],
        [{"i": "x:0xdeadbeef:171:0:f", "Mf": 0.0, "Tf": 0.0}, "taker"],
        [{"i": -1, "Mf": 0.0, "Tf": 0.0}, "maker"],
        [{"i": 3.7, "Mf": 0.0, "Tf": 0.0}, "maker"],
    ]
    kept, dropped = sanitize_miner_trades(trades)
    assert [t[0]["i"] for t in kept] == [601]
    assert dropped == 3


def test_a_numeric_string_id_is_coerced_here_too():
    """Consistent with sanitize_recent_trades: a msgpack-stringified number IS joinable."""
    kept, dropped = sanitize_miner_trades([[{"i": "601", "Mf": 0.0, "Tf": 0.0}, "taker"]])
    assert kept[0][0]["i"] == 601
    assert dropped == 0


def test_a_missing_id_is_kept_because_absent_is_not_wrong():
    """A pre-settlement trade legitimately has no id yet; dropping it would lose a real trade."""
    kept, dropped = sanitize_miner_trades([[{"Mf": 0.0, "Tf": 0.0}, "taker"]])
    assert len(kept) == 1
    assert dropped == 0
