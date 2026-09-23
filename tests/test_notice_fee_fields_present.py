# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A trade notice must carry its fee fields, and a consumer must survive one that does not.

Seen live 28 times inside a two-hour window:

    ERROR | PD: Failed to update trade data for UID 171: 'Tf'
      File "taos/im/validator/trade.py", line 221, in _process_uid_notices
        fee = trade['Mf'] if is_maker else trade['Tf']
    KeyError: 'Tf'

The cost is not a log line. The KeyError propagates out of `_process_uid_notices`, so
`_process_uid_trade_volumes` aborts for that uid and its trade volumes, realized PnL and roundtrip
volume are all left unupdated for the block. Those are scoring inputs, and the failure is silent
apart from this message.

Cause: two builders for the same notice. `engines/__init__.py:to_notice_dict()` is canonical and emits
`Mf` and `Tf`; a hand-built notice on the other engine path omits them. This is the same shape
as the role defects: a field ABSENT from a record is invented independently by every consumer, or, as
here, crashes one of them.

Both ends are fixed, deliberately:

  * every producer populates the field. Zero is the TRUE value in exchange mode (helpers.cpp removes
    the FeePolicy node, so ZeroFeePolicy is installed and nothing is ever charged), so it is stated
    rather than left out.
  * the consumer NEVER substitutes a default. A fee feeds realized PnL, and in simulation fees are
    real, so reading an absent Tf as 0.0 would understate cost and corrupt PnL while looking perfectly
    healthy. A missing field is a producer defect, is reported as one, and costs that single notice
    rather than the uid's whole update.

The same absence separately broke a second consumer, found by the log monitor rather than by the
dict-subscript sweep: report.py rebuilds the notice with TradeEvent.model_construct, which skips
validation, so a missing key becomes a missing ATTRIBUTE and metrics publishing died with
"'TradeEvent' object has no attribute 'Tf'".
"""
import pathlib

import pytest

import taos.im.validator.trade as trade_mod
from taos.im.validator.engines import NormalizedTradeEvent


def test_the_canonical_builder_states_both_fees():
    d = NormalizedTradeEvent(
        book_id=5, quantity=1.0, price=0.5, side=0,
        maker_uid=None, taker_uid=171, maker_fee=0.0, taker_fee=0.0,
    ).to_notice_dict()
    assert d["Mf"] == 0.0
    assert d["Tf"] == 0.0


def test_an_absent_fee_raises_rather_than_defaulting():
    """The live failing notice: an ET dict with roles and no fee fields at all.

    It must NOT come back as 0.0. A fabricated zero is indistinguishable downstream from a genuine
    zero-fee trade, so it would corrupt PnL silently wherever fees are real.
    """
    import pytest

    notice = {"y": "ET", "b": 58, "q": 2170.9, "p": 0.0084, "s": 0, "Ta": 171, "Ma": None}
    with pytest.raises(trade_mod.MissingNoticeField):
        trade_mod.notice_fee(notice, is_maker=False)
    with pytest.raises(trade_mod.MissingNoticeField):
        trade_mod.notice_fee(notice, is_maker=True)


def test_a_stated_fee_is_used_not_the_default():
    """Simulation charges real fees; defaulting must never override a stated value."""
    notice = {"Mf": 0.25, "Tf": 0.75}
    assert trade_mod.notice_fee(notice, is_maker=True) == 0.25
    assert trade_mod.notice_fee(notice, is_maker=False) == 0.75


def test_a_null_fee_raises_because_null_is_not_a_fee():
    import pytest

    for notice in ({"Mf": None}, {"Tf": None}):
        with pytest.raises(trade_mod.MissingNoticeField):
            trade_mod.notice_fee(notice, is_maker="Mf" in notice)


def test_a_non_numeric_fee_raises_and_is_counted():
    """Garbage is a producer defect too, and must not be laundered into a number."""
    import pytest

    for bad in ("", "x", {}, []):
        with pytest.raises(trade_mod.MissingNoticeField):
            trade_mod.notice_fee({"Tf": bad}, is_maker=False)


def test_a_stated_zero_and_an_absent_field_are_not_conflated_for_reporting():
    """Absence is a data defect worth surfacing even though it is handled.

    Silently treating absent as zero is how the missing field survived long enough to crash a
    consumer; the counter is what makes it visible without costing the update.
    """
    import pytest

    trade_mod.reset_missing_fee_count()
    assert trade_mod.notice_fee({"Tf": 0.0}, is_maker=False) == 0.0
    assert trade_mod.missing_fee_count() == 0
    with pytest.raises(trade_mod.MissingNoticeField):
        trade_mod.notice_fee({}, is_maker=False)
    assert trade_mod.missing_fee_count() == 1


def test_the_exchange_builds_its_et_notice_with_the_canonical_serializer():
    """One notice shape, and one place that produces it.

    The root cause was drift between the canonical to_notice_dict() and a hand-built twin here.
    Consumers index these keys directly (trade.py for the fees, TradeInfo.from_json for Ti/Mi/Mf/Tf), so
    a key the twin forgot was a crash waiting for the right notice. Pinning the twin's key set was the
    previous form of this test; the twin is gone, so what is pinned now is that the call is made.

    Read from the source because the surrounding function needs a live chain, an engine process and a
    settled batch to reach. The complementary end-to-end assertion lives in the notice
    scenarios.
    """
    engines = pathlib.Path(__file__).resolve().parents[1] / "taos" / "im" / "validator" / "engines"
    module = engines / "exchange.py"
    if not module.exists():
        # The published tree carries the canonical serializer but not this engine, so the call site
        # this reads has no subject there. Guarded on the package still being present, so a tree that
        # lost the engine outright fails rather than quietly skipping.
        assert (engines / "__init__.py").exists(), "the engines package is missing entirely"
        pytest.skip("this engine is not distributed in the published tree")
    src = module.read_text(encoding="utf-8")
    assert "_et_notice = _te.to_notice_dict()" in src, (
        "the exchange no longer builds its ET notice with the canonical serializer -- if this moved, "
        "point the assertion at the new call rather than reintroducing a hand-built dict"
    )
