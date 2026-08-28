# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner must be able to see what it can actually SELL, not just what it holds.

`base_balance.free` is alpha summed across every delegate the account holds stake with. A pool SELL is
checked against ONE delegate's stake (Pool.cpp: "rejected; delegateStake ({}) less than alphaSpent ({})"),
so an order sized from the total is refused INSUFFICIENT_FUNDS on an account that visibly holds enough.

dTAO makes the split ordinary: a coldkey accumulates alpha under whichever hotkeys it staked to, so the
total can exceed any single delegate's stake by a wide margin.

These tests pin the contract of the field that closes that gap, including the two things most likely to
be "fixed" into bugs later: the empty map means NOT REPORTED, and sum(ds) == free is NOT an invariant.
"""

import pytest

from taos.im.protocol.models import Account as SimAccount
from taos.im.protocol.exchange.models import Account as XchAccount
from taos.im.agents import UnifiedAccount

ACCOUNTS = pytest.mark.parametrize("Acct", [SimAccount, XchAccount], ids=["simulation", "exchange"])


def _mk(Acct, ds=None):
    kw = dict(i=4, b=5, bb={"f": 1.5, "r": 0.0}, qb={"f": 10.0, "r": 0.0}, f=None)
    if ds is not None:
        kw["ds"] = ds
    return Acct.model_construct(**kw)


@ACCOUNTS
def test_the_breakdown_is_exposed_on_both_protocol_paths(Acct):
    """Exchange mode is the path this suite runs on; simulation is the other. Both or neither."""
    a = _mk(Acct, {"5GZ2": 0.737, "5GMy": 0.013})
    assert a.delegate_stakes == {"5GZ2": 0.737, "5GMy": 0.013}


@ACCOUNTS
def test_sellable_alpha_is_the_largest_single_delegate_not_the_sum(Acct):
    """The whole reason the field exists. 0.75 is sellable in one order; 0.75 + 0.013 is not."""
    a = _mk(Acct, {"5GZ2": 0.737, "5GMy": 0.013})
    assert a.sellable_alpha == pytest.approx(0.737)
    assert a.sellable_alpha < sum(a.delegate_stakes.values())


@ACCOUNTS
def test_absent_means_not_reported_and_never_means_no_stake(Acct):
    """An older validator sends no 'ds'. Reading that as "holds nothing" would stop a miner trading."""
    a = _mk(Acct)
    assert a.delegate_stakes == {}
    assert a.sellable_alpha == 0.0, "0.0 here means UNKNOWN; callers must fall back to base_balance.free"


def test_unified_account_reads_both_the_dict_and_object_branches():
    """The defect traded_volume already suffered: an accessor handling one branch reaches half the miners.

    The exchange path hands UnifiedAccount a raw dict keyed 'ds'; simulation hands it an Account.
    """
    from_dict = UnifiedAccount({"bb": {"f": 1.5}, "qb": {"f": 10.0}, "ds": {"5GZ2": 0.737}})
    assert from_dict.delegate_stakes == {"5GZ2": 0.737}
    assert from_dict.sellable_alpha == pytest.approx(0.737)
    assert from_dict.ds == from_dict.delegate_stakes

    from_obj = UnifiedAccount(_mk(XchAccount, {"5GZ2": 0.737}))
    assert from_obj.delegate_stakes == {"5GZ2": 0.737}
    assert from_obj.sellable_alpha == pytest.approx(0.737)


def test_unified_account_tolerates_a_producer_that_sends_nothing():
    assert UnifiedAccount({"bb": {"f": 1.5}}).delegate_stakes == {}
    assert UnifiedAccount({"bb": {"f": 1.5}}).sellable_alpha == 0.0


def test_the_sum_is_not_asserted_to_equal_the_free_balance():
    """Documented divergence, pinned so nobody "fixes" it into an assertion.

    delegate_stakes is chain stake at snapshot time; base_balance is the engine's accounting including
    reservations and its per-batch reconcile. They track closely and diverge transiently, and a test or
    a check that requires equality would fail intermittently for correct behaviour.
    """
    a = _mk(XchAccount, {"5GZ2": 0.737, "5GMy": 0.013})
    assert sum(a.delegate_stakes.values()) != pytest.approx(a.bb["f"]), (
        "this inequality is the documented case; if the model ever guarantees equality, "
        "update the field docs and this test together"
    )
