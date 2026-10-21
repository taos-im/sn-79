# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Account.from_json carries delegate_stakes, as the other state parse path does.

An account rebuilt through from_json without `ds` reports an empty breakdown, which a miner reads as
"not reported" and so falls back to the summed free balance, the figure a pool sell can be refused on.
"""
from taos.im.protocol.models import Account, Balance


def _account(ds=None):
    kw = dict(
        i=4, b=5,
        bb=Balance.model_construct(c="BASE", t=1.5, f=1.5, r=0.0, i=None),
        qb=Balance.model_construct(c="QUOTE", t=10.0, f=10.0, r=0.0, i=None),
        f=None, v=12.5,
    )
    if ds is not None:
        kw["ds"] = ds
    return Account.model_construct(**kw)


def test_delegate_stakes_survive_a_round_trip():
    a = _account({"5GZ2": 0.737, "5GMy": 0.013})
    b = Account.from_json(a.model_dump())
    assert b.delegate_stakes == {"5GZ2": 0.737, "5GMy": 0.013}
    assert b.sellable_alpha == a.sellable_alpha
    assert b.model_dump() == a.model_dump()


def test_an_account_without_ds_reads_as_not_reported():
    raw = _account().model_dump()
    raw.pop("ds", None)
    assert Account.from_json(raw).delegate_stakes == {}
