# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The counterparty concentration behind the P11 discount is computed once per counterparty map and reused.

debeta_scores applies the discount twice on the same map, to the making leg and, at unit strength, as the skill leg's
factor. p11_ec computes each maker's excess concentration once, and p11_discount applies it unchanged when handed it,
so the discounted values are the same floats."""
import inspect
import random

from taos.im.validator import debeta


def _cp(seed):
    rnd = random.Random(seed)
    cp = {}
    for m in range(60):
        cp[m] = {t: rnd.uniform(0, 10) for t in rnd.sample(range(-5, 80), rnd.randint(0, 12))}
        if rnd.random() < 0.3:
            cp[m][debeta.MARKET_FLOW] = rnd.uniform(0, 30)
    cp[3] = {}
    return cp


def test_the_discount_with_the_concentration_handed_in_is_the_same_float():
    for seed in range(5):
        cp = _cp(seed)
        uids = list(range(70))
        own = {u: random.Random(seed + u).uniform(0, 100) for u in uids}
        for topk in (1, 2, 3):
            ec = debeta.p11_ec(cp, uids, topk)
            for strength in (0.3, 1.0):
                assert debeta.p11_discount(dict(own), cp, uids, strength, topk=topk, ec=ec) == \
                    debeta.p11_discount(dict(own), cp, uids, strength, topk=topk)
            unit = {u: 1.0 for u in uids}
            assert debeta.p11_discount(unit, cp, uids, 1.0, topk=topk, ec=ec) == \
                debeta.p11_discount(unit, cp, uids, 1.0, topk=topk)


def test_debeta_scores_computes_the_concentration_once():
    src = inspect.getsource(debeta.debeta_scores)
    assert src.count("p11_ec(") == 1
    assert src.count("ec=_ec") == 2
