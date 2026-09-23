# SPDX-License-Identifier: MIT
"""The counterparty factor's scope: how many of a maker's top takers P11 looks at.

P11 was built for the dedicated-feeder case, one or two takers carrying a maker, and looks at the top two.
On 23 September 2026 the mainnet tape showed a ring that spreads the same feeding over ten to twenty takers
under one operator: each feeder is 4 to 15 per cent of the fed maker's fills, so the top-two excess reads
0.10 to 0.27 and the factor stays at 0.73 to 0.90, while the operator took 65 per cent of incentive.
Raw concentration cannot see this (the ring is more diverse in counterparties than honest makers), but
P11 is not raw concentration: it subtracts the same takers' share of everyone else's flow, so common
takers cancel and private feeders do not. Widening the scope exposes the ring: measured at top-10 the fed
makers fell to 0.25 to 0.68 while honest makers stayed at 0.90 to 0.97 and 5.6 per cent of other active
makers dipped below 0.90, none below 0.70.

The dial is `scoring.debeta.p11_topk`. The SCORER's default scope of 2 reproduces the shipped leg exactly; the CONFIG
default is the 0.6.2 launch value 10, because a deployed validator restarts on its registered command line and
runs the code defaults, so the launch configuration has to be the default. It scopes
both the making discount and the skill-leg factor, which are one measurement at two strengths.
"""
import os
import random
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from taos.im.validator.debeta import MARKET_FLOW, debeta_scores, p11_discount  # noqa: E402


def _consistent(n, level=100.0):
    return [level + (i % 3) - 1 for i in range(n)]


def _ring_board(feeders=12):
    """Maker 1 is fed by `feeders` private takers in equal parts and sees no market flow. Maker 2 is a
    diverse maker: mostly market flow plus one big taker (9) that every maker in the field also trades
    with. Makers 30..49 are the field, all market flow plus the same big taker. Feeders trade with nobody
    but maker 1."""
    uids = [1, 2] + list(range(30, 50))
    alphas = {u: _consistent(30) for u in uids}
    caps = {u: {b: 10.0 for b in range(8)} for u in uids}
    cp = {1: {100 + i: 800.0 / feeders for i in range(feeders)}}
    cp[2] = {MARKET_FLOW: 600.0, 9: 120.0, 5: 40.0, 6: 40.0}
    for u in range(30, 50):
        cp[u] = {MARKET_FLOW: 700.0, 9: 100.0}
    return uids, alphas, caps, cp


def test_default_scope_reproduces_the_shipped_scorer_exactly():
    uids, alphas, caps, cp = _ring_board()
    a, b = {}, {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, p11_strength=1.0, skill_p11_strength=1.0, detail=a)
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, p11_strength=1.0, skill_p11_strength=1.0, detail=b,
                  p11_topk=2)
    assert {u: d["p11_factor"] for u, d in a.items()} == {u: d["p11_factor"] for u, d in b.items()}
    assert {u: d["skill_p11_factor"] for u, d in a.items()} == {u: d["skill_p11_factor"] for u, d in b.items()}


def test_a_twelve_feeder_ring_hides_from_top_two_and_not_from_a_wider_scope():
    uids, alphas, caps, cp = _ring_board(feeders=12)
    unit2 = p11_discount({u: 1.0 for u in uids}, cp, uids, 1.0, topk=2)
    unit12 = p11_discount({u: 1.0 for u in uids}, cp, uids, 1.0, topk=12)
    # twelve equal feeders: the top two are a sixth of the maker's fills, so the top-two excess is ~0.17
    assert 0.75 < unit2[1] < 0.9
    # the whole ring in scope: nearly all of the maker's fills are private feeders nobody else trades with
    assert unit12[1] < 0.05


def test_a_diverse_maker_with_a_common_big_taker_is_untouched_at_every_scope():
    """The field's big taker (9) is common to everyone, so the leave-one-out baseline cancels it; a
    maker must not be discounted for trading with it, at any scope."""
    uids, alphas, caps, cp = _ring_board()
    for k in (2, 5, 12, 10**6):
        unit = p11_discount({u: 1.0 for u in uids}, cp, uids, 1.0, topk=k)
        # the field makers trade only with the market and the common taker; the common taker's share of
        # their own fills is a hair above its share of everyone else's, so they sit at 0.99, not below
        assert all(unit[u] > 0.98 for u in range(30, 50)), k
        # the diverse maker also has two small private takers (5 and 6, a tenth of its fills); a wide
        # scope prices those in, which is the collateral pattern seen on mainnet (honest makers at
        # 0.87 to 0.97). It must stay far from the fed maker's factor.
        assert unit[2] > 0.85, (k, unit[2])


def test_the_dial_scopes_the_making_discount_and_the_skill_factor_together():
    uids, alphas, caps, cp = _ring_board(feeders=12)
    narrow, wide = {}, {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, p11_strength=1.0, skill_p11_strength=1.0, detail=narrow,
                  p11_topk=2)
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, p11_strength=1.0, skill_p11_strength=1.0, detail=wide,
                  p11_topk=12)
    assert wide[1]["p11_factor"] < 0.05 < 0.75 < narrow[1]["p11_factor"]
    assert wide[1]["skill_p11_factor"] == pytest.approx(wide[1]["p11_factor"], abs=1e-9)
    assert narrow[1]["skill_p11_factor"] == pytest.approx(narrow[1]["p11_factor"], abs=1e-9)
    assert wide[2]["p11_factor"] > 0.85 and wide[2]["skill_p11_factor"] > 0.85   # the honest collateral pattern


def test_the_factor_is_monotone_in_scope_for_a_fed_maker():
    uids, alphas, caps, cp = _ring_board(feeders=12)
    prev = 1.0
    for k in (1, 2, 4, 8, 12, 20):
        f = p11_discount({u: 1.0 for u in uids}, cp, uids, 1.0, topk=k)[1]
        assert f <= prev + 1e-12, (k, f, prev)
        prev = f


def test_randomised_boards_default_scope_is_the_off_path():
    rng = random.Random(23)
    for _ in range(15):
        uids = list(range(rng.randint(20, 60)))
        alphas = {u: [rng.gauss(30, 60) for _ in range(rng.randint(0, 30))] for u in uids}
        caps = {u: {b: max(0.0, rng.gauss(50, 40)) for b in range(rng.randint(0, 10))} for u in uids}
        cp = {u: {rng.choice(uids + [MARKET_FLOW]): rng.uniform(1, 500) for _ in range(rng.randint(0, 6))} for u in uids}
        kw = dict(floor=rng.choice([0.0, 5.0]), w_make=0.5, cp=cp, p11_strength=1.0, skill_p11_strength=rng.choice([0.0, 1.0]))
        assert debeta_scores(caps, caps, alphas, **kw) == debeta_scores(caps, caps, alphas, p11_topk=2, **kw)


def test_the_call_site_reads_the_dial_and_the_config_defaults_to_the_launch_value():
    import argparse
    import pathlib

    import bittensor as bt

    from taos.im.config import add_im_validator_args

    p = argparse.ArgumentParser()
    add_im_validator_args(None, p)
    assert bt.Config(p).scoring.debeta.p11_topk == 10, "the config default is the 0.6.2 launch value"
    assert bt.Config(p, args=["--scoring.debeta.p11_topk", "2"]).scoring.debeta.p11_topk == 2, "the shipped scope stays reachable"
    src = pathlib.Path(REPO_ROOT, "taos", "im", "validator", "reward.py").read_text()
    body = src[src.index("def compute_debeta_scores"):]
    body = body[:body.index("\ndef ", 1)]
    assert "p11_topk" in body, "compute_debeta_scores does not pass the scope to the scorer"
