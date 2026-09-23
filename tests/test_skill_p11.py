# SPDX-License-Identifier: MIT
"""The counterparty factor extended to the skill leg.

P11 discounts a maker's captured spread by the excess concentration of the takers that filled it, so a
maker fed by a dedicated counterparty cannot rank-buy a making slot. It never reached the skill leg,
and on 22 September the tape showed that is where a feeding ring is paid: fed makers with making ranks
of 0.003 to 0.36 held skill ranks of 0.94 to 1.00 on four to nine books, because feeding at chosen
prices manufactures per-book alpha on the fed side. Against real alphas, applying the same factor
to skill touches no account outside the pattern (every genuine skill winner at exactly 1.000) and is
cloning-invariant under a proportional skill pool.

The dial is `scoring.debeta.skill_p11_strength`, default 0, which reproduces the shipped leg exactly.
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


def _board():
    """Four uids. 1 is a fed maker: all its resting fills come from taker 9. 2 is a diverse maker with
    the same alphas and the same capture. 3 and 4 are broad makers with market flow only."""
    alphas = {1: _consistent(30), 2: _consistent(30), 3: _consistent(60, 40.0), 4: _consistent(60, 35.0)}
    caps = {u: {b: 10.0 for b in range(8)} for u in alphas}
    cp = {
        1: {9: 800.0},                                   # fed: one dedicated taker
        2: {MARKET_FLOW: 700.0, 5: 40.0, 6: 30.0, 7: 30.0},  # diverse
        3: {MARKET_FLOW: 900.0},
        4: {MARKET_FLOW: 900.0},
        9: {},
    }
    return alphas, caps, cp


def test_off_by_default_reproduces_the_shipped_skill_leg_exactly():
    alphas, caps, cp = _board()
    off, explicit = {}, {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, p11_strength=1.0, detail=off)
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, p11_strength=1.0, detail=explicit,
                  skill_p11_strength=0.0)
    assert {u: d["skill_raw"] for u, d in off.items()} == {u: d["skill_raw"] for u, d in explicit.items()}
    assert all(d["skill_p11_factor"] == 1.0 for d in off.values())


def test_a_fed_maker_loses_its_skill_and_an_identical_diverse_one_keeps_it():
    alphas, caps, cp = _board()
    detail = {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, p11_strength=1.0, detail=detail,
                  skill_p11_strength=1.0)
    # 1 and 2 have identical alphas, so identical kappa before the factor
    assert detail[1]["skill_p11_factor"] < 0.2
    assert detail[2]["skill_p11_factor"] > 0.9
    assert detail[1]["skill_raw"] < detail[2]["skill_raw"] * 0.25
    assert detail[3]["skill_p11_factor"] == 1.0 and detail[4]["skill_p11_factor"] == 1.0


def test_the_skill_factor_is_the_same_excess_concentration_p11_uses_for_making():
    """One measurement, two strengths. At equal strength the two factors coincide."""
    alphas, caps, cp = _board()
    detail = {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, p11_strength=1.0, detail=detail,
                  skill_p11_strength=1.0)
    for u in (1, 2, 3, 4):
        assert detail[u]["skill_p11_factor"] == pytest.approx(detail[u]["p11_factor"], abs=1e-9)


def test_it_works_with_the_making_discount_switched_off():
    """The skill factor must not depend on p11_strength being on for making."""
    alphas, caps, cp = _board()
    detail = {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, p11_strength=0.0, detail=detail,
                  skill_p11_strength=1.0)
    assert detail[1]["p11_factor"] == 1.0                 # making untouched
    assert detail[1]["skill_p11_factor"] < 0.2             # skill discounted


def test_strength_scales_the_discount_linearly_in_excess_concentration():
    alphas, caps, cp = _board()
    unit = p11_discount({u: 1.0 for u in (1, 2, 3, 4)}, cp, [1, 2, 3, 4], 1.0)
    ec1 = 1.0 - unit[1]
    for s in (0.25, 0.5, 1.0):
        detail = {}
        debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, p11_strength=0.0, detail=detail,
                      skill_p11_strength=s)
        assert detail[1]["skill_p11_factor"] == pytest.approx(max(0.0, 1.0 - s * ec1), abs=1e-9)


def test_net_alpha_is_published_over_every_filled_book():
    alphas, caps, cp = _board()
    detail = {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, detail=detail)
    for u, a in alphas.items():
        assert detail[u]["skill_net_alpha"] == pytest.approx(sum(a))


def test_without_counterparty_data_the_factor_is_one():
    alphas, caps, _ = _board()
    detail = {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=None, detail=detail, skill_p11_strength=1.0)
    assert all(d["skill_p11_factor"] == 1.0 for d in detail.values())


def _split_fed_maker(background_makers, dedicated, market_leak, k):
    """A fed maker (uid 1) with `dedicated` flow from taker 9 and `market_leak` from the market, on a
    field of `background_makers` diverse makers; returns (paid as one account, paid as k clones,
    clone factors). The clones split alphas, capture and the counterparty row identically."""
    alphas = {1: _consistent(30), 2: _consistent(30)}
    caps = {1: {b: 10.0 for b in range(8)}, 2: {b: 10.0 for b in range(8)}}
    cp = {1: {9: dedicated, MARKET_FLOW: market_leak}, 2: {MARKET_FLOW: 700.0, 5: 40.0, 6: 30.0, 7: 30.0}, 9: {}}
    for j in range(background_makers):
        u = 1000 + j
        alphas[u] = _consistent(60, 40.0)
        caps[u] = {b: 10.0 for b in range(8)}
        cp[u] = {MARKET_FLOW: 900.0, 9: 5.0}
    one = {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, cp=cp, detail=one, skill_p11_strength=1.0)
    alphas2 = {u: a for u, a in alphas.items() if u != 1}
    caps2 = {u: c for u, c in caps.items() if u != 1}
    cp2 = {u: dict(c) for u, c in cp.items() if u != 1}
    clones = list(range(100, 100 + k))
    for c in clones:
        alphas2[c] = [a / k for a in alphas[1]]
        caps2[c] = {b: v / k for b, v in caps[1].items()}
        cp2[c] = {t: v / k for t, v in cp[1].items()}
    many = {}
    debeta_scores(caps2, caps2, alphas2, floor=0.0, w_make=0.5, cp=cp2, detail=many, skill_p11_strength=1.0)
    paid_one = max(0.0, one[1]["skill_net_alpha"]) * one[1]["skill_p11_factor"]
    paid_many = sum(max(0.0, many[c]["skill_net_alpha"]) * many[c]["skill_p11_factor"] for c in clones)
    return paid_one, paid_many, [many[c]["skill_p11_factor"] for c in clones]


def test_on_a_realistic_field_the_factor_is_near_invariant_under_cloning():
    """Sixty diverse makers, the ring about one per cent of all flow, as on the mainnet board where
    the real-board split of fed maker 230 measured x1.001. Every clone carries the same excess
    concentration, so the same factor, and additive alpha times the same factor barely moves. The
    residual (measured 1.065 at k=16) is P11's leave-one-out market share: part of the dedicated
    taker's flow now sits with the other clones and reads as market flow to each. Pinned at 1.10."""
    paid_one, paid_many, factors = _split_fed_maker(60, 800.0, 200.0, 16)
    assert max(factors) - min(factors) < 1e-9
    assert paid_one > 0
    assert paid_many / paid_one < 1.10


def test_the_limit_when_a_ring_is_a_large_share_of_the_whole_market():
    """Documented so it is not rediscovered. With no background field the ring is near half of all
    flow, and leave-one-out breaks: each clone sees most of the dedicated taker's flow as market
    flow, the factor rises from 0.20 to 0.63 and the operator is paid about three times as much for
    splitting. P11 weakens as a ring grows toward dominating the market's miner flow. On mainnet the
    largest ring is a few per cent of miner flow and this does not bind; it would on an exchange
    board with few participants, which is one more reason exchange mode needs its own calibration."""
    paid_one, paid_many, factors = _split_fed_maker(0, 800.0, 200.0, 16)
    assert paid_one > 0
    assert paid_many / paid_one > 2.0


def test_randomised_boards_default_is_the_off_path():
    rng = random.Random(22)
    for _ in range(15):
        uids = list(range(rng.randint(20, 80)))
        alphas = {u: [rng.gauss(30, 60) for _ in range(rng.randint(0, 40))] for u in uids}
        caps = {u: {b: max(0.0, rng.gauss(50, 40)) for b in range(rng.randint(0, 12))} for u in uids}
        cp = {u: {rng.choice(uids + [MARKET_FLOW]): rng.uniform(1, 500) for _ in range(rng.randint(0, 5))}
              for u in uids}
        kw = dict(floor=rng.choice([0.0, 5.0, 20.0]), w_make=0.5, cp=cp, p11_strength=rng.choice([0.0, 1.0]))
        assert debeta_scores(caps, caps, alphas, **kw) == debeta_scores(caps, caps, alphas, skill_p11_strength=0.0, **kw)


def test_the_call_site_builds_the_counterparty_map_for_the_skill_discount_too():
    """The scorer receives `cp` from compute_debeta_scores. Until 22 September that map was built only
    when the MAKING discount was on, so with p11_strength 0 the skill-leg factor never saw a map and
    silently stayed at 1.0, while the dial's help text promised independence. Read from source
    because compute_debeta_scores is a validator method; the property is that the line building `cp`
    is conditioned on the skill strength as well as the making one."""
    import pathlib
    src = pathlib.Path(REPO_ROOT, "taos", "im", "validator", "reward.py").read_text()
    start = src.index("def compute_debeta_scores")
    body = src[start:src.index("\ndef ", start + 1)]
    line = next(ln for ln in body.splitlines() if "debeta_cp" in ln and "cp = " in ln.replace("(", " ").replace("cp = (", "cp = "))
    cond = body[body.index(line):body.index("else None", body.index(line))]
    assert "skill_p11_strength" in cond, "the counterparty map is built for the making discount only"
