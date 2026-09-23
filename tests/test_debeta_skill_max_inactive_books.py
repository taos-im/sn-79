"""The skill leg weighted by how much of the market it covers.

kappa is a consistency ratio normalised by the median absolute deviation of the per-book alphas, so
it is maximised at its smallest admissible sample: `kappa_floored`'s four-book minimum is a floor a
narrow agent sits on rather than a bar it clears. These pin the shape of the correction, and above
all that the dial's default changes nothing.
"""

import pytest

from taos.im.validator.debeta import coverage_weight, debeta_scores, kappa_of_alpha


def _consistent(n, level=100.0):
    return [level + (i % 3) - 1 for i in range(n)]


def test_off_by_default_returns_the_input_untouched():
    skill = [9.0, 1.0, -0.5, 0.0]
    books = [4, 90, 60, 0]
    assert coverage_weight(skill, books, 0.0) == skill
    assert coverage_weight(skill, books, None) == skill


def test_a_narrow_agent_loses_most_of_its_skill_and_a_broad_one_keeps_all_of_it():
    skill = [9.0, 9.0]
    books = [4, 120]
    out = coverage_weight(skill, books, 0.375)
    required = (1 - 0.375) * 120                           # widest coverage in the field
    assert out[0] == pytest.approx(9.0 * 4 / required)
    assert out[1] == 9.0                                   # at or above the bar, untouched
    assert out[0] < out[1] / 4


def test_everything_at_or_above_the_bar_is_left_exactly_alone():
    """The free allowance is the point: this must not tax the agents it is protecting."""
    skill = [5.0, 5.0, 5.0]
    books = [60, 70, 128]
    assert coverage_weight(skill, books, 0.75) == skill


def test_the_bar_follows_the_field_not_an_absolute():
    """The same agent is weighted differently depending on what the rest of the field covers."""
    thin_field = coverage_weight([9.0], [4], 0.3)
    # with the same agent inside a field of broad coverage the median rises, so n0 rises
    wide = coverage_weight([9.0, 9.0, 9.0], [4, 120, 120], 0.3)
    assert wide[0] < thin_field[0]


def test_it_is_monotone_in_coverage():
    skill = [5.0] * 5
    books = [4, 10, 40, 80, 120]
    out = coverage_weight(skill, books, 0.375)
    assert out == sorted(out)


def test_zero_coverage_scores_zero_rather_than_dividing_by_nothing():
    """Inside a field that has coverage, an agent with none earns nothing on the leg."""
    assert coverage_weight([3.0, 5.0], [0, 60], 0.3)[0] == 0.0


def test_a_field_with_no_coverage_at_all_is_left_alone():
    assert coverage_weight([1.0, 2.0], [0, 0], 0.3) == [1.0, 2.0]


def _board():
    """One narrow agent with four consistent books against three broad agents."""
    alphas = {
        1: _consistent(4),                                   # the narrow agent
        2: [40.0 + (i * 7 % 33) - 16 for i in range(90)],
        3: [35.0 + (i * 11 % 41) - 20 for i in range(110)],
        4: [30.0 + (i * 5 % 29) - 14 for i in range(70)],
    }
    caps = {u: {b: 10.0 for b in range(4)} for u in alphas}
    return alphas, caps


def test_default_reproduces_the_shipped_leg_exactly():
    alphas, caps = _board()
    base = debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5)
    same = debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, skill_max_inactive_books=0.0)
    assert base == same


def test_the_narrow_agent_leads_the_skill_leg_today_and_does_not_once_the_dial_turns():
    alphas, caps = _board()
    off, on = {}, {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, detail=off)
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, detail=on, skill_max_inactive_books=0.3)
    assert off[1]["skill_rank"] == max(d["skill_rank"] for d in off.values())
    assert on[1]["skill_rank"] < max(d["skill_rank"] for d in on.values())


def test_the_counterfactual_is_published_whichever_way_the_dial_is_set():
    alphas, caps = _board()
    for frac in (0.0, 0.3):
        detail = {}
        debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, detail=detail,
                      skill_max_inactive_books=frac)
        for d in detail.values():
            assert d["skill_coverage_factor"] is not None
            assert d["skill_books"] is not None
        # off every factor is 1.0; on, the narrow agent carries the multiplier it was charged
        if frac == 0.0:
            assert all(d["skill_coverage_factor"] == 1.0 for d in detail.values())
        else:
            assert detail[1]["skill_coverage_factor"] < 1.0
            assert detail[2]["skill_coverage_factor"] == 1.0


def test_the_published_book_count_is_the_one_the_weighting_used():
    alphas, caps = _board()
    detail = {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, detail=detail,
                  skill_max_inactive_books=0.375)
    assert detail[1]["skill_books"] == 4
    assert detail[2]["skill_books"] == 90
    required = (1 - 0.375) * 110                           # widest coverage in the field
    assert detail[1]["skill_coverage_factor"] == pytest.approx(min(1.0, 4 / required))


def test_it_applies_to_a_skill_values_form_too():
    """A sub-window or hurdle form supplies skill; coverage still weights it."""
    alphas, caps = _board()
    supplied = {1: 9.0, 2: 9.0, 3: 9.0, 4: 9.0}
    detail = {}
    debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, detail=detail,
                  skill_values=supplied, skill_max_inactive_books=0.3)
    assert detail[1]["skill_raw"] < detail[2]["skill_raw"]


def test_the_premise_holds_that_a_clean_narrow_sample_beats_a_real_broad_one():
    """Not a property of the fix: the reason it is needed. A handful of alphas can be made to look
    consistent; a real book of a hundred carries the dispersion of the market it was earned in, and
    the ratio is normalised by that dispersion."""
    import random

    rng = random.Random(11)
    narrow = kappa_of_alpha(_consistent(4))
    broad = kappa_of_alpha([rng.gauss(40, 120) for _ in range(120)])
    assert narrow > broad * 3


def test_coverage_reads_books_filled_not_books_that_cleared_the_floor():
    """The defect this guards: the post-floor count falls with SIZE per book, not with breadth.
    An agent quoting the whole field at modest size clears the floor on a handful of books, and
    feeding that count to the coverage rule punishes it for being small rather than narrow."""
    # both agents fill in 100 books; one is large so its alphas clear the floor, the other is tiny
    big = [50.0 + (i % 7) for i in range(100)]
    small = [0.5 + (i % 7) * 0.01 for i in range(100)]
    alphas = {1: small, 2: big, 3: big, 4: big}
    caps = {u: {b: 10.0 for b in range(4)} for u in alphas}
    detail = {}
    debeta_scores(caps, caps, alphas, floor=10.0, w_make=0.5, detail=detail,
                  skill_max_inactive_books=0.375)
    # the small agent cleared the floor on nothing, but it covered the field
    assert detail[1]["skill_books"] == 0
    assert detail[1]["coverage_books"] == 100
    assert detail[2]["coverage_books"] == 100
    # so the coverage rule must not penalise it: same breadth, same factor
    assert detail[1]["skill_coverage_factor"] == detail[2]["skill_coverage_factor"] == 1.0


def test_a_genuinely_narrow_agent_is_still_penalised():
    """The rule must still bite on the case it exists for: few books, whatever the size."""
    alphas = {1: [100.0, 101.0, 99.0, 100.0],            # four books, large
              2: [50.0 + (i % 7) for i in range(100)],
              3: [50.0 + (i % 7) for i in range(100)]}
    caps = {u: {b: 10.0 for b in range(4)} for u in alphas}
    detail = {}
    debeta_scores(caps, caps, alphas, floor=10.0, w_make=0.5, detail=detail,
                  skill_max_inactive_books=0.375)
    assert detail[1]["coverage_books"] == 4
    assert detail[1]["skill_coverage_factor"] == pytest.approx(4 / ((1 - 0.375) * 100))
    assert detail[2]["skill_coverage_factor"] == 1.0


def test_the_kappa_minimum_defaults_to_the_shipped_four_and_never_goes_below_it():
    from taos.im.validator.debeta import kappa_floored
    four = [100.0, 101.0, 99.0, 100.0]
    assert kappa_floored(four, 0.0) == kappa_floored(four, 0.0, 4) == kappa_floored(four, 0.0, 1)
    assert kappa_floored(four[:3], 0.0) == 0.0


def test_raising_the_minimum_zeroes_a_thin_sample_and_leaves_a_broad_one():
    from taos.im.validator.debeta import kappa_floored
    thin = [100.0, 101.0, 99.0, 100.0, 100.5]
    broad = [50.0 + (i * 7 % 33) - 16 for i in range(60)]
    assert kappa_floored(thin, 0.0) > 0
    assert kappa_floored(thin, 0.0, 20) == 0.0
    assert kappa_floored(broad, 0.0, 20) == pytest.approx(kappa_floored(broad, 0.0))


def test_the_minimum_counts_books_that_cleared_the_floor_not_books_traded():
    """An agent trading a hundred books at negligible size has four alphas over the floor, and it is
    those four the minimum judges: the defect is the sample kappa runs on, not the breadth."""
    from taos.im.validator.debeta import kappa_floored
    alphas = [100.0, 101.0, 99.0, 100.0] + [0.1] * 96      # 100 books, 4 clear a floor of 10
    assert kappa_floored(alphas, 10.0) > 0
    assert kappa_floored(alphas, 10.0, 20) == 0.0


def test_the_dial_reaches_the_scorer():
    alphas, caps = _board()
    on = debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5, skill_min_books=20)
    off = debeta_scores(caps, caps, alphas, floor=0.0, w_make=0.5)
    assert on != off


def test_the_minimum_is_a_hard_cliff_and_that_is_the_residual_risk():
    """Exactly at the bar the full kappa is paid; one book below, nothing. Pinned deliberately.

    `kappa_of_alpha` has no sample-size term, so the minimum is a step rather than a taper, and a step
    is somewhere to stand: an operator whose edge survived to the bar would sit exactly on it. The
    setting works because the measured consistency of the accounts it targets has collapsed by the
    bar, not because the gate is structurally sound. If a sample-size aware estimator ever replaces
    this, this test should fail and the gate should be reconsidered rather than re-pinned.
    """
    from taos.im.validator.debeta import kappa_floored
    at_bar = _consistent(20)
    below = _consistent(19)
    assert kappa_floored(at_bar, 0.0, 20) == pytest.approx(kappa_of_alpha(at_bar))
    assert kappa_floored(below, 0.0, 20) == 0.0
    # and the estimator itself does not discount the smaller sample: that is why the gate exists
    assert kappa_of_alpha(_consistent(5)) > kappa_of_alpha(_consistent(60))


def test_the_defaults_are_the_off_path_on_randomised_boards():
    """The deploy-safety property, at board scale rather than on a four-uid fixture.

    Both dials ship defaulting to the 0.6.1 behaviour, so the commit that adds them must be a no-op
    until one is turned. A four-uid fixture cannot show that across magnitude floors, rank scopes and
    the making-floor path, and it is the whole basis on which this goes out ahead of a run boundary.
    """
    import random

    rng = random.Random(20260922)
    for _ in range(25):
        uids = list(range(rng.randint(40, 200)))
        alphas = {u: [rng.gauss(rng.uniform(-30, 60), rng.uniform(1, 180))
                      for _ in range(rng.randint(0, 128))] for u in uids}
        cb = {u: {b: max(0.0, rng.gauss(500, 900)) for b in range(rng.randint(0, 40))} for u in uids}
        cs = {u: {b: max(0.0, rng.gauss(500, 900)) for b in range(rng.randint(0, 40))} for u in uids}
        kw = dict(floor=rng.choice([0.0, 5.0, 17.3, 40.0]), w_make=rng.choice([0.5, 0.65]),
                  making_floor_scale=rng.choice([0.0, 0.5]))
        implicit = debeta_scores(cb, cs, alphas, **kw)
        explicit = debeta_scores(cb, cs, alphas, skill_min_books=4,
                                 skill_max_inactive_books=0.0, **kw)
        assert implicit == explicit
