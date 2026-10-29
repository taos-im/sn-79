# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The base delay a response carries into the simulation decides its place in the queue among the quotes of the same
tick, since the engine orders equal timestamps by insertion, so the mapping from process time to delay is a market
design dial rather than a constant. Three policies: curve, the exponential mapping the validator has always applied;
flat, the minimum delay for every response within the timeout, the responses submitted in a random order; jitter, a
uniform band over the minimum for a prompt response and the curve beyond a cutoff. A latency profile substitutes a
configured process time per uid, so two policies can be compared on identical latencies."""
import argparse
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from taos.im.config import add_im_validator_args
from taos.im.protocol import instructions as I
from taos.im.protocol.models import OrderDirection
from taos.im.protocol.response import FinanceAgentResponse
from taos.im.validator import delays as delays_module
from taos.im.validator.delays import DELAY_POLICIES, base_delay, latency_profile, set_delays

TIMEOUT, MIN, MAX = 3.0, 10_000_000, 1_000_000_000
MIN_I, MAX_I = 5_000_000, 25_000_000
JITTER, CUTOFF = 40_000_000, 0.5


def _todays_curve(p_time, timeout=TIMEOUT, min_delay=MIN, max_delay=MAX):
    """The shipped mapping, operation for operation, so the default policy is checked against it and not itself."""
    t = p_time / timeout
    exp_scale = 5
    delay_frac = (np.exp(exp_scale * t) - 1) / (np.exp(exp_scale) - 1)
    return int(min_delay + delay_frac * (max_delay - min_delay))


def _validator(**scoring):
    dials = dict(min_delay=MIN, max_delay=MAX, min_instruction_delay=MIN_I, max_instruction_delay=MAX_I)
    dials.update(scoring)
    return SimpleNamespace(config=SimpleNamespace(neuron=SimpleNamespace(timeout=TIMEOUT), scoring=SimpleNamespace(**dials)))


def _limit(**kw):
    base = dict(agentId=3, bookId=0, direction=OrderDirection.BUY, quantity=1.0, price=100.0, clientOrderId=None)
    base.update(kw)
    return I.PlaceLimitOrderInstruction(**base)


def _synapses(times, books=(0,), delay=0):
    return {uid: SimpleNamespace(response=FinanceAgentResponse(agent_id=uid, instructions=[_limit(agentId=uid, bookId=b, delay=delay) for b in books]),
                                 dendrite=SimpleNamespace(process_time=t))
            for uid, t in times.items()}


def _first_delays(out):
    """Per uid, the delay of its first instruction: the base delay alone, since the first instruction of a book gets no
    instruction delay and the instructions were sent with delay 0."""
    return {r.agent_id: r.instructions[0].delay for r in out}


def test_the_default_policy_is_the_curve_the_validator_has_always_applied():
    for p in (0.0, 0.05, 0.12, 0.5, 1.2, 2.9, TIMEOUT, 4.0):
        assert base_delay(p, TIMEOUT, MIN, MAX) == _todays_curve(p)
        assert base_delay(p, TIMEOUT, MIN, MAX, policy="curve", jitter=JITTER, prompt_cutoff=CUTOFF) == _todays_curve(p)
    assert 11_400_000 <= base_delay(0.12, TIMEOUT, MIN, MAX) <= 11_600_000, "about 11.5 ms at 0.12 s"
    assert 52_500_000 <= base_delay(1.2, TIMEOUT, MIN, MAX) <= 53_500_000, "about 53 ms at 1.2 s"
    assert base_delay(TIMEOUT, TIMEOUT, MIN, MAX) == MAX


def test_a_configuration_without_the_new_dials_applies_the_curve_through_set_delays():
    times = {1: 0.0, 2: 0.12, 3: 1.2, 4: TIMEOUT, 5: 3.4}
    out = set_delays(_validator(), _synapses(times))
    assert [r.agent_id for r in out] == [1, 2, 3, 4, 5], "the curve keeps the order the responses arrived in"
    assert _first_delays(out) == {uid: _todays_curve(t) for uid, t in times.items()}


def test_flat_gives_the_minimum_delay_to_every_response_within_the_timeout_and_shuffles_the_order():
    times = {uid: 0.05 + 0.07 * uid for uid in range(40)}       # 0.05 s to 2.78 s, all within the timeout
    times[40] = TIMEOUT
    times[41] = 3.5
    random.seed(11)
    out = set_delays(_validator(delay_policy="flat"), _synapses(times))
    delays = _first_delays(out)
    assert all(delays[uid] == MIN for uid in range(40))
    assert delays[40] == _todays_curve(TIMEOUT) and delays[41] == _todays_curve(3.5), "at or over the timeout nothing changes"
    order = [r.agent_id for r in out]
    assert sorted(order) == sorted(times) and len(order) == len(times), "every response once"
    assert order != sorted(times), "the engine keeps insertion order among equal timestamps, so the list is shuffled"


def test_jitter_draws_within_the_band_under_the_cutoff_and_follows_the_curve_above_it():
    rng = random.Random(5)
    draws = {base_delay(0.2, TIMEOUT, MIN, MAX, policy="jitter", jitter=JITTER, prompt_cutoff=CUTOFF, rng=rng) for _ in range(300)}
    assert all(MIN <= d <= MIN + JITTER for d in draws)
    assert len(draws) > 1, "a band, not a constant"
    at_cutoff = base_delay(CUTOFF, TIMEOUT, MIN, MAX, policy="jitter", jitter=JITTER, prompt_cutoff=CUTOFF, rng=rng)
    assert MIN <= at_cutoff <= MIN + JITTER, "the cutoff is inclusive"
    for p in (0.51, 1.2, 2.9, TIMEOUT, 3.5):
        assert base_delay(p, TIMEOUT, MIN, MAX, policy="jitter", jitter=JITTER, prompt_cutoff=CUTOFF, rng=rng) == _todays_curve(p)
    random.seed(2)
    times = {1: 0.0, 2: 0.3, 3: 0.5, 4: 0.6, 5: 1.2, 6: TIMEOUT}
    delays = _first_delays(set_delays(_validator(delay_policy="jitter", delay_jitter=JITTER, delay_prompt_cutoff=CUTOFF), _synapses(times)))
    for uid in (1, 2, 3):
        assert MIN <= delays[uid] <= MIN + JITTER
    for uid in (4, 5, 6):
        assert delays[uid] == _todays_curve(times[uid])


def test_the_latency_profile_replaces_the_measured_time_for_listed_uids_only_and_is_read_once(tmp_path, monkeypatch):
    infos = []
    monkeypatch.setattr(delays_module.bt.logging, "info", lambda msg, *a, **k: infos.append(str(msg)))
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"3": 1.2, "5": 0.1}))
    v = _validator(latency_profile=str(path))
    delays = _first_delays(set_delays(v, _synapses({3: 0.05, 4: 0.05, 5: 0.05})))
    assert delays == {3: _todays_curve(1.2), 4: _todays_curve(0.05), 5: _todays_curve(0.1)}
    assert any("2 uids" in m for m in infos), "one line at load with the count of uids"
    path.write_text(json.dumps({"3": 2.5}))
    delays = _first_delays(set_delays(v, _synapses({3: 0.05, 4: 0.05})))
    assert delays[3] == _todays_curve(1.2), "the file is read once and kept on the validator"
    assert latency_profile(v) == {3: 1.2, 5: 0.1}


def test_a_profile_that_cannot_be_read_warns_once_and_leaves_every_measured_time_in_place(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(delays_module.bt.logging, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
    bad = tmp_path / "profile.json"
    bad.write_text("not json")
    v = _validator(latency_profile=str(bad))
    times = {3: 0.05, 4: 1.2}
    assert _first_delays(set_delays(v, _synapses(times))) == {uid: _todays_curve(t) for uid, t in times.items()}
    assert len(warnings) == 1 and str(bad) in warnings[0]
    set_delays(v, _synapses(times))
    assert len(warnings) == 1, "disabled after the warning, not retried every round"
    missing = _validator(latency_profile=str(tmp_path / "absent.json"))
    assert _first_delays(set_delays(missing, _synapses(times))) == {uid: _todays_curve(t) for uid, t in times.items()}
    assert len(warnings) == 2
    assert latency_profile(_validator()) == {}, "an unset dial is no profile and no warning"
    assert len(warnings) == 2


@pytest.mark.parametrize("policy", DELAY_POLICIES)
def test_the_first_instruction_per_book_gets_no_instruction_delay_under_every_policy(policy):
    random.seed(9)
    v = _validator(delay_policy=policy, delay_jitter=JITTER, delay_prompt_cutoff=CUTOFF)
    out = set_delays(v, _synapses({7: 0.2}, books=(0, 1, 0)))
    first_a, first_b, second_a = (i.delay for i in out[0].instructions)
    assert first_a == first_b, "the first instruction on each book carries the base delay alone"
    assert MIN_I <= second_a - first_a <= MAX_I, "a later instruction on the same book adds the instruction delay"


@pytest.mark.parametrize("policy", DELAY_POLICIES)
def test_the_wire_bound_still_holds_the_sum_under_every_policy(policy):
    random.seed(13)
    v = _validator(delay_policy=policy, delay_jitter=JITTER, delay_prompt_cutoff=CUTOFF)
    out = set_delays(v, _synapses({3: 0.2, 4: 2.9}, books=(0, 0, 1), delay=I.MAX_INSTRUCTION_DELAY))
    for r in out:
        for instruction in r.instructions:
            assert 0 <= instruction.delay <= I.MAX_INSTRUCTION_DELAY


def test_the_dials_are_declared_with_the_curve_as_the_default():
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    config = parser.parse_args([])
    assert getattr(config, "scoring.delay_policy") == "curve"
    assert getattr(config, "scoring.delay_jitter") == JITTER
    assert getattr(config, "scoring.delay_prompt_cutoff") == CUTOFF
    assert getattr(config, "scoring.latency_profile") == ""
    with pytest.raises(SystemExit):
        parser.parse_args(["--scoring.delay_policy", "fastest"])
    assert set(DELAY_POLICIES) == {"curve", "flat", "jitter"}


def test_the_query_process_delays_by_the_policy_and_the_profile_main_holds(tmp_path):
    """With neuron.orders_packed the query process applies set_delays to the settings main hands it each round, so they
    carry the policy dials and main's loaded profile: both processes delay a response the same way, and the query
    process reads no file."""
    from taos.im.validator.forward import delay_settings
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"3": 1.2}))
    v = _validator(delay_policy="jitter", delay_jitter=JITTER, delay_prompt_cutoff=CUTOFF, latency_profile=str(path))
    v.simulation = SimpleNamespace(time_unit="ns")
    settings = delay_settings(v)
    assert (settings.config.scoring.delay_policy, settings.config.scoring.delay_jitter,
            settings.config.scoring.delay_prompt_cutoff) == ("jitter", JITTER, CUTOFF)
    path.unlink()
    delays = _first_delays(set_delays(settings, _synapses({3: 0.05, 4: 0.05})))
    assert delays[3] == _todays_curve(1.2), "uid 3 takes its profile time, past the cutoff, so the curve"
    assert MIN <= delays[4] <= MIN + JITTER, "uid 4 is measured, prompt, so the jitter band"
