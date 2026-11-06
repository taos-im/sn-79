# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The fundamental's step seed is a drand quicknet round published after the miners' replies are in.

Every 30 simulated seconds the engine draws each book's fundamental step from a seed. The validator sends the seed
with the batch answering the state whose reply window crosses that boundary: the first quicknet round scheduled
strictly after the validator closed the reply window, so no reply can have been decided on it. It takes the round
from drand's HTTP endpoints, checking that the randomness is the sha256 of its signature, else from the chain's Drand
pallet, and after RANDOM_AFTER_S without either sends 32 random bytes as round 0. An HTTP round is taken only when two
endpoints serve the same checked value, a disagreement counted; the chain's pulses are verified by the chain. Each
seed sent is appended to the run's fundamental_seed.csv as round,randomness_hex,sim_time_ns, which a recorded replay
reads, and its source is counted for the gauges."""
import asyncio
import hashlib
import inspect
from types import SimpleNamespace

import pytest

from taos.im.validator import fundamental_seed as fs

S = 10 ** 9


def _beacon(r, sig=b"\x01" * 48):
    return {"round": r, "signature": sig.hex(), "randomness": hashlib.sha256(sig).hexdigest()}


def test_the_seed_goes_with_the_batch_whose_reply_window_crosses_a_boundary():
    assert fs.SEED_INTERVAL_NS == 30 * S
    assert fs.crosses_seed_boundary(29 * S, S) and fs.crosses_seed_boundary(59 * S, S)
    assert not fs.crosses_seed_boundary(30 * S, S) and not fs.crosses_seed_boundary(28 * S, S)
    assert fs.crosses_seed_boundary(27 * S, 5 * S) and not fs.crosses_seed_boundary(29 * S, 0)


def test_the_round_is_the_first_scheduled_strictly_after_the_window_closed():
    t100 = fs.round_time(100)
    assert fs.round_time(101) - t100 == fs.PERIOD_S
    assert fs.round_after(t100) == 101 and fs.round_after(t100 + 0.5) == 101 and fs.round_after(t100 - 0.1) == 100


def test_a_beacon_is_read_only_for_its_round_and_with_randomness_matching_its_signature():
    b = _beacon(7)
    assert fs.randomness_of(b, 7) == bytes.fromhex(b["randomness"])
    assert fs.randomness_of(b, 8) is None
    assert fs.randomness_of(dict(b, randomness="00" * 32), 7) is None
    assert fs.randomness_of({"round": 7}, 7) is None


class _Clock:
    def __init__(self, t):
        self.t = t

    def now(self):
        return self.t

    async def sleep(self, d):
        self.t += max(d, 0.01)


def _run(closed, http, chain):
    clock = _Clock(closed)
    asked = []

    async def http_get(r):
        asked.append((r, clock.t))
        return http(r, clock.t)

    async def chain_get(r):
        return chain(r, clock.t)

    out = asyncio.run(fs.seed_for(closed, http_get, chain_get, now=clock.now, sleep=clock.sleep))
    return out, asked, clock.t


def test_the_round_is_taken_from_drand_once_it_is_published():
    closed = fs.round_time(500) + 1.0
    (r, rnd, src), asked, t = _run(closed, lambda r, t: b"\x07" * 32, lambda r, t: None)
    assert (r, rnd, src) == (501, b"\x07" * 32, "drand")
    assert all(at >= fs.round_time(501) for _r, at in asked), "never asked before the round is scheduled"
    assert t - closed <= fs.PERIOD_S + 0.5


def test_the_chain_answers_when_drand_does_not():
    closed = fs.round_time(500) + 1.0
    (r, rnd, src), _asked, _t = _run(closed, lambda r, t: None,
                                     lambda r, t: b"\x09" * 32 if t > fs.round_time(501) + 1 else None)
    assert (r, rnd, src) == (501, b"\x09" * 32, "chain")


def test_after_the_wait_a_random_seed_goes_as_round_zero():
    closed = fs.round_time(500) + 1.0
    (r, rnd, src), _asked, t = _run(closed, lambda r, t: None, lambda r, t: None)
    assert r == 0 and src == "random" and len(rnd) == 32 and t - closed >= fs.RANDOM_AFTER_S


def test_attach_sends_the_seed_records_it_and_counts_its_source(tmp_path):
    v = SimpleNamespace(simulation=SimpleNamespace(publish_interval=S, logDir=str(tmp_path)))

    async def fake_seed(closed_at, http_get, chain_get, now=None, sleep=None):
        return 42, b"\xab" * 32, "drand"

    batch = {"responses": [1], "capped": []}
    out = asyncio.run(fs.attach(v, batch, 29 * S, closed_at=0.0, seed_fn=fake_seed))
    assert out == {"responses": [1], "capped": [], "seed": [42, b"\xab" * 32]}
    assert (tmp_path / "fundamental_seed.csv").read_text() == f"42,{'ab' * 32},{29 * S}\n"
    assert v.fundamental_seed_sources == {"drand": 1}
    same = asyncio.run(fs.attach(v, batch, 30 * S, closed_at=0.0, seed_fn=fake_seed))
    assert same is batch, "no boundary in the reply window: the batch goes as it is"


def test_the_validator_attaches_the_seed_to_the_simulation_batch_after_the_replies():
    from taos.im.neurons import validator as vmod
    src = inspect.getsource(vmod)
    i = src.index("response = with_capped(response_batch(miner_responses), getattr(self, '_capped_pairs', None))")
    j = src.index("response = await fundamental_seed.attach(self, response, state.timestamp, closed_at=_rp_fwd1)")
    assert 0 < j - i < 400


def test_the_seed_service_does_not_write_the_fundamental_seed_file():
    from taos.im.validator import seed
    assert "fundamental_seed.csv" not in inspect.getsource(seed)


def test_an_http_round_needs_two_endpoints_agreeing():
    a, b = b"\x01" * 32, b"\x02" * 32
    assert fs.agreed([a, a, None, None]) == (a, False)
    assert fs.agreed([a, None, None, None]) == (None, False), "one endpoint alone is not enough"
    assert fs.agreed([a, b, None, None]) == (None, True), "two that differ are a disagreement"
    assert fs.agreed([a, a, b, None]) == (a, True), "a majority of two is taken, the odd one counted"
    assert fs.agreed([None] * 4) == (None, False)


def test_the_sources_reach_the_validator_gauges():
    from taos.im.validator import report
    st = SimpleNamespace(fundamental_seed_sources={"drand": 3, "random": 1, "drand_disagree": 2})
    out = dict(report.fundamental_seed_gauges(st))
    assert out == {"fundamental_seed_drand": 3.0, "fundamental_seed_chain": 0.0, "fundamental_seed_random": 1.0,
                   "fundamental_seed_drand_disagree": 2.0}


@pytest.mark.parametrize("field", ["randomness", "signature"])
def test_a_chain_pulse_is_read_as_a_beacon(field):
    pulse = {"round": 9, "randomness": "0x" + hashlib.sha256(b"\x02" * 48).hexdigest(), "signature": "0x" + ("02" * 48)}
    assert fs.pulse_randomness(pulse, 9) == hashlib.sha256(b"\x02" * 48).digest()
    assert fs.pulse_randomness(dict(pulse, **{field: "0x00"}), 9) is None
