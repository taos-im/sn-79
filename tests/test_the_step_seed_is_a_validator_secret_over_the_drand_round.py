# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The fundamental's step seed, hardened in three ways.

1. The round is chosen a period past the reply window's close, not at it, so a host clock slow by up to a period still
   picks a round published after the true close; and the round is never below the first after drand's own latest at
   the close, so a clock slow by more than a period cannot reach back either. The close against drand's latest is kept
   as a clock-skew sample, the smallest of the recent samples a gauge (negative: the host clock runs slow).
2. An HTTP round is taken when endpoints of two operators serve the same checked randomness; three of the four
   endpoints are one operator. When only one operator's endpoints agree, the chain's verified pulse is asked first and
   the single-operator value goes only in its absence, counted apart.
3. The seed sent is HMAC-SHA256(secret, randomness) under a secret the validator draws when a run starts and commits
   to (its sha256 published and recorded) and reveals when the run ends, so anyone can check afterwards that every
   seed came from the public round and none was chosen, while in-run nobody can rebuild the fundamental's innovations
   from the public rounds. The engine and a recorded replay read the value sent; the public input is recorded beside.
"""
import asyncio
import hashlib
import hmac
import json
from types import SimpleNamespace

from taos.im.validator import fundamental_seed as fs

S = 10 ** 9


class _Clock:
    def __init__(self, t):
        self.t = t

    def now(self):
        return self.t

    async def sleep(self, d):
        self.t += max(d, 0.01)


def _run(closed, http, chain, **kw):
    clock = _Clock(closed)

    async def http_get(r):
        return http(r, clock.t)

    async def chain_get(r):
        return chain(r, clock.t)

    return asyncio.run(fs.seed_for(closed, http_get, chain_get, now=clock.now, sleep=clock.sleep, **kw)), clock.t


# --- 1. the clock ------------------------------------------------------------------------------------------------------
def test_the_round_is_a_period_past_the_close_so_a_clock_slow_by_a_period_still_picks_a_round_after_it():
    assert fs.MARGIN_S == fs.PERIOD_S
    closed = fs.round_time(500) + 1.0
    (r, _rnd, src), t = _run(closed, lambda r, t: (b"\x07" * 32, "drand"), lambda r, t: None)
    assert src == "drand" and fs.round_time(r) >= closed + fs.MARGIN_S
    for slow_by in (0.0, 1.0, fs.PERIOD_S - 0.01):
        true_close = closed + slow_by
        assert fs.round_time(r) > true_close, f"a clock slow by {slow_by} s still gives a round after the true close"
    assert t - closed <= 2 * fs.PERIOD_S + 0.5


def test_the_round_is_never_below_the_first_after_drands_latest_at_the_close():
    closed = fs.round_time(500) + 1.0
    (r, _rnd, _src), _t = _run(closed, lambda r, t: (b"\x07" * 32, "drand"), lambda r, t: None, min_round=600)
    assert r == 600
    (r2, _rnd, _src), _t = _run(closed, lambda r, t: (b"\x07" * 32, "drand"), lambda r, t: None, min_round=3)
    assert r2 == fs.round_after(closed + fs.MARGIN_S), "a latest far behind changes nothing"


def test_the_skew_sample_is_the_close_against_the_latest_rounds_schedule():
    latest = 700
    closed = fs.round_time(latest) + 0.4
    assert abs(fs.clock_skew_sample(closed, latest) - 0.4) < 1e-6
    assert fs.clock_skew_sample(closed, None) is None
    v = SimpleNamespace()
    for s in (0.4, -0.2, 1.1):
        fs.note_clock_skew(v, s)
    assert fs.clock_skew_gauge(v) == -0.2, "the smallest recent sample: negative means the host clock runs slow"
    for _ in range(fs.CLOCK_SAMPLES + 5):
        fs.note_clock_skew(v, 2.0)
    assert fs.clock_skew_gauge(v) == 2.0 and len(v.fundamental_seed_clock) == fs.CLOCK_SAMPLES


# --- 2. two operators ----------------------------------------------------------------------------------------------------
def test_an_http_round_needs_two_operators_agreeing_and_one_operator_alone_is_counted_apart():
    a, b = b"\x01" * 32, b"\x02" * 32
    api, api2, cf = "https://api.drand.sh", "https://api2.drand.sh", "https://drand.cloudflare.com"
    assert fs.OPERATOR[api] == fs.OPERATOR[api2] != fs.OPERATOR[cf]
    assert fs.agreed([(api, a), (cf, a)]) == (a, "drand", False)
    assert fs.agreed([(api, a), (api2, a)]) == (a, "drand_one_operator", False), "one operator's pair, counted apart"
    assert fs.agreed([(api, a)]) == (None, None, False)
    assert fs.agreed([(api, a), (cf, b)]) == (None, None, True), "two that differ are a disagreement"
    assert fs.agreed([(api, a), (api2, a), (cf, b)]) == (a, "drand_one_operator", True)
    assert fs.agreed([(api, None), (cf, None)]) == (None, None, False)


def test_a_single_operator_round_yields_to_the_chain_and_stands_without_it():
    closed = fs.round_time(500) + 1.0
    (r, rnd, src), _t = _run(closed, lambda r, t: (b"\x05" * 32, "drand_one_operator"),
                             lambda r, t: b"\x09" * 32)
    assert (rnd, src) == (b"\x09" * 32, "chain"), "the chain's verified pulse first"
    (r, rnd, src), _t = _run(closed, lambda r, t: (b"\x05" * 32, "drand_one_operator"), lambda r, t: None)
    assert (rnd, src) == (b"\x05" * 32, "drand_one_operator"), "then the single operator's value, not random"


# --- 3. the secret -------------------------------------------------------------------------------------------------------
def _v(tmp_path, run="20261108_200000"):
    d = tmp_path / run
    d.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(simulation=SimpleNamespace(publish_interval=S, logDir=str(d)))


async def _fake_seed(closed_at, http_get, chain_get, now=None, sleep=None, min_round=0):
    return 42, b"\xab" * 32, "drand"


async def _no_latest():
    return None


def test_the_first_boundary_of_a_run_commits_a_secret_and_the_seed_sent_is_its_hmac_over_the_round(tmp_path):
    v = _v(tmp_path)
    batch = {"responses": [1]}
    out = asyncio.run(fs.attach(v, batch, 29 * S, closed_at=0.0, seed_fn=_fake_seed, latest_fn=_no_latest))
    secret = bytes.fromhex(v.fundamental_seed_secret["secret"])
    assert len(secret) == 32 and v.fundamental_seed_secret["run"] == "20261108_200000"
    assert v.fundamental_seed_secret["commit"] == hashlib.sha256(secret).hexdigest()
    value = hmac.new(secret, b"\xab" * 32, hashlib.sha256).digest()
    assert out["seed"] == [42, value] and value != b"\xab" * 32
    run_dir = tmp_path / "20261108_200000"
    assert (run_dir / "fundamental_seed.csv").read_text() == f"42,{value.hex()},{29 * S}\n", "what the engine replays"
    assert (run_dir / "fundamental_seed_drand.csv").read_text() == f"42,{'ab' * 32},{29 * S},drand\n", "the public input"
    assert json.loads((run_dir / "fundamental_seed_commit.json").read_text())["commit"] == v.fundamental_seed_secret["commit"]
    assert v.fundamental_seed_sources == {"drand": 1, "commit": 1}
    assert v.fundamental_seed_reveal is None


def test_a_new_run_reveals_the_old_secret_and_commits_a_new_one(tmp_path):
    v = _v(tmp_path)
    asyncio.run(fs.attach(v, {"responses": []}, 29 * S, closed_at=0.0, seed_fn=_fake_seed, latest_fn=_no_latest))
    old = dict(v.fundamental_seed_secret)
    v.simulation.logDir = str(tmp_path / "20261109_100000")
    (tmp_path / "20261109_100000").mkdir()
    asyncio.run(fs.attach(v, {"responses": []}, 29 * S, closed_at=0.0, seed_fn=_fake_seed, latest_fn=_no_latest))
    assert v.fundamental_seed_reveal == {"run": old["run"], "commit": old["commit"], "secret": old["secret"]}
    revealed = json.loads((tmp_path / old["run"] / "fundamental_seed_secret.json").read_text())
    assert revealed["secret"] == old["secret"] and revealed["commit"] == old["commit"]
    assert v.fundamental_seed_secret["run"] == "20261109_100000" and v.fundamental_seed_secret["secret"] != old["secret"]
    assert v.fundamental_seed_sources == {"drand": 2, "commit": 2, "reveal": 1}


def test_the_secret_and_the_skew_travel_through_the_saved_state_but_the_secret_never_to_the_report(tmp_path):
    v = _v(tmp_path)
    asyncio.run(fs.attach(v, {"responses": []}, 29 * S, closed_at=0.0, seed_fn=_fake_seed, latest_fn=_no_latest))
    fs.note_clock_skew(v, -0.3)
    saved = fs.state_to_save(v)
    assert saved["fundamental_seed_secret"] == v.fundamental_seed_secret and saved["fundamental_seed_clock"] == [-0.3]
    w = SimpleNamespace()
    fs.restore_state(w, json.loads(json.dumps(saved)))
    assert w.fundamental_seed_secret == v.fundamental_seed_secret and w.fundamental_seed_clock == [-0.3]
    for_report = fs.report_fields(v)
    assert for_report["fundamental_seed_commit"] == {"run": v.fundamental_seed_secret["run"],
                                                     "commit": v.fundamental_seed_secret["commit"]}
    assert "secret" not in json.dumps(for_report["fundamental_seed_commit"])
    assert for_report["fundamental_seed_reveal"] is None and for_report["fundamental_seed_clock"] == [-0.3]


def test_the_report_publishes_the_commitment_the_reveal_and_the_skew():
    from taos.im.validator import report
    st = SimpleNamespace(fundamental_seed_sources={"drand": 3, "commit": 1},
                         fundamental_seed_commit={"run": "r2", "commit": "c2"},
                         fundamental_seed_reveal={"run": "r1", "commit": "c1", "secret": "s1"},
                         fundamental_seed_clock=[0.2, -0.1])
    gauges = dict(report.fundamental_seed_gauges(st))
    assert gauges["fundamental_seed_commit"] == 1.0 and gauges["fundamental_seed_clock_skew_s"] == -0.1
    assert gauges["fundamental_seed_reveal"] == 0.0
    info = dict(report.fundamental_seed_info(st))
    assert info["commitment"] == {"run": "r2", "commit": "c2"}
    assert info["reveal"] == {"run": "r1", "commit": "c1", "secret": "s1"}
    assert dict(report.fundamental_seed_info(SimpleNamespace())) == {}


def test_the_validator_sends_the_report_the_seed_fields_and_saves_the_secret():
    import inspect
    from taos.im.neurons import validator as vmod
    from taos.im.validator import persistence
    src = inspect.getsource(vmod)
    assert "**fundamental_seed.report_fields(self)" in src
    assert "fundamental_seed.state_to_save(self)" in inspect.getsource(persistence)
    assert "fundamental_seed.restore_state(self" in inspect.getsource(persistence)
