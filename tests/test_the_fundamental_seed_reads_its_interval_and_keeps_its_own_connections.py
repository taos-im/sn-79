# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The fundamental seed's loose ends, the validator's side.

The seed's interval is the class files' own seedInterval, not a second constant beside them: a layout whose classes
reseed every 20 simulated seconds gets a seed at every 20-second boundary. A switch skips drand: off, the validator
asks no drand endpoint, takes the chain's Drand pulse where it can and sends a random round 0 at once where it cannot.
drand's endpoints are asked through one HTTP session per event loop, not one per poll. The chain fallback asks the
Drand pallet over a connection of its own: the maintenance subtensor's websocket is not shared with the seed's
thread."""
import asyncio
import sys
import types
from types import SimpleNamespace

from taos.im.validator import fundamental_seed as fs

S = 10 ** 9


def _layout(*intervals):
    classes = [SimpleNamespace(config=SimpleNamespace(fp_seed_interval=i)) for i in intervals]
    return SimpleNamespace(publish_interval=S, logDir=None, asset_classes=lambda: classes)


# ── the interval is the class files' ─────────────────────────────────────────────────────────────────────────────

def test_the_interval_is_the_class_files_seed_interval():
    assert fs.seed_intervals(_layout(20 * S, 20 * S)) == (20 * S,)
    assert fs.seed_intervals(_layout(20 * S, 30 * S)) == (20 * S, 30 * S), "each class reseeds on its own interval"
    assert fs.seed_intervals(_layout(None)) == (fs.SEED_INTERVAL_NS,), "a layout that states none: the default"
    assert fs.seed_intervals(SimpleNamespace(publish_interval=S)) == (fs.SEED_INTERVAL_NS,)


def test_the_published_layout_reseeds_every_30_seconds_in_every_class():
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from taos.im.protocol.config import MultiAssetSimulationConfig
    cfg = Path(__file__).resolve().parents[1] / "simulate" / "trading" / "run" / "config"
    sim = MultiAssetSimulationConfig.from_multiasset_xml(ET.parse(cfg / "multiasset_simulation_0.xml").getroot(), cfg)
    assert fs.seed_intervals(sim) == (30 * S,)


def test_a_window_crossing_any_classs_boundary_carries_the_seed():
    assert fs.crosses_seed_boundary(19 * S, S, (20 * S,)) and not fs.crosses_seed_boundary(29 * S, S, (20 * S,))
    assert fs.crosses_seed_boundary(29 * S, S, (20 * S, 30 * S))
    assert fs.crosses_seed_boundary(29 * S, S) and not fs.crosses_seed_boundary(19 * S, S), "default 30 s"


def test_attach_sends_on_the_layouts_interval():
    v = SimpleNamespace(simulation=_layout(20 * S))

    async def fake_seed(closed_at, http_get, chain_get, now=None, sleep=None, min_round=0):
        return 42, b"\xab" * 32, "drand"

    async def no_latest():
        return None

    batch = {"responses": []}
    out = asyncio.run(fs.attach(v, batch, 19 * S, closed_at=0.0, seed_fn=fake_seed, latest_fn=no_latest))
    assert out["seed"][0] == 42
    assert asyncio.run(fs.attach(v, batch, 29 * S, closed_at=0.0, seed_fn=fake_seed, latest_fn=no_latest)) is batch


# ── the switch ────────────────────────────────────────────────────────────────────────────────────────────────────

def _switched(on, subtensor=None):
    cfg = SimpleNamespace(simulation=SimpleNamespace(seeding=SimpleNamespace(drand=on)),
                          subtensor=SimpleNamespace(chain_endpoint="ws://seed-test:9944"))
    return SimpleNamespace(simulation=_layout(30 * S), config=cfg, subtensor=subtensor)


def test_the_switch_reads_the_config_and_defaults_on():
    assert fs.drand_enabled(_switched(True)) and not fs.drand_enabled(_switched(False))
    assert fs.drand_enabled(SimpleNamespace()), "a validator without the option asks drand"


def test_drand_off_asks_no_endpoint_and_without_a_chain_sends_round_0_at_once():
    asked = []

    async def latest():
        asked.append("latest")
        return 99

    v = _switched(False)
    out = asyncio.run(fs.attach(v, {"responses": []}, 29 * S, closed_at=0.0, latest_fn=latest,
                                seed_fn=lambda *a, **k: asked.append("seed_for")))
    assert asked == [], "neither drand's latest round nor the round wait"
    assert out["seed"][0] == 0 and v.fundamental_seed_sources.get("random") == 1


def test_drand_off_with_a_chain_asks_the_chain_and_never_http(monkeypatch):
    seen = {}

    async def fake_seed(closed_at, http_get, chain_get, now=None, sleep=None, min_round=0):
        seen["http"] = await http_get(5)
        seen["min_round"] = min_round
        return 5, b"\xcd" * 32, "chain"

    async def latest():
        raise AssertionError("drand's latest round asked with drand off")

    monkeypatch.setattr(fs, "http_randomness", lambda *a, **k: (_ for _ in ()).throw(AssertionError("http asked")))
    v = _switched(False, subtensor=object())
    out = asyncio.run(fs.attach(v, {"responses": []}, 29 * S, closed_at=0.0, seed_fn=fake_seed, latest_fn=latest))
    assert seen == {"http": None, "min_round": 0} and out["seed"][0] == 5 and v.fundamental_seed_sources["chain"] == 1


# ── one HTTP session ──────────────────────────────────────────────────────────────────────────────────────────────

def test_drands_endpoints_are_asked_through_one_session_per_loop(monkeypatch):
    made = []

    class _Resp:
        status = 404

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def __init__(self, timeout=None):
            made.append(self)

        def get(self, url):
            return _Resp()

    mod = types.ModuleType("aiohttp")
    mod.ClientTimeout = lambda total=None: None
    mod.ClientError = type("ClientError", (Exception,), {})
    mod.ClientSession = _Session
    monkeypatch.setitem(sys.modules, "aiohttp", mod)

    async def polls():
        for r in (7, 8, 9):
            await fs.http_randomness(r, verifier=None)
        await fs.http_latest()
        await fs.http_latest()

    asyncio.run(polls())
    assert len(made) == 1, f"{len(made)} sessions for five polls in one loop"
    asyncio.run(polls())
    assert len(made) == 2, "a new loop gets a session of its own"


# ── the chain fallback's own connection ───────────────────────────────────────────────────────────────────────────

def test_the_chain_fallback_asks_a_connection_of_its_own_on_the_validators_endpoint(monkeypatch):
    sig = b"\x02" * 48
    import hashlib
    pulse = {"round": 7, "signature": sig.hex(), "randomness": hashlib.sha256(sig).hexdigest()}
    opened = []

    class _Own:
        def __init__(self, network=None, config=None):
            opened.append(network)
            self.substrate = SimpleNamespace(query=lambda pallet, item, args: SimpleNamespace(value=pulse))

    class _Maintenance:
        @property
        def substrate(self):
            raise AssertionError("the maintenance subtensor's websocket was used")

    monkeypatch.setattr(fs.bt, "Subtensor", _Own)
    v = _switched(True, subtensor=_Maintenance())
    get = fs.seed_chain_getter(v)
    got = asyncio.run(get(7))
    assert got == fs.pulse_randomness(pulse, 7) and got is not None
    assert asyncio.run(get(7)) == got
    assert opened == ["ws://seed-test:9944"], "one connection, opened once, on the validator's chain endpoint"


def test_a_failed_chain_connection_answers_none_and_is_opened_again_next_time(monkeypatch):
    tries = []

    def _broken(network=None, config=None):
        tries.append(network)
        raise ConnectionError("no chain")

    monkeypatch.setattr(fs.bt, "Subtensor", _broken)
    v = _switched(True, subtensor=object())
    get = fs.seed_chain_getter(v)
    assert asyncio.run(get(7)) is None and asyncio.run(get(8)) is None
    assert len(tries) == 2


def test_the_switch_is_an_option_on_by_default_and_turned_off_by_false():
    import argparse
    from taos.im.config.simulation import add_simulation_args
    parser = argparse.ArgumentParser()
    add_simulation_args(None, parser)
    key = "simulation.seeding.drand"
    assert getattr(parser.parse_args([]), key) is True
    assert getattr(parser.parse_args(["--simulation.seeding.drand", "false"]), key) is False
    assert getattr(parser.parse_args(["--simulation.seeding.drand"]), key) is True
