# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator queries the miners with their payloads built in the main process and the checked orders delayed and
encoded in the query process, and it freezes its loaded heap with larger young collections: each on by default.
Building and signing a round's requests before the first is sent stays off by default. Each is an option with its
own on and off switch, so one can be changed without the others, and each is read from the configuration the
validator was started with, never from the environment. The query process parses its own command line, so the one
switch it reads reaches it from the validator's, in both directions."""
import argparse
import inspect
import pathlib
import re
from types import SimpleNamespace

import bittensor as bt

from taos.im.config import add_im_validator_args
from taos.im.neurons import validator
from taos.im.validator import forward, query

SWITCHES = ("query_prebuilt", "query_prepayload", "orders_packed")
DEFAULTS = {"query_prebuilt": False, "query_prepayload": True, "orders_packed": True, "gc_freeze": True}
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _parse(args):
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    return parser.parse_args(args)


def _on(ns, name):
    return getattr(ns, f"neuron.{name}")


def _values(ns):
    return {name: _on(ns, name) for name in DEFAULTS}


def test_the_defaults():
    ns = _parse([])
    assert _values(ns) == DEFAULTS
    assert _on(ns, "gc_threshold") == "50000,20,100"
    assert _on(ns, "skip_repeated_state") is False


def test_each_switch_turns_off_alone():
    for name in DEFAULTS:
        assert _values(_parse([f"--neuron.no_{name}"])) == {**DEFAULTS, name: False}, name


def test_each_switch_turns_on_alone():
    for name in DEFAULTS:
        assert _values(_parse([f"--neuron.{name}"])) == {**DEFAULTS, name: True}, name


def test_the_on_flags_launchers_pass_still_parse():
    ns = _parse(["--neuron.gc_freeze", "--neuron.gc_threshold", "50000,20,100"]
                + [f"--neuron.{s}" for s in SWITCHES])
    assert all(_on(ns, s) is True for s in SWITCHES + ("gc_freeze",))
    assert _on(ns, "gc_threshold") == "50000,20,100"
    assert _on(_parse(["--neuron.gc_threshold", ""]), "gc_threshold") == ""


def _launch_source():
    src = inspect.getsource(validator)
    start = src.index("Starting query service from")
    return src[start:src.index("self.query_process = subprocess.Popen", start)]


def test_the_validator_passes_the_prebuilt_switch_to_the_query_process():
    launch = _launch_source()
    assert "if getattr(self.config.neuron, 'query_prebuilt', False):" in launch
    assert "cmd.append('--neuron.query_prebuilt')" in launch
    assert bt.Config(query.service_parser(), args=["--neuron.query_prebuilt"]).neuron.query_prebuilt is True
    assert bt.Config(query.service_parser(), args=[]).neuron.query_prebuilt is False


def test_the_query_process_follows_the_validators_default_and_both_switches():
    for args, want in (([], False), (["--neuron.query_prebuilt"], True), (["--neuron.no_query_prebuilt"], False)):
        parser = argparse.ArgumentParser()
        add_im_validator_args(None, parser)
        started = bt.Config(parser, args=args)
        passed = ["--neuron.query_prebuilt"] if getattr(started.neuron, "query_prebuilt", False) else []
        assert bt.Config(query.service_parser(), args=passed).neuron.query_prebuilt is want, args


def _service(**neuron):
    return SimpleNamespace(config=SimpleNamespace(neuron=SimpleNamespace(**neuron)))


def test_the_query_process_reads_its_config_not_the_environment(monkeypatch):
    monkeypatch.setenv("QUERY_PREBUILT", "1")
    assert query._prebuilt(_service(query_prebuilt=False)) is False
    assert query._prebuilt(_service()) is False
    monkeypatch.delenv("QUERY_PREBUILT")
    assert query._prebuilt(_service(query_prebuilt=True)) is True


def test_no_environment_variable_carries_a_switch():
    names = "|".join(s.upper() for s in SWITCHES)
    read = re.compile(rf"os\.environ(\.get)?\s*[\[(]\s*['\"]({names})['\"]|getenv\(\s*['\"]({names})['\"]")
    hits = [f"{p.relative_to(ROOT)}:{n}" for p in sorted((ROOT / "taos").rglob("*.py"))
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if read.search(line)]
    assert not hits, f"a switch is still read from the environment: {hits}"


def test_the_main_process_reads_its_two_switches_from_its_config_in_simulation_mode_only():
    src = inspect.getsource(forward)
    assert "getattr(self.config.neuron, 'orders_packed', False) and self.engine.mode != 'exchange'" in src
    assert ("getattr(self.config.neuron, 'query_prepayload', False) and request_data['engine_mode'] != 'exchange'"
            in src)
