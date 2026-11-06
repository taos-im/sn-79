# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A neuron logs its config at start; a secret in it is logged as "set", never its value.

The config line goes to the process log, and the log travels further than the host. The PagerDuty integration key
and the gradient server's API key are settings like any other, so the logged config shows each as "set".
"""
import pathlib
import re
import sys

import bittensor as bt
import pytest

from taos.common.config import loggable_config, prune_config
from taos.common.neurons import BaseNeuron
from taos.im.neurons.validator import Validator

PD_KEY = "DUMMYPAGERDUTYKEY0123456789abcdef"
GX_KEY = "DUMMYGENTRXKEY0123456789"


class _Neuron(BaseNeuron):
    def run(self):
        pass

    def resync_metagraph(self):
        pass


class _Started(Exception):
    pass


def _argv(tmp_path, *extra):
    return ["neuron.py", "--netuid", "79", "--logging.logging_dir", str(tmp_path), "--neuron.dont_save_events", *extra]


def _config(monkeypatch, tmp_path, *extra):
    """The validator's config, built as the neuron's start builds it."""
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, *extra))
    base = BaseNeuron.config()
    cfg = Validator.config()
    cfg.merge(base)
    prune_config(cfg, Validator)
    return cfg


def test_a_secret_that_is_set_is_logged_as_set(monkeypatch, tmp_path):
    cfg = _config(
        monkeypatch, tmp_path, "--alerting.pagerduty.integration_key", PD_KEY, "--gentrx.api_key", GX_KEY
    )
    shown = loggable_config(cfg)
    assert PD_KEY not in shown and GX_KEY not in shown
    assert re.search(r"integration_key: set\b", shown) and re.search(r"api_key: set\b", shown)
    assert "netuid: 79" in shown


def test_the_config_itself_keeps_its_secrets(monkeypatch, tmp_path):
    cfg = _config(monkeypatch, tmp_path, "--alerting.pagerduty.integration_key", PD_KEY)
    loggable_config(cfg)
    assert cfg.alerting.pagerduty.integration_key == PD_KEY


def test_a_secret_that_is_not_set_is_logged_as_it_is(monkeypatch, tmp_path):
    shown = loggable_config(_config(monkeypatch, tmp_path))
    assert not re.search(r"integration_key: set\b", shown) and not re.search(r"api_key: set\b", shown)
    assert re.search(r"integration_key: null\b", shown) and re.search(r"api_key: ''", shown), "the unset value is shown"


def test_the_neurons_start_logs_no_secret(monkeypatch, tmp_path):
    logged = []

    def info(msg, *args, **kwargs):
        logged.append(str(msg))
        if str(msg).startswith("Setting up bittensor objects"):
            raise _Started

    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--alerting.pagerduty.integration_key", PD_KEY))
    monkeypatch.setattr(bt.logging, "info", info)
    with pytest.raises(_Started):
        _Neuron()
    config_lines = [line for line in logged if line.startswith("Config:")]
    assert config_lines and not any(PD_KEY in line for line in config_lines)


def test_no_module_logs_a_whole_config_but_through_loggable_config():
    root = pathlib.Path(__file__).resolve().parents[1] / "taos"
    whole = re.compile(r"""logging\.\w+\(\s*f?["'][^"']*\{\s*(self\.)?config\s*\}""")
    found = [
        f"{path.relative_to(root.parent)}:{n}"
        for path in root.rglob("*.py")
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if whole.search(line)
    ]
    assert found == []
