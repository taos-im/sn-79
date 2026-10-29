# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""0.6.5's --neuron.reuse_connections keeps the query process's dendrite, and with it the miners' connections, across
rounds. That process parses its own command line, so the switch only takes effect if the validator passes it there
and the process's parser takes it; without both the child reads it unset and opens fresh connections every round,
whatever the validator was started with (found on the test bed: no kept connection in any round)."""
import argparse
import inspect
import shlex

import bittensor as bt

from taos.common.config import add_validator_args
from taos.im.neurons import validator
from taos.im.validator import query


def _launch_source():
    src = inspect.getsource(validator)
    start = src.index("Starting query service from")
    return src[start:src.index("self.query_process = subprocess.Popen", start)]


def test_the_validator_passes_the_switch_it_was_started_with():
    launch = _launch_source()
    assert "if getattr(self.config.neuron, 'reuse_connections', False):" in launch
    assert "cmd.append('--neuron.reuse_connections')" in launch


def test_the_query_process_reads_what_the_validator_passes():
    passed = shlex.split("--neuron.reuse_connections")
    assert bt.Config(query.service_parser(), args=passed).neuron.reuse_connections is True
    assert bt.Config(query.service_parser(), args=[]).neuron.reuse_connections is False


def test_the_query_process_follows_the_validators_default_and_its_off_switch():
    for args, want in (([], True), (["--neuron.no_reuse_connections"], False)):
        p = argparse.ArgumentParser()
        add_validator_args(None, p)
        started = bt.Config(p, args=args)
        passed = ["--neuron.reuse_connections"] if getattr(started.neuron, "reuse_connections", False) else []
        assert bt.Config(query.service_parser(), args=passed).neuron.reuse_connections is want, args
