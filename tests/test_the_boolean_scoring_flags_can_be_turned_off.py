# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The reward floor and the benchmark agents are switched by flags that can be turned off.

Both parse as --scoring.debeta.publish_book_gauges does (_flag, nargs="?", const=True): the bare flag and "true" are
on, false|0|no|off is off, and the defaults stay (the floor on, the benchmark off)."""
import argparse

import pytest

from taos.im.config import add_im_validator_args

FLAGS = (("rewarding.floor.enabled", True), ("benchmark.enabled", False))
NAMES = [name for name, _ in FLAGS]


def _parse(*argv):
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    known, _ = parser.parse_known_args(list(argv))
    return known


@pytest.mark.parametrize("name,default", FLAGS)
def test_the_defaults_stay(name, default):
    assert getattr(_parse(), name) is default


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off"])
def test_an_off_value_turns_the_flag_off(name, value):
    assert getattr(_parse(f"--{name}", value), name) is False


@pytest.mark.parametrize("name", NAMES)
def test_the_bare_flag_and_true_turn_it_on(name):
    assert getattr(_parse(f"--{name}"), name) is True
    assert getattr(_parse(f"--{name}", "true"), name) is True
