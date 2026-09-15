# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The per-book de-beta gauges are ON by default for the 0.6.1 ladder and must stay switchable: a
store_true flag with default=True has no off switch, and the reporting child parses its own argv,
so the validator forwards the value explicitly in both directions (validator.py). The option must
therefore accept the bare flag, an explicit true and an explicit false, on the real parser."""
import argparse

import pytest

from taos.im.config import add_im_validator_args

KEY = "scoring.debeta.publish_book_gauges"


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    ns, _unknown = parser.parse_known_args(argv)
    return getattr(ns, KEY)


@pytest.mark.parametrize("argv, expected", [
    ([], True),
    ([f"--{KEY}"], True),
    ([f"--{KEY}", "true"], True),
    ([f"--{KEY}", "false"], False),
    ([f"--{KEY}", "0"], False),
    ([f"--{KEY}", "off"], False),
])
def test_publish_book_gauges_default_on_and_switchable(argv, expected):
    assert _parse(argv) is expected


def test_bare_flag_does_not_swallow_the_next_option():
    assert _parse([f"--{KEY}", "--netuid", "366"]) is True


def test_bittensor_config_sees_the_same_value():
    bt = pytest.importorskip("bittensor")
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    on = bt.Config(parser, args=[])
    off = bt.Config(parser, args=[f"--{KEY}", "false"])
    assert on.scoring.debeta.publish_book_gauges is True
    assert off.scoring.debeta.publish_book_gauges is False
