# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The first milestone's small cuts, each checked against what it replaces.

A state notification the engine sends again after a late answer is skipped once, unanswered. The garbage collector
settings are validator config, off by default. A trade event built from a precomputed field map is the object
model_construct builds.
"""
import argparse
import pathlib

from taos.im.config import add_im_validator_args
from taos.im.protocol.events import TradeEvent, _build_trade_event
from taos.im.validator.engines import RepeatedState


def test_a_repeated_state_is_skipped_once_and_a_further_copy_is_answered():
    repeats = RepeatedState()
    assert [repeats.skip(ts) for ts in (1, 2, 2, 3, 3, 3, 3, 4)] == [False, False, True, False, True, False, False, False]


def test_skipping_a_repeated_state_is_an_option_and_off_by_default():
    """Off is 0.6.5's behaviour: every state answered. The skip costs a full receive timeout when the late answer
    lands before the resend, and after a resume whose first state repeats."""
    parser = argparse.ArgumentParser()
    add_im_validator_args(object, parser)
    assert getattr(parser.parse_known_args([])[0], "neuron.skip_repeated_state") is False
    assert getattr(parser.parse_known_args(["--neuron.skip_repeated_state"])[0], "neuron.skip_repeated_state") is True
    src = (pathlib.Path(__file__).resolve().parents[1] / "taos" / "im" / "neurons" / "validator.py").read_text()
    assert "and skip_repeats and repeats.skip(" in src, "the listener skips a repeat only under the option"


def test_the_garbage_collector_settings_are_config_and_on_by_default():
    parser = argparse.ArgumentParser()
    add_im_validator_args(object, parser)
    args, _ = parser.parse_known_args([])
    assert getattr(args, "neuron.gc_freeze") is True and getattr(args, "neuron.gc_threshold") == "50000,20,100"
    args, _ = parser.parse_known_args(["--neuron.no_gc_freeze", "--neuron.gc_threshold", ""])
    assert getattr(args, "neuron.gc_freeze") is False and getattr(args, "neuron.gc_threshold") == ""


def test_a_trade_event_from_the_field_map_is_the_one_model_construct_builds():
    base = {"y": "ET", "t": 1, "b": 0, "i": 7, "Ti": 1, "Mi": 2, "q": 1.0, "p": 2.0, "s": 0,
            "Ma": 3, "Ta": 4, "Mf": 0.0, "Tf": 0.0}
    for wire in (base, {**base, "cr": "SL"}, {**base, "extra_key": 5}, {k: v for k, v in base.items() if k != "Mf"}):
        fast, ref = _build_trade_event(wire), TradeEvent.model_construct(**wire)
        assert fast.__dict__ == ref.__dict__
        assert fast.model_fields_set == ref.model_fields_set
        assert fast.__pydantic_extra__ == ref.__pydantic_extra__

