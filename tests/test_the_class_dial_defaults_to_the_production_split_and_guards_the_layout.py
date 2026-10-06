# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""scoring.debeta.class_weights defaults to the production two-background split (0.95 simulation_0, 0.05 regime B) for
0.6.3, and the dial is read against the layout: on a single market there is no class map and it is ignored; on a
layout whose background count does not match the dial's length the scorer falls back to the flat rule and says so,
rather than indexing a class that has no weight. Decided 1 October 2026: the second class starts small and steps up
once observed (0.75/0.25 next), and a default that only fits one layout must never crash another."""
import argparse


def test_the_default_is_the_production_split():
    from taos.im.config import add_im_validator_args

    p = argparse.ArgumentParser()
    add_im_validator_args(None, p)
    a = next(x for x in p._actions if "--scoring.debeta.class_weights" in x.option_strings)
    assert a.default == "0.95,0.05"


def test_a_mismatched_dial_falls_back_to_flat():
    from taos.im.validator.reward import class_weights_for_layout

    two = {0: 0, 1: 0, 2: 1, 3: 1}
    assert class_weights_for_layout("0.95,0.05", two) == [0.95, 0.05]
    three = {0: 0, 1: 1, 2: 2}
    assert class_weights_for_layout("0.95,0.05", three) == [], "two weights on three classes: flat, not a crash"
    assert class_weights_for_layout("0.95,0.05", {}) == [], "a single market has no class map and no dial"
    assert class_weights_for_layout("", two) == []
    assert class_weights_for_layout("1,1", two) == [0.5, 0.5]
