# SPDX-License-Identifier: MIT
"""The skill leg's magnitude basis is size-neutral by default: the pool-relative floor is off
(`floor_scale` 0) and a book's alpha counts only above 2.3 basis points of the agent's own filled
notional on that book (`skill_hurdle_bps`). The pool-relative floor pooled raw |alpha| over every
traded pair, so an operator trading far larger size across many books could set it for everyone; the
hurdle is measured against each agent's own notional and cannot be moved by anyone else's size. Both
dials remain settable, so the previous basis is one flag away.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taos.im.config import add_im_validator_args  # noqa: E402


def _defaults():
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    known, _ = parser.parse_known_args([])
    return known


def test_the_floor_is_off_and_the_hurdle_is_two_point_three_bps_by_default():
    known = _defaults()
    assert getattr(known, "scoring.debeta.floor_scale") == 0.0
    assert getattr(known, "scoring.debeta.skill_hurdle_bps") == 2.3


def test_the_previous_basis_is_one_flag_away():
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    known, _ = parser.parse_known_args(["--scoring.debeta.floor_scale", "0.5", "--scoring.debeta.skill_hurdle_bps", "0"])
    assert getattr(known, "scoring.debeta.floor_scale") == 0.5
    assert getattr(known, "scoring.debeta.skill_hurdle_bps") == 0.0


def test_the_rest_of_the_skill_leg_defaults_are_unchanged():
    known = _defaults()
    assert getattr(known, "scoring.debeta.making_pool") == "proportional_both"
    # 0.6.3: the count derives from a share of the layout (0.15625 of 128 is the launch bar of 20)
    assert getattr(known, "scoring.debeta.skill_min_books") == 0
    assert getattr(known, "scoring.debeta.skill_min_books_share") == 0.15625
    assert getattr(known, "scoring.debeta.min_books") == 4
    assert getattr(known, "scoring.debeta.w_make") == 0.5
    assert getattr(known, "scoring.debeta.weight") == 1.0
