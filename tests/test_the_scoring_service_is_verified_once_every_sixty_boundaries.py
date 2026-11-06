# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Main recomputes the scoring service's rewards in-process once every 60 scoring boundaries by default.

The check is a second full reward calculation under the reward lock. The parity digest compares the service's
accumulators with main's every parity interval and re-initialises it on two mismatches in a row, so the in-process
check is the end-to-end backstop for the inputs the digest does not cover, not the first line.
SCORING_PROC_VERIFY_EVERY overrides the default."""
import inspect
import os
import subprocess
import sys
from pathlib import Path

from taos.im.validator import scoring_shadow

ROOT = Path(__file__).resolve().parents[1]


def _fresh(env):
    # a fresh interpreter, not a reload: other tests hold this module's classes
    e = {k: v for k, v in os.environ.items() if k != "SCORING_PROC_VERIFY_EVERY"}
    if env is not None:
        e["SCORING_PROC_VERIFY_EVERY"] = env
    out = subprocess.run([sys.executable, "-c", "from taos.im.validator import scoring_shadow as s; "
                          "print(s.SCORING_PROC_VERIFY_EVERY)"], cwd=ROOT, env=e, capture_output=True, text=True,
                         check=True)
    return int(out.stdout.strip())


def test_the_default_is_every_sixtieth_boundary_and_the_variable_overrides_it():
    assert _fresh(None) == 60
    assert _fresh("10") == 10


def test_a_check_falls_due_once_in_sixty_boundaries_on_an_idle_box():
    sched = scoring_shadow.VerifyScheduler(60)
    assert [n for n in range(1, 241) if sched.boundary(n, 0)] == [60, 120, 180, 240]


def test_the_validator_builds_its_schedule_from_the_module_setting():
    from taos.im.neurons import validator
    src = inspect.getsource(validator)
    assert 'os.environ.get("SCORING_PROC_VERIFY_EVERY"' not in src
    assert "VerifyScheduler(SCORING_PROC_VERIFY_EVERY)" in src
