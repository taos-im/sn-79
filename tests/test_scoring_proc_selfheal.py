# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A validator restore racing the simulator restart can leave the scoring service unable to score.

The service's INIT snapshot carries the sim clock as base_ts. When the clock has since moved, the
child reads every new-clock frame as pre-snapshot replay, applies none of them and produces no
score, so each boundary costs the full child timeout plus an in-process compute while the query
loop waits on it. Source-level wiring assertions for the two guards that close it."""
from pathlib import Path

DEV = Path(__file__).resolve().parents[1]
VALIDATOR = (DEV / "taos/im/neurons/validator.py").read_text()
ENGINE = (DEV / "taos/im/validator/engines/simulation.py").read_text()
REWARD = (DEV / "taos/im/validator/reward.py").read_text()


def test_sim_restart_stamps_the_new_clock_for_a_late_init():
    block = ENGINE[ENGINE.index("Shifting timestamps for simulation restart"):]
    block = block[:block.index("last_state_time")]
    assert "_shadow_applied_ts = new_simulation_timestamp" in block, (
        "an INIT built after the restart shift must stamp the NEW clock, or the child "
        "discards every new-sim frame as pre-snapshot replay"
    )


def test_consecutive_child_misses_trigger_reinit():
    block = VALIDATOR[VALIDATOR.index("child scoring unavailable"):]
    block = block[:block.index("_scoring_proc_n += 1")]
    assert "request_reinit()" in block, (
        "an alive+initialized child that never delivers must be re-INIT'd, not retried forever"
    )
    assert "_scoring_proc_consec_unavail" in block


def test_debeta_weight_is_published_every_cycle():
    line = [ln for ln in REWARD.splitlines() if "uid_kappa['debeta_weight']" in ln]
    assert line and "debeta_applied" not in line[0], (
        "the weight is the config dial: warming cycles must still publish it, or the live "
        "blend invariant cannot reconstruct the renormalized composition"
    )
