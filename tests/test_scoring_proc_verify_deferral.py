# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""With shadow parity clean, seven 'Waiting for rewarding to
catch up' bursts of 13 to 23 s, one at every VERIFY boundary. The in-process VERIFY is a second full
get_rewards under the reward lock (35 to 50 s in total against 15 s for a plain adoption); rounds queue
behind it, the pending count crosses the query-blocking threshold, and miners go unqueried for the
rest of the check. A diagnostic must not cost the miners their query windows.

The first cut deferred a due check to the NEXT boundary, where the queue is the same size because the
adoption wait alone queues three rounds, so it slid three boundaries and ran at the old cadence
(deferred at boundaries 10, 11, 12, ran at 13, one blackout per due period as before).

Pinned here: the pure decision rule; the scheduler that skips a whole due period on a busy box and
forces one after enough skips, so a permanently busy box verifies once per (force_after + 1) * every
boundaries and an idle box every `every`; and the validator wiring through the scheduler.
"""
from pathlib import Path

import taos.im.validator.scoring_shadow as ss

DEV = Path(__file__).resolve().parents[1]
VALIDATOR = (DEV / "taos/im/neurons/validator.py").read_text()


def _knobs(monkeypatch, max_pending=2, force_after=3):
    monkeypatch.setattr(ss, "SCORING_PROC_VERIFY_MAX_PENDING", max_pending)
    monkeypatch.setattr(ss, "SCORING_PROC_VERIFY_FORCE_AFTER", force_after)


def test_due_verify_runs_when_the_queue_is_short(monkeypatch):
    _knobs(monkeypatch)
    assert ss.scoring_proc_verify_decision(pending_reward_tasks=1, deferrals=0) == (True, 0)
    assert ss.scoring_proc_verify_decision(pending_reward_tasks=2, deferrals=2) == (True, 0)


def test_due_verify_is_deferred_behind_a_queue_then_forced(monkeypatch):
    _knobs(monkeypatch)
    deferrals = 0
    for expected in (1, 2, 3):
        run, deferrals = ss.scoring_proc_verify_decision(pending_reward_tasks=4, deferrals=deferrals)
        assert (run, deferrals) == (False, expected)
    assert ss.scoring_proc_verify_decision(pending_reward_tasks=4, deferrals=deferrals) == (True, 0), (
        "the fourth due check runs even on a busy box"
    )


def test_force_after_zero_means_never_defer(monkeypatch):
    _knobs(monkeypatch, force_after=0)
    assert ss.scoring_proc_verify_decision(pending_reward_tasks=9, deferrals=0) == (True, 0)


def test_scheduler_skips_whole_due_periods_on_a_busy_box(monkeypatch):
    _knobs(monkeypatch)
    sched = ss.VerifyScheduler(every=10)
    ran = [n for n in range(1, 121) if sched.boundary(n, pending_reward_tasks=4)]
    assert ran == [40, 80, 120], (
        f"a busy box verifies once per (force_after + 1) * every boundaries, got {ran}; the first cut "
        f"ran at 13, 23, 33 because the due flag was carried to the next boundary"
    )


def test_scheduler_runs_every_period_on_an_idle_box(monkeypatch):
    _knobs(monkeypatch)
    sched = ss.VerifyScheduler(every=10)
    ran = [n for n in range(1, 41) if sched.boundary(n, pending_reward_tasks=1)]
    assert ran == [10, 20, 30, 40]


def test_scheduler_recovers_at_the_next_due_period_when_the_queue_clears(monkeypatch):
    _knobs(monkeypatch)
    sched = ss.VerifyScheduler(every=10)
    ran = [n for n in range(1, 41) if sched.boundary(n, pending_reward_tasks=4 if n <= 25 else 1)]
    assert ran == [30, 40], "skipped 10 and 20 behind the queue; runs at the first due boundary with a short queue"
    assert sched.deferrals == 0


def test_scheduler_reports_the_decision_for_the_log(monkeypatch):
    _knobs(monkeypatch)
    sched = ss.VerifyScheduler(every=10)
    assert sched.boundary(10, 4) is False and sched.last == "deferred" and sched.deferrals == 1
    assert sched.skips_before_force == 2
    assert sched.boundary(11, 4) is False and sched.last is None, "a boundary that is not due decides nothing"
    assert sched.boundary(20, 1) is True and sched.last == "ran" and sched.deferrals == 0
    assert ss.VerifyScheduler(every=0).every == 1, "every is clamped to at least one boundary"


def test_validator_schedules_verify_through_the_scheduler():
    block = VALIDATOR[VALIDATOR.index("self._scoring_proc_n += 1"):]
    block = block[:block.index("if adopted is None or _verify:")]
    assert "_verify = adopted is not None and (self._scoring_proc_n % _verify_every == 0)" not in block
    assert "VerifyScheduler(" in block and ".boundary(self._scoring_proc_n, self._pending_reward_tasks)" in block
    assert "VERIFY deferred" in block, "a skipped period must be visible in the log"
    assert "_scoring_proc_verify_sched = None" in VALIDATOR, "the scheduler slot is initialised with the cutover state"
