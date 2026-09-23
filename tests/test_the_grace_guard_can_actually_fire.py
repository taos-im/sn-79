# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The grace guard must be able to return True, or it is decoration.

`_within_grace` exists to stop the IPC reattach firing while the engine is deliberately silent: the
engine publishes nothing during its grace period, so a cold-started simulation logs
"No state for 60s on /taosim-req: re-opened the queue" every minute until it speaks, and each of
those re-opens the queue under a process that is mid-startup.

As written it could never return True, for two independent reasons:

  1. It read `validator.start_time`, which is assigned inside the STATE HANDLER
     (validator.py:2947) and is therefore None for exactly the window the guard covers. The first
     `if not _started: return False` ended it.
  2. It read `v.simulation.config.grace_period`. `grace_period` lives on `v.simulation` itself --
     `reward.py:1478` uses `self.simulation.grace_period` -- so the `.config` hop yielded None and
     the second guard ended it too.

Confirmed live: a cold start logged the reattach every 60s straight through
its grace period, which is the line the guard exists to suppress.

The replacement asks the only two questions that can be answered during grace: has any state ever
arrived (grace is over the instant one has), and has the suppression already run longer than the
configured grace allows. The second is a COUNT of consecutive suppressions rather than a clock, so
it is self-limiting and needs no new dependency -- a genuinely dead engine is still recovered, just
one grace period later than a hung one.
"""

import taos.im.validator.engines.simulation as S


class _Sim:
    def __init__(self, grace_ns):
        self.grace_period = grace_ns


class _V:
    def __init__(self, grace_ns=600_000_000_000, last_state=None, start_time=None):
        self.simulation = _Sim(grace_ns)
        self.last_state = last_state
        self.start_time = start_time


class _Engine:
    """Only the attributes the guard reads."""

    _within_grace = S.SimulationEngine._within_grace

    def __init__(self, validator):
        self.validator = validator
        self._grace_suppressions = 0


def test_it_fires_before_any_state_has_arrived():
    """The whole point: during grace, start_time is None and no state has come."""
    e = _Engine(_V())
    assert e._within_grace() is True, (
        "the guard cannot fire during grace, so the reattach it exists to suppress runs anyway"
    )


def test_it_stops_once_state_has_arrived():
    """Grace is over the instant the engine speaks, whatever any clock says."""
    e = _Engine(_V(last_state=object()))
    assert e._within_grace() is False


def test_it_gives_up_rather_than_suppressing_for_ever():
    """A genuinely dead engine must still be recovered."""
    e = _Engine(_V(grace_ns=600_000_000_000))
    fired = 0
    for _ in range(10_000):
        if not e._within_grace():
            break
        fired += 1
    assert fired > 0, "it never suppressed at all"
    assert fired < 10_000, "it suppresses without bound, so a dead engine is never reattached"


def test_no_configured_grace_means_no_suppression():
    e = _Engine(_V(grace_ns=0))
    assert e._within_grace() is False


def test_it_does_not_consult_start_time():
    """start_time is set in the state handler, so it is None for exactly the window this covers."""
    src = open(S.__file__).read()
    i = src.index("def _within_grace")
    body = src[i:i + 1800]
    # Code, not prose: the docstring names start_time precisely to say it must not be read.
    code = [ln for ln in body.splitlines()
            if "start_time" in ln and not ln.lstrip().startswith("#") and '"""' not in ln]
    reads = [ln for ln in code if "getattr" in ln or "v.start_time" in ln or "= self" in ln]
    assert not reads, (
        "the guard is reading start_time again, which is None until the first state arrives and so "
        f"ends the guard before it can fire: {reads}"
    )


def test_it_reads_grace_period_off_the_simulation_not_a_config_hop():
    src = open(S.__file__).read()
    i = src.index("def _within_grace")
    body = src[i:i + 1400]
    assert "simulation" in body and '"config"' not in body, (
        "grace_period lives on validator.simulation; the .config hop yields None"
    )
