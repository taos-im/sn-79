"""A state from a run OLDER than the one the validator has locked is refused, not merged.

If the engine's `-c latest` resumes an older run in place of the one the validator has locked, merging it
would score and save the older run's books into the locked run's state over its rotating backups. A
genuinely new run always has a LATER id, so only a step back to an older id is refused.
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import taos.im.validator.engines.simulation as sim  # noqa: E402

S = 1_000_000_000
H = 3600 * S


def _engine(locked, tmp_path="/tmp"):
    v = SimpleNamespace()
    v.config = SimpleNamespace(subtensor=SimpleNamespace(network="local"), neuron=SimpleNamespace(full_path=str(tmp_path)))
    v.simulation_timestamp = 10 * H
    v.simulation = SimpleNamespace(logDir=f"/r/logs/{locked}_00" if locked else None,
                                   simulation_id=locked, log_window=H, publish_interval=1)
    v.update_repo = lambda end=False: None
    e = sim.SimulationEngine.__new__(sim.SimulationEngine)
    e.validator = v
    e._fresh_threshold = lambda: H
    e._notify_seed_log_dir_change = lambda d: None
    e.compress_outputs = lambda start=False: None
    return e, v


def test_a_state_from_an_older_run_is_refused_and_changes_nothing():
    e, v = _engine("20261006_1200")
    with pytest.raises(sim.RunWentBackwardError):
        e.on_tick(SimpleNamespace(timestamp=10 * H + S, logDir="/r/logs/20261002_120000"))
    assert v.simulation.simulation_id == "20261006_1200"
    assert v.simulation.logDir == "/r/logs/20261006_1200_00"
    assert v.simulation_timestamp == 10 * H


def test_a_newer_run_and_the_same_run_are_not_refused():
    e, v = _engine("20261006_1200")
    e.on_tick(SimpleNamespace(timestamp=10 * H + S, logDir="/r/logs/20261006_120000"))
    e.on_tick(SimpleNamespace(timestamp=10 * H + 2 * S, logDir="/r/logs/20261009_120000"))


def test_nothing_is_refused_before_a_run_is_locked(tmp_path):
    e, _ = _engine(None, tmp_path)
    e.on_tick(SimpleNamespace(timestamp=10 * H + S, logDir="/r/logs/20261002_120000"))
