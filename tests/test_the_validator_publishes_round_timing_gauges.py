# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator publishes its round timing as gauges, so the timing can be read from its metrics.

Per uid, the published block carries the mean response time and the block's 5th percentile (the uid's floor) and
median. The validator itself publishes, over each publish interval, the round period's 5th percentile, median and
95th percentile (the wall time between consecutive states), the query wait's median and 95th percentile (dispatch to
the last answer, from the query service), the reward tasks pending and the longest wait for the reward lock.
Quantiles are the inclusive kind the store computes.
"""
import inspect
from types import SimpleNamespace

from taos.im.validator import forward, report


def test_the_inclusive_quantile_matches_the_stores():
    assert report.round_quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert report.round_quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == 4.8
    assert report.round_quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.05) == 1.2
    assert report.round_quantile([7.0], 0.95) == 7.0
    assert report.round_quantile([], 0.5) is None


def test_a_uids_block_publishes_its_floor_and_median_beside_the_mean():
    block = {"requests": 120, "timeouts": 6, "failures": 1, "rejections": 0,
             "call_time": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 2.4]}
    g = dict(report.round_block_gauges(block))
    assert g["call_time"] == sum(block["call_time"]) / 10
    assert g["call_time_median"] == 0.95
    assert abs(g["call_time_p5"] - 0.545) < 1e-9
    assert g["requests"] == 120 and g["success"] == 113
    silent = dict(report.round_block_gauges({"requests": 100, "timeouts": 100, "failures": 0, "rejections": 0, "call_time": []}))
    assert silent["call_time"] == 0 and silent["call_time_median"] == 0 and silent["call_time_p5"] == 0


def test_the_validators_round_gauges_come_from_the_payloads_lists():
    self = SimpleNamespace(round_gaps=[5.0, 5.2, 5.1, 9.0, 5.3], query_waits=[1.0, 1.2, 3.0], reward_gate_waits=[0.0, 0.4],
                           pending_reward_tasks=2)
    g = dict(report.round_timing_gauges(self))
    assert g["round_period_median"] == 5.2 and g["round_period_p5"] == 5.02 and abs(g["round_period_p95"] - 8.26) < 1e-9
    assert g["query_wait_median"] == 1.2 and abs(g["query_wait_p95"] - 2.82) < 1e-9
    assert g["reward_gate_wait_max"] == 0.4 and g["reward_tasks_pending"] == 2
    empty = dict(report.round_timing_gauges(SimpleNamespace()))
    assert empty == {"reward_tasks_pending": 0}, "an older payload publishes only what it carries"


def test_the_query_wait_is_recorded_from_the_services_stats():
    self = SimpleNamespace(query_waits=[])
    forward.update_query_wait(self, {"validation_stats": {"wait": 1.75}})
    forward.update_query_wait(self, {"validation_stats": {}})
    forward.update_query_wait(self, {})
    assert self.query_waits == [1.75]
    bare = SimpleNamespace()
    forward.update_query_wait(bare, {"validation_stats": {"wait": 0.5}})
    assert bare.query_waits == [0.5], "a validator without the list gets one"


def test_the_service_reports_the_wait_and_the_validator_ships_the_lists():
    from taos.im.validator import query
    from taos.im.neurons import validator

    qsrc = inspect.getsource(query)
    assert qsrc.count("self._last_wait_s = elapsed") == 3, "the threaded, fanout and prebuilt paths record the wait"
    assert '"wait": getattr(self, "_last_wait_s", None)' in qsrc
    vsrc = inspect.getsource(validator)
    for key in ("'round_gaps': list(self.round_gaps[-100:])", "'query_waits': list(self.query_waits[-100:])",
                "'reward_gate_waits': list(self.reward_gate_waits[-100:])", "'pending_reward_tasks': self._pending_reward_tasks"):
        assert key in vsrc, key
    assert "self.round_gaps.append(_now - self._round_clock)" in vsrc
    assert "self.reward_gate_waits.append(waited)" in vsrc
    fsrc = inspect.getsource(forward.forward)
    assert "update_query_wait(self, result)" in fsrc
    rsrc = inspect.getsource(report)
    assert "for _name, _value in round_block_gauges(_ms):" in rsrc
    assert "for _name, _value in round_timing_gauges(self):" in rsrc
