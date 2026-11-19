# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Each state's accounting time, main's and the scoring service's, is published as validator gauges.

Main's per-state accounting has a budget (under 0.15 s a state at 256 books), and the per-entry legs run in the
scoring service, which must keep up with the states. The [ACCOUNTING-PROFILE] log line says so on a host whose logs
are read; the gauges say so on every host a monitor reads: main's wall time (median, 95th percentile) and
thread CPU (median), and the service's wall time (mean, whether it keeps up; 95th percentile, the states that run the
legs). Main keeps the last thousand of its own, the service ships its own with each score.
"""
import os
from types import SimpleNamespace

import pytest

import taos.im.validator.report as report
import taos.im.validator.scoring_shadow as ss

HERE = os.path.dirname(os.path.abspath(__file__))


def _src(*parts):
    return open(os.path.join(HERE, "..", *parts)).read()


def test_the_gauges_read_main_and_the_service():
    main = [(0.10, 0.05), (0.20, 0.10), (0.30, 0.20)]
    service = [(0.05, 0.04, 0), (0.05, 0.04, 0), (0.30, 0.28, 1)]
    got = dict(report.accounting_gauges(main, service))
    assert got["accounting_wall_median"] == pytest.approx(report.round_quantile([0.1, 0.2, 0.3], 0.5))
    assert got["accounting_wall_p95"] == pytest.approx(report.round_quantile([0.1, 0.2, 0.3], 0.95))
    assert got["accounting_cpu_median"] == pytest.approx(report.round_quantile([0.05, 0.1, 0.2], 0.5))
    assert got["service_accounting_wall_mean"] == pytest.approx(0.4 / 3)
    assert got["service_accounting_wall_p95"] == pytest.approx(report.round_quantile([0.05, 0.05, 0.3], 0.95))
    assert report.accounting_gauges([], None) == [], "no times, no gauges"


def test_the_round_timing_gauges_carry_them():
    v = SimpleNamespace(accounting_times=[(0.2, 0.1)], service_accounting_times=[(0.05, 0.04, 0)])
    names = [n for n, _ in report.round_timing_gauges(v)]
    assert {"accounting_wall_median", "accounting_cpu_median", "service_accounting_wall_mean"} <= set(names)


def test_the_service_ships_its_states_times_with_each_score_once():
    shadow = SimpleNamespace(inventory_history={}, legs_clock={"last": 5}, legs_here=True)
    for wall, due in ((0.04, 0), (0.25, 1)):
        shadow.accounting_profile = {"wall": wall, "cpu": wall * 0.9, "due": due}
        ss.note_accounting(shadow)
    first = ss.legs_report(shadow)
    assert first["accounting"] == [(0.04, 0.036, 0), (0.25, 0.225, 1)]
    assert ss.legs_report(shadow)["accounting"] == [], "each state's time goes with one score"
    child = _src("taos", "im", "validator", "scoring_shadow.py")
    assert "apply_state_bytes(shadow, raw)\n            note_accounting(shadow)" in child


def test_main_keeps_its_own_and_the_services_and_ships_both():
    main = _src("taos", "im", "neurons", "validator.py")
    i = main.index("bt.logging.info(accounting_profile_line(_aprof))")
    assert "self.accounting_times.append(" in main[i:i + 300]
    assert "self.service_accounting_times.extend(adopted['legs'].get('accounting') or ())" in main
    assert "'accounting_times': list(self.accounting_times[-100:])" in main
    assert "'service_accounting_times': list(self.service_accounting_times[-100:])" in main
    rep = _src("taos", "im", "validator", "report.py")
    assert "self.accounting_times = list(data.get('accounting_times') or [])" in rep
    assert "self.service_accounting_times = list(data.get('service_accounting_times') or [])" in rep
