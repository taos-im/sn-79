# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The queued (interface or API) orders a validator drains are published as totals of every drain.

After each drain the engine carries that drain's counts (queued_order_counts: the orders drained, those without the
service's receipt stamp, the largest delay given). A report samples every half minute while a drain runs every round,
so each drain's counts are added up as they come and the report publishes the totals: the counts since the process
started, the largest delay over the last hour of drains and the latest drain's delay. A reading resets nothing: the
validator prepares a report every minute but publishes metrics less often, and a largest-since-the-last-read figure
read by a report that was never published was lost (the exchange validator published 0 for an order that went in at
0.42 s). An engine whose drains carry no counts publishes nothing."""
import os
import re
from types import SimpleNamespace

from taos.im.validator import queued_orders as qo
from taos.im.validator.report import publish_queued_order_gauges

DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _drain(engine, drained, unstamped=0, delay_max_ns=0, now=0.0, **more):
    engine.queued_order_counts = {"drained": drained, "unstamped": unstamped, "delay_max_ns": delay_max_ns, **more}
    qo.tally(engine, now=now)


def test_each_drain_adds_to_the_totals_and_the_largest_delay_holds_over_the_window():
    engine = SimpleNamespace()
    _drain(engine, 3, delay_max_ns=2_000_000_000, now=0.0)
    _drain(engine, 0, now=10.0)
    _drain(engine, 2, unstamped=1, delay_max_ns=500_000_000, now=20.0)
    assert qo.health(engine, now=30.0) == {
        "drained": 5, "unstamped": 1, "delay_max_ns": 2_000_000_000, "delay_last_ns": 500_000_000}
    _drain(engine, 1, delay_max_ns=700_000_000, now=40.0)
    assert qo.health(engine, now=50.0) == {
        "drained": 6, "unstamped": 1, "delay_max_ns": 2_000_000_000, "delay_last_ns": 700_000_000}, (
        "the totals run on from the process's start; the largest delay is the hour's, the last one the latest drain's")


def test_a_reading_resets_nothing_so_a_report_that_is_never_published_loses_nothing():
    engine = SimpleNamespace()
    _drain(engine, 1, delay_max_ns=420_000_000, now=0.0)
    for minute in range(1, 7):
        qo.health(engine, now=60.0 * minute)
    assert qo.health(engine, now=420.0)["delay_max_ns"] == 420_000_000, (
        "six reports prepared and not published must not take the delay from the seventh, which is")
    assert qo.health(engine, now=420.0)["delay_last_ns"] == 420_000_000


def test_a_delay_older_than_the_window_leaves_the_largest_and_the_latest_holds_until_the_next_drain_with_orders():
    engine = SimpleNamespace()
    _drain(engine, 1, delay_max_ns=900_000_000, now=0.0)
    _drain(engine, 1, delay_max_ns=100_000_000, now=qo.WINDOW_S - 60.0)
    _drain(engine, 0, now=qo.WINDOW_S + 30.0)
    out = qo.health(engine, now=qo.WINDOW_S + 60.0)
    assert out["delay_max_ns"] == 100_000_000, "the first drain is more than the window old"
    assert out["delay_last_ns"] == 100_000_000, "a drain that carried no orders leaves the latest delay as it was"
    assert qo.health(engine, now=3 * qo.WINDOW_S)["delay_max_ns"] == 0


def test_a_count_the_drain_adds_later_is_totalled_with_the_rest():
    engine = SimpleNamespace()
    _drain(engine, 4, late=1)
    _drain(engine, 2, late=2)
    assert qo.health(engine)["late"] == 3


def test_an_engine_whose_drains_carry_no_counts_has_nothing_to_publish():
    engine = SimpleNamespace()
    qo.tally(engine)
    qo.tally(engine)
    assert qo.health(engine) is None
    assert qo.health(None) is None


def test_a_malformed_count_costs_that_count_only():
    engine = SimpleNamespace(queued_order_counts={"drained": "x", "unstamped": 2, "delay_max_ns": None})
    qo.tally(engine)
    assert qo.health(engine) == {"drained": 0, "unstamped": 2, "delay_max_ns": 0}


def test_both_engines_total_each_drain_right_after_it():
    """Each engine the tree carries: the public tree has the simulation engine alone, the exchange's being private."""
    present = [n for n in ("exchange", "simulation") if os.path.exists(os.path.join(DEV, "taos/im/validator/engines", f"{n}.py"))]
    assert "simulation" in present
    for name in present:
        src = open(os.path.join(DEV, "taos/im/validator/engines", f"{name}.py")).read()
        assert re.search(r"self\._external_instructions = self\._drain_external_orders\(\)\n\s*queued_orders\.tally\(self\)",
                         src), f"engines/{name}.py does not total its drain's counts right after the drain"


def test_the_report_hands_the_totals_to_the_reporting_service():
    validator = open(os.path.join(DEV, "taos/im/neurons/validator.py")).read()
    assert "'queued_orders': queued_orders.health(getattr(self, 'engine', None))" in validator
    report = open(os.path.join(DEV, "taos/im/validator/report.py")).read()
    assert "self.queued_orders = data.get('queued_orders')" in report
    assert "publish_queued_order_gauges(self)" in report.split("def publish_validator_gauges")[1].split("\ndef ")[0]


class _Gauges:
    def __init__(self):
        self.values = {}

    def labels(self, **kw):
        gauges = self

        class _G:
            def set(self, v):
                gauges.values[kw["validator_gauge_name"]] = v
        return _G()


def _service(totals):
    return SimpleNamespace(queued_orders=totals, prometheus_validator_gauges=_Gauges(),
                           wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5x")),
                           config=SimpleNamespace(netuid=79), simulation=SimpleNamespace(simulation_id="r"))


def test_the_gauges_carry_the_totals_under_their_names():
    svc = _service({"drained": 6, "unstamped": 1, "delay_max_ns": 700_000_000, "delay_last_ns": 400_000_000, "late": 2})
    publish_queued_order_gauges(svc)
    assert svc.prometheus_validator_gauges.values == {
        "queued_orders": 6.0, "queued_orders_unstamped": 1.0, "queued_order_delay_max_ns": 7e8,
        "queued_order_delay_last_ns": 4e8, "queued_orders_late": 2.0}


def test_no_totals_publish_no_gauges():
    svc = _service(None)
    publish_queued_order_gauges(svc)
    assert svc.prometheus_validator_gauges.values == {}
