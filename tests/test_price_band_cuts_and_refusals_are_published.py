# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""What the price band cut and refused is published as totals for the validator's gauges.

An engine whose exchange carries a price band reports each order or close the band cut in its response's 'band' list,
and each order the band refused whole among its rejects, with the reason PRICE_BAND. The validator adds them up as each
response arrives, so a report sampled every half minute publishes every cut and refusal since the process started:
price_band_cuts (orders and closes), price_band_cut_closes (the closes among them) and price_band_refusals. An engine whose responses never carried a band list (the
simulation engine, or an engine from before the band) publishes nothing, so a reader sees 'not measured', not zero."""
import os
from types import SimpleNamespace

from taos.im.validator import price_band as pb
from taos.im.validator.report import publish_price_band_gauges

DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _response(cuts=(), rejects=()):
    return {"band": list(cuts), "rejects": list(rejects)}


def test_each_response_adds_its_cuts_and_band_refusals_to_the_totals():
    engine = SimpleNamespace()
    pb.count(engine, _response(
        cuts=[{"a": 1, "b": 5, "k": "order"}, {"a": 2, "b": 5, "k": "close"}],
        rejects=[{"a": 3, "m": "PRICE_BAND"}, {"a": 4, "m": "LIMIT_PRICE_UNMET"}]))
    pb.count(engine, _response(cuts=[{"a": 1, "b": 7, "k": "order"}]))
    assert pb.health(engine) == {"cuts": 3, "cut_closes": 1, "refusals": 1}
    assert pb.health(engine) == {"cuts": 3, "cut_closes": 1, "refusals": 1}, "the totals run on; reading resets none"


def test_a_response_with_a_band_list_but_nothing_cut_starts_the_totals_at_zero():
    engine = SimpleNamespace()
    pb.count(engine, _response())
    assert pb.health(engine) == {"cuts": 0, "cut_closes": 0, "refusals": 0}


def test_an_engine_that_never_sent_a_band_list_has_nothing_to_publish():
    engine = SimpleNamespace()
    pb.count(engine, {"rejects": [{"m": "PRICE_BAND"}]})
    pb.count(engine, None)
    assert pb.health(engine) is None
    assert pb.health(None) is None


def test_a_malformed_entry_costs_that_entry_only():
    engine = SimpleNamespace()
    pb.count(engine, {"band": [None, "x", {"k": "close"}], "rejects": [None, {"m": "PRICE_BAND"}]})
    assert pb.health(engine) == {"cuts": 1, "cut_closes": 1, "refusals": 1}


def test_the_report_hands_the_totals_to_the_reporting_service():
    validator = open(os.path.join(DEV, "taos/im/neurons/validator.py")).read()
    assert "'price_band': price_band.health(getattr(self, 'engine', None))" in validator
    report = open(os.path.join(DEV, "taos/im/validator/report.py")).read()
    assert "self.price_band = data.get('price_band')" in report
    assert "publish_price_band_gauges(self)" in report.split("def publish_validator_gauges")[1].split("\ndef ")[0]


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
    return SimpleNamespace(price_band=totals, prometheus_validator_gauges=_Gauges(),
                           wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5x")),
                           config=SimpleNamespace(netuid=79), simulation=SimpleNamespace(simulation_id="r"))


def test_the_gauges_carry_the_totals_under_their_names():
    svc = _service({"cuts": 3, "cut_closes": 1, "refusals": 2})
    publish_price_band_gauges(svc)
    assert svc.prometheus_validator_gauges.values == {
        "price_band_cuts": 3.0, "price_band_cut_closes": 1.0, "price_band_refusals": 2.0}


def test_no_totals_publish_no_gauges():
    svc = _service(None)
    publish_price_band_gauges(svc)
    assert svc.prometheus_validator_gauges.values == {}
