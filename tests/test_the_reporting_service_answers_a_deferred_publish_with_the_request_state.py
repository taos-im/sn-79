# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""On a cold run the simulation id is not yet known at the first report cycles, so publish_metrics defers and returns
before it copies the request's state onto the service. The reporting loop then read `initial_balances_published` and
`miner_stats` from a service that had never set them and failed every cycle until the id appeared (localnet deploy of
0.6.3, 7 October 2026: 'ReportingService' object has no attribute 'initial_balances_published'). The response to a
deferred publish carries the request's own state back, and a service that has published answers with its own."""
from taos.im.validator.report import ReportingService


def _bare():
    return ReportingService.__new__(ReportingService)


def test_a_service_that_has_not_published_answers_with_the_request_state():
    svc = _bare()
    data = {"step": 3, "initial_balances_published": {"7": False}, "miner_stats": {"7": {"requests": 1}}}
    out = svc._response_for(data)
    assert out == {"step": 3, "initial_balances_published": {"7": False}, "miner_stats": {"7": {"requests": 1}}}


def test_a_service_that_has_published_answers_with_its_own_state():
    svc = _bare()
    svc.initial_balances_published = {"7": True}
    svc.miner_stats = {"7": {"requests": 2}}
    out = svc._response_for({"step": 4, "initial_balances_published": {"7": False}, "miner_stats": {}})
    assert out["initial_balances_published"] == {"7": True} and out["miner_stats"] == {"7": {"requests": 2}} and out["step"] == 4


def test_a_request_without_the_fields_still_gets_a_well_formed_answer():
    out = _bare()._response_for({"step": 1})
    assert out == {"step": 1, "initial_balances_published": {}, "miner_stats": {}}
