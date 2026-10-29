# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The exposed-surface check a deploy runs from outside a validator host: each validator API port refuses or drops
the connection, the metrics port answers its validator family, and no server answers /openapi.json, /docs or /redoc.
The family renders the cycle before its first byte, so its read waits a longer budget than the connects and the doc
paths: a slow first byte within it passes, a family that never answers still fails. Its verdict is driven here through
stand-in probes, so no socket is opened."""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "runbooks", "deploy", "exposed_surface_check.py")


@pytest.fixture(scope="module")
def check():
    if not os.path.exists(SCRIPT):
        pytest.skip("the deploy runbooks are not in this tree")
    spec = importlib.util.spec_from_file_location("exposed_surface_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _probes(open_ports=(), statuses=None):
    statuses = statuses or {}

    def tcp(host, port, timeout):
        return port in open_ports

    def http(host, port, path, timeout):
        return statuses.get((port, path))

    return tcp, http


def _healthy_metrics(port=9091):
    s = {(port, "/metrics/validator"): 200}
    s.update({(port, p): 404 for p in ("/openapi.json", "/docs", "/redoc")})
    return s


def test_a_host_as_intended_passes(check):
    tcp, http = _probes(open_ports={9091}, statuses=_healthy_metrics())
    results = check.run_checks("validator.example", [8000, 8001], 9091, tcp=tcp, http=http)
    assert results and all(ok for ok, _line in results), results
    assert check.exit_code(results) == 0


def test_a_reachable_api_port_fails(check):
    statuses = _healthy_metrics()
    statuses[(8000, "/openapi.json")] = 404
    tcp, http = _probes(open_ports={8000, 9091}, statuses=statuses)
    results = check.run_checks("validator.example", [8000, 8001], 9091, tcp=tcp, http=http)
    failed = [line for ok, line in results if not ok]
    assert any("8000" in line for line in failed), results
    assert check.exit_code(results) == 1


@pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
def test_a_route_list_on_the_metrics_port_fails(check, path):
    statuses = _healthy_metrics()
    statuses[(9091, path)] = 200
    tcp, http = _probes(open_ports={9091}, statuses=statuses)
    results = check.run_checks("validator.example", [8000], 9091, tcp=tcp, http=http)
    assert any(path in line for ok, line in results if not ok), results
    assert check.exit_code(results) == 1


def test_a_silent_metrics_port_fails(check):
    tcp, http = _probes(open_ports=set(), statuses={})
    results = check.run_checks("validator.example", [8000], 9091, tcp=tcp, http=http)
    assert any("9091" in line for ok, line in results if not ok), results
    assert check.exit_code(results) == 1


def _timed(first_byte_s, seen=None):
    """A metrics port whose family takes first_byte_s to its first byte (None: it never answers) and whose doc paths
    answer 404 at once. A read given less time than the first byte takes sees no answer, as http_status does."""

    def http(host, port, path, timeout):
        if seen is not None:
            seen[path] = timeout
        if path != "/metrics/validator":
            return 404
        return 200 if first_byte_s is not None and first_byte_s <= timeout else None

    return http


def _metrics_only(host, port, timeout):
    return port == 9091


def test_a_family_slow_to_its_first_byte_passes_within_its_budget(check):
    results = check.run_checks("validator.example", [8000], 9091, tcp=_metrics_only, http=_timed(45.0),
                               timeout=30, metrics_timeout=120)
    assert all(ok for ok, _line in results), results
    results = check.run_checks("validator.example", [8000], 9091, tcp=_metrics_only, http=_timed(45.0))
    assert all(ok for ok, _line in results), "the default budget waits out a slow render"


def test_only_the_family_read_waits_the_long_budget(check):
    seen, connects = {}, []

    def tcp(host, port, timeout):
        connects.append(timeout)
        return port == 9091

    check.run_checks("validator.example", [8000, 8001], 9091, tcp=tcp, http=_timed(1.0, seen),
                     timeout=30, metrics_timeout=120)
    assert seen.pop("/metrics/validator") == 120
    assert seen == {p: 30 for p in check.DOC_PATHS} and connects == [30, 30, 30], "the control port and the two API ports"


def test_a_family_that_does_not_answer_within_its_budget_still_fails(check):
    for first_byte_s in (None, 150.0):
        results = check.run_checks("validator.example", [8000], 9091, tcp=_metrics_only, http=_timed(first_byte_s),
                                   timeout=30, metrics_timeout=120)
        failed = [line for ok, line in results if not ok]
        assert len(failed) == 1 and "/metrics/validator answered None" in failed[0], results
        assert check.exit_code(results) == 1


def test_the_command_line_hands_the_family_budget_to_the_checks(check, monkeypatch):
    seen = {}

    def run_checks(host, api_ports, metrics_port, **kw):
        seen.update(kw)
        return [(True, "ok")]

    monkeypatch.setattr(check, "run_checks", run_checks)
    assert check.main(["validator.example", "--timeout", "30", "--metrics-timeout", "120"]) == 0
    assert seen == {"timeout": 30.0, "metrics_timeout": 120.0}
    check.main(["validator.example"])
    assert seen["metrics_timeout"] == check.METRICS_TIMEOUT_S


def test_an_upstream_that_accepts_every_port_is_judged_by_the_http_answer(check):
    """A network in front of the host that completes the handshake on every port (a SYN proxy,
    where ports nothing listens on accept and reset) says nothing about the API ports. With the
    control port accepting too, an API port fails only when an HTTP answer comes back from behind it."""
    def tcp(host, port, timeout):
        return True
    statuses = _healthy_metrics()

    def http(host, port, path, timeout):
        return statuses.get((port, path))
    results = check.run_checks("validator.example", [8000, 8001], 9091, tcp=tcp, http=http)
    assert check.exit_code(results) == 0, results
    assert any("every port" in line for ok, line in results if ok), results
    statuses[(8000, "/openapi.json")] = 404
    results = check.run_checks("validator.example", [8000, 8001], 9091, tcp=tcp, http=http)
    failed = [line for ok, line in results if not ok]
    assert any("8000" in line and "answered 404" in line for line in failed), results


def test_a_closed_control_port_keeps_the_handshake_as_the_verdict(check):
    tcp, http = _probes(open_ports={8000, 9091}, statuses=_healthy_metrics())
    results = check.run_checks("validator.example", [8000, 8001], 9091, tcp=tcp, http=http)
    assert any("8000" in line for ok, line in results if not ok), "the control port refuses: an accept on 8000 is exposure"

