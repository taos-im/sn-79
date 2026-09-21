# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Compressing the Prometheus exposition is a deployment choice, not a default.

Whether it pays depends on the size of the exposition, the scrape interval and the network between
the scraper and the host, and those do not match between a development box and a production
deployment. So it is available and off, and the level is pinned below the library default.

The level matters more than it looks. Compression runs inside the ASGI middleware, on the event
loop serving the scrape, so its cost is paid before the response starts. On Prometheus text,
level 9 and level 6 land within a few percent of each other on size while 9 costs several times
the CPU -- enough, on a large exposition, for the compression to take longer than the transfer it
is meant to shorten, which would make overlapping scrapes worse rather than better.
"""

import importlib
import os

import pytest


def _reload(monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import taos.im.validator.report as R

    return importlib.reload(R)


def test_it_is_off_unless_asked_for(monkeypatch):
    R = _reload(monkeypatch, MVTRX_METRICS_GZIP=None)
    assert R._METRICS_GZIP_ENABLED is False, "compression must not switch itself on in a deployment"


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_the_flag_turns_it_on(monkeypatch, value):
    R = _reload(monkeypatch, MVTRX_METRICS_GZIP=value)
    assert R._METRICS_GZIP_ENABLED is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "nonsense"])
def test_anything_else_leaves_it_off(monkeypatch, value):
    R = _reload(monkeypatch, MVTRX_METRICS_GZIP=value)
    assert R._METRICS_GZIP_ENABLED is False


def test_the_level_defaults_below_the_library_default(monkeypatch):
    R = _reload(monkeypatch, MVTRX_METRICS_GZIP_LEVEL=None)
    assert R._METRICS_GZIP_LEVEL == 6, (
        "9 is the library default and costs several times the CPU of 6 for a few percent of size"
    )


def test_the_level_is_clamped_and_survives_nonsense(monkeypatch):
    assert _reload(monkeypatch, MVTRX_METRICS_GZIP_LEVEL="99")._METRICS_GZIP_LEVEL == 9
    assert _reload(monkeypatch, MVTRX_METRICS_GZIP_LEVEL="0")._METRICS_GZIP_LEVEL == 1
    assert _reload(monkeypatch, MVTRX_METRICS_GZIP_LEVEL="banana")._METRICS_GZIP_LEVEL == 6


def test_the_server_only_adds_the_middleware_when_enabled():
    """The flag has to reach the app, not just the module."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "taos", "im", "validator", "report.py")).read()
    i = src.index("GZipMiddleware")
    window = src[max(0, i - 400):i]
    assert "_METRICS_GZIP_ENABLED" in window, "the middleware is added without consulting the flag"
    assert "compresslevel=_METRICS_GZIP_LEVEL" in src, "the level is not pinned, so it takes the library default"
