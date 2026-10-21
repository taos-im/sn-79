# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The simulation proxy posts state to a data service only when its command line names one.

A proxy started for a test or a calibration arm inherits whatever environment its shell carries. If
that environment could arm the push, synthetic fills would reach whatever service the variable points
at. The URL comes from --data-service-url alone.

The HTTP client is replaced in every test here, so nothing leaves the host. Where the optional push
module is absent the push is a no-op, and the test of the flag path is skipped.
"""
import asyncio
import ast
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

PROXY = Path(__file__).resolve().parents[1] / "agents" / "proxy" / "proxy.py"


class _Posts:
    def __init__(self):
        self.urls = []
        outer = self

        class _Resp:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _Session:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def post(self, url, **k):
                outer.urls.append(url)
                return _Resp()

        self.Session = _Session


class _EnvWithUrls(dict):
    """An environment that answers any data-service URL variable, whatever its name."""

    def _url(self, key):
        return "http://127.0.0.1:9" if "DATA_SERVICE" in str(key).upper() else None

    def get(self, key, default=None):
        return self._url(key) or super().get(key, default)

    def __getitem__(self, key):
        return self._url(key) or super().__getitem__(key)

    def __contains__(self, key):
        return bool(self._url(key)) or super().__contains__(key)


def _state():
    return SimpleNamespace(notices={}, accounts={}, books={}, timestamp=0)


@pytest.fixture
def posts(monkeypatch):
    import aiohttp
    p = _Posts()
    monkeypatch.setattr(aiohttp, "ClientSession", p.Session)
    return p


def _push_installed():
    return any(PROXY.parent.glob("*_push.py"))


def test_the_proxy_reads_no_data_service_url_from_the_environment():
    reads = []
    for node in ast.walk(ast.parse(PROXY.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("get", "getenv"):
            key = node.args[0] if node.args else None
            if isinstance(key, ast.Constant) and "DATA_SERVICE" in str(key.value):
                reads.append(node.lineno)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if "DATA_SERVICE" in str(node.slice.value):
                reads.append(node.lineno)
    assert not reads, f"proxy.py reads a data-service URL from the environment at lines {reads}"


def test_an_inherited_url_does_not_arm_the_push(monkeypatch, posts):
    import agents.proxy.proxy as proxy
    with monkeypatch.context() as m:
        m.setattr(os, "environ", _EnvWithUrls(os.environ))
        proxy = importlib.reload(proxy)
    try:
        asyncio.run(proxy._push_proxy_state(_state(), 1))
    finally:
        importlib.reload(proxy)
    assert posts.urls == [], f"the proxy posted to {posts.urls} with no --data-service-url"


@pytest.mark.skipif(not _push_installed(), reason="the optional push module is not installed")
def test_the_flag_arms_the_push(posts):
    import agents.proxy.proxy as proxy
    try:
        proxy._parse_args(["--data-service-url", "http://127.0.0.1:9"])
        asyncio.run(proxy._push_proxy_state(_state(), 1))
    finally:
        importlib.reload(proxy)
    assert posts.urls == ["http://127.0.0.1:9/ingest"]
