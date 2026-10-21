"""Every HTTP client the validator-side GenTRX service builds shares one SSL context, and the sync calls reuse one
client per thread.

httpx builds an SSL context for each client it creates, reading the CA bundle from disk each time, even for a plain
HTTP URL, and the service built a fresh client for every state post, health check and bucket registration. The
context is now built once per process; sync calls reuse a client per thread and its connection pool.
"""
from __future__ import annotations

import asyncio
import threading

import httpx

from GenTRX.src import service as svc


class _Resp:
    status_code = 200

    def json(self):
        return {"status": "ok"}


class _FakeClient:
    made: list = []

    def __init__(self, *a, **kw):
        self.kw = kw
        _FakeClient.made.append(self)

    def post(self, *a, **kw):
        return _Resp()

    def get(self, *a, **kw):
        return _Resp()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeAsyncClient(_FakeClient):
    async def post(self, *a, **kw):
        return _Resp()

    async def get(self, *a, **kw):
        return _Resp()


def _service():
    s = svc.GenTRXService.__new__(svc.GenTRXService)
    s._server_url = "http://127.0.0.1:1"
    s._api_key = None
    return s


def _patch(monkeypatch):
    _FakeClient.made = []
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(svc, "_HTTP", threading.local())


def test_the_ssl_context_is_built_once():
    assert svc._http_ssl_context() is svc._http_ssl_context()


def test_sync_calls_reuse_one_client_per_thread(monkeypatch):
    _patch(monkeypatch)
    s = _service()
    s._http_post_sync("/state", b"x")
    s._http_post_sync("/state", b"y")
    assert s.is_healthy()
    assert s.register_benchmark_bucket(3, {"bucket_name": "b"})
    sync = [c for c in _FakeClient.made if type(c) is _FakeClient]
    assert len(sync) == 1
    other = []
    t = threading.Thread(target=lambda: other.append(s._http_post_sync("/state", b"z")))
    t.start()
    t.join()
    assert len([c for c in _FakeClient.made if type(c) is _FakeClient]) == 2, "a second thread gets its own client"


def test_every_client_carries_the_one_context(monkeypatch):
    _patch(monkeypatch)
    s = _service()
    s._http_post_sync("/state", b"x")
    asyncio.run(s._http_get_async("/data-status"))
    asyncio.run(s._http_post_async("/round", b"x"))
    ctx = svc._http_ssl_context()
    assert _FakeClient.made and all(c.kw.get("verify") is ctx for c in _FakeClient.made)
