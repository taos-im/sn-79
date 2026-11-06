"""Validator ↔ gradient server API key middleware.

Confirms that:
  - Every request is gated on X-API-Key when a key is configured.
  - Missing / wrong keys produce 401 before any route handler runs.
  - No key configured = pass-through (for loopback/dev deployments).

We do NOT exercise the actual gradient server routes here — those need a
checkpoint, val data, chain stubs, etc. Auth is orthogonal to all of that,
so we test it against a stub route.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from GenTRX.src.gradient_server import add_api_key_middleware


SECRET = "test-secret-abc123"


def _build_app(api_key: str) -> FastAPI:
    app = FastAPI()

    @app.get("/gentrx/version")
    async def version():
        return {"version": 0}

    @app.post("/gentrx/state")
    async def state():
        return {"status": "ok"}

    add_api_key_middleware(app, api_key)
    return app


# ---------------------------------------------------------------------------
# With a key set
# ---------------------------------------------------------------------------

def test_missing_header_rejected_401():
    client = TestClient(_build_app(SECRET))
    r = client.get("/gentrx/version")
    assert r.status_code == 401
    assert "X-API-Key" in r.json()["error"]


def test_wrong_header_rejected_401():
    client = TestClient(_build_app(SECRET))
    r = client.get("/gentrx/version", headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401


def test_correct_header_accepted():
    client = TestClient(_build_app(SECRET))
    r = client.get("/gentrx/version", headers={"X-API-Key": SECRET})
    assert r.status_code == 200
    assert r.json() == {"version": 0}


def test_post_state_gated_same_as_get():
    client = TestClient(_build_app(SECRET))
    r_bad = client.post("/gentrx/state", content=b"\x00\x01")
    r_ok = client.post(
        "/gentrx/state", content=b"\x00\x01", headers={"X-API-Key": SECRET}
    )
    assert r_bad.status_code == 401
    assert r_ok.status_code == 200


def test_case_sensitive_key_match():
    """Header name is case-insensitive (HTTP), but the secret itself must match."""
    client = TestClient(_build_app(SECRET))
    r = client.get("/gentrx/version", headers={"x-api-key": SECRET})  # lowercase header
    assert r.status_code == 200
    r = client.get("/gentrx/version", headers={"X-API-Key": SECRET.upper()})
    assert r.status_code == 401  # wrong secret, despite same letters


# ---------------------------------------------------------------------------
# With no key (loopback / dev mode)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("empty", ["", None])
def test_no_key_configured_is_passthrough(empty):
    client = TestClient(_build_app(empty or ""))
    r = client.get("/gentrx/version")
    assert r.status_code == 200
    r = client.post("/gentrx/state", content=b"\x00")
    assert r.status_code == 200


def test_the_key_is_compared_in_constant_time(monkeypatch):
    import hmac

    calls = []
    real = hmac.compare_digest

    def _recording(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", _recording)
    client = TestClient(_build_app(SECRET))
    assert client.get("/gentrx/version", headers={"X-API-Key": SECRET}).status_code == 200
    assert client.get("/gentrx/version", headers={"X-API-Key": "wrong-key"}).status_code == 401
    assert client.get("/gentrx/version").status_code == 401
    assert calls and all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in calls)
