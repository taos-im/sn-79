# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""When the exposition IS compressed, the scraper must see exactly the same bytes.

Compression is opt-in and off by default -- test_metrics_gzip_is_opt_in.py owns that contract.
This owns the other half: that turning it on changes only the transfer. A scraper asking for
identity gets the body, a scraper asking for gzip gets the same body through a different encoding,
and neither reading of the exposition differs from the other. Prometheus sends
Accept-Encoding: gzip on every scrape, so in a deployment that enables this, the compressed path
is the only one that is ever exercised and a difference between them would be invisible until it
had already corrupted a dashboard.

Pinned at the level the server actually uses, so this exercises the shipped configuration rather
than the library's default.
"""

import gzip

from fastapi import FastAPI, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.testclient import TestClient

from taos.im.validator.report import _METRICS_GZIP_LEVEL


def test_a_prometheus_exposition_travels_compressed_and_identical():
    app = FastAPI()
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=_METRICS_GZIP_LEVEL)
    body = b"".join(
        b'agent_gauges{uid="%d",book="%d",agent_gauge_name="pnl"} 1.5\n' % (u, b) for u in range(50) for b in range(40)
    )

    @app.get("/metrics/agent")
    def agent():
        return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")

    c = TestClient(app)
    plain = c.get("/metrics/agent", headers={"Accept-Encoding": "identity"})
    zipped = c.get("/metrics/agent", headers={"Accept-Encoding": "gzip"})
    assert plain.status_code == 200 and plain.content == body
    assert zipped.headers.get("content-encoding") == "gzip"
    assert zipped.content == body, "the client sees the same bytes"


def test_the_shipped_level_still_earns_its_cpu():
    """A level that barely compresses would spend the scrape's CPU for nothing."""
    body = b"".join(
        b'agent_gauges{uid="%d",book="%d",agent_gauge_name="pnl"} 1.5\n' % (u, b) for u in range(50) for b in range(40)
    )
    squeezed = gzip.compress(body, compresslevel=_METRICS_GZIP_LEVEL)
    assert len(squeezed) * 5 < len(body), "the exposition should compress at least fivefold at the shipped level"
