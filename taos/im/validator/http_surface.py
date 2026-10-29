# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator's two HTTP servers: how each app is built and how each is served.

The API (`--port`, `--api.host`) answers the simulator and the SL/TP service on this host, so it listens on loopback
unless a deployment names another address. The metrics server (`--prometheus.port`, `--prometheus.host`) is scraped
from other hosts and listens on every interface by default. Neither app serves a route list and neither server writes
a line per request. The metrics app bounds the requests each client address (the TCP peer: the port is served
directly) has in flight, MVTRX_METRICS_PER_CLIENT_CONCURRENCY (default 8, read once at import), and answers 503 at
once past it. There is no global cap, so no set of other clients can push one address out.
"""
import os

API_HOST_DEFAULT = "127.0.0.1"
METRICS_HOST_DEFAULT = "0.0.0.0"
SERVER_LOG_LEVEL = "warning"
# In simulation mode the API server takes the stop signal first and hands it to the validator, which then saves its
# state, only once its graceful shutdown ends; past this it cancels the requests still running. The API's handlers
# answer from memory, and the longest await among them, the forwarding of the engine's notices to miners, allows one
# second per notice.
API_GRACEFUL_SHUTDOWN_S = 5
_PER_CLIENT_DEFAULT = 8


def per_client_limit_from_env(raw) -> int:
    """The per-address bound from its environment value; the default for anything but a positive integer."""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return _PER_CLIENT_DEFAULT
    return value if value > 0 else _PER_CLIENT_DEFAULT


METRICS_PER_CLIENT_CONCURRENCY = per_client_limit_from_env(os.environ.get("MVTRX_METRICS_PER_CLIENT_CONCURRENCY"))


class PerClientConcurrency:
    """ASGI middleware: at most `limit` requests in flight per client address; one past it answers 503 at once.

    The count changes only between awaits on the server's event loop, so the check and the increment are one step.
    """

    _REFUSAL = b"too many requests in flight from this address\n"

    def __init__(self, app, limit: int = METRICS_PER_CLIENT_CONCURRENCY):
        self.app = app
        self.limit = limit
        self.in_flight: dict[str, int] = {}

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        address = client[0] if client else ""
        count = self.in_flight.get(address, 0)
        if count >= self.limit:
            await send({
                "type": "http.response.start",
                "status": 503,
                "headers": [(b"content-type", b"text/plain; charset=utf-8"), (b"retry-after", b"1"),
                            (b"content-length", str(len(self._REFUSAL)).encode())],
            })
            await send({"type": "http.response.body", "body": self._REFUSAL})
            return
        self.in_flight[address] = count + 1
        try:
            await self.app(scope, receive, send)
        finally:
            left = self.in_flight.get(address, 1) - 1
            if left > 0:
                self.in_flight[address] = left
            else:
                self.in_flight.pop(address, None)


def make_app():
    """A FastAPI app without the generated route list and its two viewers."""
    from fastapi import FastAPI

    return FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def api_server_options(config) -> dict:
    """uvicorn.run keyword arguments for the validator API."""
    host = getattr(getattr(config, "api", None), "host", None) or API_HOST_DEFAULT
    return {
        "host": host,
        "port": int(config.port),
        "access_log": False,
        "log_level": SERVER_LOG_LEVEL,
        "timeout_graceful_shutdown": API_GRACEFUL_SHUTDOWN_S,
    }


def metrics_server_options(config) -> dict:
    """uvicorn.run keyword arguments for the metrics server."""
    host = getattr(config.prometheus, "host", None) or METRICS_HOST_DEFAULT
    return {
        "host": host,
        "port": int(config.prometheus.port),
        "access_log": False,
        "log_level": SERVER_LOG_LEVEL,
    }
