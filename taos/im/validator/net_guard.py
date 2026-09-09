"""Restrict the simulator state-ingest routes to the local machine.

`/orderbook` and `/account` accept simulator state and notice batches and feed them into the
validator, so the only legitimate sender is the simulator process on this host.

WHY AN ADDRESS CHECK RATHER THAN A SHARED SECRET. The sender is the C++ simulator, and its HTTP client
(`net::AsyncSendContext` / `net.cpp`) sets only `host` and `content_type` - it has no way to attach a
header, so a secret would require a C++ change deployed in lockstep with the validator. The simulator
runs on the same machine as the validator, so the address check needs no coordination at all.

MINERS ARE NOT AFFECTED. The validator queries miners OUTBOUND through its dendrite to
`metagraph.axons`; miners never open a connection to this port. Nor is monitoring: only these two
paths are restricted, so `/metrics/*`, `/sltp` and the rest are untouched.
"""
import os

# The two routes that ingest simulator state. Everything else on the API port is left alone.
LOCAL_ONLY_ROUTES = frozenset({"/orderbook", "/account"})

# Loopback as seen by uvicorn. "localhost" appears when a client resolves it that way.
LOCAL_PEERS = frozenset({"127.0.0.1", "::1", "localhost"})

# Deployments running the simulator in a separate container reach the validator over the bridge
# network, so the peer is a bridge address rather than loopback. Comma-separated.
_ENV_ALLOW = "TAOS_STATE_INGEST_ALLOW"


def extra_allowed_peers() -> frozenset:
    """Additional peers permitted on the ingest routes, from the environment."""
    raw = os.environ.get(_ENV_ALLOW, "") or ""
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def is_local_peer(peer: str) -> bool:
    """True when `peer` is this machine, or an operator-declared additional source."""
    if not peer:
        # A missing client address is not evidence of locality. Refuse.
        return False
    return peer in LOCAL_PEERS or peer in extra_allowed_peers()


def state_ingest_allowed(path: str, peer: str) -> bool:
    """Whether a request for `path` from `peer` may proceed.

    Only the ingest routes are considered; every other path is allowed through untouched, so this
    cannot affect miner traffic, metrics scraping, or the SL/TP routes.
    """
    if path not in LOCAL_ONLY_ROUTES:
        return True
    return is_local_peer(peer)


def install_state_ingest_guard(app, log=None) -> None:
    """Attach the guard to a FastAPI app.

    Production and the tests both call this, so the tests exercise the shipped middleware rather than
    a copy of it.
    """
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def _restrict_state_ingest(request, call_next):
        peer = request.client.host if request.client else ""
        if not state_ingest_allowed(request.url.path, peer):
            if log:
                log(f"REFUSED {request.url.path} from {peer or '<unknown>'} (state ingest is local-only)")
            # 404, NOT 403: a 403 confirms the route exists and is merely guarded, which tells a
            # prober exactly what to come back for. 404 is indistinguishable from an absent route.
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        return await call_next(request)
