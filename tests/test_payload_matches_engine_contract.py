"""Every field the C++ engine expects must actually be packed by Python.

Four separate defects tonight were the same shape: a field declared in the engine's
MSGPACK_DEFINE_MAP that Python never put on the wire. msgpack ignores unknown keys and silently
leaves absent ones at their struct default, so each one failed quietly and looked like something
else entirely:

* `stp`   -- emitted as "stpFlag", so the engine kept STPFlag::CO for every order ever placed and
             a miner could not select any self-trade policy.
* `currency` (limit) -- never packed, so a TAO-denominated limit arrived in the engine's default
             unit: BUY 0.006 TAO was read as 0.006 ALPHA, a notional of 0.00004 TAO, and refused
             with MINIMUM_ORDER_SIZE_VIOLATION. A miner could not place one at all.
* `leverage` (market) -- never packed, so leverage on a market order was discarded and the order
             executed UNLEVERAGED rather than being refused, which is what the limit path does.
* `settleFlag` (market) -- never packed, leaving the engine on FIFO while simulation, the sim limit
             path and the exchange limit path all send NONE.

Each was found by hand, hours apart, from a different downstream symptom. This test does the diff
directly: parse the engine's field list out of the header and compare it against what payload()
emits. It is deliberately mechanical -- it does not know or care what any field means.
"""

import re
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

from taos.im.protocol.exchange.instructions import (  # noqa: E402
    PlaceLimitOrderInstruction,
    PlaceMarketOrderInstruction,
)

HEADER = (
    Path(__file__).resolve().parents[1]
    / "simulate/trading/src/cpp/message/include/taosim/message/ExchangeAgentMessagePayloads.hpp"
)

# Filled in by the ENGINE, never sent by a client.
#   placeholder       - engine-internal padding
#   interfaceOrderId  - assigned engine-side for UI-submitted orders
# Sent only when the miner sets them, so absence from a bare payload is correct.
#   stopLoss / takeProfit
ENGINE_FILLED = {"placeholder", "interfaceOrderId"}
CONDITIONAL = {"stopLoss", "takeProfit"}


def _engine_fields(struct: str) -> set[str]:
    """Wire keys from the struct's MSGPACK_DEFINE_MAP.

    MSGPACK_NVP("wire", member) means the WIRE key is "wire"; a bare member is its own key. Taking
    both (as a naive regex does) invents fields like "stpFlag" that no longer exist on the wire.
    """
    src = HEADER.read_text()
    m = re.search(r"struct\s+" + struct + r"\b", src)
    assert m, f"{struct} not found in {HEADER.name}"
    seg = src[m.end():]
    nxt = re.search(r"\nstruct\s+\w+", seg)
    if nxt:
        seg = seg[: nxt.start()]
    dm = re.search(r"MSGPACK_DEFINE_MAP\((.*?)\);", seg, re.S)
    assert dm, f"{struct} has no MSGPACK_DEFINE_MAP"
    body = re.sub(r"//[^\n]*", "", dm.group(1))

    entries, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        if ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            entries.append(cur)
            cur = ""
        else:
            cur += ch
    entries.append(cur)

    keys = set()
    for e in entries:
        e = e.strip()
        if not e:
            continue
        nvp = re.match(r'MSGPACK_NVP\(\s*"([^"]+)"', e)
        keys.add(nvp.group(1) if nvp else e)
    return keys


def _instruction(kind: str):
    base = dict(agentId=1, bookId=5, direction=0, quantity=1.0, clientOrderId=None, delegate="")
    return PlaceMarketOrderInstruction(**base) if kind == "market" else PlaceLimitOrderInstruction(
        price=0.01, **base
    )


@pytest.mark.parametrize(
    "kind,struct",
    [("market", "PlaceOrderMarketPayload"), ("limit", "PlaceOrderLimitPayload")],
)
def test_python_packs_every_field_the_engine_expects(kind, struct):
    want = _engine_fields(struct) - ENGINE_FILLED - CONDITIONAL
    got = set(_instruction(kind).payload().keys())
    missing = sorted(want - got)
    assert not missing, (
        f"{kind} orders never send {missing}, which {struct} declares. The engine will silently use "
        f"its struct default and the miner's value is discarded with no error on either side. That "
        f"is how stp, currency, leverage and settleFlag were each lost."
    )


@pytest.mark.parametrize(
    "kind,struct",
    [("market", "PlaceOrderMarketPayload"), ("limit", "PlaceOrderLimitPayload")],
)
def test_python_sends_nothing_the_engine_ignores(kind, struct):
    """An unknown key is dropped by msgpack, so a typo'd name fails exactly like omission."""
    extra = sorted(set(_instruction(kind).payload().keys()) - _engine_fields(struct))
    assert not extra, (
        f"{kind} orders send {extra}, which {struct} does not declare. msgpack discards unknown "
        f"keys silently, so this value never reaches the engine -- the same failure as omitting it. "
        f"'stpFlag' was exactly this: spelled wrong, dropped, and unnoticed for the life of "
        f"exchange mode."
    )


def test_conditional_fields_appear_once_set():
    """stopLoss/takeProfit are legitimately absent when unset; prove they are packed when present."""
    base = dict(agentId=1, bookId=5, direction=0, quantity=1.0, clientOrderId=None, delegate="")
    p = PlaceMarketOrderInstruction(stop_loss=0.9, take_profit=1.1, **base).payload()
    assert p.get("stopLoss") == 0.9 and p.get("takeProfit") == 1.1


def test_exchange_api_offers_no_leverage_to_miners():
    """Exchange mode has no way to REFUSE leverage on a market order, so it must not offer it.

    Confirmed 2026-08-07 by whole-path trace: in exchange mode both order types go through
    Exchange::handle*Instruction -> pool.update(), but only a LIMIT order is forwarded on to the LOB,
    where OrderPlacementValidator returns INVALID_LEVERAGE. A MARKET order is executed as an AMM pool
    swap and returns a signal directly, so validateMarketOrderPlacement never runs on it and the
    leverage field is never read -- neither honoured nor rejected.

    What keeps that safe is this API surface: neither exchange method exposes `leverage`, while the
    SIMULATION API (taos/im/protocol/response.py) exposes it on both. Adding the parameter back here
    would compile, ship, and silently do nothing on the market path. This test is the tripwire.
    """
    import inspect

    from taos.im.protocol.exchange.response import ExchangeAgentResponse

    offenders = sorted(
        m
        for m in ("market_order", "limit_order")
        if "leverage" in inspect.signature(getattr(ExchangeAgentResponse, m)).parameters
    )
    assert not offenders, (
        f"ExchangeAgentResponse.{offenders} now accept `leverage`. Exchange-mode market orders are "
        f"AMM pool swaps that never reach OrderPlacementValidator, so a leveraged market order "
        f"cannot be rejected -- it would be silently ignored. Either remove the parameter or add a "
        f"guard to Exchange::handleMarketOrderInstruction first."
    )
