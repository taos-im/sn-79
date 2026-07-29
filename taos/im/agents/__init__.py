# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
import os
import json
import msgpack
import traceback
import time
import csv
import urllib.request
import urllib.error
from datetime import datetime
from typing import cast
import bittensor as bt
from threading import Thread, Lock
from abc import ABC, abstractmethod
from taos.common.agents import SimulationAgent
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse, FinanceEventNotification
# taos.im.protocol.exchange is excluded from the public release (exchange-mode
# only). Sentinel classes keep type hints and isinstance() checks valid; code
# paths that operate on real instances are gated by self._exchange_mode and
# never invoked in public sim-only deployments.
try:
    from taos.im.protocol.exchange import ExchangeStateUpdate, ExchangeAgentResponse
    from taos.im.protocol.exchange.models import OrderCurrency as ExchangeOrderCurrency
except ImportError:
    # Sentinels inherit from pydantic.BaseModel (not plain classes) so FastAPI
    # can register routes whose param/return types are Union[..., ExchangeStateUpdate]
    # — a bare class fails get_dependant() with "Invalid args for response field".
    from pydantic import BaseModel as _SentinelBase

    class ExchangeStateUpdate(_SentinelBase):  # type: ignore[no-redef]
        pass

    class ExchangeAgentResponse(_SentinelBase):  # type: ignore[no-redef]
        pass
    class ExchangeOrderCurrency:  # type: ignore[no-redef]
        ALPHA = None  # only referenced inside the exchange-mode market_order branch
from taos.im.protocol.events import *
from taos.im.protocol.models import *
from taos.im.utils import duration_from_timestamp, timestamp_from_duration

@dataclass
class RollingWindow:
    """
    Rolling window configuration for price sampling.
    Attributes:
        min (int): Minimum number of samples required before signals are considered reliable.
        max (int): Maximum length of the rolling buffer (in samples).
        samples (int): If max is not used or multi-scale approach
        num_windows (int): Multi-scale approaches
        sampling_interval (int): timestamps per sample (sec or nanosec)
    
    Parameter Tuning Guidelines:
        - TODO
    """
    min: int
    max: int
    samples: int
    num_windows: int
    sampling_interval: int = 1

@dataclass
class Thresholds:
    """
    Threshold configuration for generating trading signals.

    Attributes:
        signal (float): 
        tolerance (float): 
        model (float): 

    Parameter Tuning Guidelines:
        - Increase signal_threshold to reduce trading frequency, focus on strong trends.
        - Increase tolerance to reduce overtrading in noisy markets.
        - Adjust model_threshold to filter out unreliable signals.
    """
    signal: float
    tolerance: float
    model: float

@dataclass
class TimestampedPrice:
    """Container for midquote prices with timestamps."""
    price: float
    timestamp: int = 0

@dataclass
class Positions:
    """Container for positions for different books"""
    open: bool = False
    direction: OrderDirection = OrderDirection.BUY
    amount: int = 0
    qty: float = 0
    
class Signals(IntEnum):
    """
    Enum to represent signals coming from the model

    Attributes:
        ENTRY (int): Open a new position
        EXIT (int): Close current position 
        HOLD (int): Hold or extend current position
        NOISE (int): Ignore
        -- Simplified
        BULLISH (int): Rising
        BEARISH (int): Falling
    """
    REVERSION=0
    MOMENTUM=1
    HOLD=3
    NOISE=4
    BULLISH=5
    BEARISH=6


# ─────────────────────────────────────────────────────────────────────────────
# Account proxy helpers
# ─────────────────────────────────────────────────────────────────────────────

class _BalanceProxy:
    """Minimal Balance-like object wrapping a free float from an exchange account dict."""
    def __init__(self, free: float, total: float | None = None):
        self.free     = free
        self.total    = total if total is not None else free
        self.reserved = max(0.0, self.total - self.free)

class _FeesProxy:
    """Minimal Fees-like object wrapping exchange fee fields (or zero defaults)."""
    def __init__(self, maker: float = 0.0, taker: float = 0.0):
        self.maker_fee_rate = maker
        self.taker_fee_rate = taker

class _OrderProxy:
    """Minimal Order-like object wrapping an open order dict from exchange mode."""
    def __init__(self, raw: dict):
        self.i    = raw.get('i', raw.get('id', 0))
        self.s    = raw.get('s', raw.get('side', 0))
        self.p    = raw.get('p', raw.get('price', 0.0))
        self.q    = raw.get('q', raw.get('quantity', 0.0))

    @property
    def id(self) -> int:
        return self.i

    @property
    def side(self) -> int:
        return self.s

    @property
    def price(self) -> float:
        return self.p

    @property
    def quantity(self) -> float:
        return self.q

class UnifiedAccount:
    """
    Normalises simulation Account Pydantic objects and exchange plain dicts to a
    single interface so agents written for simulation work unchanged in exchange mode.

    Simulation path:  wraps an Account object — delegates all attribute access to it.
    Exchange path:    wraps a plain dict (rich: bb/qb/bl/ql/bc/qc/f/BASE/QUOTE,
                      or fallback: BASE/QUOTE/WEALTH) via proxy objects.
    """
    def __init__(self, raw):
        self._raw    = raw
        self._is_dict = isinstance(raw, dict)

    # ── Balance ───────────────────────────────────────────────────────────────

    @property
    def base_balance(self) -> _BalanceProxy:
        if self._is_dict:
            bb = self._raw.get('bb')
            free = bb.get('f', 0.0) if isinstance(bb, dict) else self._raw.get('BASE', 0.0)
            return _BalanceProxy(free)
        return self._raw.base_balance

    @property
    def quote_balance(self) -> _BalanceProxy:
        if self._is_dict:
            qb = self._raw.get('qb')
            free = qb.get('f', 0.0) if isinstance(qb, dict) else self._raw.get('QUOTE', 0.0)
            return _BalanceProxy(free)
        return self._raw.quote_balance

    # ── Loans / collateral ────────────────────────────────────────────────────

    @property
    def base_loan(self) -> float:
        return self._raw.get('bl', 0.0) if self._is_dict else self._raw.base_loan

    @property
    def quote_loan(self) -> float:
        return self._raw.get('ql', 0.0) if self._is_dict else self._raw.quote_loan

    @property
    def base_collateral(self) -> float:
        return self._raw.get('bc', 0.0) if self._is_dict else self._raw.base_collateral

    @property
    def quote_collateral(self) -> float:
        return self._raw.get('qc', 0.0) if self._is_dict else self._raw.quote_collateral

    # ── Orders / loans ────────────────────────────────────────────────────────

    @property
    def orders(self) -> list:
        if self._is_dict:
            raw_orders = self._raw.get('o', [])
            return [_OrderProxy(o) for o in raw_orders if isinstance(o, dict)]
        return self._raw.orders

    @property
    def loans(self) -> dict:
        return {} if self._is_dict else self._raw.loans

    # ── Fees ──────────────────────────────────────────────────────────────────

    @property
    def fees(self) -> _FeesProxy:
        if self._is_dict:
            f = self._raw.get('f', {})
            if isinstance(f, dict):
                return _FeesProxy(f.get('m', 0.0), f.get('t', 0.0))
            return _FeesProxy()
        return self._raw.fees

    @property
    def traded_volume(self) -> float:
        if self._is_dict:
            return self._raw.get('tv', 0.0)
        return getattr(self._raw, 'traded_volume', 0.0)

    @property
    def raw(self):
        """Escape hatch for mode-specific attribute access."""
        return self._raw


# ─────────────────────────────────────────────────────────────────────────────
# Unified response wrapper
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedAgentResponse:
    """
    Mode-agnostic response wrapper with the simulation FinanceAgentResponse API.

    Builds simulation or exchange Pydantic instruction objects depending on mode.
    Call finalize() to obtain the correctly-typed Pydantic response for serialization.
    """

    def __init__(self, agent_id: int, exchange_mode: bool, delegate: str = ''):
        self.agent_id       = agent_id
        self._exchange_mode = exchange_mode
        self._delegate      = delegate
        self.instructions   = []

    def limit_order(
        self,
        book_id,
        direction,
        quantity: float,
        price: float,
        delay: int = 0,
        clientOrderId=None,
        stp=STP.CANCEL_OLDEST,
        postOnly: bool = False,
        timeInForce=TimeInForce.GTC,
        expiryPeriod=None,
        leverage: float = 0.0,
        settlement_option=LoanSettlementOption.NONE,
    ) -> None:
        if self._exchange_mode:
            from taos.im.protocol.exchange.instructions import PlaceLimitOrderInstruction
            self.instructions.append(PlaceLimitOrderInstruction(
                agentId=self.agent_id, delay=delay, bookId=book_id,
                delegate=self._delegate, direction=direction,
                quantity=quantity, price=price, clientOrderId=clientOrderId,
                stp=stp, postOnly=postOnly, timeInForce=timeInForce,
                expiryPeriod=expiryPeriod, settleFlag=settlement_option,
                # leverage silently dropped — exchange does not support margin
            ))
        else:
            from taos.im.protocol.instructions import PlaceLimitOrderInstruction as _Sim
            self.instructions.append(_Sim(
                agentId=self.agent_id, delay=delay, bookId=book_id,
                direction=direction, quantity=quantity, price=price,
                clientOrderId=clientOrderId, stp=stp, postOnly=postOnly,
                timeInForce=timeInForce, expiryPeriod=expiryPeriod,
                leverage=leverage, settleFlag=settlement_option,
            ))

    def market_order(
        self,
        book_id,
        direction,
        quantity: float,
        delay: int = 0,
        clientOrderId=None,
        stp=STP.CANCEL_OLDEST,
        currency=None,
        leverage: float = 0.0,
        settlement_option=LoanSettlementOption.NONE,
        max_slippage: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        if self._exchange_mode:
            from taos.im.protocol.exchange.instructions import PlaceMarketOrderInstruction
            self.instructions.append(PlaceMarketOrderInstruction(
                agentId=self.agent_id, delay=delay, bookId=book_id,
                delegate=self._delegate, direction=direction, quantity=quantity,
                max_slippage=max_slippage if max_slippage is not None else 0.01,
                clientOrderId=clientOrderId, stp=stp,
                currency=currency if currency is not None else ExchangeOrderCurrency.ALPHA,
                stop_loss=stop_loss, take_profit=take_profit,
                # leverage and settlement_option silently dropped
            ))
        else:
            from taos.im.protocol.instructions import PlaceMarketOrderInstruction as _Sim
            self.instructions.append(_Sim(
                agentId=self.agent_id, delay=delay, bookId=book_id,
                direction=direction, quantity=quantity,
                clientOrderId=clientOrderId, stp=stp,
                currency=currency if currency is not None else OrderCurrency.BASE,
                leverage=leverage, settleFlag=settlement_option,
                max_slippage=max_slippage,
                stop_loss=stop_loss, take_profit=take_profit,
            ))

    def cancel_order(self, book_id, order_id, quantity=None, delay: int = 0) -> None:
        # quantity=None → full cancel; a value → partial cancel of that many BASE units.
        self.cancel_orders(book_id, [order_id], volume=quantity, delay=delay)

    def cancel_orders(self, book_id, order_ids, volume=None, delay: int = 0) -> None:
        # volume applies to every order id (None = full cancel); pass a list to vary per order.
        vols = volume if isinstance(volume, (list, tuple)) else [volume] * len(order_ids)
        if self._exchange_mode:
            from taos.im.protocol.exchange.instructions import (
                CancelOrdersInstruction, CancelOrderInstruction)
            self.instructions.append(CancelOrdersInstruction(
                agentId=self.agent_id, delay=delay, bookId=book_id,
                cancellations=[CancelOrderInstruction(orderId=oid, volume=v)
                               for oid, v in zip(order_ids, vols)],
            ))
        else:
            from taos.im.protocol.instructions import (
                CancelOrdersInstruction as _Sim, CancelOrderInstruction as _SimC)
            self.instructions.append(_Sim(
                agentId=self.agent_id, delay=delay, bookId=book_id,
                cancellations=[_SimC(orderId=oid, volume=v) for oid, v in zip(order_ids, vols)],
            ))

    def close_position(self, book_id, order_id, quantity=None, delay: int = 0) -> None:
        if self._exchange_mode:
            bt.logging.warning("close_position is not supported in exchange mode — ignored")
            return
        from taos.im.protocol.instructions import ClosePositionsInstruction, ClosePositionInstruction
        self.instructions.append(ClosePositionsInstruction(
            agentId=self.agent_id, delay=delay, bookId=book_id,
            positions=[ClosePositionInstruction(orderId=order_id, volume=quantity)],
        ))

    def add_instruction(self, instruction) -> None:
        self.instructions.append(instruction)

    def finalize(self) -> FinanceAgentResponse | ExchangeAgentResponse:
        """Return the correctly-typed Pydantic response for serialization."""
        if self._exchange_mode:
            r = ExchangeAgentResponse(agent_id=self.agent_id)
        else:
            r = FinanceAgentResponse(agent_id=self.agent_id)
        r.instructions = self.instructions
        return r


# Base class for agents operating in intelligent market simulations
class FinanceSimulationAgent(SimulationAgent):
    # Populated each tick from the (decompressed) state in update(). Declared here
    # so subclasses see a concrete type instead of the synapse field's wire union
    # (MarketSimulationConfig | str | ExchangeConfig | dict | None).
    simulation_config: MarketSimulationConfig
    # Per-tick event notices for this agent. Events are dispatched by their string
    # `.type` tag (match/case), which the type system cannot correlate with the
    # concrete event subclass, so the element type is intentionally untyped.
    events: list

    def __init__(self, uid : int, config : object, log_dir : str | None = None) -> None:
        """
        Initializer method that sets up the agent's unique ID and configuration, and initializes common objects for storing agent data.

        Args:
            uid (int): The UID of the agent in the subnet.
            config (obj): Config object for the agent.

        Returns:
            None
        """
        self.history = []
        # State snapshots kept for handlers that read self.history (e.g. last
        # bid/ask). 0 keeps none — set it on agents that never read history to
        # avoid retaining full state copies.
        self.history_len = int(getattr(config, "history_len", 10))
        self.accounts = {}
        self.event_history : dict[str, AgentEventHistory | None] = {}
        if not hasattr(config, "lazy_load"):
            config.lazy_load = False
        else:
            config.lazy_load = bool(config.lazy_load)
        super().__init__(uid, config, log_dir)
        self._live_config_start()

    # ── Live config (Tier 2) ───────────────────────────────────────────────────
    # Opt-in remote param reload with no restart. Enabled when the agent param
    # `live_config_poll_s` is > 0 AND the scheduler injects TAOS_LIVE_CONFIG_URL
    # (+ optional TAOS_LIVE_CONFIG_TOKEN) into the container. Because Hangar agents
    # expose no inbound port, the agent PULLS: a daemon thread polls the endpoint
    # off the hot path; the next handle() applies any change to self.config and
    # calls on_config_reload(changed). See runbooks/planning/hangar-live-config.md.

    @staticmethod
    def _coerce_live_value(v):
        # Match ParseKwargs: numeric strings become floats, everything else stays.
        if not isinstance(v, str):
            return v
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    @staticmethod
    def _parse_kv_csv(s: str) -> dict:
        out = {}
        for pair in str(s or "").split(","):
            if "=" in pair:
                k, _, v = pair.partition("=")
                k = k.strip()
                if k:
                    out[k] = FinanceSimulationAgent._coerce_live_value(v.strip())
        return out

    def _parse_live_config_body(self, body: str) -> dict:
        body = (body or "").strip()
        if not body:
            return {}
        if body[0] in "{[":
            try:
                data = json.loads(body)
            except ValueError:
                data = None
            if isinstance(data, dict):
                src = data.get("agent_params", data)
                if isinstance(src, str):
                    return self._parse_kv_csv(src)
                if isinstance(src, dict):
                    return {str(k): self._coerce_live_value(v) for k, v in src.items()}
        return self._parse_kv_csv(body)

    def _live_config_start(self) -> None:
        poll_s = int(getattr(self.config, "live_config_poll_s", 0) or 0)
        url = os.environ.get("TAOS_LIVE_CONFIG_URL", "").strip()
        if poll_s <= 0 or not url:
            return
        self._live_cfg_lock = Lock()
        self._live_cfg_pending = None
        self._live_cfg_etag = None
        Thread(target=self._live_config_loop, args=(url, poll_s), daemon=True, name="taos-live-config").start()
        bt.logging.info(f"live-config: polling {url} every {poll_s}s")

    def _live_config_loop(self, url: str, poll_s: int) -> None:
        token = os.environ.get("TAOS_LIVE_CONFIG_TOKEN", "").strip()
        while True:
            time.sleep(poll_s)
            try:
                req = urllib.request.Request(url)
                if token:
                    req.add_header("Authorization", f"Bearer {token}")
                if self._live_cfg_etag:
                    req.add_header("If-None-Match", self._live_cfg_etag)
                try:
                    with urllib.request.urlopen(req, timeout=10) as r:
                        etag = r.headers.get("ETag")
                        body = r.read().decode("utf-8", "replace")
                except urllib.error.HTTPError as e:
                    if e.code == 304:
                        continue
                    raise
                params = self._parse_live_config_body(body)
                with self._live_cfg_lock:
                    self._live_cfg_pending = params
                    self._live_cfg_etag = etag
            except Exception as e:
                bt.logging.debug(f"live-config: poll failed: {e}")

    def _apply_pending_live_config(self) -> None:
        # Cheap fast-path: the attribute only exists once the poller is running.
        if getattr(self, "_live_cfg_pending", None) is None:
            return
        with self._live_cfg_lock:
            pending = self._live_cfg_pending
            self._live_cfg_pending = None
        if not pending:
            return
        changed = {}
        for k, v in pending.items():
            if getattr(self.config, k, None) != v:
                setattr(self.config, k, v)
                changed[k] = v
        if changed:
            bt.logging.info(f"live-config: applied {sorted(changed.keys())}")
            try:
                self.on_config_reload(changed)
            except Exception as e:
                bt.logging.warning(f"live-config: on_config_reload raised: {e}")

    def on_config_reload(self, changed: dict) -> None:
        """Hook: a live strategy-param change was applied to self.config.

        `changed` maps each changed param name to its new (coerced) value. The
        config attributes are already updated; override this to re-derive cached
        state computed in __init__ (e.g. ranges, intervals). Runs on the handle
        thread before the tick's response is produced. Default: no-op.
        """
        return None

    def handle(self, state: MarketSimulationStateUpdate | ExchangeStateUpdate) -> FinanceAgentResponse | ExchangeAgentResponse:
        self._apply_pending_live_config()
        return super().handle(state)

    def process(self, notification: FinanceEventNotification) -> FinanceEventNotification:
        """
        Method to handle a new event notification.
        """
        notification.acknowledged = True
        if notification.event.type == 'EVENT_SIMULATION_END':
            self.onEnd(notification.event)
        return notification
    
    def simulation_output_dir(self, state : MarketSimulationStateUpdate | ExchangeStateUpdate):
        simulation_output_dir = os.path.join(self.output_dir, state.dendrite.hotkey, state.config.simulation_id)
        os.makedirs(simulation_output_dir, exist_ok=True)
        return simulation_output_dir
    
    def load_event_history(self, state) -> None:
        """
        Load per-agent event history from CSV files into an AgentEventHistory object.

        This method:
        - Loads CSVs (orders, cancellations, trades).
        - Constructs event objects directly from CSV columns (no from_json).
        - Aggregates into an AgentEventHistory instance.
        - Applies optional retention logic based on `event_lookback_minutes`.

        Populates:
            self.event_history[state.dendrite.hotkey] (AgentEventHistory | None)
        """
        self.event_lookback_minutes = getattr(self.config, "event_lookback_minutes", 60)

        base_dir = self.simulation_output_dir(state)

        def _load_orders(path: str):
            events = []
            if os.path.isfile(path):
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            if row['price']:
                                ev = LimitOrderPlacementEvent(
                                    timestamp=timestamp_from_duration(row["timestamp"]),
                                    agentId=self.uid,
                                    bookId=int(row["bookId"]),
                                    orderId=int(row["orderId"]) if row["orderId"] else None,
                                    clientOrderId=int(row["clientOrderId"]) if row["clientOrderId"] else None,
                                    side=int(row["side"]),
                                    p=float(row["price"]),
                                    quantity=float(row["quantity"]),
                                    leverage=float(row["leverage"]),
                                    settleFlag=row.get("settleFlag"),
                                    success=row["success"].lower() == "true",
                                    message=row.get("message", ""),
                                )
                            else:
                                ev = MarketOrderPlacementEvent(
                                    timestamp=timestamp_from_duration(row["timestamp"]),
                                    agentId=self.uid,
                                    bookId=int(row["bookId"]),
                                    orderId=int(row["orderId"]) if row["orderId"] else None,
                                    clientOrderId=int(row["clientOrderId"]) if row["clientOrderId"] else None,
                                    side=int(row["side"]),
                                    r=(row["currency"] if row["currency"].isnumeric() else OrderCurrency[row["currency"].split('.')[1]]) if "currency" in row and row["currency"] else OrderCurrency.BASE,
                                    quantity=float(row["quantity"]),
                                    leverage=float(row["leverage"]),
                                    settleFlag=row.get("settleFlag"),
                                    success=row["success"].lower() == "true",
                                    message=row.get("message", ""),
                                )
                            events.append(ev)
                        except Exception as e:
                            bt.logging.warning(f"Failed to parse order row: {e}")
            return events

        def _load_cancellations(path: str):
            events = []
            if os.path.isfile(path):
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            ev = OrderCancellationEvent(
                                t=timestamp_from_duration(row["timestamp"]),
                                b=int(row["bookId"]),
                                o=int(row["orderId"]),
                                q=float(row["quantity"]) if row["quantity"] else None,
                                u=row["success"].lower() == "true",
                                m=row.get("message", ""),
                            )
                            events.append(ev)
                        except Exception as e:
                            bt.logging.warning(f"Failed to parse cancellation row: {e}")
            return events

        def _load_trades(path: str):
            events = []
            if os.path.isfile(path):
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            ev = TradeEvent(
                                timestamp=timestamp_from_duration(row["timestamp"]),
                                agentId=self.uid,
                                b=int(row["bookId"]) if row["bookId"] else None,
                                i=int(row["tradeId"]),
                                c=int(row["clientOrderId"]) if row["clientOrderId"] else None,
                                Ta=int(row["takerAgentId"]),
                                Ti=int(row["takerOrderId"]),
                                Tf=float(row["takerFee"]),
                                Ma=int(row["makerAgentId"]),
                                Mi=int(row["makerOrderId"]),
                                Mf=float(row["makerFee"]),
                                s=int(row["side"]),
                                p=float(row["price"]),
                                q=float(row["quantity"]),
                            )
                            events.append(ev)
                        except Exception as e:
                            bt.logging.warning(f"Failed to parse trade row: {e}")
            return events

        # Load everything
        orders = _load_orders(os.path.join(base_dir, "orders.csv"))
        cancels = _load_cancellations(os.path.join(base_dir, "cancellations.csv"))
        trades = _load_trades(os.path.join(base_dir, "trades.csv"))

        all_events = orders + cancels + trades
        if not all_events:
            self.event_history[state.dendrite.hotkey] = AgentEventHistory(
                uid=self.uid,
                start=state.timestamp - state.config.publish_interval,
                end=state.timestamp,
                events=[],
                publish_interval=self.simulation_config.publish_interval,
                retention_mins=self.event_lookback_minutes,
            )
        else:
            # Sort by timestamp
            all_events.sort(key=lambda e: e.timestamp)

            start = all_events[0].timestamp
            end = all_events[-1].timestamp
            self.event_history[state.dendrite.hotkey] = AgentEventHistory(
                uid=self.uid,
                start=start,
                end=end,
                events=all_events,
                publish_interval=getattr(self.config, "publish_interval", 1_000_000_000),
                retention_mins=self.event_lookback_minutes,
            )
        
    
    def log_order_event(self, event : LimitOrderPlacementEvent | MarketOrderPlacementEvent, state : MarketSimulationStateUpdate | ExchangeStateUpdate):
        """Log LimitOrderPlacementEvent or MarketOrderPlacementEvent to CSV."""
        orders_log_file = os.path.join(self.simulation_output_dir(state), 'orders.csv')
        file_exists = os.path.exists(orders_log_file)
        with open(orders_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'bookId', 'orderId', 'clientOrderId',
                    'side', 'price', 'currency', 'quantity', 'leverage', 'settleFlag',
                    'success', 'message'
                ])
            writer.writerow([
                duration_from_timestamp(event.timestamp),
                getattr(event, 'bookId', None),
                getattr(event, 'orderId', None),
                getattr(event, 'clientOrderId', None),
                getattr(event, 'side', None),
                getattr(event, 'price', None),
                getattr(event, 'currency', None),
                getattr(event, 'quantity', None),
                getattr(event, 'leverage', None),
                getattr(event, 'settleFlag', None),
                event.success,
                event.message
            ])

    def log_cancellation_event(self, event : OrderCancellationEvent, state : MarketSimulationStateUpdate | ExchangeStateUpdate):
        """Log OrderCancellationEvent to CSV."""
        cancellations_log_file = os.path.join(self.simulation_output_dir(state), 'cancellations.csv')
        file_exists = os.path.exists(cancellations_log_file)
        with open(cancellations_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'bookId', 'orderId', 'quantity', 'success', 'message'
                ])
            writer.writerow([
                duration_from_timestamp(event.timestamp),
                event.bookId,
                event.orderId,
                event.quantity,
                event.success,
                event.message
            ])

    def log_trade_event(self, event : TradeEvent, state : MarketSimulationStateUpdate | ExchangeStateUpdate):
        """Log TradeEvent to CSV."""
        trades_log_file = os.path.join(self.simulation_output_dir(state), 'trades.csv')
        file_exists = os.path.exists(trades_log_file)
        with open(trades_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'bookId', 'tradeId', 'clientOrderId',
                    'takerAgentId', 'takerOrderId', 'takerFee',
                    'makerAgentId', 'makerOrderId', 'makerFee',
                    'side', 'price', 'quantity'
                ])
            writer.writerow([
                duration_from_timestamp(event.timestamp),
                event.bookId,
                event.tradeId,
                event.clientOrderId,
                event.takerAgentId,
                event.takerOrderId,
                event.takerFee,
                event.makerAgentId,
                event.makerOrderId,
                event.makerFee,
                event.side,
                event.price,
                event.quantity
            ])

    def update(self, state : MarketSimulationStateUpdate | ExchangeStateUpdate) -> None:
        """
        Method to update the stored agent data, print relevant state information and trigger handlers for reported events.

        Args:
            state (taos.im.protocol.MarketSimulationStateUpdate): The UID of the agent in the subnet.

        Returns:
            None
        """
        if self.history_len:
            self.history.append(state.model_copy())
            self.history = self.history[-self.history_len:]
        else:
            self.history = []
        self.simulation_config = cast(MarketSimulationConfig, state.config)
        self.accounts = state.accounts[self.uid]
        self.events = cast(list, state.notices[self.uid])
        
        if state.dendrite.hotkey not in self.event_history or not self.event_history[state.dendrite.hotkey]:            
            self.load_event_history(state)        
        self.event_history[state.dendrite.hotkey].append(state)
        
        simulation_ended = False
        update_text = ''
        update_text += "\n" + '-' * 50 + "\n"
        update_text += f'VALIDATOR : {state.dendrite.hotkey} | SIMULATION TIME : {duration_from_timestamp(state.timestamp)} (T={state.timestamp})' + "\n"
        update_text += '-' * 50 + "\n"
        if len(self.events) > 0:
            global_events = False
            for event in self.events:
                match event.type:
                    case"RESET_AGENTS" | "RA":
                        if not global_events:
                            update_text += 'GLOBAL EVENTS' + "\n"
                            update_text += '-' * 50 + "\n"                            
                            global_events = True
                        update_text += f"{event}" + "\n"
                    case "EVENT_SIMULATION_START" | "ESS":
                        if not global_events:
                            update_text += 'GLOBAL EVENTS' + "\n"
                            update_text += '-' * 50 + "\n"                            
                            global_events = True
                        update_text += f"{event}" + "\n"
                        self.onStart(event)
                    case "EVENT_SIMULATION_END" | "ESE":
                        if not global_events:
                            update_text += 'GLOBAL EVENTS' + "\n"
                            update_text += '-' * 50 + "\n"                            
                            global_events = True
                        update_text += f"{event}" + "\n"
                        simulation_ended = True
                    case _:
                        pass
            if global_events:                
                update_text += '-' * 50 + "\n"
        debug_text = update_text
        for book_id in range(self.simulation_config.book_count):
            debug_text += f"BOOK {book_id}" + "\n"            
            debug_text += '-' * 50 + "\n"
            debug_text += 'EVENTS' + "\n"
            debug_text += '-' * 50 + "\n"
            for event in self.events:
                if hasattr(event, 'bookId') and event.bookId == book_id:
                    if event.type not in ["EVENT_TRADE", "ET"]:
                        debug_text += f"{event}" + "\n"
                        update_text += f"BOOK {book_id} : {event}" + "\n"
                    match event.type:
                        case "RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT" | "RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET" | "RDPOL" | "RDPOM":
                            self.onOrderAccepted(event)
                            self.log_order_event(event, state)
                        case "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT" | "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET" | "ERDPOL" | "ERDPOM":
                            self.onOrderRejected(event)
                        case "RESPONSE_DISTRIBUTED_CANCEL_ORDERS" | "RDCO":
                            for cancellation in event.cancellations:
                                self.onOrderCancelled(cancellation)
                                self.log_cancellation_event(cancellation, state)
                        case "ERROR_RESPONSE_DISTRIBUTED_CANCEL_ORDERS" | "ERDCO":
                            for cancellation in event.cancellations:
                                self.onOrderCancellationFailed(cancellation)
                        case "RESPONSE_DISTRIBUTED_CLOSE_POSITIONS" | "RDCP":
                            for close in event.closes:
                                self.onPositionClosed(close)
                        case "ERROR_RESPONSE_DISTRIBUTED_CLOSE_POSITIONS" | "ERDCP":
                            for close in event.closes:
                                self.onPositionCloseFailed(close)
                        case "EVENT_TRADE" | "ET":
                            role = "taker" if self.uid == event.takerAgentId else "maker"
                            trade_text = f"{'BUY ' if event.side == 0 else 'SELL'} TRADE #{event.tradeId} : YOUR {'AGGRESSIVE' if role=='taker' else 'PASSIVE'} " + \
                                f"ORDER #{event.takerOrderId if role=='taker' else event.makerOrderId} (AGENT {event.takerAgentId if role=='taker' else event.makerAgentId}) " + \
                                f"MATCHED AGAINST #{event.makerOrderId if role=='taker' else event.takerOrderId} (AGENT {event.makerAgentId if role=='taker' else event.takerAgentId}) " + \
                                f"FOR {event.quantity}@{event.price} AT {duration_from_timestamp(event.timestamp)} (T={event.timestamp})"
                            debug_text += f"{trade_text}" + "\n"
                            update_text += f"BOOK {book_id} : {trade_text}" + "\n"
                            self.onTrade(event,state.dendrite.hotkey)
                            self.log_trade_event(event, state)
                        case _:
                            bt.logging.warning(f"Unknown event : {event}")
            if len(self.events) == 0: 
                debug_text += "NO EVENTS\n"
            debug_text += '-' * 50 + "\n"
            if not self.config.lazy_load:
                account= self.accounts[book_id]
                debug_text += "TOP LEVELS" + "\n"
                debug_text += '-' * 50 + "\n"
                debug_text += ' | '.join([f"{level.quantity:.4f}@{level.price}" for level in reversed(state.books[book_id].bids[:5])]) + '||' + ' | '.join([f"{level.quantity:.4f}@{level.price}" for level in state.books[book_id].asks[:5]]) + "\n"
                debug_text += '-' * 50 + "\n"
                debug_text += 'BALANCES' + "\n"
                debug_text += '-' * 50 + "\n"
                debug_text += f"BASE  : TOTAL={account.base_balance.total:.8f} FREE={account.base_balance.free:.8f} RESERVED={account.base_balance.reserved:.8f} | LOAN={account.base_loan:.8f} COLLATERAL={account.base_collateral}" + "\n"
                debug_text += f"QUOTE : TOTAL={account.quote_balance.total:.8f} FREE={account.quote_balance.free:.8f} RESERVED={account.quote_balance.reserved:.8f} | LOAN={account.quote_loan:.8f} COLLATERAL={account.quote_collateral}" + "\n"
                if len(account.orders) > 0:
                    debug_text += '-' * 50 + "\n"
                    debug_text += 'ORDERS' + "\n"
                    debug_text += '-' * 50 + "\n"
                    for order in sorted(account.orders, key=lambda x: x.timestamp):
                        debug_text += f"#{order.id} : {'BUY ' if order.side == 0 else 'SELL'} {f'{1+order.leverage:.2f}x' if order.leverage > 0 else ''}{order.quantity}@{order.price} [PLACED AT {duration_from_timestamp(order.timestamp)} (T={order.timestamp})]" + "\n"
                if len(account.loans) > 0:
                    debug_text += '-' * 50 + "\n"
                    debug_text += 'LOANS' + "\n"
                    debug_text += '-' * 50 + "\n"
                    for order_id, loan in account.loans.items():
                        debug_text += f"#{order_id} : {loan}\n"
                if account.fees:
                    debug_text += '-' * 50 + "\n"
                    debug_text += f'FEES : TRADED {account.fees.volume_traded} | MAKER {account.fees.maker_fee_rate * 100}% | TAKER {account.fees.taker_fee_rate * 100}%' + "\n"
                    debug_text += '-' * 50 + "\n"
                debug_text += '-' * 50 + "\n"
        if simulation_ended:
            update_text += f"{event}" + "\n"
            update_text += '-' * 50 + "\n"
            self.onEnd(event)
        bt.logging.debug("." + debug_text)
        if bt.logging.current_state_value == 'Info':
            bt.logging.info("." + update_text)

    # Handler functions for various simulation events, to be overridden in agent implementations.
    def onStart(self, event : SimulationStartEvent) -> None:
        """
        Handler for simulation start event.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.SimulationStartEvent): The event class representing start of simulation.

        Returns:
            None
        """
        pass

    def onOrderAccepted(self, event : OrderPlacementEvent) -> None:
        """
        Handler for event where order is accepted to the book by simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderPlacementEvent): The event class representing order placement.

        Returns:
            None
        """
        pass

    def onOrderRejected(self, event : OrderPlacementEvent) -> None:
        """
        Handler for event where order is rejected for placement by simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderPlacementEvent): The event class representing order rejection.

        Returns:
            None
        """
        match event.message:
            case 'EXCEEDING_MAX_ORDERS':
                bt.logging.warning(f"FAILED TO PLACE {'BUY' if event.side == 0 else 'SELL'} ORDER FOR {event.quantity}@{event.price if event.type.endswith('L') else 'MARKET'} ON BOOK {event.bookId} : You already have the maximum allowed number of open orders ({self.simulation_config.max_open_orders}) on this book.  You will not be able to place any more orders until you either cancel existing orders, or they are traded.")
            case 'EXCEEDING_LOAN':
                bt.logging.warning(f"FAILED TO PLACE {'BUY' if event.side == 0 else 'SELL'} ORDER FOR {event.quantity}@{event.price if event.type.endswith('L') else 'MARKET'} ON BOOK {event.bookId} : You have exceeded the maximum allowed loan value ({self.simulation_config.max_loan}) on this book.  You need to close positions using the close_position method to repay some of the loan amount before you can place more leveraged orders.")
            case 'DUAL_POSITION':
                bt.logging.warning(f"FAILED TO PLACE {'BUY' if event.side == 0 else 'SELL'} ORDER FOR {event.quantity}@{event.price if event.type.endswith('L') else 'MARKET'} ON BOOK {event.bookId} : You already have a margin position in the opposite direction.  You must close this position before you can take leverage on this side of the book.")
            case _:
                pass

    def onOrderCancelled(self, event : OrderCancellationEvent) -> None:
        """
        Handler for event where order is cancelled in the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderCancellationEvent): The event class representing order cancellation.

        Returns:
            None
        """
        pass

    def onOrderCancellationFailed(self, event : OrderCancellationEvent) -> None:
        """
        Handler for event where order cancellation request is rejected by the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderCancellationEvent): The event class representing order cancellation.

        Returns:
            None
        """
        pass
    
    def onPositionClosed(self, event : ClosePositionEvent) -> None:
        """
        Handler for event where position is closed in the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.ClosePositionEvent): The event class representing position closure.

        Returns:
            None
        """
        pass

    def onPositionCloseFailed(self, event : ClosePositionEvent) -> None:
        """
        Handler for event where close position request is rejected by the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.ClosePositionEvent): The event class representing position closure.

        Returns:
            None
        """
        pass

    def onTrade(self, event : TradeEvent, validator: str = None) -> None:
        """
        Handler for event where an order is traded in the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.TradeEvent): The event class representing a trade.
            validator: Validator identifier (optional)
        Returns:
            None
        """
        pass

    def onEnd(self, event : SimulationEndEvent) -> None:
        """
        Handler for simulation end event.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.SimulationEndEvent): The event class representing end of simulation.

        Returns:
            None
        """
        pass

    def respond(self, state : MarketSimulationStateUpdate | ExchangeStateUpdate) -> FinanceAgentResponse | ExchangeAgentResponse:
        """
        Abstract method for handling generation of response to new state update.  To be implemented by subclasses.

        Args:
            state (taos.im.protocol.MarketSimulationStateUpdate): The class representing the latest state of the simulation.

        Returns:
            taos.im.protocol.FinanceAgentResponse : The response which will be attached to the synapse for return to the querying validator.
        """
        ...

    def report(self, state : MarketSimulationStateUpdate | ExchangeStateUpdate, response : FinanceAgentResponse | ExchangeAgentResponse) -> None:
        """
        Method for reporting the latest simulation state and the response generated by the agent.

        Args:
            state (taos.im.protocol.MarketSimulationStateUpdate): The class representing the latest state of the simulation.
            response (taos.im.protocol.FinanceAgentResponse): The class representing the response of the agent.

        Returns:
            None
        """
        update_text = '-' * 50 + "\n"
        if len(response.instructions) > 0:
            update_text += 'SIMULATION INSTRUCTIONS' + "\n"
            update_text += '-' * 50 + "\n"
            for instruction in response.instructions:
                update_text += f"{instruction}" + "\n"
        else:
            update_text += 'NO SIMULATION INSTRUCTIONS' + "\n"
        update_text += '-' * 50
        bt.logging.info(".\n" + update_text)
    

# ─────────────────────────────────────────────────────────────────────────────
# Unified base class (simulation + exchange)
# ─────────────────────────────────────────────────────────────────────────────

class FinanceAgent(FinanceSimulationAgent):
    """
    Unified base class for agents that must operate in both simulation and
    exchange modes without modification.

    Extends FinanceSimulationAgent so all simulation event hooks, history,
    and debug logging are preserved in simulation mode.  In exchange mode the
    update() path is minimal (no event processing) and self.accounts is
    populated with UnifiedAccount wrappers keyed by netuid.

    Agents should:
      - Call  make_response()         instead of constructing a response directly
      - Loop  `for book_id, book in (state.books or {}).items()` to iterate books
      - Access self.accounts[book_id] via .base_balance.free / .quote_balance.free
      - For empty books in exchange mode, use self._pools[book_id]['price'] as fallback

    Optional agent attributes (read in make_response):
      self.delegate     (str, default '')   — exchange delegate hotkey; auto-resolved if empty
    """

    def __init__(self, uid, config, log_dir=None):
        self._exchange_mode = False
        self._pools: dict = {}
        super().__init__(uid, config, log_dir)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def handle(self, state: MarketSimulationStateUpdate | ExchangeStateUpdate) -> FinanceAgentResponse | ExchangeAgentResponse:
        """
        Override SimulationAgent.handle() to finalize UnifiedAgentResponse into
        the correct Pydantic type before FastAPI serializes it.

        Dispatches to respond_exchange() or respond_simulation() based on the
        state type so that the exchange_mode flag is determined from the call
        itself, not from a shared instance attribute that concurrent requests
        could overwrite.
        """
        self._apply_pending_live_config()
        exchange_mode = isinstance(state, ExchangeStateUpdate)
        if exchange_mode:
            header = (
                "\n" + '-' * 50 + "\n"
                f"VALIDATOR : {state.dendrite.hotkey} | BLOCK : {state.block} | EXCHANGE TIME : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                + '-' * 50
            )
            bt.logging.info("." + header)
        self.update(state)
        if exchange_mode:
            response = self.respond_exchange(state)
        else:
            response = self.respond_simulation(state)
        self.report(state, response)
        if isinstance(response, UnifiedAgentResponse):
            return response.finalize()
        return response

    def report(self, state: MarketSimulationStateUpdate | ExchangeStateUpdate, response: FinanceAgentResponse | ExchangeAgentResponse) -> None:
        exchange_mode = isinstance(state, ExchangeStateUpdate)
        label = 'EXCHANGE' if exchange_mode else 'SIMULATION'
        update_text = '-' * 50 + "\n"
        if len(response.instructions) > 0:
            update_text += f'{label} INSTRUCTIONS' + "\n"
            update_text += '-' * 50 + "\n"
            for instruction in response.instructions:
                update_text += f"{instruction}" + "\n"
        else:
            update_text += f'NO {label} INSTRUCTIONS' + "\n"
        update_text += '-' * 50
        bt.logging.info(".\n" + update_text)

    def respond_simulation(self, state) -> "FinanceAgentResponse":
        """
        Called for MarketSimulationStateUpdate requests.

        Override in subclasses to handle simulation state.  Default delegates
        to respond() for backward compatibility with agents that implement the
        single-method API.
        """
        return self.respond(state)

    def respond_exchange(self, state) -> "ExchangeAgentResponse":
        """
        Called for ExchangeStateUpdate requests.

        Override in subclasses to handle exchange state.  Default delegates to
        respond() for backward compatibility with agents that implement the
        single-method API.
        """
        return self.respond(state)

    def update(self, state) -> None:
        if isinstance(state, ExchangeStateUpdate):
            # Minimal exchange-mode setup — no simulation event processing
            self.simulation_config = cast(MarketSimulationConfig, state.config)
            raw = (state.accounts or {}).get(self.uid, {})
            self.accounts = {bid: UnifiedAccount(a) for bid, a in raw.items()}
            self.events   = list((state.notices or {}).get(self.uid, []))
            self._exchange_mode = True
            # Cache pools so empty-book fallback works even when state.pools is
            # None (pools can be lost during bt.Synapse JSON serialisation).
            # Prefer state.pools; fall back to 'price' embedded in each account.
            if state.pools:
                self._pools = {int(k): v for k, v in state.pools.items()}
            else:
                pools = {}
                for bid, a in self.accounts.items():
                    raw = a._raw if isinstance(a, UnifiedAccount) else a
                    if isinstance(raw, dict):
                        price = raw.get('price', 0.0)
                    else:
                        # LazyAccount: access underlying raw dict before it is consumed
                        inner = getattr(raw, '_raw', None)
                        price = inner.get('price', 0.0) if isinstance(inner, dict) else 0.0
                    pools[int(bid)] = {'price': price}
                self._pools = pools
        else:
            # Full simulation update: event hooks, history, debug logging
            super().update(state)
            # Re-wrap raw Account objects with UnifiedAccount
            self.accounts = {bid: UnifiedAccount(a) for bid, a in self.accounts.items()}
            self._exchange_mode = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def make_response(self, exchange_mode: "bool | None" = None) -> UnifiedAgentResponse:
        """Return a mode-aware response object with the simulation-compatible API.

        Pass exchange_mode explicitly from respond_simulation/respond_exchange to
        avoid reading the shared instance attribute, which concurrent requests
        can overwrite.  Falls back to self._exchange_mode when not provided.
        """
        return UnifiedAgentResponse(
            agent_id=self.uid,
            exchange_mode=self._exchange_mode if exchange_mode is None else exchange_mode,
            delegate=getattr(self, 'delegate', ''),
        )



from taos.im.utils.history import history, batch_history
class StateHistoryManager:
    """
    Manages the state history for market simulations, including reconstruction of L2 states,
    constructing and maintaining histories over multiple state updates, as well as
    saving and loading of history data for later retrieval.
    """

    def __init__(self, history_retention_mins: int, log_dir: str, depth: int = 21, parallel_workers: int = 0, save: bool = False):
        """
        Initialize the StateHistoryManager.

        Args:
            history_retention_mins (int): Retention period for history in minutes.
            log_dir (str): Directory to store history files.
            depth (int): Number of levels of the order book to retain in history.
            parallel_workers (int): Number of parallel processes for reconstruction; if 0, disables parallelization.
            save (bool): If True, persist history to file after each update.
        """
        self.history_retention_mins: int = history_retention_mins  # Retention duration for history data
        self.log_dir: str = log_dir  # Directory path for log and history files
        self.state_file: str = os.path.join(log_dir, "history.mp")  # Path to serialized state file

        self.gap: dict[str, dict[int, bool]] = {}  # Tracks gaps in validator/book histories
        self.depth: int = depth  # Number of book levels to store
        self.parallel_workers: int = parallel_workers  # Number of processes for parallel reconstruction
        self.parallel: bool = parallel_workers > 0  # Whether parallel processing is enabled

        self.should_save: bool = save  # Whether to automatically save history after updates
        self.saving: bool = False  # Flag: True if a save operation is in progress
        self.updating: bool = False  # Flag: True if an update operation is in progress

        self.last_snapshot: dict[str, dict[int, L2Snapshot]] = {}  # Last known snapshot per validator/book
        self.history: dict[str, dict[int, L2History]] = {}  # Full history per validator/book
        self.publish_interval: int = None # Publishing interval

        self.load()  # Attempt to load existing history from disk

    def update(self, state: MarketSimulationStateUpdate) -> None:
        """
        Update the history with a new market simulation state.

        Args:
            state (MarketSimulationStateUpdate): The latest simulation state to process.
        """
        self.updating = True
        self.publish_interval = state.config.publish_interval
        try:
            validator: str = state.dendrite.hotkey  # Validator identifier

            # Ensure internal structures for this validator exist
            if validator not in self.last_snapshot:
                self.last_snapshot[validator] = {}
            if validator not in self.history:
                self.history[validator] = {}
            if validator not in self.gap:
                self.gap[validator] = {}

            # Wait for any ongoing save operation to complete
            while self.saving:
                bt.logging.info("Waiting for history saving to complete...")
                time.sleep(0.5)

            snapshots: dict[int, L2Snapshot | None] = {}
            for book_id, book in state.books.items():
                snapshots[book_id] = self._prepare_snapshot(state, book)

            # If all snapshots were successfully prepared
            if all(snapshots.values()):
                bt.logging.info(f"Updating state history for {validator} at {duration_from_timestamp(state.timestamp)}...")
                start_time = time.time()

                # Parallel or sequential history reconstruction
                if self.parallel:
                    num_processes = min(self.parallel_workers, len(state.books))
                    if len(state.books) % num_processes != 0:
                        raise ValueError(f"`parallel_workers` ({self.parallel_workers}) must divide number of books ({len(state.books)}).")

                    # Divide books into batches for parallel processing
                    batch_size = len(state.books) // num_processes
                    batches = [list(state.books.keys())[i:i + batch_size] for i in range(0, len(state.books), batch_size)]
                    histories = batch_history(
                        {book_id: snapshot.model_dump() for book_id, snapshot in snapshots.items()},
                        {book_id: [event.model_dump() for event in book.events] for book_id, book in state.books.items()},
                        batches,
                        state.config.volumeDecimals
                    )
                else:
                    # Process sequentially
                    histories = {
                        book_id: history(
                            snapshot.model_dump(),
                            [event.model_dump() for event in state.books[book_id].events],
                            state.config.volumeDecimals
                        )
                        for book_id, snapshot in snapshots.items()
                    }

                # Validate and process histories
                processed_histories = {
                    book_id: state.books[book_id].process_history(
                        {t: L2Snapshot.model_validate(snapshot).sort(self.depth) for t, snapshot in hist.items()},
                        {t: TradeInfo.model_validate(trade) for t, trade in trades.items()},
                        state.timestamp,
                        state.config,
                        self.history_retention_mins,
                        self.depth
                    )
                    for book_id, (hist, trades) in histories.items()
                }

                # Update all books in the state
                for book_id, book in state.books.items():
                    history_obj, matched, discrepancies = processed_histories[book_id]
                    self._update_book_history(state, book, history_obj, matched, discrepancies)

                bt.logging.info(f"Updated State History ({time.time() - start_time:.2f}s)")

            if self.should_save:
                # Trigger asynchronous save
                self.save()

        except Exception as ex:
            bt.logging.error(f"Exception processing state update for {validator} at {duration_from_timestamp(state.timestamp)}: {ex}")
        finally:
            self.updating = False

    def update_async(self, state: MarketSimulationStateUpdate) -> None:
        """
        Update the history asynchronously in a separate thread.
        Allows non-blocking updates while other operations continue.

        Args:
            state (MarketSimulationStateUpdate): The latest simulation state to process.
        """
        if not self.updating:
            Thread(target=self.update, args=(state,), daemon=True, name=f'update_history_{state.timestamp}').start()

    def _prepare_snapshot(self, state: MarketSimulationStateUpdate, book: Book) -> L2Snapshot | None:
        """
        Prepare a snapshot for a specific book, handling gaps if necessary.

        Args:
            state (MarketSimulationStateUpdate): The current simulation state.
            book (Book): The book data to process.

        Returns:
            L2Snapshot | None: The prepared snapshot, or None if unavailable.
        """
        book_id: int = book.id
        validator: str = state.dendrite.hotkey
        snapshot: L2Snapshot | None = None

        try:
            # Detect gaps or inconsistencies in history
            has_gap = (
                (book_id not in self.gap[validator] or not self.gap[validator][book_id])
                and (
                    (book_id in self.history[validator] and self.history[validator][book_id].end != state.timestamp - state.config.publish_interval)
                    or (book_id in self.last_snapshot[validator] and self.last_snapshot[validator][book_id].timestamp != state.timestamp - state.config.publish_interval)
                )
            )
            if has_gap:
                # Clear outdated snapshots and handle small/large gaps
                if book_id in self.last_snapshot[validator]:
                    del self.last_snapshot[validator][book_id]
                if book_id in self.history[validator]:
                    if state.timestamp > self.history[validator][book_id].end and state.timestamp < self.history[validator][book_id].end + (self.history_retention_mins * 60_000_000_000) // 10:
                        self.gap[validator][book_id] = True
                        if book_id == 0:
                            bt.logging.warning(
                                f"VALI {validator}: Small gap detected in L2 history from {duration_from_timestamp(self.history[validator][book_id].end)} to {duration_from_timestamp(state.timestamp - state.config.publish_interval)}. Continuing history."
                            )
                    else:
                        self.gap[validator][book_id] = False
                        if book_id == 0:
                            bt.logging.warning(
                                f"VALI {validator}: Large gap detected in L2 history. Resetting history for book {book_id}."
                            )
                        del self.history[validator][book_id]

            # Determine appropriate snapshot to use
            if book_id in self.last_snapshot[validator]:
                if book_id not in self.history[validator]:
                    snapshot = self.last_snapshot[validator][book_id]
                else:
                    if self.history[validator][book_id].end == state.timestamp - state.config.publish_interval:
                        snapshot = list(self.history[validator][book_id].snapshots.values())[-1]
                    else:
                        snapshot = self.last_snapshot[validator][book_id]

            # Update last snapshot for this book
            self.last_snapshot[validator][book_id] = book.snapshot(state.timestamp)

        except Exception as ex:
            bt.logging.error(
                f"VALI {validator} BOOK {book_id}: Exception while processing state at {duration_from_timestamp(state.timestamp)} (T={state.timestamp}): {ex}\n{traceback.format_exc()}"
            )
        finally:
            return snapshot

    def _update_book_history(self, state: MarketSimulationStateUpdate, book: Book, history: L2History, matched: bool, discrepancies: list[str]) -> None:
        """
        Commit an updated book history to the state manager.

        Args:
            state (MarketSimulationStateUpdate): The current simulation state.
            book (Book): The book being updated.
            history (L2History): The reconstructed book history.
            matched (bool): Whether reconstructed snapshot matches published state.
            discrepancies (list[str]): List of discrepancies found during reconstruction.
        """
        book_id: int = book.id
        validator: str = state.dendrite.hotkey

        try:
            if book_id in self.last_snapshot[validator]:
                self.gap[validator][book_id] = False
                if book_id not in self.history[validator]:
                    self.history[validator][book_id] = history
                    if book_id == 0:
                        bt.logging.info(
                            f"VALI {validator}: Initialized new L2 history at {duration_from_timestamp(self.history[validator][book_id].end)} "
                            f"(Available: {duration_from_timestamp(self.history[validator][book_id].start)}-{duration_from_timestamp(self.history[validator][book_id].end)})"
                        )
                else:
                    if self.history[validator][book_id].end == state.timestamp - state.config.publish_interval:
                        self.history[validator][book_id] = self.history[validator][book_id].append(history)
                        if book_id == 0:
                            bt.logging.info(
                                f"VALI {validator}: Appended L2 history at {duration_from_timestamp(state.timestamp)} "
                                f"(Available: {duration_from_timestamp(self.history[validator][book_id].start)}-{duration_from_timestamp(self.history[validator][book_id].end)})"
                            )
                    else:
                        # Recover after history gap
                        self.history[validator][book_id] = self.history[validator][book_id].append(history)
                        if book_id == 0:
                            bt.logging.info(
                                f"VALI {validator}: Recovered after history gap at {duration_from_timestamp(state.timestamp)} "
                                f"(Available: {duration_from_timestamp(self.history[validator][book_id].start)}-{duration_from_timestamp(self.history[validator][book_id].end)})"
                            )

                if not matched:
                    bt.logging.error(
                        f"VALI {validator} BOOK {book_id}: Mismatch between reconstructed and published book state:\n" + "\n".join(discrepancies)
                    )

            # Always update the last snapshot
            self.last_snapshot[validator][book_id] = book.snapshot(state.timestamp)

        except Exception as ex:
            bt.logging.error(
                f"VALI {validator} BOOK {book_id}: Exception while processing state at {duration_from_timestamp(state.timestamp)} (T={state.timestamp}): {ex}\n{traceback.format_exc()}"
            )

    def __getitem__(self, validator: str) -> dict[int, L2History]:
        """
        Access the history for a specific validator.

        Args:
            validator (str): Validator identifier.

        Returns:
            dict[int, L2History]: Mapping of book IDs to their histories.
        """
        return self.history[validator]

    def __contains__(self, validator: str) -> bool:
        """
        Check if a validator has history stored.

        Args:
            validator (str): Validator identifier.

        Returns:
            bool: True if history exists, False otherwise.
        """
        return validator in self.history

    def serialize(self) -> dict[str, Any]:
        """
        Serialize the current state history into a dictionary.

        Returns:
            dict[str, Any]: Serialized representation of the state history.
        """
        return {
            "publish_interval" : self.publish_interval,
            "last_snapshot": {
                validator: {book_id: snapshot.model_dump() for book_id, snapshot in validator_snapshot.items()}
                for validator, validator_snapshot in self.last_snapshot.items()
            },
            "history": {
                validator: {
                    book_id: {
                        'snapshots': {t: snapshot.model_dump() for t, snapshot in history.snapshots.items()},
                        'trades': {t: trade.model_dump() for t, trade in history.trades.items()}
                    }
                    for book_id, history in validator_history.items()
                }
                for validator, validator_history in self.history.items()
            }
        }

    def populate(self, serialized: dict[str, Any]) -> None:
        """
        Populate the state history from a serialized dictionary.

        Args:
            serialized (dict[str, Any]): Serialized history data.
        """
        self.last_snapshot = {
            validator: {book_id: L2Snapshot.model_validate(snapshot) for book_id, snapshot in validator_snapshot.items()}
            for validator, validator_snapshot in serialized["last_snapshot"].items()
        }
        self.history = {
            validator: {
                book_id: L2History(
                    snapshots={t: L2Snapshot.model_validate(snapshot) for t, snapshot in history_data["snapshots"].items()},
                    trades={t: TradeInfo.model_validate(trade) for t, trade in history_data["trades"].items()},
                    retention_mins=self.history_retention_mins,
                    publish_interval=serialized["publish_interval"]
                )
                for book_id, history_data in validator_history.items()
            }
            for validator, validator_history in serialized["history"].items()
        }

    def _save(self) -> None:
        """
        Save the current state history to disk synchronously.
        """
        self.saving = True
        bt.logging.info("Saving history...")
        start_time = time.time()

        # Write serialized data to a temporary file
        with open(self.state_file + ".tmp", 'wb') as file:
            packed_data = msgpack.packb(self.serialize(), use_bin_type=True)
            file.write(packed_data)

        # Replace old state file with new one
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        os.rename(self.state_file + ".tmp", self.state_file)

        bt.logging.info(f"History saved to {self.state_file} ({time.time() - start_time:.2f}s)")
        self.saving = False

    def save(self) -> None:
        """
        Save the state history asynchronously in a separate thread.
        """
        if not self.saving:
            Thread(target=self._save, daemon=True, name='save_history').start()

    def load(self) -> None:
        """
        Load the state history from disk if available, otherwise initialize empty structures.
        """
        if os.path.exists(self.state_file):
            bt.logging.info(f"Loading history from {self.state_file}...")
            with open(self.state_file, 'rb') as file:
                byte_data = file.read()
            state_data = msgpack.unpackb(byte_data, use_list=False, strict_map_key=False)
            self.populate(state_data)
            bt.logging.success("Loaded history! Available data:")
            for validator, validator_history in self.history.items():
                for book_id, book_history in validator_history.items():
                    bt.logging.info(
                        f"VALI {validator} BOOK {book_id}: {duration_from_timestamp(book_history.start)} - {duration_from_timestamp(book_history.end)}"
                    )
        else:
            # No saved state found; start fresh
            bt.logging.info(f"No history file found at {self.state_file}. Initializing empty history.")
            self.last_snapshot = {}
            self.history = {}

# GenTRXAgent extends FinanceSimulationAgent with distributed training.
# Import is deferred to end of file to avoid the circular-import that would
# result from gentrx.py importing FinanceSimulationAgent from this module.
try:
    from taos.im.agents.gentrx import GenTRXAgent  # noqa: F401
except ImportError:
    pass  # GenTRX optional dependencies (torch, boto3, etc.) not installed
