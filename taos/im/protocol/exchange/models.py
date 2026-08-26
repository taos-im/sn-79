# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Classes representing models of objects occurring within intelligent market simulations are defined here.
"""
import numpy as np
from collections.abc import Mapping, Sequence
from xml.etree.ElementTree import Element
from pydantic import Field
from ypyjson import YpyObject
from enum import IntEnum
from itertools import accumulate
from typing import Literal, Any
from taos.common.protocol import BaseModel

class FeeTier(BaseModel):
    """One tier of a volume-tiered fee schedule.

    Attributes:
        volume_required (float): Rolling traded volume required to qualify for this tier.
        maker_fee (float): Maker fee rate charged at this tier.
        taker_fee (float): Taker fee rate charged at this tier.
    """
    volume_required : float
    maker_fee : float
    taker_fee : float

class FeePolicy(BaseModel):
    """The venue's fee schedule as published on the state update.

    Attributes:
        fee_type (str): Which fee scheme is active.
        params (dict): Scheme-specific parameters, exactly as the engine published them.
        tiers (list[FeeTier]): Volume tiers when the scheme is tiered; empty otherwise.
    """
    fee_type : str
    params : dict
    tiers : list[FeeTier]

    @classmethod
    def from_xml(cls, xml : Element):
        """
        Constructs an instance of the class from the XML simulation configuration element.
        """
        if xml:
            fee_policy = FeePolicy(fee_type=xml.attrib['type'], params={k : v for k, v in xml.attrib.items() if k != 'type'}, tiers=[FeeTier(volume_required=0, maker_fee=0.0, taker_fee=0.0 )])
            match fee_policy.fee_type:
                case 'static':
                    fee_policy.tiers = [FeeTier(volume_required=0, maker_fee=xml.attrib['makerFee'], taker_fee=xml.attrib['takerFee'] )]
                case 'tiered':
                    fee_policy.tiers = [FeeTier(volume_required=tier.attrib['volumeRequired'], maker_fee=tier.attrib['makerFee'], taker_fee=tier.attrib['takerFee']) for tier in xml.findall("Tier")]
        else:
            fee_policy = FeePolicy(fee_type=xml.attrib['type'], params={k : v for k, v in xml.attrib.items() if k != 'type'}, tiers=[FeeTier(volume_required=0, maker_fee=0.0, taker_fee=0.0)]  )
            fee_policy.tiers = [FeeTier(volume_required=0, maker_fee=0.0, taker_fee=0.0 )]
        return fee_policy

    def to_prom_info(self) -> dict:
        """
        Creates a dictionary containing the details of the fee policy specification in format suitable for publishing via Prometheus Info metric
        """
        prometheus_info = {}
        prometheus_info['simulation_fee_policy_type'] = self.fee_type
        for name, value in self.params.items():
            prometheus_info[f'simulation_fee_policy_{name}'] = str(value)
        if self.fee_type == 'tiered':
            for i, tier in enumerate(self.tiers):
                prometheus_info[f'simulation_fee_policy_tier_{i}_volume_required'] = f"{tier.volume_required:.2f}"
                prometheus_info[f'simulation_fee_policy_tier_{i}_maker_rate'] = f"{tier.maker_fee * 100:.4f}"
                prometheus_info[f'simulation_fee_policy_tier_{i}_taker_rate'] = f"{tier.taker_fee * 100:.4f}"
        return prometheus_info

class Order(BaseModel):
    """
    Represents an order.

    Attributes:
        type (str): The type of the instruction; fixed to `"o"` (used for parallelized history reconstruction).
        id (int): The ID of the order as assigned by the simulator.
        client_id (int | None): Optional agent-assigned identifier for the order.
        timestamp (int): Simulation timestamp at which the order was placed.
        quantity (float): The size of the order in base currency.
        side (int): The side of the book on which the order was attempted to be placed (`0=BID`, `1=ASK`).
        price (float | None): Price of the order (`None` for market orders).
        leverage (float): Leverage ratio applied to the order. Defaults to 0.0 (unleveraged).
    """
    y : str = "o"
    i : int = Field(alias='id')
    c : int | None = Field(alias='client_id', default=None)
    d : str = Field(alias='delegate')
    t : int = Field(alias='timestamp')
    q : float = Field(alias='quantity')
    s : int = Field(alias='side')
    p : float | None = Field(alias='price')
    l : float = Field(alias="leverage", default=0.0)

    @property
    def type(self) -> str:
        """Readable accessor for wire field ``y``."""
        return self.y

    @property
    def id(self) -> int:
        """Readable accessor for wire field ``i``; ``id`` is its serialized alias."""
        return self.i

    @property
    def client_id(self) -> int | None:
        """Readable accessor for wire field ``c``; ``client_id`` is its serialized alias."""
        return self.c

    @property
    def delegate(self) -> str:
        """Readable accessor for wire field ``d``; ``delegate`` is its serialized alias."""
        return self.d

    @property
    def timestamp(self) -> int:
        """Readable accessor for wire field ``t``; ``timestamp`` is its serialized alias."""
        return self.t

    @property
    def quantity(self) -> float:
        """Readable accessor for wire field ``q``; ``quantity`` is its serialized alias."""
        return self.q

    @property
    def side(self) -> int:
        """Readable accessor for wire field ``s``; ``side`` is its serialized alias."""
        return self.s

    @property
    def price(self) -> float | None:
        """Readable accessor for wire field ``p``; ``price`` is its serialized alias."""
        return self.p
    
    @property
    def leverage(self) -> float:
        """Readable accessor for wire field ``l``; ``leverage`` is its serialized alias."""
        return self.l

    @classmethod
    def from_event(self, event : dict):
        """
        Method to extract model data from simulation event in the format required by the MarketSimulationStateUpdate synapse.
        """
        return Order(order_type="limit" if event['price'] else 'market', id=event['orderId'],client_id=event['clientOrderId'], 
                     delegate=event['delegate'],
                     timestamp=event['timestamp'],
                     quantity=event['volume'], side=event['direction'], price=event['price'], 
                     leverage=event['leverage'])

    @classmethod
    def from_json(self, json : dict):
        """
        Method to extract model data from simulation account representation in the format required by the MarketSimulationStateUpdate synapse.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return Order.model_construct(order_type="limit", id=json['i'], client_id=json['c'], 
                    delegate=json['d'],
                    timestamp=json['t'],
                    quantity=json['q'], side=json['s'], price=json['p'], 
                    leverage=json['l'])

class LevelInfo(BaseModel):
    """
    Represents a level in the order book.

    Attributes:
        price (float): The price level in the order book.
        quantity (float): Total quantity in base currency at this price level.
        orders (list[Order] | None): List of individual orders at this level (if available).
    """
    
    p : float = Field(alias='price')
    q : float = Field(alias='quantity')
    o: list[Order] | None = Field(alias='orders', default=None)

    @property
    def price(self) -> float:
        """Readable accessor for wire field ``p``; ``price`` is its serialized alias."""
        return self.p

    @property
    def quantity(self) -> float:
        """Readable accessor for wire field ``q``; ``quantity`` is its serialized alias."""
        return self.q

    @property
    def orders(self) -> list[Order]:
        """Readable accessor for wire field ``o``; ``orders`` is its serialized alias."""
        return self.o

    @classmethod
    def from_json(self, json : dict):
        """
        Method to transform simulator format model to the format required by the MarketSimulationStateUpdate synapse.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        if not 'o' in json:
            orders = None
        else:
            orders = [Order.model_construct(id=order['i'], timestamp=order['t'],quantity=order['q'],side=order['s'],order_type="limit",price=json['p'],leverage=json['l'] if 'l' in json else 0.0) for order in json['o']]
        return LevelInfo.model_construct(price = json['p'], quantity=json['q'], orders=orders)

class TradeInfo(BaseModel):
    """
    Represents a trade.

    Attributes:
        type (str): The type of instruction; fixed to `t` (used for parallelized history reconstruction).
        id (int): Simulator-assigned ID of the trade.
        side (int): Direction in which the trade was initiated (0 = BUY, 1 = SELL).
        timestamp (int): Simulation timestamp at which the trade occurred.
        quantity (float): Quantity in base currency that was traded.
        price (float): Price at which the trade occurred.
        taker_id (int): ID of the aggressing order.
        taker_agent_id (int): ID of the agent placing the aggressing order.
        taker_fee (float | None): Transaction fee paid by the taker agent.
        maker_id (int): ID of the resting order.
        maker_agent_id (int): ID of the agent placing the resting order.
        maker_fee (float | None): Transaction fee paid by the maker agent.
    """
    y : str = "t"
    i : int = Field(alias='id')
    s : int = Field(alias='side')
    t : int = Field(alias='timestamp')
    q : float = Field(alias='quantity')
    p : float = Field(alias='price')
    Ti : int = Field(alias='taker_id')
    Ta : int = Field(alias='taker_agent_id')
    Tf : float | None = Field(alias='taker_fee', default=None)
    Mi : int = Field(alias='maker_id')
    Ma : int = Field(alias='maker_agent_id')
    Mf : float | None = Field(alias='maker_fee', default=None)

    @property
    def type(self) -> str:
        """Readable accessor for wire field ``y``."""
        return self.y

    @property
    def id(self) -> int:
        """Readable accessor for wire field ``i``; ``id`` is its serialized alias."""
        return self.i

    @property
    def side(self) -> int:
        """Readable accessor for wire field ``s``; ``side`` is its serialized alias."""
        return self.s

    @property
    def timestamp(self) -> int:
        """Readable accessor for wire field ``t``; ``timestamp`` is its serialized alias."""
        return self.t

    @property
    def quantity(self) -> float:
        """Readable accessor for wire field ``q``; ``quantity`` is its serialized alias."""
        return self.q

    @property
    def price(self) -> float:
        """Readable accessor for wire field ``p``; ``price`` is its serialized alias."""
        return self.p

    @property
    def taker_id(self) -> int:
        """Readable accessor for wire field ``Ti``; ``taker_id`` is its serialized alias."""
        return self.Ti

    @property
    def taker_agent_id(self) -> int:
        """Readable accessor for wire field ``Ta``; ``taker_agent_id`` is its serialized alias."""
        return self.Ta

    @property
    def taker_fee(self) -> float | None:
        """Readable accessor for wire field ``Tf``; ``taker_fee`` is its serialized alias."""
        return self.Tf

    @property
    def maker_id(self) -> int:
        """Readable accessor for wire field ``Mi``; ``maker_id`` is its serialized alias."""
        return self.Mi

    @property
    def maker_agent_id(self) -> int:
        """Readable accessor for wire field ``Ma``; ``maker_agent_id`` is its serialized alias."""
        return self.Ma

    @property
    def maker_fee(self) -> float | None:
        """Readable accessor for wire field ``Mf``; ``maker_fee`` is its serialized alias."""
        return self.Mf

    @classmethod
    def from_event(self, event : dict):
        """
        Method to extract model data from simulation event in the format required by the MarketSimulationStateUpdate synapse.
        """
        return TradeInfo(id=event['tradeId'],timestamp=event['timestamp'],quantity=event['volume'],side=event['direction'],price=event['price'],
                         taker_agent_id=event['aggressingAgentId'], taker_id=event['aggressingOrderId'], maker_agent_id=event['restingAgentId'], maker_id=event['restingOrderId'],
                         maker_fee=event['fees']['maker'], taker_fee=event['fees']['taker'])

    @classmethod
    def from_json(self, json : dict):
        """
        Method to extract model data from simulation event in the format required by the MarketSimulationStateUpdate synapse.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return TradeInfo.model_construct(id=json['i'],timestamp=json['t'],quantity=json['q'],side=json['s'],price=json['p'],
                         taker_agent_id=json['Ta'], taker_id=json['Ti'], maker_agent_id=json['Ma'], maker_id=json['Mi'],
                         maker_fee=json['Mf'], taker_fee=json['Tf'])

class Cancellation(BaseModel):
    """
    Represents an order cancellation.

    Attributes:
        type (str): The type of instruction; fixed to `c` (used for parallelized history reconstruction).
        orderId (int): ID of the cancelled order.
        timestamp (int | None): Simulation timestamp at which the cancellation occurred.
        price (float | None): Price of the order that was cancelled.
        quantity (float | None): Quantity cancelled (None if the entire order was cancelled).
    """
    y : str = "c"
    i: int = Field(alias="orderId")
    t: int | None = Field(alias='timestamp', default=None)
    p: float | None = Field(alias="price", default=None)
    q: float | None = Field(alias="quantity")

    @property
    def type(self) -> str:
        """Readable accessor for wire field ``y``."""
        return self.y

    @property
    def orderId(self) -> int:
        """Readable accessor for wire field ``i``; ``orderId`` is its serialized alias."""
        return self.i

    @property
    def timestamp(self) -> int:
        """Readable accessor for wire field ``t``; ``timestamp`` is its serialized alias."""
        return self.t

    @property
    def price(self) -> float:
        """Readable accessor for wire field ``p``; ``price`` is its serialized alias."""
        return self.p

    @property
    def quantity(self) -> float | None:
        """Readable accessor for wire field ``q``; ``quantity`` is its serialized alias."""
        return self.q

    @classmethod
    def from_event(self, event : dict):
        """
        Method to extract model data from simulation event in the format required by the MarketSimulationStateUpdate synapse.
        """
        return Cancellation(orderId=event['orderId'], timestamp=event['timestamp'], price=event['price'], quantity=event['volume'])

    @classmethod
    def from_json(self, json : dict):
        """
        Method to extract model data from simulation event in the format required by the MarketSimulationStateUpdate synapse.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return Cancellation.model_construct(orderId=json['i'], timestamp=json['t'], price=json['p'], quantity=json['q'])

class History:
    """A rolling window of events on one book.

    Attributes:
        start (int): Timestamp of the oldest retained event.
        end (int): Timestamp of the newest event.
        retention_mins (int | None): Retention horizon in minutes; None keeps everything.
    """
    start : int
    end : int
    retention_mins : int | None

    def is_full(self) -> bool:
        """
        Check whether the history covers the full retention window.

        Returns:
            bool: True if the history is full (matches retention window), False otherwise.
        """
        if self.retention_mins:
            return self.start == self.end - self.retention_mins * 60_000_000_000
        return False
    
    def bucket(self, series: dict[int, Any], interval: float) -> dict[int, list[Any]]:
        """
        Buckets a time series into intervals based on timestamp.

        Args:
            series (dict[int, Any]): Time series mapping timestamps to values.
            interval (float): Bucket size in seconds.

        Returns:
            dict[int, list[Any]]: Buckets indexed by upper-bound timestamps.
        """
        interval_ns = int(interval * 1_000_000_000)
        bucketed: dict[int, list[Any]] = {}

        for timestamp, value in series.items():
            if timestamp < self.start or timestamp > self.end:
                continue  # ignore out-of-range timestamps

            # Compute upper bound of bucket interval
            bucket_index = ((timestamp - self.start) // interval_ns) + 1
            bucket_ts = self.start + bucket_index * interval_ns

            if bucket_ts not in bucketed:
                bucketed[bucket_ts] = []

            bucketed[bucket_ts].append(value)

        return bucketed

    def sample(
        self,
        series: dict[int, float],
        interval: float,
        method: Literal['open', 'high', 'low', 'close', 'ohlc'] = 'close'
    ) -> dict[int, Any]:
        """
        Sample a time series at regular intervals.

        Args:
            series (dict[int, float]): Original time series (timestamp → value).
            interval (float): Interval between samples in seconds.
            method (str): Sampling method; one of 'open', 'high', 'low', 'close', 'ohlc'.

        Returns:
            dict[int, float | dict]: Sampled series with requested method.
        """
        buckets = self.bucket(series, interval)
        sampled: dict[int, Any] = {}
        last_val = None

        if method == 'ohlc':
            for ts, bucket in buckets.items():
                if bucket:
                    open_ = last_val if last_val is not None else bucket[0]
                    high = max(bucket + [open_])
                    low = min(bucket + [open_])
                    close = bucket[-1]
                    sampled[ts] = {'open': open_, 'high': high, 'low': low, 'close': close}
                    last_val = close
                elif last_val is not None:
                    sampled[ts] = {'open': last_val, 'high': last_val, 'low': last_val, 'close': last_val}
                else:
                    sampled[ts] = None
        else:
            pick_fn = {
                'open': lambda b: b[0],
                'high': max,
                'low': min,
                'close': lambda b: b[-1]
            }[method]

            for ts, bucket in buckets.items():
                if bucket:
                    sampled[ts] = pick_fn(bucket)
                    last_val = sampled[ts]
                elif last_val is not None:
                    sampled[ts] = last_val
                else:
                    sampled[ts] = None

        return sampled
                
from typing import Union, Optional
from itertools import accumulate
import numpy as np

class EventHistory(History):
    """
    EventHistory is a specialized history tracker for market events, including:
    - Trades
    - Orders
    - Cancellations

    It allows filtering and analysis of these events for use in modeling,
    feature extraction, and simulation.
    """

    events: dict[int, Union[Order, TradeInfo, Cancellation]]

    def __init__(
        self,
        start: int,
        end: int,
        events: list[Union[Order, TradeInfo, Cancellation]],
        publish_interval: int,
        retention_mins: Optional[int] = None
    ):
        """
        Initializes the EventHistory object.

        Args:
            start (int): Start timestamp in nanoseconds.
            end (int): End timestamp in nanoseconds.
            events (list[Order | TradeInfo | Cancellation]): Initial market events.
            publish_interval (int): Interval at which states are published.
            retention_mins (int | None): Optional retention window in minutes.
        """
        self.events = {e.timestamp: e for e in events}
        self.start = start
        self.end = end
        self.retention_mins = retention_mins
        self.publish_interval = publish_interval

    @property
    def trades(self) -> dict[int, TradeInfo]:
        """Returns all trades indexed by timestamp."""
        return {ts: t for ts, t in self.events.items() if t.type == 't'}

    @property
    def orders(self) -> dict[int, Order]:
        """Returns all orders indexed by timestamp."""
        return {ts: o for ts, o in self.events.items() if o.type == 'o'}

    @property
    def cancellations(self) -> dict[int, Cancellation]:
        """Returns all cancellations indexed by timestamp."""
        return {ts: c for ts, c in self.events.items() if c.type == 'c'}

    @property
    def last_trade(self) -> TradeInfo:
        """Returns the most recent trade."""
        return self.trades[max(self.trades)]

    @property
    def trade_prices(self) -> dict[int, float]:
        """Returns trade prices indexed by timestamp."""
        return {ts: t.price for ts, t in self.trades.items()}

    @property
    def OHLC(self) -> Optional[dict[str, float]]:
        """
        Computes OHLC (Open, High, Low, Close) prices from trade data.

        Returns:
            dict[str, float] | None: OHLC structure or None if no trades.
        """
        trade_prices = self.trade_prices
        if trade_prices:
            values = list(trade_prices.values())
            return {
                "open": values[0],
                "high": max(values),
                "low": min(values),
                "close": values[-1],
            }
        return None

    @property
    def traded_volume(self) -> float:
        """
        Computes the total traded volume (price * quantity).

        Returns:
            float: Total traded value.
        """
        return sum(t.quantity * t.price for t in self.trades.values())

    @property
    def traded_volumes(self) -> dict[int, float]:
        """Returns traded volume per timestamp."""
        return {ts: t.quantity * t.price for ts, t in self.trades.items()}

    @property
    def trade_imbalance(self) -> float:
        """
        Computes net trade imbalance (BUY - SELL quantity).

        Returns:
            float: Net trade imbalance.
        """
        return (
            sum(t.quantity for t in self.trades.values() if t.side == OrderDirection.BUY)
            - sum(t.quantity for t in self.trades.values() if t.side == OrderDirection.SELL)
        )

    @property
    def trade_imbalances(self) -> dict[int, float]:
        """
        Returns cumulative trade imbalance over time.

        Returns:
            dict[int, float]: Time-indexed cumulative trade imbalance.
        """
        return dict(zip(
            self.trades.keys(),
            accumulate(
                t.quantity if t.side == OrderDirection.BUY else -t.quantity
                for t in self.trades.values()
            )
        ))

    @property
    def order_volume(self) -> float:
        """
        Computes total order volume.

        Returns:
            float: Total order volume.
        """
        return sum(o.quantity for o in self.orders.values())

    @property
    def order_volumes(self) -> dict[int, float]:
        """Returns order volume per timestamp."""
        return {ts: o.quantity for ts, o in self.orders.items()}

    @property
    def order_imbalance(self) -> float:
        """
        Computes net order imbalance (BUY - SELL quantity).

        Returns:
            float: Net order imbalance.
        """
        return (
            sum(o.quantity for o in self.orders.values() if o.side == OrderDirection.BUY)
            - sum(o.quantity for o in self.orders.values() if o.side == OrderDirection.SELL)
        )

    @property
    def order_imbalances(self) -> dict[int, float]:
        """
        Returns cumulative order imbalance over time.

        Returns:
            dict[int, float]: Time-indexed cumulative order imbalance.
        """
        return dict(zip(
            self.orders.keys(),
            accumulate(
                o.quantity if o.side == OrderDirection.BUY else -o.quantity
                for o in self.orders.values()
            )
        ))

    def append(self, new_history: 'EventHistory') -> 'EventHistory':
        """
        Efficiently appends a new EventHistory to this instance and applies retention logic.

        Args:
            new_history (EventHistory): New event history to append.

        Returns:
            EventHistory: Self, with updated events and time range.
        """
        # Fast in-place update of event dict (assumes timestamps are unique)
        self.events.update(new_history.events)
        self.end = new_history.end

        # Apply retention window if specified
        if self.retention_mins is not None:
            retention_threshold = self.end - self.retention_mins * 60_000_000_000
            self.events = {ts: event for ts, event in self.events.items() if ts >= retention_threshold}
            self.start = max(self.start, retention_threshold)

        return self

    def trade_price(self, sampling_secs: Optional[float] = None) -> dict[int, float]:
        """
        Returns sampled or raw trade price series.

        Args:
            sampling_secs (float | None): Optional sampling interval in seconds.

        Returns:
            dict[int, float]: Time-series of trade prices.
        """
        trades = {time: trade.price for time, trade in self.trades.items()}
        return self.sample(trades, sampling_secs) if sampling_secs else trades

    def ohlc(self, interval: float) -> dict[int, dict[str, float]]:
        """
        Computes OHLC over sampled intervals.

        Args:
            interval (float): Sampling interval in seconds.

        Returns:
            dict[int, dict[str, float]]: OHLC per bucket.
        """
        return self.sample(self.trade_price(), interval, 'ohlc')

    def mean_trade_price(self, interval: float) -> dict[int, Optional[float]]:
        """
        Computes mean trade price per time bucket.

        Args:
            interval (float): Sampling interval in seconds.

        Returns:
            dict[int, float | None]: Time-indexed average trade price.
        """
        sampled: dict[int, Optional[float]] = {}
        last_val: Optional[float] = None

        for ts, prices in self.bucket(self.trade_price(), interval).items():
            if prices:
                sampled[ts] = float(np.mean(prices))
                last_val = prices[-1]
            elif last_val is not None:
                sampled[ts] = last_val
            else:
                sampled[ts] = None

        return sampled

class Book(BaseModel):
    """
    Represents an order book at a specific point in time, including events
    (orders, trades, cancellations) that have occurred since the last update.

    Attributes:
        id (int): Internal book identifier.
        bids (list[LevelInfo]): List of LevelInfo objects representing bid levels.
        asks (list[LevelInfo]): List of LevelInfo objects representing ask levels.
        events (list[Order | TradeInfo | Cancellation] | None): List of events applied to the book 
            since the last snapshot.
    """

    i: int = Field(alias="id")
    r: float | None = Field(alias="MTR", default=None)
    b: list[LevelInfo] = Field(alias="bids")
    a: list[LevelInfo] = Field(alias="asks")
    e: list[Order | TradeInfo | Cancellation] | None = Field(alias="events")

    @property
    def id(self) -> int:
        """
        Get the ID of the order book.

        Returns:
            int: The book's unique identifier.
        """
        return self.i

    @property
    def MTR(self) -> float:
        """
        Get the current maker-taker ratio for the order book.

        Returns:
            int: The book's unique identifier.
        """
        return self.r

    @property
    def bids(self) -> list[LevelInfo]:
        """
        Get the list of bid levels.

        Returns:
            list[LevelInfo]: Bid levels in descending price order.
        """
        return self.b

    @property
    def asks(self) -> list[LevelInfo]:
        """
        Get the list of ask levels.

        Returns:
            list[LevelInfo]: Ask levels in ascending price order.
        """
        return self.a

    @property
    def events(self) -> list[Order | TradeInfo | Cancellation] | None:
        """
        Get the list of recent events applied to the book.

        Returns:
            list[Order | TradeInfo | Cancellation] | None: List of events or None.
        """
        return self.e
    
    @property
    def trades(self) -> dict[int, TradeInfo]:
        """Trades in this book's event window, keyed by timestamp."""
        return {t.timestamp : t for t in self.events if t.type == 't'}
    
    @property
    def orders(self) -> dict[int, Order]:
        """Order placements in this book's event window, keyed by timestamp."""
        return {o.timestamp : o for o in self.events if o.type == 'o'}
    
    @property
    def cancellations(self) -> dict[int, Cancellation]:
        """Cancellations in this book's event window, keyed by timestamp."""
        return {c.timestamp : c for c in self.events if c.type == 'c'}
    
    @property
    def trade_prices(self) -> dict[int, float]:
        """Trade prices keyed by timestamp."""
        return {ts : t.price for ts, t in self.trades.items()}
    
    @property
    def last_trade(self) -> TradeInfo:
        """The most recent trade in the window."""
        return self.trades[max(self.trades)]
    
    @property
    def OHLC(self) -> dict:       
        """Open/high/low/close of the window's trade prices; empty dict when no trades exist."""
        trade_prices = self.trade_prices 
        if len(trade_prices) > 0:
            return {
                "open" : list(trade_prices.values())[0],
                "high" : max(trade_prices.values()),
                "low" : min(trade_prices.values()),
                "close" : list(trade_prices.values())[-1],
            }
        else:
            return None
        
    @property
    def traded_volume(self) -> float:       
        """Total quote-denominated volume traded in the window (sum of quantity x price)."""
        return sum([t.quantity * t.price for t in self.trades.values()])
    
    @property
    def traded_volumes(self) -> dict:
        """Quote-denominated volume per trade, keyed by timestamp."""
        return {ts: t.quantity * t.price for ts,t in self.trades.items()}
        
    @property
    def trade_imbalance(self) -> float:       
        """Net signed base quantity traded over the window: buys minus sells."""
        return sum([t.quantity for t in self.trades.values() if t.side == OrderDirection.BUY]) - sum([t.quantity for t in self.trades.values() if t.side == OrderDirection.SELL])
    
    @property 
    def trade_imbalances(self) -> dict[int,float]:        
        """Running cumulative trade imbalance, keyed by trade timestamp."""
        return dict(zip(
            self.trades.keys(),
            accumulate(t.quantity if t.side == OrderDirection.BUY else -t.quantity for t in self.trades.values())
        ))
    
    @property
    def order_volume(self) -> float:       
        """Total base quantity across order placements in the window."""
        return sum([o.quantity for o in self.orders.values()])

    @property 
    def order_volumes(self) -> dict[int,float]:
        """Placed order quantity per placement, keyed by timestamp."""
        return {ts : o.quantity for ts, o in self.orders.items()}
    
    @property
    def order_imbalance(self) -> float:       
        """Net signed placed quantity over the window: buy orders minus sell orders."""
        return sum([o.quantity for o in self.orders.values() if o.side == OrderDirection.BUY]) - sum([o.quantity for o in self.orders.values() if o.side == OrderDirection.SELL])
   
    # THIS IS NOT NEEDED MOST LIKELY 
    @property 
    def order_imbalances(self) -> dict[int,float]:
        """Running cumulative order imbalance, keyed by placement timestamp."""
        return dict(zip(
            self.orders.keys(),
            accumulate(o.quantity if o.side == OrderDirection.BUY else -o.quantity for o in self.orders.values())
        ))
        
    @classmethod
    def from_json(cls, json: dict, depth : int = 21) -> 'Book':
        """
        Convert a JSON object from the simulator format into a Book instance.

        Args:
            json (dict): JSON dictionary with book details.
            depth (int): Number of book levels to retain in the bids and asks arrays.

        Returns:
            Book: A new Book instance populated with bids, asks, and events.
        """
        id = json['i']
        bids = []
        asks = []
        if json['b']:
            bids = [LevelInfo.from_json(bid) for bid in json['b']][:depth]
        if json['a']:
            asks = [LevelInfo.from_json(ask) for ask in json['a']][:depth]

        events = []
        if json['e']:
            # Parse events: orders, trades, cancellations
            events = [
                Order.from_json(event) if event['y'] == 'o' else
                TradeInfo.from_json(event) if event['y'] == 't' else
                Cancellation.from_json(event) if event['y'] == 'c' else
                None
                for event in json['e']
            ]

        return cls.model_construct(id=id, bids=bids, asks=asks, events=events)
    
    @classmethod
    def from_ypy(cls, json: YpyObject, depth : int = 21) -> 'Book':
        """Build a Book from a ypy shared object, keeping at most ``depth`` levels per side.

        Args:
            json (YpyObject): The shared-document book node.
            depth (int): Maximum levels per side to retain.

        Returns:
            Book: The parsed book.
        """
        book_id = json['bookId']
        bids = []
        for i, lvl in enumerate(json['bid']):
            if i >= 21:
                break
            bids.append(LevelInfo.model_construct(
                p=lvl['price'],
                q=lvl['volume'],
                o=[Order.model_construct(
                        id=o['orderId'],
                        timestamp=o['timestamp'],
                        quantity=o['volume'],
                        side=o['direction'],
                        order_type="limit",
                        price=lvl['price']
                    ) for o in lvl['orders']] if i < 5 else None
            ))
        asks = []
        for i, lvl in enumerate(json['ask']):
            if i >= 21:
                break
            asks.append(LevelInfo.model_construct(
                p=lvl['price'],
                q=lvl['volume'],
                o=[Order.model_construct(
                        id=o['orderId'],
                        timestamp=o['timestamp'],
                        quantity=o['volume'],
                        side=o['direction'],
                        order_type="limit",
                        price=lvl['price']
                    ) for o in lvl['orders']] if i < 5 else None
            ))
        events = []
        for ev in json['record']:
            ev_type = ev['event']
            if ev_type == 'place':
                events.append(Order.from_event(ev))
            elif ev_type == 'trade':
                events.append(TradeInfo.from_event(ev))
            elif ev_type == 'cancel':
                events.append(Cancellation.from_event(ev))

        return Book.model_construct(
            i=book_id,
            b=bids,
            a=asks,
            e=events if events else None
        )

class Balance(BaseModel):
    """
    Represents an account balance for a specific currency.

    Attributes:
        currency (str): String identifier for the currency (e.g., "USD", "BTC").
        total (float): Total currency balance in the account.
        free (float): Free currency balance available for order placement.
        reserved (float): Reserved currency balance tied up in resting orders.
        initial (float | None): Initial balance for the currency at the start of the simulation or session.
    """
    c : str = Field(alias="currency")
    t : float = Field(alias="total")
    f : float = Field(alias="free")
    r : float = Field(alias="reserved")
    i : float = Field(alias="initial", default=None)

    @property
    def currency(self) -> str:
        """Readable accessor for wire field ``c``; ``currency`` is its serialized alias."""
        return self.c

    @property
    def total(self) -> float:
        """Readable accessor for wire field ``t``; ``total`` is its serialized alias."""
        return self.t

    @property
    def free(self) -> float:
        """Readable accessor for wire field ``f``; ``free`` is its serialized alias."""
        return self.f

    @property
    def reserved(self) -> float:
        """Readable accessor for wire field ``r``; ``reserved`` is its serialized alias."""
        return self.r

    @property
    def initial(self) -> float:
        """Readable accessor for wire field ``i``; ``initial`` is its serialized alias."""
        return self.i

    @classmethod
    def from_json(self, currency : str, json : dict):
        """
        Method to transform simulator format model to the format required by the MarketSimulationStateUpdate synapse.

        Args:
            currency: The order's quantity currency.
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return Balance.model_construct(currency=currency,total=json['t'],free=json['f'],reserved=json['r'], initial=json['i'])

class Fees(BaseModel):
    """
    Represents account fees for a specific agent and book.

    Attributes:
        volume_traded (float): Total volume traded in the aggregation period for tiered fee assignment.
        maker_fee_rate (float): The current maker fee rate for the agent.
        taker_fee_rate (float): The current taker fee rate for the agent.
    """
    v : float | None = Field(alias="volume_traded", default=None)
    m : float = Field(alias="maker_fee_rate")
    t : float = Field(alias="taker_fee_rate")

    @property
    def volume_traded(self) -> float | None:
        """Readable accessor for wire field ``v``; ``volume_traded`` is its serialized alias."""
        return self.v

    @property
    def maker_fee_rate(self) -> float:
        """Readable accessor for wire field ``m``; ``maker_fee_rate`` is its serialized alias."""
        return self.m

    @property
    def taker_fee_rate(self) -> float:
        """Readable accessor for wire field ``t``; ``taker_fee_rate`` is its serialized alias."""
        return self.t

    @classmethod
    def from_json(self, json : dict):
        """
        Method to transform simulator format model to the format required by the MarketSimulationStateUpdate synapse.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return Fees.model_construct(volume_traded=json['v'],maker_fee_rate=json['m'],taker_fee_rate=json['t'])
    
class OrderCurrency(IntEnum):
    """
    Enum to represent the currency in which the quantity of an order is specified.

    The venue's own names are ALPHA and TAO; BASE and QUOTE are the simulation's names for the same two
    currencies and are accepted as aliases, so one agent can use either spelling in either mode.

    Attributes:
        BASE (int): Quantity is specified in the base currency, i.e. subnet alpha. Alias: ALPHA.
        QUOTE (int): Quantity is specified in the quote currency, i.e. TAO. Alias: TAO.
        ALPHA (int): Alias of BASE.
        TAO (int): Alias of QUOTE.
    """
    # Both spellings resolve in both trees and modes; the wire values are 0 (BASE/ALPHA) and 1 (QUOTE/TAO).
    # IntEnum makes the later name an alias of the earlier one when the values match, so
    # `OrderCurrency.ALPHA is OrderCurrency.BASE` holds and either name is accepted everywhere.
    BASE=0
    QUOTE=1
    ALPHA=0
    TAO=1
    
class Loan(BaseModel):
    """
    Represents a loan associated with an open position for the agent.

    Attributes:
        order_id (int): ID of the order associated with the loan.
        amount (float): Total loan amount.
        currency (OrderCurrency): Currency in which the loan is denominated.
        base_collateral (float): Amount of base currency collateral posted for the loan.
        quote_collateral (float): Amount of quote currency collateral posted for the loan.
    """
    i : int = Field(alias="order_id")
    a : float = Field(alias="amount")
    c : OrderCurrency = Field(alias="currency")    
    bc : float = Field(alias="base_collateral")    
    qc : float = Field(alias="quote_collateral")

    @property
    def order_id(self) -> int:
        """Readable accessor for wire field ``i``; ``order_id`` is its serialized alias."""
        return self.i

    @property
    def amount(self) -> float:
        """Readable accessor for wire field ``a``; ``amount`` is its serialized alias."""
        return self.a

    @property
    def currency(self) -> OrderCurrency:
        """Readable accessor for wire field ``c``; ``currency`` is its serialized alias."""
        return self.c

    @property
    def base_collateral(self) -> float:
        """Readable accessor for wire field ``bc``; ``base_collateral`` is its serialized alias."""
        return self.bc

    @property
    def quote_collateral(self) -> float:
        """Readable accessor for wire field ``qc``; ``quote_collateral`` is its serialized alias."""
        return self.qc

    @classmethod
    def from_json(self, json : dict):
        """
        Method to transform simulator format model to the format required by the MarketSimulationStateUpdate synapse.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return Loan.model_construct(order_id=json['i'],amount=json['a'],currency=OrderCurrency(json['c']),base_collateral=json['bc'],quote_collateral=json['qc'])
    
    def __str__(self):
        return f"{self.amount} {self.currency.name} [COLLAT : {self.base_collateral} BASE | {self.quote_collateral} QUOTE]"

class Account(BaseModel):
    """
    Represents an agent's trading account.

    Attributes:
        agent_id (int): The agent ID which owns the account.
        book_id (int): ID of the book on which the account is able to trade.
        base_balance (Balance): Balance object for the base currency.
        quote_balance (Balance): Balance object for the quote currency.
        base_loan (float): Amount of base currency currently borrowed.
        quote_loan (float): Amount of quote currency currently borrowed.
        base_collateral (float): Amount of base currency posted as collateral.
        quote_collateral (float): Amount of quote currency posted as collateral.
        orders (list[Order]): List of the current open orders associated to the agent.
        loans (dict[int, Loan]): Mapping from order ID to Loan objects representing open loans.
        fees (Fees | None): The current fee structure for the account.
        traded_volume (float | None): Total volume traded by the account. Defaults to None.
    """
    i : int = Field(alias="agent_id")
    b : int = Field(alias="book_id")
    bb : Balance = Field(alias="base_balance")
    qb : Balance = Field(alias="quote_balance")
    bl : float = Field(alias="base_loan", default=0.0)
    ql : float = Field(alias="quote_loan", default=0.0)
    bc : float = Field(alias="base_collateral", default=0.0)
    qc : float = Field(alias="quote_collateral", default=0.0)    
    o : list[Order] = Field(alias="orders", default=[])
    l : dict[int, Loan] = Field(alias="loans", default={})
    f : Fees | None = Field(alias="fees")
    v : float | None = Field(alias="traded_volume", default=None)
    # EXCHANGE-MODE TWIN of Account.ds in protocol/models.py. This is the model the exchange path
    # actually delivers, so omitting it here would publish the breakdown in simulation mode only --
    # exactly the surface-disagreement class this field exists to end. Same semantics: alpha per
    # delegate hotkey for THIS account on THIS book; {} means NOT REPORTED, not "no stake"; and
    # sum(ds) == base_balance.free is NOT an invariant (chain snapshot vs engine accounting).
    ds : dict[str, float] = Field(alias="delegate_stakes", default_factory=dict)

    @property
    def delegate_stakes(self) -> dict[str, float]:
        """This account's alpha on this book, per delegate hotkey. Empty means NOT REPORTED."""
        return self.ds

    @property
    def sellable_alpha(self) -> float:
        """The most alpha ONE order can sell here: the largest single delegate's stake.

        0.0 when the breakdown was not reported, which callers must read as "unknown" and fall back to
        base_balance.free -- the pre-existing contract -- rather than as "nothing to sell".
        """
        return max(self.ds.values()) if self.ds else 0.0

    @property
    def agent_id(self) -> int:
        """Readable accessor for wire field ``i``; ``agent_id`` is its serialized alias."""
        return self.i

    @property
    def book_id(self) -> int:
        """Readable accessor for wire field ``b``; ``book_id`` is its serialized alias."""
        return self.b

    @property
    def base_balance(self) -> Balance:
        """Readable accessor for wire field ``bb``; ``base_balance`` is its serialized alias."""
        return self.bb

    @property
    def quote_balance(self) -> Balance:
        """Readable accessor for wire field ``qb``; ``quote_balance`` is its serialized alias."""
        return self.qb

    @property
    def base_loan(self) -> float:
        """Readable accessor for wire field ``bl``; ``base_loan`` is its serialized alias."""
        return self.bl

    @property
    def quote_loan(self) -> float:
        """Readable accessor for wire field ``ql``; ``quote_loan`` is its serialized alias."""
        return self.ql

    @property
    def base_collateral(self) -> float:
        """Readable accessor for wire field ``bc``; ``base_collateral`` is its serialized alias."""
        return self.bc

    @property
    def quote_collateral(self) -> float:
        """Readable accessor for wire field ``qc``; ``quote_collateral`` is its serialized alias."""
        return self.qc

    @property
    def orders(self) -> list[Order]:
        """Readable accessor for wire field ``o``; ``orders`` is its serialized alias."""
        return self.o

    @property
    def loans(self) -> dict[int, Loan]:
        """Readable accessor for wire field ``l``; ``loans`` is its serialized alias."""
        return self.l

    @property
    def fees(self) -> Fees | None:
        """Readable accessor for wire field ``f``; ``fees`` is its serialized alias."""
        return self.f

    @property
    def traded_volume(self) -> float | None:
        """Readable accessor for wire field ``v``; ``traded_volume`` is its serialized alias."""
        return self.v
    
    @property
    def own_quote(self) -> float:
        """Quote the account actually owns: total minus loan plus collateral."""
        return self.quote_balance.total - self.quote_loan + self.quote_collateral
    
    @property
    def own_base(self) -> float:
        """Base the account actually owns: total minus loan plus collateral."""
        return self.base_balance.total - self.base_loan + self.base_collateral
    
    @classmethod
    def from_json(cls, json: dict) -> "Account":
        """
        Construct an Account from simulator JSON into an Account model,
        using model_construct and manually populating nested classes.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return cls.model_construct(
            i=json["i"],
            b=json["b"],
            bb=Balance.model_construct(**json["bb"]),
            qb=Balance.model_construct(**json["qb"]),
            bl=json.get("bl", 0.0),
            ql=json.get("ql", 0.0),
            bc=json.get("bc", 0.0),
            qc=json.get("qc", 0.0),
            o=[Order.from_json(o) for o in json.get("o", [])],
            l={int(k): Loan.from_json(v) for k, v in json.get("l", {}).items()},
            f=Fees.model_construct(**json["f"]) if json.get("f") else None,
            v=json.get("v"),
        )

class OrderDirection(IntEnum):
    """
    Enum to represent order direction.

    Attributes:
        BUY (int): Associated with an order placed in the BUY direction.
        SELL (int): Associated with an order placed in the SELL direction.
    """
    BUY=0
    SELL=1

class STP(IntEnum):
    """
    Enum to represent self-trade prevention options.

    Attributes:
        NO_STP (int): No self-trade prevention.
        CANCEL_OLDEST (int): If self-trade would occur when placing an order, cancel the resting order.
        CANCEL_NEWEST (int): If self-trade would occur when placing an order, cancel the aggressive order.
        CANCEL_BOTH (int): If self-trade would occur when placing an order, cancel both orders.
        DECREASE_CANCEL (int): If self-trade would occur when placing an order, cancel the quantity of the smaller order from the larger.
    """
    NO_STP=0
    CANCEL_OLDEST=1
    CANCEL_NEWEST=2
    CANCEL_BOTH=3
    DECREASE_CANCEL=4

class TimeInForce(IntEnum):
    """
    Enum to represent order time-in-force options.

    Attributes:
        GTC (int): Order remains on the book until cancelled by the agent, or executed in a trade.
        GTT (int): Order remains on the book until specified expiry period elapses, unless traded or cancelled before expiry.
        IOC (int): Any part of the order which is not immediately traded will be cancelled.
        FOK (int): If the order will not be executed in its entirety immediately upon receipt by the simulator, the order will be rejected.
    """
    GTC=0
    GTT=1
    IOC=2
    FOK=3
    
class LoanSettlementOption(IntEnum):
    """
    Enum to represent options for repayment of margin loans when submitting an order.

    Attributes:
        NONE (int): Do not settle outstanding margin loans with proceeds from this order.
        FIFO (int): Settle outstanding margin loans in a FIFO (First-In-First-Out) manner
                    using proceeds from this order.
    """
    NONE = -2
    FIFO = -1
    
    @classmethod
    def from_string(cls, name):
        """Parse a LoanSettlementOption from its name.

        Args:
            name: The option's name, e.g. ``'NONE'``.

        Returns:
            LoanSettlementOption: The matching option.
        """
        match name:
            case 'NONE':
                return LoanSettlementOption.NONE
            case 'FIFO':
                return LoanSettlementOption.FIFO
            case _:
                try:
                    order_id = int(name)
                    return order_id
                except Exception:
                    return None

class LazyLevel(Sequence):
    """
    Lazily-parsed order book level.

    This class defers construction of the `LevelInfo` and `Order` objects until their data is accessed.

    Attributes:
        _raw (dict): Raw data for the level.
        _parsed (LevelInfo | None): Parsed LevelInfo object once loaded.
    """
    __slots__ = ("_raw", "_parsed")

    def __init__(self, raw_level):
        self._raw = raw_level
        self._parsed = None

    def _load(self):
        if self._parsed is None:
            orders = [Order.model_construct(**o) for o in self._raw.get("o", [])] if self._raw.get("o") else []
            self._parsed = LevelInfo.model_construct(
                p=self._raw.get("p"),
                q=self._raw.get("q"),
                o=orders
            )
            self._raw = None

    def __getattr__(self, name):
        self._load()
        return getattr(self._parsed, name)

    def __getitem__(self, index):
        self._load()
        return self._parsed[index]

    def __len__(self):
        self._load()
        return len(self._parsed)

    def parse(self) -> LevelInfo:
        """Return fully parsed LevelInfo object."""
        self._load()
        return self._parsed


class LazyLevels(Sequence):
    """
    Collection of lazily-parsed order book levels.

    Attributes:
        _raw_levels (list[dict]): Raw level data.
        _parsed (dict[int, LazyLevel]): Cache of parsed LazyLevel objects.
    """
    def __init__(self, raw_levels):
        self._raw_levels = raw_levels
        self._parsed = {}

    def __getitem__(self, i):
        if i not in self._parsed:
            self._parsed[i] = LazyLevel(self._raw_levels[i])
        return self._parsed[i]

    def __iter__(self):
        for i in range(len(self._raw_levels)):
            yield self[i]

    def __len__(self):
        return len(self._raw_levels)

    def parse(self) -> list[LevelInfo]:
        """Parse all levels and return list of LevelInfo objects."""
        return [lvl.parse() for lvl in self]


class LazyBook(Book):
    """
    Lazily-parsed order book.

    Attributes:
        _raw (dict): Raw order book data.
        _bids (LazyLevels | None): Lazily-parsed bid levels.
        _asks (LazyLevels | None): Lazily-parsed ask levels.
        _events (list | None): Parsed events (Orders, Trades, Cancellations).
    """
    def __init__(self, raw_book):
        self._raw = raw_book
        self._bids = None
        self._asks = None
        self._events = None

    @property
    def id(self) -> int:
        """The book id, read from the raw wire dict without parsing the rest."""
        return self._raw.get("i")

    @property
    def MTR(self) -> float | None:
        """The maker-taker ratio, read straight off the raw payload.

        MTR is the accessor a miner is meant to use: short key on the wire, clearer name in the agent API.
        On the eager Book it is a property returning the `r` field, but LazyBook never populates fields, so
        that property raised AttributeError on the very model a miner receives. Overridden here for the
        same reason id/bids/asks are, and it cannot be done by defining `r` instead: `r` is a pydantic
        FIELD on Book, so a subclass property does not take precedence over it.
        """
        return self._raw.get("r")

    @property
    def bids(self):
        """Bid levels, parsed lazily on first access."""
        if self._bids is None:
            self._bids = LazyLevels(self._raw.get("b", []))
        return self._bids

    @property
    def asks(self):
        """Ask levels, parsed lazily on first access."""
        if self._asks is None:
            self._asks = LazyLevels(self._raw.get("a", []))
        return self._asks

    @property
    def events(self):
        """Book events, parsed lazily on first access into their event models."""
        if self._events is None:
            raw_events = self._raw.get("e", [])
            parsed_events = []
            for e in raw_events:
                ty = e.get("y")
                if ty == "o":
                    parsed_events.append(Order.model_construct(**e))
                elif ty == "t":
                    parsed_events.append(TradeInfo.model_construct(**e))
                elif ty == "c":
                    parsed_events.append(Cancellation.model_construct(**e))
                else:
                    parsed_events.append(e)
            self._events = parsed_events
        return self._events

    def parse(self) -> Book:
        """Return fully parsed Book object."""
            # r IS THE MAKER-TAKER RATIO, and it has to be passed explicitly: model_construct sets only
            # the fields named here, so an omitted one stays at its default of None however faithfully the
            # wire carried it. MTR is a property returning this field, so the accessor a miner is
            # documented to use returned None on every book in BOTH mechanisms.
        return Book.model_construct(
            i=self._raw.get("i"),
            r=self._raw.get("r"),
            b=self.bids.parse(),
            a=self.asks.parse(),
            e=self.events
        )


class LazyBooks(Mapping):
    """
    Lazily-parsed collection of order books.

    Attributes:
        _raw_books (dict[int, dict]): Raw book data keyed by book_id.
        _parsed_books (dict[int, LazyBook]): Cache of parsed LazyBook objects.
    """
    def __init__(self, raw_books: dict):
        self._raw_books = {int(k): v for k, v in raw_books.items()}
        self._parsed_books = {}

    def __getitem__(self, book_id: int):
        if book_id not in self._parsed_books:
            self._parsed_books[book_id] = LazyBook(self._raw_books[book_id])
        return self._parsed_books[book_id]

    def __iter__(self):
        return iter(self._raw_books)

    def __len__(self):
        return len(self._raw_books)

    def items(self):
        """Iterate ``(book_id, book)`` pairs, parsing each book lazily."""
        for k in self._raw_books:
            yield k, self[k]

    def values(self):
        """Iterate books, parsing each lazily."""
        for k in self._raw_books:
            yield self[k]

    def parse(self) -> dict[int, Book]:
        """Return dict of fully parsed Book objects keyed by book_id."""
        return {book_id: lb.parse() for book_id, lb in self.items()}


class LazyAccount:
    """
    Lazily-parsed trading account.

    Attributes:
        _raw (dict): Raw account data.
        _parsed (Account | None): Parsed Account object.
    """
    def __init__(self, raw_acc):
        self._raw = raw_acc
        self._parsed = None

    @property
    def data(self):
        """The parsed account, built on first access via ``model_construct`` (no re-validation)."""
        if self._parsed is None:
            bb = Balance.model_construct(**self._raw.get("bb", {}))
            qb = Balance.model_construct(**self._raw.get("qb", {}))
            orders = [Order.model_construct(**o) for o in self._raw.get("o", [])]

            loans = {}
            for k, v in self._raw.get("l", {}).items():
                loan = Loan.model_construct(**v)
                loan.c = OrderCurrency(loan.c)
                loans[int(k)] = loan

            fees = Fees.model_construct(**self._raw["f"]) if self._raw.get("f") else None

            self._parsed = Account.model_construct(
                i=self._raw.get("i"),
                b=self._raw.get("b"),
                bb=bb,
                qb=qb,
                bl=self._raw.get("bl", 0.0),
                ql=self._raw.get("ql", 0.0),
                bc=self._raw.get("bc", 0.0),
                qc=self._raw.get("qc", 0.0),
                o=orders,
                l=loans,
                f=fees,
                v=self._raw.get("v"),
                ds=self._raw.get("ds", {}) or {}
            )
            self._raw = None
        return self._parsed

    def __getattr__(self, name):
        return getattr(self.data, name)

    def parse(self) -> Account:
        """Return fully parsed Account object."""
        return self.data


class LazyAccounts(Mapping):
    """
    Lazily-parsed collection of agent accounts.

    Attributes:
        _raw_accounts (dict[int, dict[int, dict]]): Outer dict keyed by agent ID (uid), inner dict keyed by book_id.
        _parsed_accounts (dict[int, dict[int, LazyAccount]]): Cache of parsed LazyAccount objects.
    """
    def __init__(self, raw_accounts: dict):
        self._raw_accounts = {
            int(uid): {int(book_id): account for book_id, account in uid_accounts.items()}
            for uid, uid_accounts in raw_accounts.items()
        }
        self._parsed_accounts = {}

    def __getitem__(self, uid: int):
        if uid not in self._parsed_accounts:
            self._parsed_accounts[uid] = {
                book_id: LazyAccount(raw_acc)
                for book_id, raw_acc in self._raw_accounts[uid].items()
            }
        return self._parsed_accounts[uid]

    def __iter__(self):
        return iter(self._raw_accounts)

    def __len__(self):
        return len(self._raw_accounts)

    def items(self):
        """Iterate ``(agent_id, accounts)`` pairs, parsing lazily."""
        for k in self._raw_accounts:
            yield k, self[k]

    def values(self):
        """Iterate per-agent account maps, parsing lazily."""
        for k in self._raw_accounts:
            yield self[k]

    def parse(self) -> dict[int, dict[int, Account]]:
        """Return dict of fully parsed Account objects keyed by uid and book_id."""
        return {
            uid: {book_id: la.parse() for book_id, la in books.items()}
            for uid, books in self.items()
        }

# ── Book behaviour shared with the simulation mechanism ───────────────────────────────────────────
#
# The exchange Book carried NO domain methods at all -- only pydantic's. The simulation Book has
# `event_history` and `append_to_event_history`, and three SHIPPED example agents call them
# (SimpleRegressorAgent, DevAgent, MovingHurstAgent). Those agents therefore worked under mechanism 0
# and crashed under mechanism 1:
#
#     File "agents/SimpleRegressorAgent.py", line 106, in update_predictors
#         book.append_to_event_history(timestamp, self.book_event_history[validator], ...)
#     AttributeError: 'Book' object has no attribute 'append_to_event_history'
#
# That is a break in a surface miners COPY: an agent built on event history dies the moment it is
# queried on the exchange, and from the miner's side it looks identical to never being queried at all
# (the agent throws, the validator sees no response).
#
# Bound rather than copied. Both classes declare the SAME fields (i, r, b, a, e), so the simulation
# implementations work unchanged here, and binding keeps one definition instead of two that drift.
# Safe from circular import: taos.im.protocol.models does not import this module.
from taos.im.protocol.models import Book as _SimulationBook  # noqa: E402

# BIND THE WHOLE SURFACE, not the methods that happened to break first. event_history and
# append_to_event_history were bound when SimpleRegressorAgent crashed on the latter; enumerating the
# two classes afterwards showed four MORE simulation-only callables. Binding them one incident at a
# time guarantees the next agent to use history() fails the same way, in production, on a surface
# miners copy.
#
# They do not have to differ: both classes declare exactly the same fields (i, r, b, a, e), so every
# simulation implementation operates correctly on an exchange Book. The divergence is an omission, not
# a design decision -- the unified layer (UnifiedAccount, UnifiedAgentResponse) was built for accounts
# and responses and never completed for books.
for _name in (
    "event_history",
    "append_to_event_history",
    "history",
    "append_to_history",
    "process_history",
    "snapshot",
):
    _impl = getattr(_SimulationBook, _name, None)
    if _impl is not None and not hasattr(Book, _name):
        setattr(Book, _name, _impl)
