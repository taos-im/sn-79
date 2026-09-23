# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The order book, its levels, trades, L2 snapshots and the event and L2 histories a book carries.

Split out of taos.im.protocol.models, which re-exports every name here; import from either.
"""
import numpy as np
from collections.abc import Mapping
from itertools import accumulate
from typing import Literal, Any, Union, Optional
from pydantic import Field
from ypyjson import YpyObject
from taos.common.protocol import BaseModel
from taos.im.protocol.orders import Order, Cancellation, OrderDirection
from taos.im.protocol.config import MarketSimulationConfig


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
        if 'o' not in json:
            orders = None
        else:
            orders = [Order.model_construct(i=order['i'], t=order['t'], q=order['q'], s=order['s'], order_type="limit", p=json['p'], l=json['l'] if 'l' in json else 0.0) for order in json['o']]
        return LevelInfo.model_construct(p=json['p'], q=json['q'], o=orders)


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
    Ti : int | None = Field(alias='taker_id', default=None)
    Ta : int | None = Field(alias='taker_agent_id', default=None)
    Tf : float | None = Field(alias='taker_fee', default=None)
    Mi : int | None = Field(alias='maker_id', default=None)
    Ma : int | None = Field(alias='maker_agent_id', default=None)
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
        return TradeInfo.model_construct(i=json['i'], t=json['t'], q=json['q'], s=json['s'], p=json['p'],
                         Ta=json['Ta'], Ti=json['Ti'], Ma=json['Ma'], Mi=json['Mi'],
                         Mf=json['Mf'], Tf=json['Tf'])


class L2Snapshot(BaseModel):
    """
    Represents a level-2 order book snapshot at a specific timestamp.

    Attributes:
        timestamp (int): Simulation timestamp of the snapshot in nanoseconds.
        bids (dict[float, LevelInfo]): Bid side of the order book (price → LevelInfo).
        asks (dict[float, LevelInfo]): Ask side of the order book (price → LevelInfo).
    """

    timestamp: int
    bids: dict[float, LevelInfo]
    asks: dict[float, LevelInfo]

    def best_bid(self) -> float:
        """
        Get the highest bid price in the snapshot.

        Returns:
            float: The best (highest) bid price.
        """
        return max(self.bids.keys())

    def best_ask(self) -> float:
        """
        Get the lowest ask price in the snapshot.

        Returns:
            float: The best (lowest) ask price.
        """
        return min(self.asks.keys())

    def bid_level(self, index: int) -> LevelInfo:
        """
        Get a specific bid level sorted by price descending.

        Args:
            index (int): The index of the bid level to retrieve.

        Returns:
            LevelInfo: The bid level at the specified index.
        """
        return self.bids[list(sorted(self.bids.values(), reverse=True))[index]]

    def ask_level(self, index: int) -> LevelInfo:
        """
        Get a specific ask level sorted by price ascending.

        Args:
            index (int): The index of the ask level to retrieve.

        Returns:
            LevelInfo: The ask level at the specified index.
        """
        return self.asks[list(sorted(self.asks.values()))[index]]

    def imbalance(self, depth: int | None = None) -> float:
        """
        Calculate the order book imbalance at a given depth.

        Imbalance formula:
            (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

        Args:
            depth (int | None): Optional number of levels to include in the calculation. If None, uses all levels.

        Returns:
            float: The imbalance ratio.
        """
        total_bid_vol = sum(
            [bid.quantity for bid in list(self.bids.values())[:(depth if depth else len(self.bids))]]
        )
        total_ask_vol = sum(
            [ask.quantity for ask in list(self.asks.values())[:(depth if depth else len(self.asks))]]
        )
        return (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

    def compare(self, target: 'L2Snapshot', config: MarketSimulationConfig) -> tuple[bool, list[str], dict[str, dict[float, float]]]:
        """
        Compare this snapshot to a target snapshot, and return a list of discrepancies as well as a dictionary mapping price level to the volume determined to already exist at that level
        prior to the original snapshot being constructed.  This is necessary as some new price levels may enter the top levels due to cancellations and trades.

        Args:
            target (L2Snapshot): The snapshot to compare against.
            config (MarketSimulationConfig): Simulation configuration with rounding and volume precision.

        Returns:
            tuple:
                - bool: True if snapshots match (no discrepancies), False otherwise.
                - list[str]: List of textual discrepancy descriptions.
                - dict: Dictionary of existing volumes needed to reconcile (bids and asks).
        """
        discrepancies = []
        existing_volumes = {'bid': {}, 'ask': {}}

        # Compare bids
        for price, bid in self.bids.items():
            if price in target.bids:
                if bid.quantity != target.bids[price].quantity:
                    discrepancies.append(f"BID : RECON {bid.quantity}@{price} vs. TARGET {target.bids[price].quantity}@{price}")
                if bid.quantity < target.bids[price].quantity:
                    existing_volumes['bid'][price] = round(target.bids[price].quantity - bid.quantity, config.volumeDecimals)
            else:
                discrepancies.append(f"BID : RECON {bid.quantity}@{price} vs. TARGET 0.0@{price}")
                if bid.quantity < 0:
                    existing_volumes['bid'][price] = round(-bid.quantity, config.volumeDecimals)

        # Add missing bids from target
        for price, bid in target.bids.items():
            if price not in self.bids:
                discrepancies.append(f"BID : RECON 0.0@{price} vs. TARGET {bid.quantity}@{price}")
                existing_volumes['bid'][price] = bid.quantity

        # Compare asks
        for price, ask in self.asks.items():
            if price in target.asks:
                if ask.quantity != target.asks[price].quantity:
                    discrepancies.append(f"ASK : RECON {ask.quantity}@{price} vs. TARGET {target.asks[price].quantity}@{price}")
                if ask.quantity < target.asks[price].quantity:
                    existing_volumes['ask'][price] = round(target.asks[price].quantity - ask.quantity, config.volumeDecimals)
            else:
                discrepancies.append(f"ASK : RECON {ask.quantity}@{price} vs. TARGET 0.0@{price}")
                if ask.quantity < 0:
                    existing_volumes['ask'][price] = round(-ask.quantity, config.volumeDecimals)

        # Add missing asks from target
        for price, ask in target.asks.items():
            if price not in self.asks:
                discrepancies.append(f"ASK : RECON 0.0@{price} vs. TARGET {ask.quantity}@{price}")
                existing_volumes['ask'][price] = ask.quantity

        return len(discrepancies) == 0, discrepancies, existing_volumes

    def sort(self, depth: int | None = None, in_place : bool = True) -> 'L2Snapshot':
        """
        Sort bids descending and asks ascending, and truncates levels to the specified depth.

        Args:
            depth (int | None): Optional number of levels to keep after sorting.
        """
        if in_place:
            self.bids = dict(list(sorted(self.bids.items(), reverse=True))[:(depth if depth else len(self.bids))])
            self.asks = dict(list(sorted(self.asks.items()))[:(depth if depth else len(self.asks))])
            return self
        else:
            return self.model_copy(update = {
                "bids" : dict(list(sorted(self.bids.items(), reverse=True))[:(depth if depth else len(self.bids))]),
                "asks" : dict(list(sorted(self.asks.items()))[:(depth if depth else len(self.asks))])
            })

    def reconcile(self, existing_volumes: dict[str, dict[float, float]], config: MarketSimulationConfig, depth: int) -> 'L2Snapshot':
        """
        Reconcile snapshot levels with specified volume adjustments.

        Args:
            existing_volumes (dict): Volumes to adjust for bids and asks.
            config (MarketSimulationConfig): Simulation configuration, for rounding.
            depth (int): Depth of levels to retain.

        Returns:
            L2Snapshot: The updated snapshot after reconciliation.
        """
        if len(existing_volumes['bid']) > 0 or len(existing_volumes['ask']) > 0:
            # Adjust bid levels
            for price, volume in existing_volumes['bid'].items():
                if price in self.bids:
                    self.bids[price].q = round(self.bids[price].q + volume, config.volumeDecimals)
                    if self.bids[price].q == 0:
                        del self.bids[price]
                else:
                    self.bids[price] = LevelInfo(price=price, quantity=volume, orders=None)
            # Adjust ask levels
            for price, volume in existing_volumes['ask'].items():
                if price in self.asks:
                    self.asks[price].q = round(self.asks[price].q + volume, config.volumeDecimals)
                    if self.asks[price].q == 0:
                        del self.asks[price]
                else:
                    self.asks[price] = LevelInfo(price=price, quantity=volume, orders=None)
        self.sort(depth)
        return self


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


class L2History(History):
    """
    Represents the historical record of L2Snapshots and trades over time.

    Attributes:
        snapshots (dict[int, L2Snapshot]): Mapping of timestamps to L2Snapshot instances.
        trades (dict[int, TradeInfo]): Mapping of timestamps to TradeInfo instances.
        start (int): The earliest timestamp in the history.
        end (int): The latest timestamp in the history.
        retention_mins (int | None): Optional retention window in minutes. If set, older data will be purged.
    """

    snapshots: dict[int, L2Snapshot]
    trades: dict[int, TradeInfo]
    start : int
    end : int
    retention_mins : int | None

    def __init__(
        self,
        snapshots: dict[int, L2Snapshot],
        trades: dict[int, TradeInfo],
        publish_interval: int,
        retention_mins: int | None = None
    ):
        """
        Initialize an L2History object.

        Args:
            snapshots (dict[int, L2Snapshot]): Initial snapshots to populate history.
            trades (dict[int, TradeInfo]): Initial trades to populate history.
            publish_interval (int): Interval between snapshot publications; the history start is set one interval before the first snapshot.
            retention_mins (int | None): Optional retention window in minutes.
        """
        self.snapshots = snapshots
        self.trades = trades
        self.start = list(snapshots.keys())[0] - publish_interval
        self.end = list(snapshots.keys())[-1]
        self.retention_mins = retention_mins

    def append(self, new_history: 'L2History') -> 'L2History':
        """
        Append another L2History instance to this history.

        Merges snapshots and trades, then applies retention logic if enabled.

        Args:
            new_history (L2History): The history instance to append.

        Returns:
            L2History: Updated history instance with merged data.
        """
        # Merge and sort snapshots and trades
        self.snapshots = dict(list(sorted((self.snapshots | new_history.snapshots).items())))
        self.trades = dict(list(sorted((self.trades | new_history.trades).items())))
        self.end = list(self.snapshots.keys())[-1]

        # Apply retention if configured
        if self.retention_mins:
            min_time = self.end - self.retention_mins * 60_000_000_000  # nanoseconds
            # Remove old snapshots
            for t in list(self.snapshots):
                if t < min_time:
                    del self.snapshots[t]
                else:
                    break
            # Remove old trades
            for t in list(self.trades):
                if t < min_time:
                    del self.trades[t]
                else:
                    break
        self.start = list(self.snapshots.keys())[0]
        return self

    def insert(self, snapshot : L2Snapshot):
        """
        Insert a snapshot to the history, sorting and updating start/end times.
        
        Args:
            snapshot (L2Snapshot): The snapshot to insert.
        """
        self.snapshots[snapshot.timestamp] = snapshot
        self.snapshots = dict(list(sorted((self.snapshots).items())))
        self.end = list(self.snapshots.keys())[-1]
        self.start = list(self.snapshots.keys())[0]

    def reconcile(self, existing_volumes: dict[str, dict[float, float]], config: MarketSimulationConfig, depth: int) -> None:
        """
        Reconcile all snapshots in history with specified volume adjustments.

        Args:
            existing_volumes (dict): Dictionary of volume adjustments (bids and asks).
            config (MarketSimulationConfig): Simulation configuration.
            depth (int): Depth of order book to retain.
        """
        for time in self.snapshots:
            self.snapshots[time] = self.snapshots[time].reconcile(existing_volumes, config, depth)
    
    def ohlc(self, interval: float):
        """Sample the L2 trade series into OHLC bars.

        Args:
            interval (float): Bar width in seconds.

        Returns:
            dict: Timestamp-keyed OHLC bars.
        """
        return self.sample(self.trade(), interval, 'ohlc')

    def midquote(self, sampling_secs: float | None = None) -> dict[int, float]:
        """
        Compute the midquote (average of best bid and ask) over time.

        Args:
            sampling_secs (float | None): Optional sampling interval in seconds.

        Returns:
            dict[int, float]: Time series of midquotes.
        """
        midquotes = {
            time: (snapshot.best_bid() + snapshot.best_ask()) / 2
            for time, snapshot in self.snapshots.items()
        }
        return self.sample(midquotes, sampling_secs) if sampling_secs else midquotes

    def bid(self, sampling_secs: float | None = None) -> dict[int, float]:
        """
        Get the best bid prices over time.

        Args:
            sampling_secs (float | None): Optional sampling interval in seconds.

        Returns:
            dict[int, float]: Time series of best bid prices.
        """
        bids = {time: snapshot.best_bid() for time, snapshot in self.snapshots.items()}
        return self.sample(bids, sampling_secs) if sampling_secs else bids

    def ask(self, sampling_secs: float | None = None) -> dict[int, float]:
        """
        Get the best ask prices over time.

        Args:
            sampling_secs (float | None): Optional sampling interval in seconds.

        Returns:
            dict[int, float]: Time series of best ask prices.
        """
        asks = {time: snapshot.best_ask() for time, snapshot in self.snapshots.items()}
        return self.sample(asks, sampling_secs) if sampling_secs else asks

    def trade(self, sampling_secs: float | None = None) -> dict[int, float]:
        """
        Get the trade prices over time.

        Args:
            sampling_secs (float | None): Optional sampling interval in seconds.

        Returns:
            dict[int, float]: Time series of trade prices.
        """
        trades = {time: trade.price for time, trade in self.trades.items()}
        return self.sample(trades, sampling_secs) if sampling_secs else trades

    def imbalance(self, depth: int | None = None, sampling_secs: float | None = None) -> dict[int, float]:
        """
        Get the order book imbalance over time.

        Args:
            depth (int | None): Depth of order book to consider.
            sampling_secs (float | None): Optional sampling interval in seconds.

        Returns:
            dict[int, float]: Time series of imbalance values.
        """
        imbalance = {time: snapshot.imbalance(depth) for time, snapshot in self.snapshots.items()}
        return self.sample(imbalance, sampling_secs) if sampling_secs else imbalance

    def mean_imbalance(self, depth: int | None = None) -> float:
        """
        Compute the mean order book imbalance over the history.

        Args:
            depth (int | None): Depth of order book to consider.

        Returns:
            float: Mean imbalance value.
        """
        imbalance_history = self.imbalance(depth)
        return sum(imbalance_history.values()) / len(imbalance_history)


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

        Populated only under the dynamic fee policy (simulation); a tiered or
        zero-fee policy (including the zero-fee exchange) reports 0.

        Returns:
            float: The book's maker-taker ratio in [0, 1], or None if unset.
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

        return cls.model_construct(id=id, bids=bids, asks=asks, events=events, r=json.get("r"))
    
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

    def snapshot(self, timestamp: int) -> L2Snapshot:
        """
        Generate an L2Snapshot of the current book state.

        Args:
            timestamp (int): Timestamp to assign to the snapshot.

        Returns:
            L2Snapshot: Snapshot representing current bids and asks.
        """
        return L2Snapshot(
            timestamp=timestamp,
            bids={l.price: LevelInfo.model_construct(price=l.price, quantity=l.quantity, orders=l.orders) for l in self.bids},
            asks={l.price: LevelInfo.model_construct(price=l.price, quantity=l.quantity, orders=l.orders) for l in self.asks}
        )
        
    def process_history(
        self, 
        history: dict[int, L2Snapshot], 
        trades: dict[int, TradeInfo], 
        timestamp: int, 
        config: MarketSimulationConfig, 
        retention_mins: int, 
        depth: int | None = None
    ) -> tuple[L2History, bool, list[str]]:
        """
        Processes an existing L2 history with the current book state.

        Args:
            history (dict[int, L2Snapshot]): Dictionary of previous snapshots indexed by timestamp.
            trades (dict[int, TradeInfo]): Dictionary of trades indexed by timestamp.
            timestamp (int): Current timestamp for the new snapshot.
            config (MarketSimulationConfig): Configuration settings for volume precision and publish intervals.
            retention_mins (int): Retention period for keeping history (in minutes).
            depth (int | None): Optional depth to limit order book levels.

        Returns:
            tuple:
                - L2History: The updated history object including the new snapshot.
                - bool: True if the reconstructed snapshot matches the target snapshot.
                - list[str]: List of discrepancies detected between reconstructed and target snapshot.
        """
        # Generate a snapshot of the current book state at the given timestamp
        target_snapshot: L2Snapshot = self.snapshot(timestamp)

        # Compare the last snapshot in history with the target snapshot to check for discrepancies
        pre_matched, pre_discrepancies, pre_existing_volumes = (
            list(history.values())[-1].compare(target_snapshot, config)
        )

        # Build a new history object from the provided snapshots and trades
        history_obj: L2History = L2History(
            snapshots=history, 
            trades=trades, 
            retention_mins=retention_mins,
            publish_interval=config.publish_interval
        )

        # Attempt to reconcile discrepancies by applying existing volume corrections
        history_obj.reconcile(pre_existing_volumes, config, depth)

        # After reconciliation, compare again to detect any remaining mismatches
        matched, discrepancies, existing_volumes = (
            list(history_obj.snapshots.values())[-1].compare(target_snapshot, config)
        )

        # Insert the target snapshot into the history
        history_obj.insert(target_snapshot)

        return history_obj, matched, discrepancies

    def history(
        self,
        snapshot: L2Snapshot,
        config: MarketSimulationConfig,
        retention_mins: int | None = None,
        depth: int | None = None
    ) -> tuple[L2History, bool, list[str]]:
        """
        Build an L2History from the current book and apply all events.

        Args:
            snapshot (L2Snapshot): The initial snapshot to start from.
            config (MarketSimulationConfig): Simulation configuration.
            retention_mins (int | None): Optional retention window in minutes.
            depth (int | None): Optional depth to limit order book levels.

        Returns:
            tuple:
                - L2History: The resulting history after applying events.
                - bool: True if the resulting snapshot matches the target.
                - list[str]: List of discrepancies found during reconciliation.
        """
        if not depth:
            depth = len(snapshot.bids)
        # Create history dictionary and add the starting snapshot
        history = {snapshot.timestamp: snapshot.model_copy(deep=True)}
        trades = {}

        # Generate target snapshot for comparison
        target_snapshot = self.snapshot(snapshot.timestamp + config.publish_interval)
        # Apply events in chronological order
        for event in sorted(self.events, key=lambda x: x.timestamp):
            match event:
                case o if isinstance(event, Order):
                    # Place new order
                    if o.side == OrderDirection.BUY:
                        if o.price not in snapshot.bids:
                            snapshot.bids[o.price] = LevelInfo(price=o.price, quantity=0.0, orders=None)
                        snapshot.bids[o.price].q = round(
                            snapshot.bids[o.price].q + o.quantity,
                            config.volumeDecimals
                        )
                    else:
                        if o.price not in snapshot.asks:
                            snapshot.asks[o.price] = LevelInfo(price=o.price, quantity=0.0, orders=None)
                        snapshot.asks[o.price].q = round(
                            snapshot.asks[o.price].q + o.quantity,
                            config.volumeDecimals
                        )

                case t if isinstance(event, TradeInfo):
                    # Record trade
                    trades[t.timestamp] = t
                    if t.side == OrderDirection.BUY:
                        if t.price in snapshot.asks:
                            snapshot.asks[t.price].q = round(
                                snapshot.asks[t.price].q - t.quantity,
                                config.volumeDecimals
                            )
                            if snapshot.asks[t.price].quantity == 0.0:
                                del snapshot.asks[t.price]
                    else:
                        if t.price in snapshot.bids:
                            snapshot.bids[t.price].q = round(
                                snapshot.bids[t.price].q - t.quantity,
                                config.volumeDecimals
                            )
                            if snapshot.bids[t.price].quantity == 0.0:
                                del snapshot.bids[t.price]

                case c if isinstance(event, Cancellation):
                    # Cancel existing order
                    if c.price >= snapshot.best_ask():
                        if c.price in snapshot.asks:
                            snapshot.asks[c.price].q = round(
                                snapshot.asks[c.price].q - c.quantity,
                                config.volumeDecimals
                            )
                            if snapshot.asks[c.price].quantity == 0.0:
                                del snapshot.asks[c.price]
                    else:
                        if c.price in snapshot.bids:
                            snapshot.bids[c.price].q = round(
                                snapshot.bids[c.price].q - c.quantity,
                                config.volumeDecimals
                            )
                            if snapshot.bids[c.price].quantity == 0.0:
                                del snapshot.bids[c.price]

            # Add snapshot to history after each update
            history[event.timestamp] = snapshot.model_copy(deep=True)
        # Compare resulting snapshot to target
        pre_matched, pre_discrepancies, pre_existing_volumes = snapshot.compare(target_snapshot, config)
        history_obj = L2History(snapshots=history, trades=trades, retention_mins=retention_mins, publish_interval=config.publish_interval)
        # Apply determined existing volumes to attempt to reconcile any discrepancies
        history_obj.reconcile(pre_existing_volumes, config, depth)
        # Check if any remaining discrepancies after reconciliation
        matched, discrepancies, existing_volumes = list(history_obj.snapshots.values())[-1].compare(target_snapshot, config)
        # Add the target snapshot to the history
        history_obj.insert(target_snapshot)

        return history_obj, matched, discrepancies

    def append_to_history(
        self,
        history: L2History,
        config: MarketSimulationConfig,
        depth: int | None = None
    ) -> tuple[L2History, bool, list[str]]:
        """
        Append the book's events to an existing L2History.

        Args:
            history (L2History): Existing history to append to.
            config (MarketSimulationConfig): Simulation configuration.
            depth (int | None): Optional depth to limit levels.

        Returns:
            tuple:
                - L2History: Updated history including new events.
                - bool: True if final snapshot matches target.
                - list[str]: List of discrepancies found.
        """
        new_history, matched, discrepancies = self.history(
            snapshot=list(history.snapshots.values())[-1],
            config=config,
            retention_mins=history.retention_mins,
            depth=depth
        )
        return history.append(new_history=new_history), matched, discrepancies
    
    def event_history(
        self,
        timestamp : int,
        config: MarketSimulationConfig,
        retention_mins: int | None = None
    ) -> EventHistory:
        """
        Build an EventHistory from the current book and apply all events.

        Args:
            timestamp (int): Timestamp of the state update associated with the book instance.
            config (MarketSimulationConfig): Simulation configuration.
            retention_mins (int | None): Optional retention window in minutes.

        Returns:
            tuple:
                - EventHistory: The resulting history after applying events.
                - bool: True if the resulting snapshot matches the target.
                - list[str]: List of discrepancies found during reconciliation.
        """
        return EventHistory(timestamp - config.publish_interval, timestamp, self.events, config.publish_interval, retention_mins)
    
    def append_to_event_history(
        self,
        timestamp : int,
        history: EventHistory,
        config: MarketSimulationConfig
    ) -> tuple[EventHistory, bool, list[str]]:
        """
        Append the book's events to an existing EventHistory.

        Args:
            timestamp (int): Timestamp up to which the book's events are collected.
            history (EventHistory): Existing history to append to.
            config (MarketSimulationConfig): Simulation configuration.

        Returns:
            tuple:
                - EventHistory: Updated history including new events.
                - bool: True if final snapshot matches target.
                - list[str]: List of discrepancies found.
        """
        new_history = self.event_history(
            timestamp=timestamp,
            config=config,
            retention_mins=history.retention_mins
        )
        return history.append(new_history=new_history)
