# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT

import time
import traceback
import numpy as np
import bittensor as bt
from collections import defaultdict
from dataclasses import dataclass
from taos.common.agents import launch
from taos.im.agents import RollingWindow
from taos.im.protocol.events import SimulationEndEvent
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
import uuid  # For generating unique trade IDs
from taos.im.agents import GenTRXAgent

@dataclass
class RollingWindowHurst(RollingWindow):
    """
    Rolling window configuration for price sampling and Hurst exponent estimation.

    Attributes:
        min (int): Minimum number of samples required before signals are considered reliable.
        max (int): Maximum length of the rolling buffer (in samples).
        lag_min (int): Minimum lag (in samples) for Hurst exponent calculation.
        lag_max (int): Maximum lag (in samples) for Hurst exponent calculation (advanced mode).
        samples (int): Number of random samples per window in advanced Hurst mode.
        num_windows (int): Number of scales/windows to use for multi-scale Hurst estimation.
    
    Parameter Tuning Guidelines:
        - Increase max to smooth Hurst signals in volatile markets.
        - Increase num_windows and samples for advanced mode for more statistical robustness.
        - Adjust lag_min/lag_max depending on expected momentum duration.
    """
    # Defaults match the call site below; needed so @dataclass inheritance
    # doesn't reject non-default fields appearing after the parent class's
    # `sampling_interval: int = 1`.
    lag_min: int = 2
    lag_max: int = 60

@dataclass
class Thresholds:
    """
    Threshold configuration for generating trading signals.

    Attributes:
        signal (float): Hurst threshold above/below which momentum is actionable.
        tolerance (float): Neutral band around the signal threshold where HOLD signals are emitted.
        model (float): Minimum Hurst level considered trustworthy for trading decisions.

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
    timestamp: int
    price: float
@dataclass
class Positions:
    """Container for positions for different books"""
    open: bool
    direction: OrderDirection
    amount: int
    
class HurstSignals(IntEnum):
    """
    Enum to represent signals coming from Hurst estimates.

    Attributes:
        ENTRY (int): Hurst estimate indicates that there is momentum 
        EXIT (int): Hurst estimate indicates mean-reversion
        HOLD (int): Hurst estimate within the tolerance of threshold
        NOISE (int): Discard too low values, safer to ignore it
    """
    ENTRY=0
    EXIT=1
    HOLD=3
    NOISE=4


class MovingHurstAgent(GenTRXAgent):
    """
    Momentum/Mean-Reversion Agent using Hurst exponent.

    Workflow:
        1. Collect rolling OHLC prices.
        2. Estimate Hurst exponent (simple or advanced).
        3. Convert Hurst to ENTRY/EXIT/HOLD/NOISE signal.
        4. Execute market orders for ENTRY/EXIT events.
    """

    def initialize(self):
        """Initialize agent configuration, thresholds, rolling windows, and buffers."""
        # GenTRX is opt-in: only activates when explicitly configured.
        if not hasattr(self.config, 'gtx_training_enabled'):
            self.config.gtx_training_enabled = False
        if not hasattr(self.config, 'gtx_collect_data'):
            self.config.gtx_collect_data = False
        super().initialize()
        self.quantity = getattr(self.config, 'quantity', 10.0)
        self.expiry_period = getattr(self.config, 'expiry_period', 120e9)
        self.predKeys = ['Close', 'Timestamp']

        self.rolling_window = RollingWindowHurst(
            min=int(getattr(self.config, 'rolling_window_min', 60)),
            max=int(getattr(self.config, 'rolling_window_max', 1800)),
            lag_min=2,
            lag_max=60,
            samples=100,
            num_windows=int(getattr(self.config, 'num_windows', 20))
        )

        self.advanced_mode = getattr(self.config, 'mode', None) == 'advanced'

        self.thresholds = Thresholds(
            model=getattr(self.config, 'model_threshold', 0.4),
            signal=getattr(self.config, 'signal_threshold', 0.5),
            tolerance=getattr(self.config, 'signal_tolerance', 0.01)
        )

        self.sampling_interval = int(getattr(self.config, 'sampling_interval', 1))

        # Internal buffers
        self.predictors = {}
        self.last_signal = {}
        self.midquotes = {}
        self.directions = {}
        self.book_event_history : dict[str, EventHistory | None] = {}
        self.trade_counter = defaultdict(int)

    def simple_hurst(self, price_history):
        """Standard deviation of lagged differences method for Hurst exponent."""
        lags = range(self.rolling_window.lag_min, self.rolling_window.num_windows)
        tau = [np.nanstd(np.subtract(price_history[lag:], price_history[:-lag])) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]

    def advanced_hurst(self, price_history):
        """Random-sampled R/S method for advanced Hurst exponent estimation."""
        log_returns = np.diff(np.log(np.array(price_history)))
        window_sizes = np.linspace(
            self.rolling_window.lag_min,
            self.rolling_window.lag_max,
            self.rolling_window.num_windows,
            dtype=int
        )
        R_S = []
        valid_windows = 0
        for window_size in window_sizes:
            if len(log_returns) <= window_size:
                continue
            valid_windows += 1
            R, S = [], []
            for _ in range(self.rolling_window.samples):
                start = np.random.randint(0, len(log_returns) - window_size)
                seq = log_returns[start:start + window_size]
                R.append(np.max(seq) - np.min(seq))
                S.append(np.std(seq))
            R_S.append(np.mean(R) / np.mean(S))
        log_window_sizes = np.log(window_sizes[:valid_windows])
        log_R_S = np.log(R_S)
        return np.polyfit(log_window_sizes, log_R_S, 1)[0]

    def estimate_hurst(self, price_history):
        """Wrapper to select simple or advanced Hurst calculation."""
        if self.advanced_mode:
            H = self.advanced_hurst(price_history)
        else:
            H = self.simple_hurst(price_history)
        return H

    def update_predictors(self, validator : str, book: Book, timestamp: int) -> None:
        """
        Gathers and processes historical event data to update predictors and targets.
        Features are computed based on sampling intervals.

        Args:
            book (Book): Book object from the state update.
            timestamp (int): Simulation timestamp of the associated state update.
        """
        if validator not in self.book_event_history or not self.book_event_history[validator]:
            lookback_minutes = max(
                (self.simulation_config.publish_interval // 1_000_000_000) // 60,
                self.sampling_interval * 2 // 60,
                1
            )
            self.book_event_history[validator] = book.event_history(timestamp, self.simulation_config, lookback_minutes)
        else:
            book.append_to_event_history(timestamp, self.book_event_history[validator], self.simulation_config)

        ohlc = self.book_event_history[validator].ohlc(self.sampling_interval)
        n_new = max((self.simulation_config.publish_interval // 1_000_000_000) // self.sampling_interval, 1)
        latest_timestamps = sorted(ohlc.keys())[-n_new:]

        new_predictors = defaultdict(list)
        for ts in latest_timestamps:
            ohlc_data = ohlc.get(ts)
            if not ohlc_data:
                continue
            _, _, _, close = ohlc_data.values()
            new_predictors['Timestamp'].append(ts)
            new_predictors['Close'].append(close)

        book_id = book.id
        if len(self.predictors.get(validator, {}).get(book_id, {}).get('Timestamp', [])) > 0:
            if latest_timestamps and self.predictors[validator][book_id]['Timestamp'][-1] > latest_timestamps[-1]:
                bt.logging.info(f"[RESET] Timestamp mismatch in book {book_id}, clearing history")
                self.reset(validator)


        self.predictors[validator][book_id] = {
            k: self.predictors.get(validator, {}).get(book_id, {}).get(k, []) + new_predictors[k]
            for k in self.predKeys
        }
        self.predictors[validator][book_id] = {k: v[-self.rolling_window.max:] for k, v in self.predictors[validator][book_id].items()}

    def signal(self, predictions: dict[str, float]) -> HurstSignals:
        """Convert Hurst prediction into discrete trading signal."""
        H = predictions['Hurst']
        if H < self.thresholds.model:
            return HurstSignals.NOISE
        elif H > self.thresholds.signal + self.thresholds.tolerance:
            return HurstSignals.ENTRY
        elif H < self.thresholds.signal - self.thresholds.tolerance:
            return HurstSignals.EXIT
        return HurstSignals.HOLD
    

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """
        Called upon new market state updates. The core strategy loop:
        - Extracts market data
        - Updates predictors
        - Predicts returns
        - Places limit orders based on predicted signal

        Args:
            state (taos.im.protocol.MarketSimulationStateUpdate): The class representing the latest state of the simulation.

        Returns:
            taos.im.protocol.FinanceAgentResponse : The response which will be attached to the synapse for return to the querying validator.
        """
        # GenTRX: data collection + training trigger (runs even when training disabled).
        response = super().respond(state)
        start = time.time()

        for book_id, book in state.books.items():
            try:
                bestBid = book.bids[0].price if book.bids else 0.0
                bestAsk = book.asks[0].price if book.asks else bestBid + 10 ** (-self.simulation_config.priceDecimals)
                midquote = (bestBid + bestAsk) / 2

                if state.dendrite.hotkey not in self.predictors:
                    self.predictors[state.dendrite.hotkey] = {}
                    self.last_signal[state.dendrite.hotkey] = {}
                    self.midquotes[state.dendrite.hotkey] = {}
                    self.directions[state.dendrite.hotkey] = {}
                # Initialize buffers if first time seeing this book
                if book_id not in self.predictors[state.dendrite.hotkey]:
                    self.predictors[state.dendrite.hotkey][book_id] = {key: [] for key in self.predKeys}
                    self.last_signal[state.dendrite.hotkey][book_id] = 0.0
                    self.midquotes[state.dendrite.hotkey][book_id] = [TimestampedPrice(0, self.simulation_config.init_price)]
                    self.directions[state.dendrite.hotkey][book_id] = Positions(open=False,direction=OrderDirection.BUY,amount=0) 

                self.update_predictors(state.dendrite.hotkey, book, state.timestamp)

                # Skip if insufficient data or rolling interval not reached
                if len(self.predictors[state.dendrite.hotkey][book_id]['Close']) < self.rolling_window.min:
                    bt.logging.info(
                        f"BOOK {book_id} | Insufficient data : {len(self.predictors[state.dendrite.hotkey][book_id]['Close'])}/{self.rolling_window.min} Observations Available"
                    )
                    continue
                if (state.timestamp // 1e9) % self.rolling_window.min != 0:
                    continue

                predictions = {
                    'Hurst': self.estimate_hurst(self.predictors[state.dendrite.hotkey][book_id]['Close']),
                    'timestamp': state.timestamp // 1e9
                }
                signal = self.signal(predictions)
                trade_id = str(uuid.uuid4())

                bt.logging.info(
                    f"[SIGNAL] Book={book_id} Hurst={predictions['Hurst']:.4f} "
                    f"Signal={signal.name} Midquote={midquote:.4f} TradeID={trade_id}"
                )

                self.last_signal[state.dendrite.hotkey][book_id] = signal
                self.midquotes[state.dendrite.hotkey][book_id].append(
                    TimestampedPrice((state.timestamp // 1e9) // self.rolling_window.min, midquote)
                )

                # Execute trades
                if signal == HurstSignals.ENTRY:
                    # Determine the order direction based on long term (rolling window max) and short term (rolling window min) returns
                    long_term_direction = OrderDirection.BUY if self.predictors[state.dendrite.hotkey][book_id]['Close'][0] <= self.predictors[state.dendrite.hotkey][book_id]['Close'][-1] else OrderDirection.SELL
                    short_term_direction = OrderDirection.BUY if self.predictors[state.dendrite.hotkey][book_id]['Close'][-self.rolling_window.min] <= self.predictors[state.dendrite.hotkey][book_id]['Close'][-1] else OrderDirection.SELL
                    if long_term_direction != short_term_direction:
                        bt.logging.info("There is momentum but the direction might have changed, general advice exit")
                        if self.directions[state.dendrite.hotkey][book_id].open:
                            response, total_amount, close_dir = self.generate_exit_response(response, state.dendrite.hotkey, book_id)
                            bt.logging.debug(
                            f"[TRADE] EXIT Vali={state.dendrite.hotkey} Book={book_id} Direction={close_dir} "
                            f"Amount={total_amount} Midquote={midquote:.4f} Hurst={predictions['Hurst']:.4f} TradeID={trade_id}"
                            )
                        continue

                    if self.directions[state.dendrite.hotkey][book_id].open:
                        if long_term_direction != self.directions[state.dendrite.hotkey][book_id].direction:
                            bt.logging.info("There is momentum but the direction is most likely wrong, Exit (stop loss) and make new entry")
                            response, total_amount, close_dir = self.generate_exit_response(response, state.dendrite.hotkey, book_id)
                            bt.logging.debug(
                            f"[TRADE] EXIT Vali={state.dendrite.hotkey} Book={book_id} Direction={close_dir} "
                            f"Amount={total_amount} Midquote={midquote:.4f} Hurst={predictions['Hurst']:.4f} TradeID={trade_id}"
                            )
                            response = self.entry_or_extend(response, state.dendrite.hotkey, book_id, long_term_direction)
                            bt.logging.debug(
                            f"[TRADE] ENTRY Vali={state.dendrite.hotkey} Book={book_id} Direction={self.directions[state.dendrite.hotkey][book_id].direction} "
                            f"Amount={self.quantity} Midquote={midquote:.4f} Hurst={predictions['Hurst']:.4f} TradeID={trade_id}"
                            )
                            continue

                    response = self.entry_or_extend(response, state.dendrite.hotkey, book_id, long_term_direction)
                    bt.logging.debug(
                        f"[TRADE] ENTRY Vali={state.dendrite.hotkey} Book={book_id} Direction={self.directions[state.dendrite.hotkey][book_id].direction} "
                        f"Amount={self.quantity} Midquote={midquote:.4f} Hurst={predictions['Hurst']:.4f} TradeID={trade_id}"
                    )
                elif signal == HurstSignals.EXIT and self.directions[state.dendrite.hotkey][book_id].open:
                    response, total_amount, close_dir = self.generate_exit_response(response, state.dendrite.hotkey, book_id)
                    bt.logging.debug(
                        f"[TRADE] EXIT Vali={state.dendrite.hotkey} Book={book_id} Direction={close_dir} "
                        f"Amount={total_amount} Midquote={midquote:.4f} Hurst={predictions['Hurst']:.4f} TradeID={trade_id}"
                    )
            except Exception as e:
                bt.logging.error(f"[ERROR] Vali {state.dendrite.hotkey} Book {book_id} processing failed: {str(e)}")
                bt.logging.error(traceback.format_exc())

        bt.logging.debug(f"[LOOP] Respond completed in {time.time() - start:.2f}s")
        return response
    
    def entry_or_extend(self, response: FinanceAgentResponse, validator : str, book_id: int, direction:  OrderDirection)-> FinanceAgentResponse:
        """Queue an entry (or extend an existing position) in the trend direction.

        Args:
            response: The response to queue on.
            validator (str): Validator this state came from.
            book_id (int): Book to trade.
            direction (OrderDirection): Side the signal points.
        """
        self.directions[validator][book_id].direction = direction
        if self.directions[validator][book_id].open:    
            self.directions[validator][book_id].amount = self.directions[validator][book_id].amount + 1
        else:
            self.directions[validator][book_id].amount = 1
            self.directions[validator][book_id].open = True
        response.market_order(book_id, self.directions[validator][book_id].direction, self.quantity)
        return response


    def generate_exit_response(self, response: FinanceAgentResponse, validator : str, book_id: int) -> tuple[FinanceAgentResponse, float, float]:
        """Queue exits for any position the signal no longer supports.

        Returns:
            tuple: The response and whether an exit was queued.
        """
        self.directions[validator][book_id].open = False
        close_dir = (
            OrderDirection.BUY
            if self.directions[validator][book_id].direction == OrderDirection.SELL
            else OrderDirection.SELL
        )
        total_amount = self.quantity * self.directions[validator][book_id].amount
        self.directions[validator][book_id].amount = 0

        response.market_order(book_id, close_dir, total_amount)
        return response, total_amount, close_dir

    def onEnd(self, event:  SimulationEndEvent):
        """Handle simulation end by resetting per-validator strategy state."""
        bt.logging.info("[SIMULATION END] Clearing history")
        for validator in list(self.predictors.keys()):
            self.reset(validator)

    def reset(self, validator : str):
        """Reset per-validator strategy state.

        Args:
            validator (str): Validator whose state to clear.
        """
        for book_id in self.predictors[validator].keys():
            self.predictors[validator][book_id] = {key: [] for key in self.predKeys}
            self.midquotes[validator][book_id] = [TimestampedPrice(0, self.simulation_config.init_price)]
            self.directions[validator][book_id] = Positions(open=False, direction=OrderDirection.BUY, amount=0)

if __name__ == "__main__":
    """
    Example command for local standalone execution (standard mode):

    python MovingHurstAgent.py \
        --port 8888 \
        --agent_id 0 \
        --params \
            quantity=10.0 \
            expiry_period=120000000000 \
            rolling_window_min=60 \
            rolling_window_max=3600 \
            num_windows=20 \
            sampling_interval=1 \
            signal_threshold=0.5 \
            signal_tolerance=0.01 \
            model_threshold=0.4

    Example for advanced mode (uses more granular rolling window sampling for Hurst estimation):

    python MovingHurstAgent.py \
        --port 8888 \
        --agent_id 1 \
        --params \
            quantity=15.0 \
            expiry_period=180000000000 \
            rolling_window_min=120 \
            rolling_window_max=7200 \
            num_windows=30 \
            sampling_interval=2 \
            signal_threshold=0.55 \
            signal_tolerance=0.02 \
            model_threshold=0.45 \
            mode=advanced
    """
    launch(MovingHurstAgent)
