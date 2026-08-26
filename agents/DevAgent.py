# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT

"""
Volume-Bucket Momentum/Mean-Reversion Agent.

This agent categorizes market activity into high/low volume buckets across buy/sell
sides, predicts directional signals based on volume imbalances, and executes
momentum or mean-reversion trades depending on the relationship between volume
imbalance and recent price movement.

Strategy Logic:
    - HIGH volume imbalance with POSITIVE returns → Momentum (persist direction)
    - HIGH volume imbalance with NEGATIVE returns → Mean-reversion (reverse direction)
    - LOW confidence or strength → Hold/Exit positions
"""

import time
import traceback
import uuid
import numpy as np
import bittensor as bt
from dataclasses import dataclass
from enum import IntEnum

from taos.common.agents import launch
from taos.im.agents import (
    GenTRXAgent,
    Positions,
    RollingWindow,
    Signals,
    Thresholds,
    TimestampedPrice
)
from taos.im.protocol.events import SimulationEndEvent
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse

class VolumeCategory(IntEnum):
    """Categorization of trade volume magnitude.
    
    Attributes:
        HIGH: Trade volume exceeds configured threshold (aggressive market participation).
        LOW: Trade volume below threshold (passive market participation).
    """
    HIGH = 0
    LOW = 1


@dataclass
class VolumeBucket:
    """Aggregated volume statistics for a single direction and category.
    
    Attributes:
        high: Total volume of HIGH category trades.
        low: Total volume of LOW category trades.
        count: Number of trades observed in this bucket.
    """
    high: float = 0.0
    low: float = 0.0
    count: int = 0
    
    def update(self, volume: float, category: VolumeCategory) -> None:
        """Accumulate volume into the appropriate category bucket.
        
        Args:
            volume: Trade volume to add.
            category: Whether this trade was HIGH or LOW volume.
        """
        if category == VolumeCategory.HIGH:
            self.high += volume
        elif category == VolumeCategory.LOW:
            self.low += volume
        self.count += 1


@dataclass
class PredictorEntry:
    """Time-bucketed volume data for predictive modeling.
    
    Each entry represents volume activity within a sampling interval, separated
    by buy/sell direction and high/low volume categories.
    
    Attributes:
        buy: Volume bucket for buy-side trades.
        sell: Volume bucket for sell-side trades.
        timestamp: Simulation timestamp when this entry was created.
    """
    buy: VolumeBucket = VolumeBucket()
    sell: VolumeBucket = VolumeBucket()
    timestamp: int = 0
    
    def update(self, volume: float, direction: OrderDirection, category: VolumeCategory) -> None:
        """Route volume update to the appropriate directional bucket.
        
        Args:
            volume: Trade volume.
            direction: BUY or SELL side of the trade.
            category: HIGH or LOW volume classification.
        """
        if direction == OrderDirection.BUY:
            self.buy.update(volume, category)
        elif direction == OrderDirection.SELL:
            self.sell.update(volume, category)


@dataclass
class AgentDataStorage:
    """Per-validator, per-book state storage for the agent.
    
    Maintains rolling historical data needed for prediction and position tracking.
    All predictors and midquotes are stored in chronological order.
    
    Attributes:
        predictors: Rolling window of PredictorEntry observations.
        midquotes: Historical midquote prices with timestamps.
        last_signal: Most recently generated trading signal.
        positions: Current open position state (direction, quantity, amount).
    """
    predictors: list[PredictorEntry]
    midquotes: list[TimestampedPrice]
    last_signal: Signals = Signals.NOISE
    positions: Positions = Positions()
    
    def appendPredictor(self, entry: PredictorEntry) -> None:
        """Add a new predictor entry to the rolling window.
        
        Args:
            entry: PredictorEntry to append.
        """
        self.predictors.append(entry)
    
    def updateLastPredictor(self, volume: float, direction: OrderDirection, 
                           category: VolumeCategory) -> None:
        """Update the most recent predictor with new trade data.
        
        Args:
            volume: Trade volume to add.
            direction: BUY or SELL.
            category: HIGH or LOW volume.
        """
        self.predictors[-1].update(volume, direction, category)
    
    def newEntry(self, volume: float, direction: OrderDirection, category: VolumeCategory,
                timestamp: int, rolling_window: RollingWindow) -> None:
        """Add trade data, creating new predictor entry if sampling interval elapsed.
        
        Args:
            volume: Trade volume.
            direction: BUY or SELL.
            category: HIGH or LOW volume.
            timestamp: Current simulation timestamp.
            rolling_window: Configuration determining when to create new entries and prune old ones.
        """
        # Create new entry if none exist or sampling interval has elapsed
        if (len(self.predictors) == 0 or 
            timestamp - self.predictors[-1].timestamp > rolling_window.sampling_interval):
            self.appendPredictor(PredictorEntry(timestamp=timestamp))
        
        # Update the current entry
        self.updateLastPredictor(volume, direction, category)
        
        # Maintain rolling window size by pruning oldest entries
        while len(self.predictors) > rolling_window.max:
            self.predictors.pop(0)


@dataclass
class Prediction:
    """Directional prediction with confidence and strength metrics.
    
    Attributes:
        key: Human-readable identifier for this prediction type.
        direction: Predicted order direction (BUY or SELL).
        confidence: Model confidence in the prediction [0, 1].
        strength: Signal strength (positive=momentum, negative=mean-reversion).
        timestamp: When this prediction was generated.
    """
    key: str
    direction: OrderDirection
    confidence: float
    strength: float
    timestamp: int
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return (f"{self.key} = Strength({self.strength:.4f}) "
                f"Confidence({self.confidence:.4f}) "
                f"Signal({self.direction.name})")

class DevAgent(GenTRXAgent):
    """Volume-Bucket Momentum/Mean-Reversion Trading Agent.
    
    Strategy Overview:
        This agent operates on the principle that volume imbalances contain
        predictive information about future price movement, but the nature of
        that prediction depends on recent price action.
        
        1. **Data Collection**: Categorizes all market trades into buckets:
           - Direction: BUY vs SELL
           - Volume: HIGH vs LOW (relative to threshold)
           
        2. **Feature Engineering**: Computes volume imbalance ratio:
           imbalance = (sell_high - buy_high) / (sell_high + buy_high)
           
        3. **Signal Generation**: Combines volume imbalance with price returns:
           - Positive imbalance + Positive returns → Momentum (persist)
           - Positive imbalance + Negative returns → Mean-reversion (reverse)
           - Low confidence or weak strength → Hold/Exit
           
        4. **Execution**: Places market orders to enter, extend, or exit positions
           based on the generated signal.
    
    Configuration Parameters:
        quantity: Base order size for trades.
        expiry_period: Time-to-live for limit orders (unused in market order mode).
        rolling_window_min: Minimum observations required before generating signals.
        rolling_window_max: Maximum observations to retain in rolling window.
        num_windows: Number of sampling windows to maintain.
        sampling_interval: Seconds between predictor entry creations.
        model_threshold: Minimum confidence required to generate non-NOISE signal.
        signal_threshold: Strength threshold for ENTRY vs EXIT decisions.
        signal_tolerance: Band around signal_threshold to avoid whipsaw.
        category_threshold: Volume level separating HIGH from LOW category.
    """
    
    def initialize(self) -> None:
        """Initialize agent configuration, thresholds, rolling windows, and buffers.

        This method is called once during agent startup to set up all strategy
        parameters and data structures.
        """
        super().initialize()
        # Order sizing
        self.quantity = getattr(self.config, 'quantity', 2.0)
        self.expiry_period = getattr(self.config, 'expiry_period', 120e9)
        
        # Rolling window configuration for predictor observations
        self.rolling_window = RollingWindow(
            min=int(getattr(self.config, 'rolling_window_min', 10)),
            max=int(getattr(self.config, 'rolling_window_max', 20)),
            samples=20,
            num_windows=int(getattr(self.config, 'num_windows', 20))
        )
        
        # Decision thresholds for signal generation
        self.thresholds = Thresholds(
            model=getattr(self.config, 'model_threshold', 0.2),
            signal=getattr(self.config, 'signal_threshold', 0.0),
            tolerance=getattr(self.config, 'signal_tolerance', 0.00)
        )
        
        # Time interval for creating new predictor entries (seconds)
        self.sampling_interval = int(getattr(self.config, 'sampling_interval', 60))
        
        # Volume threshold for HIGH/LOW categorization (absolute volume units)
        self.volumeCategoryThreshold = getattr(self.config, 'category_threshold', 1.5)
        
        # Internal state storage: buffers[validator_hotkey][book_id] → AgentDataStorage
        self.buffers: dict[str, dict[int, AgentDataStorage]] = {}
        
        # Filter to only process trades from other agents (exclude own trades)
        self.filterRange = [id_ for id_ in range(256) if id_ != self.uid]
        
        # Event history cache per validator to avoid redundant reconstructions
        self.book_event_history: dict[str, EventHistory | None] = {}
    
    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """Process market state update and generate trading instructions.
        
        This is the core strategy loop, called each time the validator sends a
        new market state update. The workflow is:
        
        1. Extract market data (bids, asks, midquote)
        2. Update volume predictors from trade history
        3. Generate directional prediction from volume imbalances
        4. Convert prediction into trading signal (ENTRY/EXIT/HOLD)
        5. Execute appropriate orders based on signal and position state
        
        Args:
            state: Current simulation state containing books, accounts, notices, etc.
        
        Returns:
            Response object containing all trading instructions for this update.
        """
        response = self.make_response()  # mode-aware: emits exchange or simulation instructions
        start = time.time()
        
        # Process each book independently
        for book_id, book in state.books.items():
            try:
                # Calculate current midquote from best bid/ask
                bestBid = book.bids[0].price if book.bids else 0.0
                bestAsk = (book.asks[0].price if book.asks 
                          else bestBid + 10 ** (-self.simulation_config.priceDecimals))
                midquote = (bestBid + bestAsk) / 2
                
                # Ensure data structures exist for this validator and book
                self.buffersInit(state, book_id)
                
                # Update volume predictors from recent trade history
                self.update_predictors(state.dendrite.hotkey, book, state.timestamp)
                buffer = self.buffers[state.dendrite.hotkey][book_id]
                
                # Skip prediction if insufficient historical data
                if len(buffer.predictors) < self.rolling_window.min:
                    # Emergency position opening if completely inactive and time-based trigger hits
                    # This ensures minimum trading activity to avoid appearing offline
                    if buffer.positions.open and state.timestamp % 600e9 == 0:
                        response = self.entry_or_extend(
                            response, state.dendrite.hotkey, book_id, OrderDirection.BUY
                        )
                        bt.logging.info(
                            f"BOOK {book_id} | No signals received, emergency position open"
                        )
                        continue
                    
                    bt.logging.info(
                        f"BOOK {book_id} | Insufficient data: "
                        f"{len(buffer.predictors)}/{self.rolling_window.min} observations"
                    )
                    continue
                
                # Only generate signals at configured intervals to avoid overtrading
                if (state.timestamp // 1e9) % self.rolling_window.min != 0:
                    continue
                
                # Generate prediction and convert to trading signal
                prediction = self.predict(buffer)
                signal = self.signal(prediction)
                trade_id = str(uuid.uuid4())
                
                bt.logging.info(
                    f"[SIGNAL] Book={book_id} {prediction} "
                    f"Signal={signal.name} Midquote={midquote:.4f} TradeID={trade_id}"
                )
                
                # Update buffer state
                buffer.last_signal = signal
                buffer.midquotes.append(
                    TimestampedPrice(
                        (state.timestamp // 1e9) // self.rolling_window.min, 
                        midquote
                    )
                )
                
                # Execute trading logic based on signal
                if signal == Signals.ENTRY:
                    # ENTRY signal: Open new position or extend existing one
                    # Direction from prediction determines long (BUY) or short (SELL)
                    response = self.entry_or_extend(
                        response, state.dendrite.hotkey, book_id, prediction.direction
                    )
                    bt.logging.debug(
                        f"[TRADE] ENTRY Vali={state.dendrite.hotkey} Book={book_id} "
                        f"Prediction={prediction} Amount={self.quantity} "
                        f"Midquote={midquote:.4f} TradeID={trade_id}"
                    )
                
                elif signal == Signals.EXIT and buffer.positions.open:
                    # EXIT signal: Close all open positions in this book
                    # Only execute if we actually have an open position
                    response, total_amount, close_dir = self.generate_exit_response(
                        response, state.dendrite.hotkey, book_id
                    )
                    bt.logging.debug(
                        f"[TRADE] EXIT Vali={state.dendrite.hotkey} Book={book_id} "
                        f"Direction={close_dir} Prediction={prediction} "
                        f"Amount={total_amount} Midquote={midquote:.4f} TradeID={trade_id}"
                    )
            
            except Exception as e:
                bt.logging.error(
                    f"[ERROR] Vali {state.dendrite.hotkey} Book {book_id} "
                    f"processing failed: {str(e)}"
                )
                bt.logging.error(traceback.format_exc())
        
        bt.logging.debug(f"[LOOP] Respond completed in {time.time() - start:.2f}s")
        return response
    
    def buffersInit(self, state: MarketSimulationStateUpdate, book_id: int) -> None:
        """Initialize data buffers for a validator-book pair if not already present.
        
        Args:
            state: Current state update containing validator hotkey.
            book_id: Book identifier to initialize.
        """
        if state.dendrite.hotkey not in self.buffers:
            self.buffers[state.dendrite.hotkey] = {}
        
        if book_id not in self.buffers[state.dendrite.hotkey]:
            self.buffers[state.dendrite.hotkey][book_id] = AgentDataStorage(
                predictors=[PredictorEntry()],
                midquotes=[TimestampedPrice(price=self.simulation_config.init_price)]
            )
    
    def update_predictors(self, validator: str, book: Book, timestamp: int) -> None:
        """Extract trade history and update volume predictor buckets.
        
        Processes all trades from the event history, categorizing each by:
        - Direction (BUY vs SELL)
        - Volume category (HIGH vs LOW based on threshold)
        
        Only trades from other agents (excluding self) are included to avoid
        self-fulfilling predictions from own order flow.
        
        Args:
            validator: Validator hotkey identifier.
            book: Book object containing event history.
            timestamp: Current simulation timestamp.
        """
        # Initialize or update event history for this validator
        if validator not in self.book_event_history or not self.book_event_history[validator]:
            # Calculate lookback window (at least 2x sampling interval or 1 minute minimum)
            lookback_minutes = max(
                (self.simulation_config.publish_interval // 1_000_000_000) // 60,
                self.sampling_interval * 2 // 60,
                self.rolling_window.min
            )
            self.book_event_history[validator] = book.event_history(
                timestamp, self.simulation_config, lookback_minutes
            )
        else:
            # Incrementally append new events to existing history
            book.append_to_event_history(
                timestamp, self.book_event_history[validator], self.simulation_config
            )
        
        # Process all trades within the current sampling window
        trades = self.book_event_history[validator].trades
        for ts, tradeInfo in trades.items():
            # Skip empty trades or trades from previous publish intervals
            if not tradeInfo or ts <= timestamp - self.simulation_config.publish_interval:
                continue
            
            # Only process trades from other agents (exclude own trades)
            if tradeInfo.taker_agent_id in self.filterRange:
                # Categorize trade volume as HIGH or LOW
                volCat = (VolumeCategory.HIGH if tradeInfo.quantity > self.volumeCategoryThreshold 
                         else VolumeCategory.LOW)
                
                # Add to rolling predictor window
                self.buffers[validator][book.id].newEntry(
                    tradeInfo.quantity, 
                    tradeInfo.side, 
                    volCat, 
                    timestamp, 
                    self.rolling_window
                )
        
        # NOTE: Order placement and cancellation events are available but not currently used
        # orders = self.book_event_history[validator].orders
        # cancellations = self.book_event_history[validator].cancellations
    
    def predict(self, buffer: AgentDataStorage) -> Prediction:
        """Generate directional prediction from volume imbalances and price returns.
        
        Strategy Logic:
            1. Compute volume imbalance: (sell_high - buy_high) / (sell_high + buy_high)
               - Positive → More aggressive selling pressure
               - Negative → More aggressive buying pressure
            
            2. Compute log returns over sampling interval
            
            3. Calculate strength = log_return * imbalance
               - Positive strength → Momentum signal (persist direction)
               - Negative strength → Mean-reversion signal (reverse direction)
            
            4. Confidence = |mean(imbalance)| across rolling window
        
        Example:
            - High sell volume + Positive returns → Strong momentum, keep selling
            - High sell volume + Negative returns → Mean reversion, flip to buying
        
        Args:
            buffer: Data storage containing predictor history and midquotes.
        
        Returns:
            Prediction object with direction, confidence, and strength.
        """
        # Calculate volume imbalance ratio for each predictor entry
        returns = []
        for p in buffer.predictors:
            # Imbalance: positive = sell pressure, negative = buy pressure
            imbalance = ((p.sell.high - p.buy.high) / 
                        (p.sell.high + p.buy.high) if (p.sell.high + p.buy.high) > 0 
                        else 0.0)
            returns.append(imbalance)
        
        returns = np.array(returns)
        mean_imbalance = np.mean(returns)
        
        # Direction based on imbalance: positive imbalance → BUY expected
        direction = OrderDirection.BUY if mean_imbalance > 0 else OrderDirection.SELL
        
        # Confidence is magnitude of average imbalance
        confidence = np.abs(mean_imbalance)
        
        # Calculate price return over sampling interval
        if len(buffer.midquotes) == 0:
            return Prediction('Missing', direction, 0, 0, 0)
        
        # Find price from sampling_interval seconds ago
        prev_price = buffer.midquotes[-1].price  # Default to current if not found
        ts = buffer.midquotes[-1].timestamp
        for midquote in reversed(buffer.midquotes):
            if ts - midquote.timestamp <= self.sampling_interval:
                prev_price = midquote.price
                break
        
        # Log return over sampling interval
        log_return = np.log(buffer.midquotes[-1].price / prev_price) if prev_price > 0 else 0.0
        
        # Strength combines return and imbalance:
        # - Positive: imbalance and return agree (momentum)
        # - Negative: imbalance and return disagree (mean-reversion)
        strength = log_return * mean_imbalance
        
        return Prediction(
            key='HighVol',
            direction=direction,
            confidence=confidence,
            strength=strength,
            timestamp=ts
        )
    
    def signal(self, prediction: Prediction) -> Signals:
        """Convert prediction into discrete trading signal using configured thresholds.
        
        Decision Logic:
            1. Low confidence → NOISE (no action)
            2. High strength (momentum) → EXIT (take profit or stop loss)
            3. Low/negative strength (reversion) → ENTRY (open or extend position)
            4. Otherwise → HOLD (maintain current state)
        
        Args:
            prediction: Prediction object with confidence and strength metrics.
        
        Returns:
            Trading signal (ENTRY, EXIT, HOLD, or NOISE).
        """
        # Insufficient confidence in prediction → do nothing
        if prediction.confidence < self.thresholds.model:
            return Signals.NOISE
        
        # Strong momentum signal → exit positions (take profit or stop loss)
        elif prediction.strength > self.thresholds.signal + self.thresholds.tolerance:
            return Signals.EXIT
        
        # Mean-reversion signal → enter or extend positions
        elif prediction.strength < self.thresholds.signal - self.thresholds.tolerance:
            return Signals.ENTRY
        
        # Within tolerance band → hold current state
        return Signals.HOLD
    
    def entry_or_extend(self, response: FinanceAgentResponse, validator: str, 
                       book_id: int, direction: OrderDirection, 
                       multiplier: float = 1.0) -> FinanceAgentResponse:
        """Open new position or extend existing position via market order.
        
        Args:
            response: Response object to append instructions to.
            validator: Validator hotkey identifier.
            book_id: Book to trade in.
            direction: BUY or SELL direction.
            multiplier: Size multiplier for the base quantity (default 1.0).
        
        Returns:
            Updated response object with market order instruction appended.
        """
        pos = self.buffers[validator][book_id].positions
        pos.direction = direction
        
        if pos.open:
            # Extend existing position
            pos.amount += 1
            pos.qty += self.quantity * multiplier
        else:
            # Open new position
            pos.amount = 1
            pos.qty = self.quantity * multiplier
            pos.open = True
        
        # Place market order to execute immediately at best available price
        response.market_order(book_id, pos.direction, self.quantity * multiplier)
        return response
    
    def generate_exit_response(self, response: FinanceAgentResponse, validator: str, 
                               book_id: int) -> tuple[FinanceAgentResponse, float, OrderDirection]:
        """Close all open positions in the specified book.
        
        Args:
            response: Response object to append instructions to.
            validator: Validator hotkey identifier.
            book_id: Book to close positions in.
        
        Returns:
            Tuple of (updated response, total quantity closed, closing direction).
        """
        pos = self.buffers[validator][book_id].positions
        pos.open = False
        
        # Close direction is opposite of position direction
        # BUY (0) → SELL (2), SELL (2) → BUY (0)
        close_dir = OrderDirection(abs(pos.direction - 2))
        total_amount = pos.qty
        pos.amount = 0
        
        # Place market order to close entire position
        response.market_order(book_id, close_dir, total_amount)
        return response, total_amount, close_dir
    
    def onEnd(self, event: SimulationEndEvent) -> None:
        """Handle simulation end event by clearing all historical data.
        
        Args:
            event: Simulation end event containing metadata.
        """
        bt.logging.info("[SIMULATION END] Clearing history")
        # NOTE: This should iterate over all validators, not just one
        for validator in self.buffers.keys():
            self.reset(validator)
    
    def reset(self, validator: str) -> None:
        """Reset all buffers for a specific validator to initial state.
        
        Args:
            validator: Validator hotkey to reset data for.
        """
        for book_id in self.buffers[validator].keys():
            self.buffers[validator][book_id] = AgentDataStorage(
                predictors=[PredictorEntry()],
                midquotes=[TimestampedPrice(price=self.simulation_config.init_price)]
            )

if __name__ == "__main__":
    """Launch the DevAgent in standalone mode for local testing.
    
    Example command:
        python DevAgent.py \\
            --port 8888 \\
            --agent_id 0 \\
            --params \\
                quantity=10.0 \\
                expiry_period=120000000000 \\
                rolling_window_min=60 \\
                rolling_window_max=3600 \\
                num_windows=20 \\
                sampling_interval=10 \\
                model_threshold=0.2 \\
                signal_threshold=0.0 \\
                signal_tolerance=0.01 \\
                category_threshold=1.5
    """
    launch(DevAgent)