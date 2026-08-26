# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Order-book imbalance agent: uses a rolling window of detailed LOB history to
compute bid/ask imbalance and drive limit order placement decisions.
Supports GenTRX distributed training via agent params.
"""
import time
import traceback
import bittensor as bt

from taos.common.agents import launch
from taos.im.utils import duration_from_timestamp
from taos.im.agents import StateHistoryManager
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse

from taos.im.agents import GenTRXAgent

class ImbalanceAgent(GenTRXAgent):
    """Example: trades in the direction of the book's order-flow imbalance."""
    def initialize(self):
        """
        Initializes properties, variables, and components needed by the agent.
        The fields attached to `self.config` are defined in the launch parameters.

        Fields:
            self.expiry_period (int): Time period (in simulation nanoseconds) after which limit orders expire.
            self.imbalance_depth (int | None): Depth of order book levels to consider for imbalance calculation (default=`None` => include all available levels).
            self.history_manager (StateHistoryManager): Tracks and manages historical market data for the agent.
        """
        # GenTRX is opt-in: only activates when explicitly configured.
        if not hasattr(self.config, 'gtx_training_enabled'):
            self.config.gtx_training_enabled = False
        if not hasattr(self.config, 'gtx_collect_data'):
            self.config.gtx_collect_data = False
        super().initialize()
        self.expiry_period: int = int(self.config.expiry_period)
        self.imbalance_depth: int = int(self.config.imbalance_depth) if hasattr(self.config, 'imbalance_depth') else None
        self.parallel_history_workers: int = int(self.config.parallel_history_workers) if hasattr(self.config, 'parallel_history_workers') else 0
        self.history_manager: StateHistoryManager = StateHistoryManager(
            history_retention_mins=self.config.history_retention_mins, 
            log_dir=self.log_dir,
            parallel_workers=self.parallel_history_workers
        )

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """
        Responds to a new market state update by analyzing order book imbalances 
        and generating buy/sell limit orders accordingly.

        Args:
            state (MarketSimulationStateUpdate): The current market state data 
                provided by the simulation validator.

        Returns:
            FinanceAgentResponse: A response object containing the list of 
                instructions (e.g., limit orders) to submit to the market.

        Process:
            1. Updates the historical state manager with the latest market data.
            2. Iterates over all order books in the market state.
            3. If sufficient history exists, computes the mean imbalance at the 
               configured depth.
            4. Places buy orders if imbalance is positive, or sell orders if 
               imbalance is negative.
        """
        # GenTRX: data collection + training trigger (runs even when training disabled).
        response = super().respond(state)
        validator = state.dendrite.hotkey        
        
        # If updating of history using previous state information is not done, wait for it to complete.
        # This is necessary to avoid changing the history object while determining trading actions.
        while self.history_manager.updating:
            # If hitting this, it is likely that the response will time out. 
            # In that case, you would need to upgrade hardware, increase parallel_history_workers,
            # or find other ways to optimize the process.
            bt.logging.info("Waiting for history update to complete...")
            time.sleep(0.5)
        # Process each order book in the current market state
        for book_id, book in state.books.items():
            try:
                # Check if sufficient history exists for this validator/book
                if (
                    validator in self.history_manager and
                    book_id in self.history_manager[validator] and
                    self.history_manager[validator][book_id].is_full()
                ):
                    # Construct history of imbalances (including the latest final state snapshot; see comment below) at the configured depth
                    imbalance_history = self.history_manager[validator][book_id].imbalance(self.imbalance_depth) | {state.timestamp : state.books[book_id].snapshot(state.timestamp).imbalance(self.imbalance_depth)}
                    # Compute the mean imbalance
                    mean_imbalance = sum(imbalance_history.values()) / len(imbalance_history)

                    # Place a BUY order if mean imbalance is positive
                    if mean_imbalance > 0.0:
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.BUY,
                            quantity=round(mean_imbalance, state.config.volumeDecimals),
                            price=book.asks[0].price,
                            stp=STP.CANCEL_BOTH,
                            timeInForce=TimeInForce.GTT,
                            expiryPeriod=self.expiry_period
                        )
                    # Place a SELL order if mean imbalance is negative
                    elif mean_imbalance < 0.0:
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.SELL,
                            quantity=round(-mean_imbalance, state.config.volumeDecimals),
                            price=book.bids[0].price,
                            stp=STP.CANCEL_BOTH,
                            timeInForce=TimeInForce.GTT,
                            expiryPeriod=self.expiry_period
                        )
            except Exception as ex:
                # Log detailed error information for debugging
                bt.logging.error(
                    f"VALI {validator} BOOK {book_id} : Exception while processing "
                    f"state at {duration_from_timestamp(state.timestamp)} (T={state.timestamp}) : {ex}\n"
                    f"{traceback.format_exc()}"
                )

        # Update historical market data with the latest state information in a background thread.
        # Note this means that the state history will always be lagged by one observation;
        # for this reason the final latest state information is included in the imbalance calculation above.
        self.history_manager.update_async(state.model_copy(deep=True))
        # Return the response containing any generated instructions
        return response

if __name__ == "__main__":
    """
    Example command for local standalone testing execution using Proxy:
    python ImbalanceAgent.py --port 8888 --agent_id 0 --params expiry_period=120000000000 history_retention_mins=10 imbalance_depth=10 parallel_history_workers=8
    """
    launch(ImbalanceAgent)