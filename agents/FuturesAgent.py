# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
A simple example agent which places orders in line with the expectation of price movement due to the futures price connection in the simulator.
"""
import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import GenTRXAgent
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse

from taos.im.utils.streams import subscribe_coinbase_trades


import logging

class FuturesAgent(GenTRXAgent):
    """Example: leveraged directional strategy exercising loans and margin."""
    def initialize(self):
        """
        Initializes properties, variables and quantities that will be used by the agent.
        The fields attached to `self.config` are defined in the launch parameters.
        """
        super().initialize()
        # Quantity of BASE to attempt to buy/sell at each round
        self.quantity = self.config.quantity
        # Expiry period for limit orders in simulation timesteps (nanosecond scale)
        self.expiry_period = self.config.expiry_period 
        # Real-world (wall clock) period at which to sample futures price in seconds
        # Should be aligned with validator config value for `simulation.seeding.external.sampling_seconds`
        self.sampling_period = self.config.sampling_period 

        self.sampled_external_prices = []

        # Define handlers triggered when new values are received from the futures price stream
        def on_trade(trade : dict):
            """
            Triggered when a new trade message is received from the stream subscription
            This simple template strategy does not use the un-sampled trade messages
            A more advanced strategy may attempt to predict the external futures price movement and use this to modify placement logic
            """
            pass
        def on_sampled(trade : dict):
            """
            Triggered when a new sampled trade value is produces via the stream subscription.
            """
            bt.logging.info(f"New external trade : {trade}")
            # Add the new sampled value to the list for use in strategy logic
            self.sampled_external_prices.append(trade['price'])
        # Initiate the subscription to the external futures trade stream
        # The symbol should be aligned with the validator config value for `simulation.seeding.external.symbol.coinbase`
        subscribe_coinbase_trades(symbol='TAO-PERP-INTX', on_trade=on_trade, inactivity_threshold_secs=120,sampling_period=self.sampling_period, on_sampled=on_sampled)

    def respond(self, state : MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """
        The main logic of the strategy executed when a new state is received from validator.
        Analyses the latest market state data and generates instructions to be submitted.
        """
        # Initialize a response class associated with the current miner
        response = self.make_response()  # mode-aware: emits exchange or simulation instructions
        # Iterate over all the book realizations in the state message
        if len(self.sampled_external_prices) >= 2:
            for book_id, book in state.books.items():
                # Calculate the change in price (return) between the previous two sampled futures price observations
                price_change = self.sampled_external_prices[-1] - self.sampled_external_prices[-2]
                # A ONE-SIDED BOOK MUST NOT COST THE WHOLE RESPONSE. bids[0]/asks[0] were indexed
                # unguarded, so any book with an empty side raises IndexError and the agent loses the
                # entire update, every other book included. Routine in exchange mode, where a sweep can
                # clear a side outright.
                if price_change > 0 and book.asks:
                    # If the price change is positive, the simulator background agents are expected to drive the price higher over the next interval
                    # Buy the configured quantity of asset, limiting the purchase price to the current best ask
                    response.limit_order(book_id, OrderDirection.BUY, self.quantity, book.asks[0].price, timeInForce=TimeInForce.GTT, expiryPeriod=self.expiry_period)
                elif price_change < 0 and book.bids:
                    # If the price change is negative, the simulator background agents are expected to drive the price lower over the next interval
                    # Sell the configured quantity of asset, limiting the sale price to the current best bid
                    response.limit_order(book_id, OrderDirection.SELL, self.quantity, book.bids[0].price, timeInForce=TimeInForce.GTT, expiryPeriod=self.expiry_period)

        # Return the response with instructions appended
        # The response will be serialized and sent back to the validator for processing
        return response

if __name__ == "__main__":
    """
    Example command for local standalone execution alongside Proxy for testing:
    python FuturesAgent.py --port 8888 --agent_id 0 --params quantity=10.0 expiry_period=120000000000 sampling_period=60
    """
    logger = logging.getLogger('uvicorn.error')
    launch(FuturesAgent)
