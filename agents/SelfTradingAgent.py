# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
A simple example agent to demonstrate self-trade prevention behaviour.
"""
from taos.common.agents import launch
from taos.im.agents import GenTRXAgent
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse

import random

class SelfTradingAgent(GenTRXAgent):
    """Example: deliberately crosses its own orders, used to verify self-trade prevention."""
    def initialize(self):
        """
        Initializes properties, variables and quantities that will be used by the agent.
        The fields attached to `self.config` are defined in the launch parameters.
        """
        super().initialize()
        self.min_quantity = self.config.min_quantity
        self.max_quantity = self.config.max_quantity
        self.lastBid = None
        self.lastAsk = None
        self.lastQty = None
        self.stp = STP.DECREASE_CANCEL

    def quantity(self):
        """
        Obtains a random quantity for order placement within the bounds defined by the agent strategy parameters.
        """
        return round(random.uniform(self.min_quantity,self.max_quantity),self.simulation_config.volumeDecimals)

    def respond(self, state : MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """
        The main logic of the strategy executed when a new state is received from validator.
        Analyses the latest market state data and generates instructions to be submitted.

        Args:
            state (MarketSimulationStateUpdate): The current market state data 
                provided by the simulation validator.

        Returns:
            FinanceAgentResponse: A response object containing the list of 
                instructions (e.g., limit orders) to submit to the market.
        """
        # Initialize a response class associated with the current miner
        response = self.make_response()  # mode-aware: emits exchange or simulation instructions
        # Iterate over all the book realizations in the state message
        for book_id, book in state.books.items():
            # If we have already placed orders, set the prices such that we expect to trade against our own orders
            if self.lastAsk and self.lastBid:
                bidprice = self.lastAsk
                askprice = self.lastBid
                quantity = self.lastQty + 0.1
                self.lastAsk = None
                self.lastBid = None
            else:
                # If the book is populated (it of course always should be)
                if len(book.bids) > 0 and len(book.asks) > 0:
                    # Calculate placement prices for new orders to be a random distance between the current best bid and best ask
                    bidprice = round(random.uniform(book.bids[0].price+10**(-1*self.simulation_config.priceDecimals),book.asks[0].price-10**(-1*self.simulation_config.priceDecimals)),self.simulation_config.priceDecimals)
                    askprice = round(random.uniform(bidprice+10**(-1*self.simulation_config.priceDecimals),book.asks[0].price-10**(-1*self.simulation_config.priceDecimals)),self.simulation_config.priceDecimals)
                else:
                    # Otherwise, place orders within 0.05 of the 100.0 price level
                    bidprice = round(random.uniform(99.95,100.05),self.simulation_config.priceDecimals)
                    askprice = round(random.uniform(bidprice,100.05),self.simulation_config.priceDecimals)
                # Obtain a random quantity
                quantity = self.quantity()
                # Populate previous quantity and placement price values
                self.lastQty = quantity
                self.lastBid = bidprice
                self.lastAsk = askprice
            # If the agent can afford to place the buy order
            if self.accounts[book_id].quote_balance.free >= quantity * bidprice:
                # Attach a buy limit order placement instruction to the response
                response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=quantity, price=bidprice, stp=self.stp)
            else:
                print(f"Cannot place BUY order for {quantity}@{bidprice} : Insufficient quote balance!")
            # If the agent can afford to place the sell order
            if self.accounts[book_id].base_balance.free >= quantity:
                # Attach a sell limit order placement instruction to the response
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=askprice, stp=self.stp)
            else:
                print(f"Cannot place SELL order for {quantity}@{askprice} : Insufficient base balance!")
        # Return the response with instructions appended
        # The response will be serialized and sent back to the validator for processing
        return response

if __name__ == "__main__":
    """
    Example command for local standalone testing execution using Proxy:
    python SelfTradingAgent.py --port 8888 --agent_id 0 --params min_quantity=0.1 max_quantity=1.0
    """
    launch(SelfTradingAgent)