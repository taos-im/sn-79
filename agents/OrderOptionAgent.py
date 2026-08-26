# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
A simple example agent to demonstrate usage of advanced order options.
GenTRX distributed training is supported: pass gtx_training_enabled=true in --agent.params to opt in.
"""
import bittensor as bt

from taos.common.agents import launch
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol.events import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse

from taos.im.agents import GenTRXAgent
import random

class OrderOptionAgent(GenTRXAgent):
    """Example: exercises every optional order flag, used to verify flags survive the wire."""
    def initialize(self):
        """
        Initializes properties, variables and quantities that will be used by the agent.
        The fields attached to `self.config` are defined in the launch parameters.
        """
        # GenTRX is opt-in: only activates when explicitly configured.
        if not hasattr(self.config, 'gtx_training_enabled'):
            self.config.gtx_training_enabled = False
        if not hasattr(self.config, 'gtx_collect_data'):
            self.config.gtx_collect_data = False
        super().initialize()
        self.min_quantity = self.config.min_quantity
        self.max_quantity = self.config.max_quantity
        # Process config flags indicating which tests are to be run
        self.tests = {
            'PO' : bool(self.config.PO) if hasattr(self.config, 'PO') else None,
            'GTT' : bool(self.config.GTT) if hasattr(self.config, 'GTT') else None,
            'IOC' : bool(self.config.IOC) if hasattr(self.config, 'IOC') else None,
            'FOK' : bool(self.config.FOK) if hasattr(self.config, 'FOK') else None,
            'QUOTE' : bool(self.config.QUOTE) if hasattr(self.config, 'QUOTE') else None,
            'MARGIN' : bool(self.config.MARGIN) if hasattr(self.config, 'MARGIN') else None,
            'SLTP' : bool(self.config.SLTP) if hasattr(self.config, 'SLTP') else None,
        }
        # If no tests explicitly specified in launch parameters, assume all tests should be run
        if all([t is None for t in self.tests.values()]):
            self.tests = {k : True for k in self.tests}
        self.round = 0
        self.response = None

    def quantity(self):
        """
        Obtains a random quantity for order placement within the bounds defined by the agent strategy parameters.
        """
        return round(random.uniform(self.min_quantity,self.max_quantity),self.simulation_config.volumeDecimals)

    def leverage(self):
        """
        Leverage to request, which is mode-dependent: exchange mode runs with maxLeverage=0 and REFUSES a
        leveraged order at placement rather than executing it unleveraged, so this agent's demonstration
        of the leverage parameter has to stand down there. Simulation keeps the 1.0 it always used.
        """
        return 0.0 if self.exchange_mode else 1.0

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
        # GenTRX: data collection + training trigger (runs even when training disabled).
        response = super().respond(state)
        # Iterate over all the book realizations in the state message
        for book_id, book in state.books.items():
            # A BOOK CAN BE ONE-SIDED. Indexing bids[0]/asks[0] without checking assumes both sides are
            # always populated, which is not true in either mode and is routinely false in EXCHANGE mode,
            # where a sweep can clear a side outright.
            # "IndexError: list index out of range" on every such tick, so the agent stopped responding
            # to the validator entirely -- 8 tracebacks in one dwell, and the whole update was lost, not
            # just the one book.
            #
            # Skipping the book keeps every OTHER book in the same update working, which is the point:
            # one empty book must not cost the agent its whole response.
            if not book.bids or not book.asks:
                continue
            bid = book.bids[0].price
            ask = book.asks[0].price
            bidvol = book.bids[0].quantity
            askvol = book.asks[0].quantity
            # Prices that cross the spread — used by PO, IOC, and FOK post-only tests.
            bidpricePOFail = ask  # buy at ask → immediately matches → PO rejected
            askpricePOFail = bid  # sell at bid → immediately matches → PO rejected
            # Obtain a random quantity
            quantity = self.quantity()

            match self.round:
                case 0:
                    response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=quantity, price=bid*0.99, postOnly=True, clientOrderId=100 + book_id)
                    response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=ask*1.01, postOnly=True, clientOrderId=200 + book_id)

            if self.tests['QUOTE']:
                response.market_order(book_id=book_id, direction=OrderDirection.BUY, quantity=round(ask * (askvol / 2),self.simulation_config.quoteDecimals), currency=OrderCurrency.QUOTE)
                response.market_order(book_id=book_id, direction=OrderDirection.BUY, quantity=round(ask * askvol,self.simulation_config.quoteDecimals), currency=OrderCurrency.QUOTE)
                response.market_order(book_id=book_id, direction=OrderDirection.SELL, quantity=round(bid * (bidvol / 2),self.simulation_config.quoteDecimals), currency=OrderCurrency.QUOTE)
                response.market_order(book_id=book_id, direction=OrderDirection.SELL, quantity=round(bid * bidvol,self.simulation_config.quoteDecimals), currency=OrderCurrency.QUOTE)

            if self.tests['PO']:
                bidpricePO = bid
                askpricePO = ask
                # Place a buy order which is expected to be opened on the book
                response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=quantity, price=bidpricePO, postOnly=True)
                # Place a buy order which is expected to be rejected due to post-only limitation
                response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=quantity, price=bidpricePOFail, postOnly=True)
                # Place a sell order which is expected to be opened on the book
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=askpricePO, postOnly=True)
                # Place a sell order which is expected to be rejected due to post-only limitation
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=askpricePOFail, postOnly=True)

            if self.tests['GTT']:
                # Place a buy order with expiry in 10 seconds
                response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=quantity, price=bid, timeInForce=TimeInForce.GTT, expiryPeriod=10_000_000_000)
                # Place a sell order with expiry in 10 seconds
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=ask, timeInForce=TimeInForce.GTT, expiryPeriod=10_000_000_000)

                # Place a sell order with TimeInForce.GTT and no expiry (INVALID)
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=ask, timeInForce=TimeInForce.GTT)
                # Place a sell order without TimeInForce.GTT and expiry given (WARNING)
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=ask, timeInForce=TimeInForce.GTC, expiryPeriod=10000000000)

            if self.tests['IOC']:
                # Populate prices which are expected to trigger key scenarios in Immediate-or-cancel order handling
                bidpriceIOCFull = ask * 2.0    # cross decisively; ABSOLUTE +10 assumed large prices
                bidpriceIOCPartial = ask
                bidqtyIOCPartial = askvol * 2
                bidpriceIOCCancel = bid
                askpriceIOCFull = bid * 0.5    # cross decisively; bid-10 went NEGATIVE at alpha prices
                askpriceIOCPartial = bid
                askqtyIOCPartial = bidvol * 2
                askpriceIOCCancel = ask
                # Place a buy IOC order which is expected to be traded in full when processed by the simulator
                response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=quantity, price=bidpriceIOCFull, timeInForce=TimeInForce.IOC)
                # Place a buy IOC order which is expected to be partially traded when processed by the simulator
                response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=bidqtyIOCPartial, price=bidpriceIOCPartial, timeInForce=TimeInForce.IOC)
                # Place a buy IOC order which is expected not to be matched (therefore rejected in full due to IOC flag)
                response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=quantity, price=bidpriceIOCCancel, timeInForce=TimeInForce.IOC)
                # Place a sell IOC order which is expected to be traded in full when processed by the simulator
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=askpriceIOCFull, timeInForce=TimeInForce.IOC)
                # Place a sell IOC order which is expected to be partially traded when processed by the simulator
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=askqtyIOCPartial, price=askpriceIOCPartial, timeInForce=TimeInForce.IOC)
                # Place a sell IOC order which is expected not to be matched (therefore rejected in full due to IOC flag)
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=askpriceIOCCancel, timeInForce=TimeInForce.IOC)

                # Place an IOC order with postOnly=True (INVALID)
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=askpricePOFail, postOnly=True, timeInForce=TimeInForce.IOC)

            if self.tests['FOK']:
                # Populate prices and quantities which are expected to trigger key scenarios in Fill-or-kill order handling
                bidpriceFOKFull = ask * 2.0    # cross decisively; ABSOLUTE +10 assumed large prices
                bidpriceFOKPartial = ask
                bidqtyFOKPartial = book.asks[0].quantity * 2
                bidpriceFOKCancel = bid
                askpriceFOKFull = bid * 0.5    # cross decisively; bid-10 went NEGATIVE at alpha prices
                askpriceFOKPartial = bid
                askqtyFOKPartial = book.bids[0].quantity * 2
                askpriceFOKCancel = ask
                # Place a buy FOK order which is expected to be traded in full when processed by the simulator
                response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=quantity, price=bidpriceFOKFull, timeInForce=TimeInForce.FOK)
                # Place a buy FOK order which is expected to attempt partial trade when processed by the simulator (therefore rejected in full due to FOK flag)
                response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=bidqtyFOKPartial, price=bidpriceFOKPartial, timeInForce=TimeInForce.FOK)
                # Place a buy FOK order which is expected not to be matched (therefore rejected in full due to FOK flag)
                response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=quantity, price=bidpriceFOKCancel, timeInForce=TimeInForce.FOK)
                # Place a sell FOK order which is expected to be traded in full when processed by the simulator
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=askpriceFOKFull, timeInForce=TimeInForce.FOK)
                # Place a sell FOK order which is expected to attempt partial trade when processed by the simulator (therefore rejected in full due to FOK flag)
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=askqtyFOKPartial, price=askpriceFOKPartial, timeInForce=TimeInForce.FOK)
                # Place a sell FOK order which is expected not to be matched (therefore rejected in full due to FOK flag)
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=askpriceFOKCancel, timeInForce=TimeInForce.FOK)

                # Place an FOK order with postOnly=True (INVALID)
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=quantity, price=askpricePOFail, postOnly=True, timeInForce=TimeInForce.FOK)
                
            if self.tests['SLTP']:
                # Round 0: BUY market with SL=2% below fill, TP=5% above fill
                # Round 1: SELL market with SL=2% above fill, TP=5% below fill
                # Round 2: BUY limit with SL=3% / TP=6%
                # Round 3: SELL limit with SL=3% / TP=6%
                match self.round:
                    case 0:
                        # BUY market: SL 2% below fill price, TP 5% above fill price
                        response.market_order(book_id=book_id, direction=OrderDirection.BUY,
                                              quantity=quantity,
                                              stop_loss=-0.02, take_profit=0.05)
                        bt.logging.info("SLTP ROUND 0: BUY MARKET SL=-2% TP=+5%")
                    case 1:
                        # SELL market: SL 2% above fill price, TP 5% below fill price
                        response.market_order(book_id=book_id, direction=OrderDirection.SELL,
                                              quantity=quantity,
                                              stop_loss=0.02, take_profit=-0.05)
                        bt.logging.info("SLTP ROUND 1: SELL MARKET SL=+2% TP=-5%")
                    case 2:
                        # BUY limit: SL 3% below price, TP 6% above price
                        response.limit_order(book_id=book_id, direction=OrderDirection.BUY,
                                             quantity=quantity, price=bid,
                                             stop_loss=-0.03, take_profit=0.06)
                        bt.logging.info("SLTP ROUND 2: BUY LIMIT SL=-3% TP=+6%")
                    case 3:
                        # SELL limit: SL 3% above price, TP 6% below price
                        response.limit_order(book_id=book_id, direction=OrderDirection.SELL,
                                             quantity=quantity, price=ask,
                                             stop_loss=0.03, take_profit=-0.06)
                        bt.logging.info("SLTP ROUND 3: SELL LIMIT SL=+3% TP=-6%")

            if self.tests['MARGIN']:
                bt.logging.info(f"BOOK {book_id} ROUND {self.round} : QUOTE : {self.accounts[book_id].quote_balance.total} [LOAN {self.accounts[book_id].quote_loan} | COLLAT {self.accounts[book_id].quote_collateral}]")
                bt.logging.info(f"BOOK {book_id} ROUND {self.round} : BASE : {self.accounts[book_id].base_balance.total} [LOAN {self.accounts[book_id].base_loan} | COLLAT {self.accounts[book_id].base_collateral}]")
                match self.round:
                    case 0:
                        response.market_order(book_id=book_id, direction=OrderDirection.BUY, quantity=0.01, leverage=self.leverage())
                    case 1:
                        loans = list(self.accounts[book_id].loans.values())
                        if len(loans) > 0:
                            loan = list(self.accounts[book_id].loans.values())[0]
                            bt.logging.info(f"CLOSING POSITION FOR ORDER #{loan.order_id} | {loan}")
                            response.close_position(book_id=book_id, order_id=loan.order_id)
                        else:
                            bt.logging.warning(f"No loans for close position on book {book_id}!")
                    case 2:
                        response.market_order(book_id=book_id, direction=OrderDirection.SELL, quantity=0.01, leverage=self.leverage())
                    case 3:
                        loans = list(self.accounts[book_id].loans.values())
                        if len(loans) > 0:
                            loan = list(self.accounts[book_id].loans.values())[0]
                            bt.logging.info(f"CLOSING POSITION FOR ORDER #{loan.order_id} | {loan}")
                            response.close_position(book_id=book_id, order_id=loan.order_id)
                        else:
                            bt.logging.warning(f"No loans for close position on book {book_id}!")
                    case 4:
                        response.market_order(book_id=book_id, direction=OrderDirection.BUY, quantity=0.01, leverage=self.leverage())
                        response.market_order(book_id=book_id, direction=OrderDirection.BUY, quantity=0.01, leverage=self.leverage())
                    case 5:
                        for order_id, loan in self.accounts[book_id].loans.items():
                            bt.logging.info(f"CLOSING POSITION FOR ORDER #{order_id} | {loan}")
                        response.close_positions(book_id=book_id, order_ids=[order_id for order_id in self.accounts[book_id].loans])
                    case 6:
                        response.market_order(book_id=book_id, direction=OrderDirection.SELL, quantity=0.01, leverage=self.leverage())
                        response.market_order(book_id=book_id, direction=OrderDirection.SELL, quantity=0.01, leverage=self.leverage())
                    case 7:
                        for order_id, loan in self.accounts[book_id].loans.items():
                            bt.logging.info(f"CLOSING POSITION FOR ORDER #{order_id} | {loan}")
                        response.close_positions(book_id=book_id, order_ids=[order_id for order_id in self.accounts[book_id].loans])
                    case 8:
                        response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=0.01, price=ask*0.99, leverage=self.leverage(), clientOrderId=1000 + book_id)
        
        if self.response:
            self.response.instructions.extend(response.instructions)
            response = self.response.model_copy()
            self.response = None
        self.round += 1
        # Return the response with instructions appended
        # The response will be serialized and sent back to the validator for processing
        return response
    
    def onOrderAccepted(self, event):
        """Record which flagged order was accepted so the exercise can assert on it."""
        if event.clientOrderId in [100 + event.bookId, 200 + event.bookId]:
            if not self.response:
                self.response = self.make_response()  # mode-aware: emits exchange or simulation instructions
            self.response.cancel_order(event.bookId, event.orderId)
    
    def onTrade(self, event : TradeEvent, validator: str = None) -> None:
        """
        Handler for event where an order is traded in the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.TradeEvent): The event class representing a trade.

        Returns:
            None
        """
        if event.clientOrderId == 1000 + event.bookId:            
            for order_id, loan in self.accounts[event.bookId].loans.items():
                if order_id == event.makerOrderId:
                    if not self.response:
                        self.response = self.make_response()  # mode-aware: emits exchange or simulation instructions
                    self.response.close_position(book_id=event.bookId, order_id=order_id)
                    self.response.limit_order(book_id=event.bookId, direction=OrderDirection.SELL, quantity=0.01, price=self.history[-1].books[event.bookId].bids[0].price+0.01, leverage=self.leverage(), clientOrderId=2000 + event.bookId)
                    bt.logging.info(f"CLOSING POSITION FOR BUY LIMIT ORDER #{order_id} | {loan}")
        if event.clientOrderId == 2000 + event.bookId:            
            for order_id, loan in self.accounts[event.bookId].loans.items():
                if order_id == event.makerOrderId:
                    if not self.response:
                        self.response = self.make_response()  # mode-aware: emits exchange or simulation instructions
                    self.response.close_position(book_id=event.bookId, order_id=order_id)
                    bt.logging.info(f"CLOSING POSITION FOR SELL LIMIT ORDER #{order_id} | {loan}")
            

if __name__ == "__main__":
    """
    Example command for local standalone testing execution using Proxy:
    python OrderOptionAgent.py --port 8888 --agent_id 0 --params min_quantity=0.1 max_quantity=1.0 PO=1 GTT=1 IOC=1 FOK=1 QUOTE=1 MARGIN=1 SLTP=1

    SLTP test runs 4 rounds per book:
      Round 0: BUY market, SL=-2%, TP=+5%
      Round 1: SELL market, SL=+2%, TP=-5%
      Round 2: BUY limit, SL=-3%, TP=+6%
      Round 3: SELL limit, SL=+3%, TP=-6%
    """
    launch(OrderOptionAgent)