# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Random market-maker agent: places limit orders at random prices between the
best bid and ask. Supports GenTRX distributed training via agent params.
"""
import bittensor as bt
from taos.common.agents import launch
from taos.im.agents import GenTRXAgent
from taos.im.protocol.models import OrderDirection, STP, TimeInForce, LoanSettlementOption
import random

class RandomMakerAgent(GenTRXAgent):
    """Example: rests random passive quotes, useful as background liquidity."""
    def initialize(self):
        """
        Initializes properties, variables and quantities that will be used by the agent.
        The fields attached to `self.config` are defined in the launch parameters.
        """
        super().initialize()
        self.min_quantity  = self.config.min_quantity
        self.max_quantity  = self.config.max_quantity
        self.min_leverage  = self.config.min_leverage if hasattr(self.config, 'min_leverage') else 0.0
        self.max_leverage  = self.config.max_leverage if hasattr(self.config, 'max_leverage') else 0.0
        self.max_fee_rate  = self.config.max_fee_rate if hasattr(self.config, 'max_fee_rate') else 0.005
        self.expiry_period = self.config.expiry_period
        self.spread        = getattr(self.config, 'spread', 0.01)
        self.delegate      = getattr(self.config, 'delegate', '')
        self.open_orders   = {}

    def quantity(self):
        """
        Obtains a random quantity for order placement within the bounds defined by the agent strategy parameters.
        """
        return round(random.uniform(self.min_quantity, self.max_quantity), getattr(self.simulation_config, 'volumeDecimals', 8))

    def leverage(self):
        """
        Obtains a random leverage value for order placement within the bounds defined by the agent strategy parameters.
        """
        # EXCHANGE MODE RUNS WITH maxLeverage=0, so a leveraged order is REFUSED at placement rather
        # than executed unleveraged. Returning 0 here rather than at each call site also fixes the
        # `quantity() * (1 + leverage())` sizing below, which would otherwise inflate an order the
        # exchange will not accept. Simulation behaviour is unchanged.
        if self.exchange_mode:
            return 0.0
        return round(random.uniform(self.min_leverage, self.max_leverage), 2)

    def respond_simulation(self, state):
        """Simulation-mode branch of this agent's per-step response; the strategy itself is shared."""
        price_decimals  = getattr(self.simulation_config, 'priceDecimals',  8)
        volume_decimals = getattr(self.simulation_config, 'volumeDecimals', 8)
        tif           = TimeInForce.GTT
        expiry_period = self.expiry_period
        # Initialize a response class associated with the current miner
        response = self.make_response()
        # Iterate over the book realizations in the state message
        books = state.books or {}
        for book_id, book in books.items():
            bids = book.bids if hasattr(book, 'bids') else []
            asks = book.asks if hasattr(book, 'asks') else []
            if bids and asks:
                bid, ask = bids[0].price, asks[0].price
            elif bids:
                bid = ask = bids[0].price
            elif asks:
                bid = ask = asks[0].price
            else:
                continue
            if book_id not in self.accounts:
                bt.logging.warning(f"No account data for book {book_id} — skipping")
                continue

            # NEW: Maker and taker fees will be dynamically moving under the DIS fee policy
            # The below demonstrates a simple approach for reacting to the changing rates in trading logic
            previous_maker_rate = self.accounts[book_id].fees.maker_fee_rate
            # Positive rate implies a fee is to be paid, negative rate results in rebates to the trader
            # Check the rate against the configured tolerance
            if previous_maker_rate > self.max_fee_rate:
                # If the rate exceeds the tolerance, cancel the order closest to the best level,
                bidOrders = {order.price : order for order in self.accounts[book_id].orders if order.side == OrderDirection.BUY}
                if bidOrders:
                    topBid = max(bidOrders.keys())
                    response.cancel_order(book_id, bidOrders[topBid].id)
                askOrders = {order.price : order for order in self.accounts[book_id].orders if order.side == OrderDirection.SELL}
                if askOrders:
                    topAsk = min(askOrders.keys())
                    response.cancel_order(book_id, askOrders[topAsk].id)
                continue

            # When bid == ask (empty book or AMM pool price), place orders at a
            # fixed spread around the price rather than between the levels.
            if bid == ask:
                bidprice = round(bid * (1 - self.spread), price_decimals)
                askprice = round(ask * (1 + self.spread), price_decimals)
            else:
                # Calculate placement prices for new orders to be a random distance between the current best bid and best ask
                bidprice = round(random.uniform(bid, ask), price_decimals)
                askprice = round(random.uniform(bidprice, ask), price_decimals)

            # If the bid and ask prices are different i.e the spread is not too small to place both orders at different prices
            if bidprice != askprice:
                bt.logging.info(f"BOOK {book_id} | QUOTE : {self.accounts[book_id].quote_balance.total} [LOAN {self.accounts[book_id].quote_loan} | COLLAT {self.accounts[book_id].quote_collateral}]")
                bt.logging.info(f"BOOK {book_id} | BASE : {self.accounts[book_id].base_balance.total} [LOAN {self.accounts[book_id].base_loan} | COLLAT {self.accounts[book_id].base_collateral}]")
                # BUY side
                # Obtain a random leverage value if there is no open margin position on sell side
                leverage   = self.leverage() if self.accounts[book_id].base_loan == 0 else 0.0
                # If an open opposite margin position exists, repay the corresponding loans in order
                # from oldest to newest by setting LoanSettlementOption.FIFO
                settlement = LoanSettlementOption.NONE if self.accounts[book_id].base_loan == 0 else LoanSettlementOption.FIFO
                # If placing unleveraged order, increase the quantity to better match the average total size of
                # leveraged orders on the other side.  This avoids accumulating too much inventory in one currency.
                quantity   = round(self.quantity() * (1 + self.leverage()), volume_decimals)
                # If the agent can afford to place the buy order
                if self.accounts[book_id].quote_balance.free >= quantity * bidprice:
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=quantity,
                        price=bidprice,
                        stp=STP.CANCEL_BOTH,
                        timeInForce=tif,
                        expiryPeriod=expiry_period,
                        leverage=leverage,
                        settlement_option=settlement
                    )
                    bt.logging.info(f"SUBMITTING BUY LIMIT ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}@{bidprice}")
                else:
                    bt.logging.error(f"CANNOT SUBMIT BUY ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}@{bidprice} : Insufficient quote balance!")
                # SELL side
                # Obtain a random leverage value if there is no open margin position on buy side
                leverage   = self.leverage() if self.accounts[book_id].quote_loan == 0 else 0.0
                # If an open opposite margin position exists, repay the corresponding loans in order
                # from oldest to newest by setting LoanSettlementOption.FIFO
                settlement = LoanSettlementOption.NONE if self.accounts[book_id].quote_loan == 0 else LoanSettlementOption.FIFO
                # If placing unleveraged order, increase the quantity to better match the average total size of
                # leveraged orders on the other side.  This avoids accumulating too much inventory in one currency.
                quantity   = round(self.quantity() * (1 + self.leverage()), volume_decimals)
                # If the agent can afford to place the sell order
                if self.accounts[book_id].base_balance.free >= quantity:
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=round(quantity * 1 + leverage, volume_decimals),
                        price=askprice,
                        stp=STP.CANCEL_NEWEST,
                        timeInForce=tif,
                        expiryPeriod=expiry_period,
                        leverage=leverage,
                        settlement_option=settlement
                    )
                    bt.logging.info(f"SUBMITTING SELL LIMIT ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}@{askprice}")
                else:
                    bt.logging.error(f"CANNOT SUBMIT SELL ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}@{askprice} : Insufficient base balance!")
        # Return the response with instructions appended
        # The response will be serialized and sent back to the validator for processing
        return response

if __name__ == "__main__":
    """
    Example command for local standalone testing execution using Proxy:
    python RandomMakerAgent.py --port 8888 --agent_id 0 --params min_quantity=0.1 max_quantity=1.0 min_leverage=0.0 max_leverage=1.0 expiry_period=200000000000 max_fee_rate=0.002
    """
    launch(RandomMakerAgent)
