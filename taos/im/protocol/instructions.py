# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Finance agent instruction classes: limit/market order placement, order
cancellation, position closing, and agent reset for the intelligent markets protocol.
"""
from pydantic import PositiveFloat, NonNegativeInt, PositiveInt, NonNegativeFloat, Field, ConfigDict
from typing import Literal, Annotated
from taos.im.protocol.simulator import *
from taos.common.protocol import AgentInstruction, BaseModel
from taos.im.protocol.models import OrderDirection, STP, TimeInForce, OrderCurrency, LoanSettlementOption

UInt32 = Annotated[int, Field(ge=0, le=2**32 - 1)]

class FinanceAgentInstruction(AgentInstruction):
    """
    Base class representing an instruction submitted by an agent in an intelligent markets simulation.

    Attributes:
        agentId (int): The ID of the agent that submitted the instruction.
        delay (NonNegativeInt): The processing delay to be assigned to the instruction. 
            This is set by validators based on the actual response time of the miner, and determines 
            how many simulation steps will elapse after submission before the agent instruction is processed.
        type (Literal["PLACE_ORDER_MARKET", "PLACE_ORDER_LIMIT", "CANCEL_ORDERS", "CLOSE_POSITIONS", "RESET_AGENT"]): 
            String identifier for the type of the submitted instruction in the simulator.
    """
    agentId: UInt32
    delay: NonNegativeInt = 0
    type: Literal["PLACE_ORDER_MARKET", "PLACE_ORDER_LIMIT", "CANCEL_ORDERS", "CLOSE_POSITIONS", "RESET_AGENT"]
    
    def serialize(self) -> dict:
        """Serialize this instruction to the wire dict the engine consumes.

        Returns:
            dict: ``agentId``, ``delay``, ``type`` and this instruction's ``payload``.
        """
        return {
            "agentId": self.agentId,
            "delay": self.delay,
            "type": self.type,
            "payload": self.payload()
        }
    
    def __str__(self):
        return f"{self.type} ON BOOK {self.bookId} : {self.payload()}"
    
class PlaceOrderInstruction(FinanceAgentInstruction):
    """
    Base class representing an instruction by an agent to place an order.

    Attributes:
        bookId (UInt32): The ID of the book on which the order is to be placed.
        direction (Literal[OrderDirection.BUY, OrderDirection.SELL]): Indicates whether the order is to buy or sell.
        quantity (PositiveFloat): The size of the order to be placed in base currency.
        clientOrderId (UInt32 | None): User-assigned client ID associated with the order.
        stp (Literal[STP.CANCEL_OLDEST, STP.CANCEL_NEWEST, STP.CANCEL_BOTH, STP.DECREASE_CANCEL]): 
            Self-trade prevention strategy to be applied for the order.
        currency (Literal[OrderCurrency.BASE, OrderCurrency.QUOTE]): Currency in which the quantity is specified (BASE or QUOTE).
        leverage (NonNegativeFloat) : The amount of leverage to take for a margin order; the effective order quantity will be `(1+leverage)`
            e.g. an order placed for 1.0 BASE with 0.5 leverage will be placed for a total quantity of 1.5 BASE, where 0.5 is borrowed from the exchange.
        settleFlag (Literal[LoanSettlementOption.NONE, LoanSettlementOption.FIFO] | NonNegativeInt):
            Strategy for settling outstanding margin loans using the proceeds of this order
            LoanSettlementOption.NONE : No loan repayments
            LoanSettlementOption.FIFO : Loans will be repaid, starting from the oldest
            NonNegativeInt : Specify a specific order id for which the associated loan will be repaid
        delegate (str): The ss58 address of the delegate hotkey to/from which stake is moved; empty when not applicable.
        max_slippage (float | None): Maximum acceptable slippage as a fraction of the best price at execution; None (serialized as 0) means no price limit.
    """
    bookId: UInt32
    direction: Literal[OrderDirection.BUY, OrderDirection.SELL]
    # WIRE NAME != FIELD NAME, and pydantic drops the difference SILENTLY.
    #
    # payload() serialises these as "volume", "stpFlag" and "allowPartial"; the fields are quantity, stp
    # and allow_partial. Anything that re-validates an instruction from its own wire form therefore loses
    # them without raising, and the field default is applied instead.
    # 598 forwarded orders reached the engine with stpFlag=CO, whatever the miner asked for, because
    # {"stpFlag": 2} validates to stp=CANCEL_OLDEST while {"stp": 2} validates to CANCEL_NEWEST.
    #
    # The aliases make the wire spelling acceptable on input. populate_by_name keeps the Python spelling
    # working, which every existing call site uses (quantity=, stp=, allow_partial=), so this is additive.
    model_config = ConfigDict(populate_by_name=True)

    quantity: PositiveFloat = Field(alias="volume")
    clientOrderId: UInt32 | None
    stp: Literal[STP.CANCEL_OLDEST, STP.CANCEL_NEWEST, STP.CANCEL_BOTH, STP.DECREASE_CANCEL] = Field(
        default=STP.CANCEL_OLDEST, alias="stpFlag")
    currency: Literal[OrderCurrency.BASE, OrderCurrency.QUOTE] = OrderCurrency.BASE
    leverage: NonNegativeFloat = 0.0
    settleFlag: Literal[LoanSettlementOption.NONE, LoanSettlementOption.FIFO] | NonNegativeInt = LoanSettlementOption.NONE
    delegate: str = ""
    max_slippage: float | None = None

    
    def __str__(self):
        return f"{'BUY ' if self.direction == OrderDirection.BUY else 'SELL'} {self.quantity} ON BOOK {self.bookId}"
    
class PlaceMarketOrderInstruction(PlaceOrderInstruction):
    """
    Class representing an instruction by an agent to place a market order.

    Attributes:
        type (Literal['PLACE_ORDER_MARKET']): Fixed to 'PLACE_ORDER_MARKET'.
        stop_loss (float | None): Stop-loss offset as a signed fraction of the entry price; None places no stop-loss.
        take_profit (float | None): Take-profit offset as a signed fraction of the entry price; None places no take-profit.
    """
    type: Literal['PLACE_ORDER_MARKET'] = 'PLACE_ORDER_MARKET'
    stop_loss:   float | None = None
    take_profit: float | None = None

    def payload(self) -> dict:
        """The market-order fields the engine expects under ``payload``.

        Returns:
            dict: Direction, volume, book and the optional execution flags this order carries.
        """
        d = {
            "direction": self.direction,
            "volume": self.quantity,
            "bookId":self.bookId,
            "clientOrderId":self.clientOrderId,
            # KEY MUST BE "stp". The C++ reads MSGPACK_NVP("stp", stpFlag)
            # (ExchangeAgentMessagePayloads.hpp:139 and :310) and the simulation engine is fed by the
            # same msgpack path (engines/simulation.py:480 packs the response with msgpack.packb).
            # Emitting "stpFlag" meant msgpack never found the field and every simulation-mode order
            # kept the struct default STPFlag::CO, so a miner asking for CANCEL_NEWEST silently got
            # CANCEL_OLDEST. The key must stay "stp".
            "stp":self.stp,
            "currency":self.currency,
            "leverage":self.leverage,
            "settleFlag":self.settleFlag,
            "delegate": self.delegate,
            "max_slippage": self.max_slippage if self.max_slippage is not None else 0.0,
        }
        if self.stop_loss is not None:
            d["stopLoss"] = self.stop_loss
        if self.take_profit is not None:
            d["takeProfit"] = self.take_profit
        return d
    
    def __str__(self):
        return f"{'BUY ' if self.direction == OrderDirection.BUY else 'SELL'} {f'{1+self.leverage:.2f}x' if self.leverage > 0 else ''}{self.quantity}{'' if self.currency==OrderCurrency.BASE else 'QUOTE'}@MARKET ON BOOK {self.bookId}"
        
class PlaceLimitOrderInstruction(PlaceOrderInstruction):
    """
    Class representing an instruction by an agent to place a limit order.

    Attributes:
        type (Literal['PLACE_ORDER_LIMIT']): Fixed to 'PLACE_ORDER_LIMIT'.
        price (PositiveFloat): The price level at which the order is to be placed.
        postOnly (bool): Boolean flag specifying if the order should be placed with Post-Only enforcement.
        timeInForce (Literal[TimeInForce.GTC, TimeInForce.GTT, TimeInForce.IOC, TimeInForce.FOK]): 
            Time-In-Force option to be applied for the order.
        expiryPeriod (PositiveInt | None): The period in simulation time after which the order should
            be cancelled (valid only with `timeInForce = TimeInForce.GTT`).
        stop_loss (float | None): Stop-loss offset as a signed fraction of the entry price; None places no stop-loss.
        take_profit (float | None): Take-profit offset as a signed fraction of the entry price; None places no take-profit.
    """
    type: Literal['PLACE_ORDER_LIMIT'] = 'PLACE_ORDER_LIMIT'
    price: PositiveFloat
    postOnly: bool = False
    timeInForce: Literal[TimeInForce.GTC, TimeInForce.GTT, TimeInForce.IOC, TimeInForce.FOK] = TimeInForce.GTC
    expiryPeriod: PositiveInt | None = None
    stop_loss:   float | None = None
    take_profit: float | None = None

    def payload(self) -> dict:
        """The limit-order fields the engine expects under ``payload``.

        Returns:
            dict: Direction, volume, price, book and the optional execution flags this order carries.
        """
        d = {
            "direction": self.direction,
            "volume": self.quantity,
            "price": self.price,
            "bookId": self.bookId,
            "clientOrderId":self.clientOrderId,
            "postOnly" : self.postOnly,
            "timeInForce" : self.timeInForce,
            "expiryPeriod" : self.expiryPeriod,
            "stp" : self.stp,
            # Every key PlaceOrderLimitPayload declares must be present: msgpack skips what it does not
            # find, leaving the C++ struct default. An absent "currency" reads as BASE, so a
            # TAO-denominated order would execute as an ALPHA one.
            "currency":self.currency,
            "leverage":self.leverage,
            "settleFlag":self.settleFlag,
            "delegate": self.delegate,
        }
        if self.stop_loss is not None:
            d["stopLoss"] = self.stop_loss
        if self.take_profit is not None:
            d["takeProfit"] = self.take_profit
        return d
    
    def __str__(self):
        return f"{'BUY ' if self.direction == OrderDirection.BUY else 'SELL'} {f'{1+self.leverage:.2f}x' if self.leverage > 0 else ''}{self.quantity}@{self.price} ON BOOK {self.bookId}"
    
class CancelOrderInstruction(BaseModel):
    """
    Class representing an instruction by an agent to cancel an open limit order.

    Attributes:
        orderId (UInt32): The simulator-assigned ID of the order to be cancelled.
        volume (PositiveFloat | None): The quantity of the order that should be cancelled 
            (`None` to cancel the entire remaining order size).
    """
    orderId: UInt32
    volume: PositiveFloat | None

    def serialize(self) -> dict:
        """Serialize one cancellation target.

        Returns:
            dict: The order id, and the volume when this is a partial cancel.
        """
        return {
            "orderId" : self.orderId,
            "volume" : self.volume
        }
    
    def __str__(self):
        return f"CANCEL ORDER #{self.orderId}{' FOR ' + str(self.volume) if self.volume else ''}"
        
class CancelOrdersInstruction(FinanceAgentInstruction):
    """
    Class representing an instruction by an agent to cancel a list of open limit orders.

    Attributes:
        type (Literal['CANCEL_ORDERS']): Fixed to 'CANCEL_ORDERS'.
        bookId (UInt32): The ID of the book on which cancellations are to be performed.
        cancellations (list[CancelOrderInstruction]): A list of CancelOrderInstruction objects.
    """
    type: Literal['CANCEL_ORDERS'] = 'CANCEL_ORDERS'
    bookId: UInt32
    cancellations: list[CancelOrderInstruction]

    def payload(self) -> dict:
        """The cancellation batch the engine expects under ``payload``.

        Returns:
            dict: The serialized cancellations and the book they rest on.
        """
        return {
            "cancellations": [cancellation.serialize() for cancellation in self.cancellations],
            "bookId": self.bookId
        }
    
    def __str__(self):
        return "\n".join([f"{c} ON BOOK {self.bookId}" for c in self.cancellations])
    
class ClosePositionInstruction(BaseModel):
    """
    Class representing an instruction by an agent to close a margin position.

    Attributes:
        orderId (UInt32): The simulator-assigned ID of the order for which position is to be closed.
        volume (PositiveFloat | None): The quantity to be closed
            (`None` to close entire remaining position).
    """
    orderId: UInt32
    volume: PositiveFloat | None

    def serialize(self) -> dict:
        """Serialize one position-close target.

        Returns:
            dict: The order id of the position being closed, and its close parameters.
        """
        return {
            "orderId" : self.orderId,
            "volume" : self.volume
        }
    
    def __str__(self):
        return f"CLOSE POSITION FOR ORDER #{self.orderId}{' FOR ' + str(self.volume) if self.volume else ''}"
        
class ClosePositionsInstruction(FinanceAgentInstruction):
    """
    Class representing an instruction by an agent to close margin positions associated with a list of orders.

    Attributes:
        type (Literal['CLOSE_POSITIONS']): Fixed to 'CLOSE_POSITIONS'.
        bookId (UInt32): The ID of the book on which closures are to be performed.
        closes (list[ClosePositionInstruction]): A list of ClosePositionInstruction objects.
    """
    type: Literal['CLOSE_POSITIONS'] = 'CLOSE_POSITIONS'
    bookId: UInt32
    closes: list[ClosePositionInstruction]

    def payload(self) -> dict:
        """The close batch the engine expects under ``payload``.

        Returns:
            dict: The serialized closes and the book they belong to.
        """
        return {
            "closes": [close_position.serialize() for close_position in self.closes],
            "bookId": self.bookId
        }
    
    def __str__(self):
        return "\n".join([f"{c} ON BOOK {self.bookId}" for c in self.closes])
        
class ResetAgentsInstruction(FinanceAgentInstruction):
    """
    Class representing an instruction to reset an agent's accounts.
    This instruction can only be submitted by validators to handle deregistration of a miner.

    Attributes:
        type (Literal['RESET_AGENT']): Fixed to 'RESET_AGENT'.
        agentIds (list[UInt32]): List of IDs of the agents for which reset should be applied.
    """
    type: Literal['RESET_AGENT'] = 'RESET_AGENT'
    agentIds: list[UInt32]

    def payload(self) -> dict:
        """The reset target list the engine expects under ``payload``.

        Returns:
            dict: The agent ids to reset.
        """
        return {
            "agentIds": self.agentIds
        }
    
    def __str__(self):
        return f"RESET AGENTS {','.join(['#' + str(agentId) for agentId in self.agentIds])}"