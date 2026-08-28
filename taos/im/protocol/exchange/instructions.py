# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Classes representing instructions that may be submitted by miner agents in a intelligent market simulation are defined here.
"""
from pydantic import PositiveFloat, NonNegativeInt, PositiveInt, NonNegativeFloat, Field, ConfigDict
from typing import Literal, Annotated
from taos.im.protocol.simulator import *
from taos.common.protocol import AgentInstruction, BaseModel
from taos.im.protocol.exchange.models import OrderDirection, STP, TimeInForce, OrderCurrency, LoanSettlementOption

UInt32 = Annotated[int, Field(ge=0, le=2**32 - 1)]

class ExchangeAgentInstruction(AgentInstruction):
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
    
class PlaceOrderInstruction(ExchangeAgentInstruction):
    """
    Base class representing an instruction by an agent to place an order.

    Attributes:
        bookId (UInt32): The ID of the book on which the order is to be placed.
        direction (Literal[OrderDirection.BUY, OrderDirection.SELL]): Indicates whether the order is to buy or sell.
        quantity (PositiveFloat): The size of the order to be placed in base currency.
        clientOrderId (UInt32 | None): User-assigned client ID associated with the order.
        stp (Literal[STP.CANCEL_OLDEST, STP.CANCEL_NEWEST, STP.CANCEL_BOTH, STP.DECREASE_CANCEL]): 
            Self-trade prevention strategy to be applied for the order.
        currency (Literal[OrderCurrency.ALPHA, OrderCurrency.TAO]): Currency in which the quantity is specified (ALPHA or TAO).
        settleFlag (Literal[LoanSettlementOption.NONE, LoanSettlementOption.FIFO] | NonNegativeInt):
            Strategy for settling outstanding margin loans using the proceeds of this order
            LoanSettlementOption.NONE : No loan repayments
            LoanSettlementOption.FIFO : Loans will be repaid, starting from the oldest
            NonNegativeInt : Specify a specific order id for which the associated loan will be repaid
    """
    bookId: UInt32
    direction: Literal[OrderDirection.BUY, OrderDirection.SELL]
    # WIRE NAME != FIELD NAME. payload() serialises these as "volume", "stpFlag" and "allowPartial" while
    # the fields are quantity, stp and allow_partial. Without the aliases below, re-validating an
    # instruction from its own wire form drops those values silently and applies the field default.
    # populate_by_name keeps the Python spelling working for direct construction.
    model_config = ConfigDict(populate_by_name=True)

    quantity: PositiveFloat = Field(alias="volume")
    clientOrderId: UInt32 | None
    delegate: str
    stp: Literal[STP.NO_STP, STP.CANCEL_OLDEST, STP.CANCEL_NEWEST, STP.CANCEL_BOTH,
                 STP.DECREASE_CANCEL] = Field(default=STP.CANCEL_OLDEST, alias="stpFlag")
    currency: Literal[OrderCurrency.ALPHA, OrderCurrency.TAO] = OrderCurrency.ALPHA
    leverage: NonNegativeFloat = 0.0
    settleFlag: Literal[LoanSettlementOption.NONE, LoanSettlementOption.FIFO] | NonNegativeInt = LoanSettlementOption.NONE
    
    def __str__(self):
        return f"{'BUY ' if self.direction == OrderDirection.BUY else 'SELL'} {self.quantity} ON BOOK {self.bookId}"
    
class PlaceMarketOrderInstruction(PlaceOrderInstruction):
    """
    Class representing an instruction by an agent to place a market order.

    Attributes:
        type (Literal['PLACE_ORDER_MARKET']): Fixed to 'PLACE_ORDER_MARKET'.
    """
    type: Literal['PLACE_ORDER_MARKET'] = 'PLACE_ORDER_MARKET'
    # A market SELL draws on ONE named delegate, so the executable size is that delegate's stake, not the
    # summed free balance an agent sizes against. True fills what the delegate can cover; False refuses
    # the shortfall rather than returning a short fill the agent did not ask for.
    allow_partial: bool = Field(default=True, alias="allowPartial")
    max_slippage: float | None = None
    stop_loss:   float | None = None
    take_profit: float | None = None
    sltp_std:    float | None = None

    def payload(self) -> dict:
        """The market-order fields the engine expects under ``payload``.

        Returns:
            dict: Direction, volume, book and the optional execution flags this order carries.
        """
        d = {
            "direction":     self.direction,
            "volume":        self.quantity,
            "max_slippage":  self.max_slippage if self.max_slippage is not None else 0.0,
            "bookId":        self.bookId,
            "clientOrderId": self.clientOrderId,
            "delegate":      self.delegate,
            # KEY MUST BE "stp": that is the name the C++ reads (MSGPACK_NVP("stp", stpFlag)). Emitting
            # the field's Python spelling instead leaves the engine on its struct default, silently.
            "stp":           self.stp,
            "currency":      self.currency,
            # Every field PlaceOrderMarketPayload declares must be sent. msgpack ignores keys it does
            # not find, so an omission leaves the C++ struct default: leverage would arrive 0 and
            # settleFlag FIFO, executing an order the miner asked for differently rather than refusing
            # it. settleFlag=NONE matches what the other three order paths send.
            "leverage":      self.leverage,
            "settleFlag":    self.settleFlag,
            "allowPartial":  self.allow_partial,
        }
        if self.stop_loss   is not None: d["stopLoss"]   = self.stop_loss
        if self.take_profit is not None: d["takeProfit"] = self.take_profit
        if self.sltp_std    is not None: d["sltpStd"]    = self.sltp_std
        return d
    
    def __str__(self):
        return f"{'BUY ' if self.direction == OrderDirection.BUY else 'SELL'} {self.quantity}{'' if self.currency==OrderCurrency.ALPHA else 'QUOTE'}@MARKET ON BOOK {self.bookId}"
        
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
    """
    type: Literal['PLACE_ORDER_LIMIT'] = 'PLACE_ORDER_LIMIT'
    price: PositiveFloat
    postOnly: bool = False
    allow_partial: bool = Field(default=True, alias="allowPartial")
    timeInForce: Literal[TimeInForce.GTC, TimeInForce.GTT, TimeInForce.IOC, TimeInForce.FOK] = TimeInForce.GTC
    expiryPeriod: PositiveInt | None = None
    stop_loss:   float | None = None
    take_profit: float | None = None
    sltp_std:    float | None = None

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
            "delegate": self.delegate,
            "postOnly" : self.postOnly,
            "allowPartial" : self.allow_partial,
            "timeInForce" : self.timeInForce,
            "expiryPeriod" : self.expiryPeriod,
            # See above: the C++ msgpack key is "stp", not "stpFlag".
            "stp" : self.stp,
            "leverage": self.leverage,
            # CURRENCY MUST TRAVEL ON THE LIMIT PATH. The C++ limit payload declares it, so an absent
            # key leaves the engine's default of ALPHA and a TAO-denominated volume is read in the wrong
            # unit -- a notional small enough to be refused for minimum order size.
            "currency": self.currency,
            "settleFlag": self.settleFlag
        }
        if self.stop_loss   is not None: d["stopLoss"]   = self.stop_loss
        if self.take_profit is not None: d["takeProfit"] = self.take_profit
        if self.sltp_std    is not None: d["sltpStd"]    = self.sltp_std
        return d
    
    def __str__(self):
        return f"{'BUY ' if self.direction == OrderDirection.BUY else 'SELL'} {self.quantity}@{self.price} ON BOOK {self.bookId}"
    
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
        
class CancelOrdersInstruction(ExchangeAgentInstruction):
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