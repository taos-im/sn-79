# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Orders, cancellations and the order enums.

Split out of taos.im.protocol.models, which re-exports every name here; import from either.
"""
from typing import Any
from enum import IntEnum
from pydantic import Field
from taos.common.protocol import BaseModel


class OrderDirection(IntEnum):
    """
    Enum to represent order direction.

    Attributes:
        BUY (int): Associated with an order placed in the BUY direction.
        SELL (int): Associated with an order placed in the SELL direction.
    """
    BUY=0
    SELL=1


class STP(IntEnum):
    """
    Enum to represent self-trade prevention options.

    Attributes:
        NO_STP (int): No self-trade prevention.
        CANCEL_OLDEST (int): If self-trade would occur when placing an order, cancel the resting order.
        CANCEL_NEWEST (int): If self-trade would occur when placing an order, cancel the aggressive order.
        CANCEL_BOTH (int): If self-trade would occur when placing an order, cancel both orders.
        DECREASE_CANCEL (int): If self-trade would occur when placing an order, cancel the quantity of the smaller order from the larger.
    """
    NO_STP=0
    CANCEL_OLDEST=1
    CANCEL_NEWEST=2
    CANCEL_BOTH=3
    DECREASE_CANCEL=4


class TimeInForce(IntEnum):
    """
    Enum to represent order time-in-force options.

    Attributes:
        GTC (int): Order remains on the book until cancelled by the agent, or executed in a trade.
        GTT (int): Order remains on the book until specified expiry period elapses, unless traded or cancelled before expiry.
        IOC (int): Any part of the order which is not immediately traded will be cancelled.
        FOK (int): If the order will not be executed in its entirety immediately upon receipt by the simulator, the order will be rejected.
    """
    GTC=0
    GTT=1
    IOC=2
    FOK=3


class OrderCurrency(IntEnum):
    """
    Enum to represent the currency in which the quantity of an order is specified.

    The venue's names for the same two currencies are ALPHA and TAO; both are accepted as aliases, so one
    agent can use either spelling in either mode.

    Attributes:
        BASE (int): Quantity is specified in BASE currency, i.e. subnet alpha. Alias: ALPHA.
        QUOTE (int): Quantity is specified in QUOTE currency, i.e. TAO. Alias: TAO.
        ALPHA (int): Alias of BASE.
        TAO (int): Alias of QUOTE.
    """
    # Both spellings resolve in both trees and modes; the wire values are 0 (BASE/ALPHA) and 1 (QUOTE/TAO).
    # IntEnum makes a later name with an equal value an alias of the earlier one, so
    # `OrderCurrency.ALPHA is OrderCurrency.BASE` holds and either spelling resolves.
    BASE=0
    QUOTE=1
    ALPHA=0
    TAO=1


class LoanSettlementOption(IntEnum):
    """
    Enum to represent options for repayment of margin loans when submitting an order.

    Attributes:
        NONE (int): Do not settle outstanding margin loans with proceeds from this order.
        FIFO (int): Settle outstanding margin loans in a FIFO (First-In-First-Out) manner
                    using proceeds from this order.
    """
    NONE = -2
    FIFO = -1
    
    @classmethod
    def from_string(cls, name):
        """Parse a LoanSettlementOption from its name.

        Args:
            name: The option's name, e.g. ``'NONE'``.

        Returns:
            LoanSettlementOption: The matching option.
        """
        match name:
            case 'NONE':
                return LoanSettlementOption.NONE
            case 'FIFO':
                return LoanSettlementOption.FIFO
            case _:
                try:
                    order_id = int(name)
                    return order_id
                except Exception:
                    return None


class Order(BaseModel):
    """
    Represents an order.

    Attributes:
        type (str): The type of the instruction; fixed to `"o"` (used for parallelized history reconstruction).
        id (int): The ID of the order as assigned by the simulator.
        client_id (int | None): Optional agent-assigned identifier for the order.
        timestamp (int): Simulation timestamp at which the order was placed.
        quantity (float): The size of the order in base currency.
        side (int): The side of the book on which the order was attempted to be placed (`0=BID`, `1=ASK`).
        price (float | None): Price of the order (`None` for market orders).
        leverage (float): Leverage ratio applied to the order. Defaults to 0.0 (unleveraged).
    """
    y : str = "o"
    i : int = Field(alias='id')
    c : int | None = Field(alias='client_id', default=None)
    t : int = Field(alias='timestamp')
    q : float = Field(alias='quantity')
    s : int = Field(alias='side')
    p : float | None = Field(alias='price')
    l : float = Field(alias="leverage", default=0.0)

    @property
    def type(self) -> str:
        """Readable accessor for wire field ``y``."""
        return self.y

    @property
    def id(self) -> int:
        """Readable accessor for wire field ``i``; ``id`` is its serialized alias."""
        return self.i

    @property
    def client_id(self) -> int | None:
        """Readable accessor for wire field ``c``; ``client_id`` is its serialized alias."""
        return self.c

    @property
    def timestamp(self) -> int:
        """Readable accessor for wire field ``t``; ``timestamp`` is its serialized alias."""
        return self.t

    @property
    def quantity(self) -> float:
        """Readable accessor for wire field ``q``; ``quantity`` is its serialized alias."""
        return self.q

    @property
    def side(self) -> int:
        """Readable accessor for wire field ``s``; ``side`` is its serialized alias."""
        return self.s

    @property
    def price(self) -> float | None:
        """Readable accessor for wire field ``p``; ``price`` is its serialized alias."""
        return self.p
    
    @property
    def leverage(self) -> float:
        """Readable accessor for wire field ``l``; ``leverage`` is its serialized alias."""
        return self.l

    @classmethod
    def from_event(self, event : dict):
        """
        Method to extract model data from simulation event in the format required by the MarketSimulationStateUpdate synapse.
        """
        return Order(order_type="limit" if event['price'] else 'market', id=event['orderId'],client_id=event['clientOrderId'], timestamp=event['timestamp'],
                     quantity=event['volume'], side=event['direction'], price=event['price'], 
                     leverage=event['leverage'])

    @classmethod
    def from_json(self, json : dict):
        """
        Method to extract model data from simulation account representation in the format required by the MarketSimulationStateUpdate synapse.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        # Use field-name kwargs (i, c, t, q, s, p, l) rather than aliases.
        # pydantic v2 model_construct does not resolve aliases; alias kwargs
        # silently become extra attributes and the short-name fields stay
        # unset, which breaks model_dump → model_validate round-trips.
        return Order.model_construct(order_type="limit", i=json['i'], c=json['c'], t=json['t'],
                     q=json['q'], s=json['s'], p=json['p'],
                     l=json['l'])


class Cancellation(BaseModel):
    """
    Represents an order cancellation.

    Attributes:
        type (str): The type of instruction; fixed to `c` (used for parallelized history reconstruction).
        orderId (int): ID of the cancelled order.
        timestamp (int | None): Simulation timestamp at which the cancellation occurred.
        price (float | None): Price of the order that was cancelled.
        quantity (float | None): Quantity cancelled (None if the entire order was cancelled).
    """
    y : str = "c"
    i: int = Field(alias="orderId")
    t: int | None = Field(alias='timestamp', default=None)
    p: float | None = Field(alias="price", default=None)
    q: float | None = Field(alias="quantity")

    @property
    def type(self) -> str:
        """Readable accessor for wire field ``y``."""
        return self.y

    @property
    def orderId(self) -> int:
        """Readable accessor for wire field ``i``; ``orderId`` is its serialized alias."""
        return self.i

    @property
    def timestamp(self) -> int:
        """Readable accessor for wire field ``t``; ``timestamp`` is its serialized alias."""
        return self.t

    @property
    def price(self) -> float:
        """Readable accessor for wire field ``p``; ``price`` is its serialized alias."""
        return self.p

    @property
    def quantity(self) -> float | None:
        """Readable accessor for wire field ``q``; ``quantity`` is its serialized alias."""
        return self.q

    @classmethod
    def from_event(self, event : dict):
        """
        Method to extract model data from simulation event in the format required by the MarketSimulationStateUpdate synapse.
        """
        return Cancellation(orderId=event['orderId'], timestamp=event['timestamp'], price=event['price'], quantity=event['volume'])

    @classmethod
    def from_json(self, json : dict):
        """
        Method to extract model data from simulation event in the format required by the MarketSimulationStateUpdate synapse.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return Cancellation.model_construct(i=json['i'], t=json['t'], p=json['p'], q=json['q'])
