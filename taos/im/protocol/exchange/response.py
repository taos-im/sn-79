# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import bittensor as bt
from pydantic import Field
from typing import Annotated, Union, List
from annotated_types import Len
from taos.im.protocol.instructions import UInt32
from taos.im.protocol.simulator import *
from taos.common.protocol import AgentResponse
from taos.im.protocol.exchange.instructions import PlaceMarketOrderInstruction, PlaceLimitOrderInstruction, CancelOrdersInstruction, CancelOrderInstruction
from taos.im.protocol.exchange.models import OrderDirection, STP, TimeInForce, OrderCurrency, LoanSettlementOption

ExchangeInstruction = Annotated[
    Union[PlaceMarketOrderInstruction, PlaceLimitOrderInstruction, CancelOrdersInstruction],
    Field(discriminator="type")
]

class ExchangeAgentResponse(AgentResponse):
    """
    Exchange agent response class.

    This class is used by miner agents to populate and attach responses to the
    `ExchangeStateUpdate.response` property in exchange mode. It encapsulates
    a list of instructions representing the agent's intended actions.

    Attributes:
        instructions (list[ExchangeInstruction]):
            A list of instructions that the miner agent wishes to execute.
            These can include market orders, limit orders or cancellations.
    """
    
    instructions: Annotated[
        list[ExchangeInstruction],
        Len(min_length=0, max_length=200_000)
    ] = []

    def market_order(
        self,
        book_id: UInt32,
        delegate: str,
        direction: OrderDirection,
        quantity: float,
        max_slippage: float,
        delay: int = 0,
        clientOrderId: UInt32 | None = None,
        stp: STP = STP.CANCEL_OLDEST,
        currency: OrderCurrency = OrderCurrency.ALPHA,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        sltp_std: float | None = None,
    ) -> None:
        """
        Add a market order instruction to the agent response.

        Args:
            book_id (UInt32): The ID of the order book to place the market order in (corresponds to netuid, kept as book_id for now).
            delegate (str): The ss58 address of the delegate hotkey to/from which stake is moved; the order draws on the alpha staked to this delegate.
            direction (OrderDirection): Direction of the order (OrderDirection.BUY or OrderDirection.SELL).
            quantity (float): Size of the order in `currency`.
            max_slippage (float): Maximum allowed slippage for the order.
            delay (int, optional): Delay in simulation nanoseconds which must elapse before the instruction is processed at the exchange.
                                This delay will be added to the delay calculated based on your response time to the validator.
                                Defaults to 0.
            clientOrderId (UInt32 | None, optional): Optional client-specified order ID for tracking.
            stp (STP, optional): Self-trade prevention strategy (`STP.NO_STP`, `STP.CANCEL_OLDEST`, `STP.CANCEL_NEWEST`, `STP.CANCEL_BOTH` or `STP.DECREASE_CANCEL`).
                                Defaults to STP.CANCEL_OLDEST.
            currency (OrderCurrency, optional): Currency to use for the order quantity (OrderCurrency.ALPHA or OrderCurrency.TAO).
                                If set to `OrderCurrency.TAO`, the `quantity` will be interpreted as the amount of TAO currency that the agent wishes to exchange.
                                The matching engine at the simulator will determine the corresponding ALPHA amount to assign based on the asset price at the time of execution.
                                Defaults to ALPHA.
            stop_loss (float | None, optional): Stop-loss offset as a signed fraction of the entry price (negative places the stop below entry, as for a BUY).
                                Defaults to None (no stop-loss).
            take_profit (float | None, optional): Take-profit offset as a signed fraction of the entry price (positive places the target above entry, as for a BUY).
                                Defaults to None (no take-profit).
            sltp_std (float | None, optional): Band width, as a fraction of price, for dynamically inferred stop-loss/take-profit placement.
                                Defaults to None.

        Returns:
            None
        """
        self.add_instruction(
            PlaceMarketOrderInstruction(
                agentId=self.agent_id,
                delay=delay,
                bookId=book_id,
                delegate=delegate,
                direction=direction,
                quantity=quantity,
                max_slippage=max_slippage,
                clientOrderId=clientOrderId,
                stp=stp,
                currency=currency,
                stop_loss=stop_loss,
                take_profit=take_profit,
                sltp_std=sltp_std,
            )
        )

    def limit_order(
        self,
        book_id: UInt32,
        delegate: str,
        direction: OrderDirection,
        quantity: float,
        price: float,
        delay: int = 0,
        clientOrderId: UInt32 | None = None,
        stp: STP = STP.CANCEL_OLDEST,
        postOnly: bool = False,
        timeInForce: TimeInForce = TimeInForce.GTC,
        expiryPeriod: int | None = None,
        settlement_option: LoanSettlementOption | int = LoanSettlementOption.NONE,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        sltp_std: float | None = None,
    ) -> None:
        """
        Add a limit order instruction to the agent response.

        Args:
            book_id (UInt32): The ID of the order book to place the limit order in.
            delegate (str): The ss58 address of the delegate hotkey to/from which stake is moved; the order draws on the alpha staked to this delegate.
            direction (OrderDirection): Direction of the order (BUY or SELL).
            quantity (float): Quantity of the asset to trade.
            price (float): Price at which to place the limit order.
            delay (int, optional): Delay in simulation nanoseconds which must elapse before the instruction is processed at the exchange.
                                This delay will be added to the delay calculated based on your response time to the validator.
                                Defaults to 0.
            clientOrderId (UInt32 | None, optional): Optional client-specified order ID for tracking.
            stp (STP, optional): Self-trade prevention strategy (`STP.NO_STP`, `STP.CANCEL_OLDEST`, `STP.CANCEL_NEWEST`, `STP.CANCEL_BOTH` or `STP.DECREASE_CANCEL`).
                                Defaults to STP.CANCEL_OLDEST.
            postOnly (bool, optional): If True, prevents the order from matching immediately.
                                If the limit order would match with any existing levels on the book at the time of processing,
                                the instruction is rejected and no trade or order placement will take place.
                                Defaults to False.
            timeInForce (TimeInForce, optional): Time-in-force option to be applied for the order (`TimeInForce.GTC`, `TimeInForce.GTT`, `TimeInForce.IOC`, `TimeInForce.FOK`).
                                Good Till Cancelled : Order remains on the book until cancelled by the agent, or executed in a trade.
                                Good Till Time : Order remains on the book for `expiryPeriod` simulation nanoseconds unless traded or cancelled before expiry.
                                Immediate Or Cancel : Any part of the order which is not immediately traded will be cancelled.
                                Fill Or Kill : If the order will not be executed in its entirety immediately upon receipt by the simulator, the order will be rejected.
                                Defaults to GTC.
            expiryPeriod (int | None, optional): Expiry period for GTT (Good Till Time) orders, in simulation nanoseconds.
            settlement_option (LoanSettlementOption | int, optional): Strategy for settling outstanding margin loans using the proceeds of this order.
                                    LoanSettlementOption.NONE : No loan repayments
                                    LoanSettlementOption.FIFO : Loans will be repaid, starting from the oldest
                                    int : An integer order id; this specifies that the proceeds of the order should be used to repay the loan associated with a specific order
                                Defaults to NONE.
            stop_loss (float | None, optional): Stop-loss offset as a signed fraction of the entry price (negative places the stop below entry, as for a BUY).
                                Defaults to None (no stop-loss).
            take_profit (float | None, optional): Take-profit offset as a signed fraction of the entry price (positive places the target above entry, as for a BUY).
                                Defaults to None (no take-profit).
            sltp_std (float | None, optional): Band width, as a fraction of price, for dynamically inferred stop-loss/take-profit placement.
                                Defaults to None.

        Returns:
            None

        Notes:
            - If `timeInForce` is GTT, `expiryPeriod` must be specified.
            - If `timeInForce` is IOC or FOK, `postOnly` must be False.
            - If `expiryPeriod` is specified but `timeInForce` is not GTT, expiry is ignored.
        """
        if timeInForce == TimeInForce.GTT and not expiryPeriod:
            bt.logging.error(
                "Invalid limit order parameters: If using TimeInForce.GTT, expiryPeriod must be specified."
            )
            return
        if timeInForce in [TimeInForce.IOC, TimeInForce.FOK] and postOnly:
            bt.logging.error(
                "Invalid limit order parameters: IOC/FOK orders cannot be postOnly."
            )
            return
        if timeInForce != TimeInForce.GTT and expiryPeriod:
            bt.logging.warning(
                "Limit order parameters: expiryPeriod is set without TimeInForce.GTT - expiry will be ignored."
            )

        self.add_instruction(
            PlaceLimitOrderInstruction(
                agentId=self.agent_id,
                delay=delay,
                bookId=book_id,
                delegate=delegate,
                direction=direction,
                quantity=quantity,
                price=price,
                clientOrderId=clientOrderId,
                stp=stp,
                postOnly=postOnly,
                timeInForce=timeInForce,
                expiryPeriod=expiryPeriod,
                settleFlag=settlement_option,
                stop_loss=stop_loss,
                take_profit=take_profit,
                sltp_std=sltp_std,
            )
        )

    def cancel_order(
        self, 
        book_id: UInt32, 
        order_id: UInt32, 
        quantity: float | None = None, 
        delay: int = 0
    ) -> None:
        """
        Add a cancellation instruction for a single order.

        Args:
            book_id (UInt32): The ID of the order book where the order exists.
            order_id (UInt32): The ID of the order to cancel.
            quantity (float | None, optional): Quantity (in BASE) to cancel (if None, cancels the entire order).
            delay (int, optional): Delay in simulation nanoseconds which must elapse before the instruction is processed at the exchange. 
                                This delay will be added to the delay calculated based on your response time to the validator.
                                Defaults to 0.

        Returns:
            None
        """
        self.add_instruction(
            CancelOrdersInstruction(
                agentId=self.agent_id, 
                delay=delay, 
                bookId=book_id, 
                cancellations=[CancelOrderInstruction(orderId=order_id, volume=quantity)]
            )
        )

    def cancel_orders(
        self, 
        book_id: UInt32, 
        order_ids: list[UInt32], 
        delay: int = 0
    ) -> None:
        """
        Add a cancellation instruction for multiple orders.

        Args:
            book_id (UInt32): The ID of the order book where the orders exist.
            order_ids (list[UInt32]): A list of order IDs to cancel.
            delay (int, optional): Delay in simulation nanoseconds which must elapse before the instruction is processed at the exchange. 
                                This delay will be added to the delay calculated based on your response time to the validator.
                                Defaults to 0.

        Returns:
            None
        """
        self.add_instruction(
            CancelOrdersInstruction(
                agentId=self.agent_id, 
                delay=delay, 
                bookId=book_id, 
                cancellations=[
                    CancelOrderInstruction(orderId=order_id, volume=None) 
                    for order_id in order_ids
                ]
            )
        )
        
class ExchangeResponseBatch(BaseModel):
    """
    Represents a batch of responses from agents.

    Attributes:
        responses (list[SimulatorAgentResponse]): List of agent responses.
    """
    responses: list[SimulatorAgentResponse]

    def __init__(self, responses: list[SimulatorAgentResponse]):
        """
        Initializes the response batch.

        Args:
        - responses: List of agent responses to be included in the batch.
        """
        instructions = []
        for response in responses:
            if response:
                instructions.extend(response.serialize())
        super().__init__(responses=instructions)

    def serialize(self) -> dict:
        """
        Serializes the batch of responses into a dictionary format.

        Returns:
        - A dictionary representation of the response batch.
        """
        return {
            "responses": [response.serialize() for response in self.responses]
        }