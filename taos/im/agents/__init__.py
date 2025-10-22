# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import os
import msgpack
import traceback
import time
import csv
import bittensor as bt
from threading import Thread
from abc import ABC, abstractmethod
from taos.common.agents import SimulationAgent
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse, FinanceEventNotification
from taos.im.protocol.events import *
from taos.im.protocol.models import *
from taos.im.utils import duration_from_timestamp, timestamp_from_duration

# Base class for agents operating in intelligent market simulations
class FinanceSimulationAgent(SimulationAgent):
    def __init__(self, uid : int, config : object, log_dir : str | None = None) -> None:
        """
        Initializer method that sets up the agent's unique ID and configuration, and initializes common objects for storing agent data.

        Args:
            uid (int): The UID of the agent in the subnet.
            config (obj): Config object for the agent.

        Returns:
            None
        """
        self.history = []
        self.accounts = {}
        self.event_history : dict[str, AgentEventHistory | None] = {}
        if not hasattr(config, "lazy_load"):
            config.lazy_load = False
        else:
            config.lazy_load = bool(config.lazy_load)
        super().__init__(uid, config, log_dir)

    def handle(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        return super().handle(state)    

    def process(self, notification: FinanceEventNotification) -> FinanceEventNotification:
        """
        Method to handle a new event notification.
        """
        notification.acknowledged = True
        if notification.event.type == 'EVENT_SIMULATION_END':
            self.onEnd(notification.event)
        return notification
    
    def simulation_output_dir(self, state : MarketSimulationStateUpdate):
        simulation_output_dir = os.path.join(self.output_dir, state.dendrite.hotkey, state.config.simulation_id)
        os.makedirs(simulation_output_dir, exist_ok=True)
        return simulation_output_dir
    
    def load_event_history(self, state) -> None:
        """
        Load per-agent event history from CSV files into an AgentEventHistory object.

        This method:
        - Loads CSVs (orders, cancellations, trades).
        - Constructs event objects directly from CSV columns (no from_json).
        - Aggregates into an AgentEventHistory instance.
        - Applies optional retention logic based on `event_lookback_minutes`.

        Populates:
            self.event_history[state.dendrite.hotkey] (AgentEventHistory | None)
        """
        self.event_lookback_minutes = getattr(self.config, "event_lookback_minutes", 60)

        base_dir = self.simulation_output_dir(state)

        def _load_orders(path: str):
            events = []
            if os.path.isfile(path):
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            if row['price']:
                                ev = LimitOrderPlacementEvent(
                                    timestamp=timestamp_from_duration(row["timestamp"]),
                                    agentId=self.uid,
                                    bookId=int(row["bookId"]),
                                    orderId=int(row["orderId"]) if row["orderId"] else None,
                                    clientOrderId=int(row["clientOrderId"]) if row["clientOrderId"] else None,
                                    side=int(row["side"]),
                                    p=float(row["price"]),
                                    quantity=float(row["quantity"]),
                                    leverage=float(row["leverage"]),
                                    settleFlag=row.get("settleFlag"),
                                    success=row["success"].lower() == "true",
                                    message=row.get("message", ""),
                                )
                            else:
                                ev = MarketOrderPlacementEvent(
                                    timestamp=timestamp_from_duration(row["timestamp"]),
                                    agentId=self.uid,
                                    bookId=int(row["bookId"]),
                                    orderId=int(row["orderId"]) if row["orderId"] else None,
                                    clientOrderId=int(row["clientOrderId"]) if row["clientOrderId"] else None,
                                    side=int(row["side"]),
                                    r=(row["currency"] if row["currency"].isnumeric() else OrderCurrency[row["currency"].split('.')[1]]) if "currency" in row and row["currency"] else OrderCurrency.BASE,
                                    quantity=float(row["quantity"]),
                                    leverage=float(row["leverage"]),
                                    settleFlag=row.get("settleFlag"),
                                    success=row["success"].lower() == "true",
                                    message=row.get("message", ""),
                                )
                            events.append(ev)
                        except Exception as e:
                            bt.logging.warning(f"Failed to parse order row: {e}")
            return events

        def _load_cancellations(path: str):
            events = []
            if os.path.isfile(path):
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            ev = OrderCancellationEvent(
                                t=timestamp_from_duration(row["timestamp"]),
                                b=int(row["bookId"]),
                                o=int(row["orderId"]),
                                q=float(row["quantity"]) if row["quantity"] else None,
                                u=row["success"].lower() == "true",
                                m=row.get("message", ""),
                            )
                            events.append(ev)
                        except Exception as e:
                            bt.logging.warning(f"Failed to parse cancellation row: {e}")
            return events

        def _load_trades(path: str):
            events = []
            if os.path.isfile(path):
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            ev = TradeEvent(
                                timestamp=timestamp_from_duration(row["timestamp"]),
                                agentId=self.uid,
                                b=int(row["bookId"]) if row["bookId"] else None,
                                i=int(row["tradeId"]),
                                c=int(row["clientOrderId"]) if row["clientOrderId"] else None,
                                Ta=int(row["takerAgentId"]),
                                Ti=int(row["takerOrderId"]),
                                Tf=float(row["takerFee"]),
                                Ma=int(row["makerAgentId"]),
                                Mi=int(row["makerOrderId"]),
                                Mf=float(row["makerFee"]),
                                s=int(row["side"]),
                                p=float(row["price"]),
                                q=float(row["quantity"]),
                            )
                            events.append(ev)
                        except Exception as e:
                            bt.logging.warning(f"Failed to parse trade row: {e}")
            return events

        # Load everything
        orders = _load_orders(os.path.join(base_dir, "orders.csv"))
        cancels = _load_cancellations(os.path.join(base_dir, "cancellations.csv"))
        trades = _load_trades(os.path.join(base_dir, "trades.csv"))

        all_events = orders + cancels + trades
        if not all_events:
            self.event_history[state.dendrite.hotkey] = AgentEventHistory(
                uid=self.uid,
                start=state.timestamp - state.config.publish_interval,
                end=state.timestamp,
                events=[],
                publish_interval=self.simulation_config.publish_interval,
                retention_mins=self.event_lookback_minutes,
            )
        else:
            # Sort by timestamp
            all_events.sort(key=lambda e: e.timestamp)

            start = all_events[0].timestamp
            end = all_events[-1].timestamp
            self.event_history[state.dendrite.hotkey] = AgentEventHistory(
                uid=self.uid,
                start=start,
                end=end,
                events=all_events,
                publish_interval=getattr(self.config, "publish_interval", 1_000_000_000),
                retention_mins=self.event_lookback_minutes,
            )
        
    
    def log_order_event(self, event : LimitOrderPlacementEvent | MarketOrderPlacementEvent, state : MarketSimulationStateUpdate):
        """Log LimitOrderPlacementEvent or MarketOrderPlacementEvent to CSV."""
        orders_log_file = os.path.join(self.simulation_output_dir(state), 'orders.csv')
        file_exists = os.path.exists(orders_log_file)
        with open(orders_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'bookId', 'orderId', 'clientOrderId',
                    'side', 'price', 'currency', 'quantity', 'leverage', 'settleFlag',
                    'success', 'message'
                ])
            writer.writerow([
                duration_from_timestamp(event.timestamp),
                getattr(event, 'bookId', None),
                getattr(event, 'orderId', None),
                getattr(event, 'clientOrderId', None),
                getattr(event, 'side', None),
                getattr(event, 'price', None),
                getattr(event, 'currency', None),
                getattr(event, 'quantity', None),
                getattr(event, 'leverage', None),
                getattr(event, 'settleFlag', None),
                event.success,
                event.message
            ])

    def log_cancellation_event(self, event : OrderCancellationEvent, state : MarketSimulationStateUpdate):
        """Log OrderCancellationEvent to CSV."""
        cancellations_log_file = os.path.join(self.simulation_output_dir(state), 'cancellations.csv')
        file_exists = os.path.exists(cancellations_log_file)
        with open(cancellations_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'bookId', 'orderId', 'quantity', 'success', 'message'
                ])
            writer.writerow([
                duration_from_timestamp(event.timestamp),
                event.bookId,
                event.orderId,
                event.quantity,
                event.success,
                event.message
            ])

    def log_trade_event(self, event : TradeEvent, state : MarketSimulationStateUpdate):
        """Log TradeEvent to CSV."""
        trades_log_file = os.path.join(self.simulation_output_dir(state), 'trades.csv')
        file_exists = os.path.exists(trades_log_file)
        with open(trades_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'bookId', 'tradeId', 'clientOrderId',
                    'takerAgentId', 'takerOrderId', 'takerFee',
                    'makerAgentId', 'makerOrderId', 'makerFee',
                    'side', 'price', 'quantity'
                ])
            writer.writerow([
                duration_from_timestamp(event.timestamp),
                event.bookId,
                event.tradeId,
                event.clientOrderId,
                event.takerAgentId,
                event.takerOrderId,
                event.takerFee,
                event.makerAgentId,
                event.makerOrderId,
                event.makerFee,
                event.side,
                event.price,
                event.quantity
            ])

    def update(self, state : MarketSimulationStateUpdate) -> None:
        """
        Method to update the stored agent data, print relevant state information and trigger handlers for reported events.

        Args:
            state (taos.im.protocol.MarketSimulationStateUpdate): The UID of the agent in the subnet.

        Returns:
            None
        """
        self.history.append(state.model_copy())
        self.history = self.history[-10:]
        self.simulation_config = state.config
        self.accounts = state.accounts[self.uid]
        self.events = state.notices[self.uid]
        
        if not state.dendrite.hotkey in self.event_history or not self.event_history[state.dendrite.hotkey]:            
            self.load_event_history(state)        
        self.event_history[state.dendrite.hotkey].append(state)
        
        simulation_ended = False
        update_text = ''
        debug_text = ''
        update_text += "\n" + '-' * 50 + "\n"
        update_text += f'VALIDATOR : {state.dendrite.hotkey} | SIMULATION TIME : {duration_from_timestamp(state.timestamp)} (T={state.timestamp})' + "\n"
        update_text += '-' * 50 + "\n"
        if len(self.events) > 0:
            update_text += 'EVENTS' + "\n"
            update_text += '-' * 50 + "\n"
            for event in self.events:
                match event.type:
                    case"RESET_AGENTS" | "RA":
                        update_text += f"{event}" + "\n"
                    case "EVENT_SIMULATION_START" | "ESS":
                        update_text += f"{event}" + "\n"
                        self.onStart(event)
                    case "EVENT_SIMULATION_END" | "ESE":
                        simulation_ended = True
                    case _:
                        pass
        else:
            update_text += 'NO EVENTS' + "\n"
            update_text += '-' * 50 + "\n"
        for book_id in range(self.simulation_config.book_count):
            debug_text += '-' * 50 + "\n"
            debug_text += f"BOOK {book_id}" + "\n"
                     
            if not self.config.lazy_load:
                account= self.accounts[book_id]
                debug_text += '-' * 50 + "\n"
                debug_text += f"TOP LEVELS" + "\n"
                debug_text += '-' * 50 + "\n"
                debug_text += ' | '.join([f"{level.quantity:.4f}@{level.price}" for level in reversed(state.books[book_id].bids[:5])]) + '||' + ' | '.join([f"{level.quantity:.4f}@{level.price}" for level in state.books[book_id].asks[:5]]) + "\n"
                debug_text += '-' * 50 + "\n"
                debug_text += 'BALANCES' + "\n"
                debug_text += '-' * 50 + "\n"
                debug_text += f"BASE  : TOTAL={account.base_balance.total:.8f} FREE={account.base_balance.free:.8f} RESERVED={account.base_balance.reserved:.8f} | LOAN={account.base_loan:.8f} COLLATERAL={account.base_collateral}" + "\n"
                debug_text += f"QUOTE : TOTAL={account.quote_balance.total:.8f} FREE={account.quote_balance.free:.8f} RESERVED={account.quote_balance.reserved:.8f} | LOAN={account.quote_loan:.8f} COLLATERAL={account.quote_collateral}" + "\n"
                if len(account.orders) > 0:
                    debug_text += '-' * 50 + "\n"
                    debug_text += 'ORDERS' + "\n"
                    debug_text += '-' * 50 + "\n"
                    for order in sorted(account.orders, key=lambda x: x.timestamp):
                        debug_text += f"#{order.id} : {'BUY ' if order.side == 0 else 'SELL'} {f'{1+order.leverage:.2f}x' if order.leverage > 0 else ''}{order.quantity}@{order.price} [PLACED AT {duration_from_timestamp(order.timestamp)} (T={order.timestamp})]" + "\n"
                if len(account.loans) > 0:
                    debug_text += '-' * 50 + "\n"
                    debug_text += 'LOANS' + "\n"
                    debug_text += '-' * 50 + "\n"
                    for order_id, loan in account.loans.items():
                        debug_text += f"#{order_id} : {loan}\n"
                if account.fees:
                    debug_text += '-' * 50 + "\n"
                    debug_text += f'FEES : TRADED {account.fees.volume_traded} | MAKER {account.fees.maker_fee_rate * 100}% | TAKER {account.fees.taker_fee_rate * 100}%' + "\n"
                    debug_text += '-' * 50 + "\n"

            debug_text += '-' * 50 + "\n"
            debug_text += 'EVENTS' + "\n"
            debug_text += '-' * 50 + "\n"
            for event in self.events:
                if hasattr(event, 'bookId') and event.bookId == book_id:
                    if not event.type in ["EVENT_TRADE", "ET"]:
                        debug_text += f"{event}" + "\n"
                        update_text += f"BOOK {book_id} : {event}" + "\n"
                    match event.type:
                        case "RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT" | "RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET" | "RDPOL" | "RDPOM":
                            self.onOrderAccepted(event)
                            self.log_order_event(event, state)
                        case "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT" | "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET" | "ERDPOL" | "ERDPOM":
                            self.onOrderRejected(event)
                        case "RESPONSE_DISTRIBUTED_CANCEL_ORDERS" | "RDCO":
                            for cancellation in event.cancellations:
                                self.onOrderCancelled(cancellation)
                                self.log_cancellation_event(cancellation, state)
                        case "ERROR_RESPONSE_DISTRIBUTED_CANCEL_ORDERS" | "ERDCO":
                            for cancellation in event.cancellations:
                                self.onOrderCancellationFailed(cancellation)
                        case "RESPONSE_DISTRIBUTED_CLOSE_POSITIONS" | "RDCP":
                            for close in event.closes:
                                self.onPositionClosed(close)
                        case "ERROR_RESPONSE_DISTRIBUTED_CLOSE_POSITIONS" | "ERDCP":
                            for close in event.closes:
                                self.onPositionCloseFailed(close)
                        case "EVENT_TRADE" | "ET":
                            role = "taker" if self.uid == event.takerAgentId else "maker"
                            trade_text = f"{'BUY ' if event.side == 0 else 'SELL'} TRADE #{event.tradeId} : YOUR {'AGGRESSIVE' if role=='taker' else 'PASSIVE'} " + \
                                f"ORDER #{event.takerOrderId if role=='taker' else event.makerOrderId} (AGENT {event.takerAgentId if role=='taker' else event.makerAgentId}) " + \
                                f"MATCHED AGAINST #{event.makerOrderId if role=='taker' else event.takerOrderId} (AGENT {event.makerAgentId if role=='taker' else event.takerAgentId}) " + \
                                f"FOR {event.quantity}@{event.price} AT {duration_from_timestamp(event.timestamp)} (T={event.timestamp})"
                            debug_text += f"{trade_text}" + "\n"
                            update_text += f"BOOK {book_id} : {trade_text}" + "\n"
                            self.onTrade(event)
                            self.log_trade_event(event, state)
                        case _:
                            bt.logging.warning(f"Unknown event : {event}")
            debug_text += '-' * 50 + "\n"
        if simulation_ended:
            update_text += f"{event}" + "\n"
            update_text += '-' * 50 + "\n"
            self.onEnd(event)
        # bt.logging.debug("." + debug_text)
        # bt.logging.info("." + update_text)

    # Handler functions for various simulation events, to be overridden in agent implementations.
    def onStart(self, event : SimulationStartEvent) -> None:
        """
        Handler for simulation start event.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.SimulationStartEvent): The event class representing start of simulation.

        Returns:
            None
        """
        pass

    def onOrderAccepted(self, event : OrderPlacementEvent) -> None:
        """
        Handler for event where order is accepted to the book by simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderPlacementEvent): The event class representing order placement.

        Returns:
            None
        """
        pass

    def onOrderRejected(self, event : OrderPlacementEvent) -> None:
        """
        Handler for event where order is rejected for placement by simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderPlacementEvent): The event class representing order rejection.

        Returns:
            None
        """
        match event.message:
            case 'EXCEEDING_MAX_ORDERS':
                bt.logging.warning(f"FAILED TO PLACE {'BUY' if event.side == 0 else 'SELL'} ORDER FOR {event.quantity}@{event.price if event.type.endswith('L') else 'MARKET'} ON BOOK {event.bookId} : You already have the maximum allowed number of open orders ({self.simulation_config.max_open_orders}) on this book.  You will not be able to place any more orders until you either cancel existing orders, or they are traded.")
            case 'EXCEEDING_LOAN':
                bt.logging.warning(f"FAILED TO PLACE {'BUY' if event.side == 0 else 'SELL'} ORDER FOR {event.quantity}@{event.price if event.type.endswith('L') else 'MARKET'} ON BOOK {event.bookId} : You have exceeded the maximum allowed loan value ({self.simulation_config.max_loan}) on this book.  You need to close positions using the close_position method to repay some of the loan amount before you can place more leveraged orders.")
            case 'DUAL_POSITION':
                bt.logging.warning(f"FAILED TO PLACE {'BUY' if event.side == 0 else 'SELL'} ORDER FOR {event.quantity}@{event.price if event.type.endswith('L') else 'MARKET'} ON BOOK {event.bookId} : You already have a margin position in the opposite direction.  You must close this position before you can take leverage on this side of the book.")
            case _:
                pass

    def onOrderCancelled(self, event : OrderCancellationEvent) -> None:
        """
        Handler for event where order is cancelled in the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderCancellationEvent): The event class representing order cancellation.

        Returns:
            None
        """
        pass

    def onOrderCancellationFailed(self, event : OrderCancellationEvent) -> None:
        """
        Handler for event where order cancellation request is rejected by the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderCancellationEvent): The event class representing order cancellation.

        Returns:
            None
        """
        pass
    
    def onPositionClosed(self, event : ClosePositionEvent) -> None:
        """
        Handler for event where position is closed in the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.ClosePositionEvent): The event class representing position closure.

        Returns:
            None
        """
        pass

    def onPositionCloseFailed(self, event : ClosePositionEvent) -> None:
        """
        Handler for event where close position request is rejected by the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.ClosePositionEvent): The event class representing position closure.

        Returns:
            None
        """
        pass

    def onTrade(self, event : TradeEvent) -> None:
        """
        Handler for event where an order is traded in the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.TradeEvent): The event class representing a trade.

        Returns:
            None
        """
        pass

    def onEnd(self, event : SimulationEndEvent) -> None:
        """
        Handler for simulation end event.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.SimulationEndEvent): The event class representing end of simulation.

        Returns:
            None
        """
        pass

    def respond(self, state : MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """
        Abstract method for handling generation of response to new state update.  To be implemented by subclasses.

        Args:
            state (taos.im.protocol.MarketSimulationStateUpdate): The class representing the latest state of the simulation.

        Returns:
            taos.im.protocol.FinanceAgentResponse : The response which will be attached to the synapse for return to the querying validator.
        """
        ...

    def report(self, state : MarketSimulationStateUpdate, response : FinanceAgentResponse) -> None:
        """
        Method for reporting the latest simulation state and the response generated by the agent.

        Args:
            state (taos.im.protocol.MarketSimulationStateUpdate): The class representing the latest state of the simulation.
            response (taos.im.protocol.FinanceAgentResponse): The class representing the response of the agent.

        Returns:
            None
        """
        update_text = '-' * 50 + "\n"
        if len(response.instructions) > 0:
            update_text += 'INSTRUCTIONS' + "\n"
            update_text += '-' * 50 + "\n"
            for instruction in response.instructions:
                update_text += f"{instruction}" + "\n"
        else:
            update_text += 'NO INSTRUCTIONS TO SUBMIT' + "\n"
        update_text += '-' * 50
        # bt.logging.info(".\n" + update_text)

from taos.im.utils.history import history, batch_history
class StateHistoryManager:
    """
    Manages the state history for market simulations, including reconstruction of L2 states,
    constructing and maintaining histories over multiple state updates, as well as
    saving and loading of history data for later retrieval.
    """

    def __init__(self, history_retention_mins: int, log_dir: str, depth: int = 21, parallel_workers: int = 0, save: bool = False):
        """
        Initialize the StateHistoryManager.

        Args:
            history_retention_mins (int): Retention period for history in minutes.
            log_dir (str): Directory to store history files.
            depth (int): Number of levels of the order book to retain in history.
            parallel_workers (int): Number of parallel processes for reconstruction; if 0, disables parallelization.
            save (bool): If True, persist history to file after each update.
        """
        self.history_retention_mins: int = history_retention_mins  # Retention duration for history data
        self.log_dir: str = log_dir  # Directory path for log and history files
        self.state_file: str = os.path.join(log_dir, "history.mp")  # Path to serialized state file

        self.gap: dict[str, dict[int, bool]] = {}  # Tracks gaps in validator/book histories
        self.depth: int = depth  # Number of book levels to store
        self.parallel_workers: int = parallel_workers  # Number of processes for parallel reconstruction
        self.parallel: bool = parallel_workers > 0  # Whether parallel processing is enabled

        self.should_save: bool = save  # Whether to automatically save history after updates
        self.saving: bool = False  # Flag: True if a save operation is in progress
        self.updating: bool = False  # Flag: True if an update operation is in progress

        self.last_snapshot: dict[str, dict[int, L2Snapshot]] = {}  # Last known snapshot per validator/book
        self.history: dict[str, dict[int, L2History]] = {}  # Full history per validator/book
        self.publish_interval: int = None # Publishing interval

        self.load()  # Attempt to load existing history from disk

    def update(self, state: MarketSimulationStateUpdate) -> None:
        """
        Update the history with a new market simulation state.

        Args:
            state (MarketSimulationStateUpdate): The latest simulation state to process.
        """
        self.updating = True
        self.publish_interval = state.config.publish_interval
        try:
            validator: str = state.dendrite.hotkey  # Validator identifier

            # Ensure internal structures for this validator exist
            if validator not in self.last_snapshot:
                self.last_snapshot[validator] = {}
            if validator not in self.history:
                self.history[validator] = {}
            if validator not in self.gap:
                self.gap[validator] = {}

            # Wait for any ongoing save operation to complete
            while self.saving:
                bt.logging.info("Waiting for history saving to complete...")
                time.sleep(0.5)

            snapshots: dict[int, L2Snapshot | None] = {}
            for book_id, book in state.books.items():
                snapshots[book_id] = self._prepare_snapshot(state, book)

            # If all snapshots were successfully prepared
            if all(snapshots.values()):
                bt.logging.info(f"Updating state history for {validator} at {duration_from_timestamp(state.timestamp)}...")
                start_time = time.time()

                # Parallel or sequential history reconstruction
                if self.parallel:
                    num_processes = min(self.parallel_workers, len(state.books))
                    if len(state.books) % num_processes != 0:
                        raise ValueError(f"`parallel_workers` ({self.parallel_workers}) must divide number of books ({len(state.books)}).")

                    # Divide books into batches for parallel processing
                    batch_size = len(state.books) // num_processes
                    batches = [list(state.books.keys())[i:i + batch_size] for i in range(0, len(state.books), batch_size)]
                    histories = batch_history(
                        {book_id: snapshot.model_dump() for book_id, snapshot in snapshots.items()},
                        {book_id: [event.model_dump() for event in book.events] for book_id, book in state.books.items()},
                        batches,
                        state.config.volumeDecimals
                    )
                else:
                    # Process sequentially
                    histories = {
                        book_id: history(
                            snapshot.model_dump(),
                            [event.model_dump() for event in state.books[book_id].events],
                            state.config.volumeDecimals
                        )
                        for book_id, snapshot in snapshots.items()
                    }

                # Validate and process histories
                processed_histories = {
                    book_id: state.books[book_id].process_history(
                        {t: L2Snapshot.model_validate(snapshot).sort(self.depth) for t, snapshot in hist.items()},
                        {t: TradeInfo.model_validate(trade) for t, trade in trades.items()},
                        state.timestamp,
                        state.config,
                        self.history_retention_mins,
                        self.depth
                    )
                    for book_id, (hist, trades) in histories.items()
                }

                # Update all books in the state
                for book_id, book in state.books.items():
                    history_obj, matched, discrepancies = processed_histories[book_id]
                    self._update_book_history(state, book, history_obj, matched, discrepancies)

                bt.logging.info(f"Updated State History ({time.time() - start_time:.2f}s)")

            if self.should_save:
                # Trigger asynchronous save
                self.save()

        except Exception as ex:
            bt.logging.error(f"Exception processing state update for {validator} at {duration_from_timestamp(state.timestamp)}: {ex}")
        finally:
            self.updating = False

    def update_async(self, state: MarketSimulationStateUpdate) -> None:
        """
        Update the history asynchronously in a separate thread.
        Allows non-blocking updates while other operations continue.

        Args:
            state (MarketSimulationStateUpdate): The latest simulation state to process.
        """
        if not self.updating:
            Thread(target=self.update, args=(state,), daemon=True, name=f'update_history_{state.timestamp}').start()

    def _prepare_snapshot(self, state: MarketSimulationStateUpdate, book: Book) -> L2Snapshot | None:
        """
        Prepare a snapshot for a specific book, handling gaps if necessary.

        Args:
            state (MarketSimulationStateUpdate): The current simulation state.
            book (Book): The book data to process.

        Returns:
            L2Snapshot | None: The prepared snapshot, or None if unavailable.
        """
        book_id: int = book.id
        validator: str = state.dendrite.hotkey
        snapshot: L2Snapshot | None = None

        try:
            # Detect gaps or inconsistencies in history
            has_gap = (
                (book_id not in self.gap[validator] or not self.gap[validator][book_id])
                and (
                    (book_id in self.history[validator] and self.history[validator][book_id].end != state.timestamp - state.config.publish_interval)
                    or (book_id in self.last_snapshot[validator] and self.last_snapshot[validator][book_id].timestamp != state.timestamp - state.config.publish_interval)
                )
            )
            if has_gap:
                # Clear outdated snapshots and handle small/large gaps
                if book_id in self.last_snapshot[validator]:
                    del self.last_snapshot[validator][book_id]
                if book_id in self.history[validator]:
                    if state.timestamp > self.history[validator][book_id].end and state.timestamp < self.history[validator][book_id].end + (self.history_retention_mins * 60_000_000_000) // 10:
                        self.gap[validator][book_id] = True
                        if book_id == 0:
                            bt.logging.warning(
                                f"VALI {validator}: Small gap detected in L2 history from {duration_from_timestamp(self.history[validator][book_id].end)} to {duration_from_timestamp(state.timestamp - state.config.publish_interval)}. Continuing history."
                            )
                    else:
                        self.gap[validator][book_id] = False
                        if book_id == 0:
                            bt.logging.warning(
                                f"VALI {validator}: Large gap detected in L2 history. Resetting history for book {book_id}."
                            )
                        del self.history[validator][book_id]

            # Determine appropriate snapshot to use
            if book_id in self.last_snapshot[validator]:
                if book_id not in self.history[validator]:
                    snapshot = self.last_snapshot[validator][book_id]
                else:
                    if self.history[validator][book_id].end == state.timestamp - state.config.publish_interval:
                        snapshot = list(self.history[validator][book_id].snapshots.values())[-1]
                    else:
                        snapshot = self.last_snapshot[validator][book_id]

            # Update last snapshot for this book
            self.last_snapshot[validator][book_id] = book.snapshot(state.timestamp)

        except Exception as ex:
            bt.logging.error(
                f"VALI {validator} BOOK {book_id}: Exception while processing state at {duration_from_timestamp(state.timestamp)} (T={state.timestamp}): {ex}\n{traceback.format_exc()}"
            )
        finally:
            return snapshot

    def _update_book_history(self, state: MarketSimulationStateUpdate, book: Book, history: L2History, matched: bool, discrepancies: list[str]) -> None:
        """
        Commit an updated book history to the state manager.

        Args:
            state (MarketSimulationStateUpdate): The current simulation state.
            book (Book): The book being updated.
            history (L2History): The reconstructed book history.
            matched (bool): Whether reconstructed snapshot matches published state.
            discrepancies (list[str]): List of discrepancies found during reconstruction.
        """
        book_id: int = book.id
        validator: str = state.dendrite.hotkey

        try:
            if book_id in self.last_snapshot[validator]:
                self.gap[validator][book_id] = False
                if book_id not in self.history[validator]:
                    self.history[validator][book_id] = history
                    if book_id == 0:
                        bt.logging.info(
                            f"VALI {validator}: Initialized new L2 history at {duration_from_timestamp(self.history[validator][book_id].end)} "
                            f"(Available: {duration_from_timestamp(self.history[validator][book_id].start)}-{duration_from_timestamp(self.history[validator][book_id].end)})"
                        )
                else:
                    if self.history[validator][book_id].end == state.timestamp - state.config.publish_interval:
                        self.history[validator][book_id] = self.history[validator][book_id].append(history)
                        if book_id == 0:
                            bt.logging.info(
                                f"VALI {validator}: Appended L2 history at {duration_from_timestamp(state.timestamp)} "
                                f"(Available: {duration_from_timestamp(self.history[validator][book_id].start)}-{duration_from_timestamp(self.history[validator][book_id].end)})"
                            )
                    else:
                        # Recover after history gap
                        self.history[validator][book_id] = self.history[validator][book_id].append(history)
                        if book_id == 0:
                            bt.logging.info(
                                f"VALI {validator}: Recovered after history gap at {duration_from_timestamp(state.timestamp)} "
                                f"(Available: {duration_from_timestamp(self.history[validator][book_id].start)}-{duration_from_timestamp(self.history[validator][book_id].end)})"
                            )

                if not matched:
                    bt.logging.error(
                        f"VALI {validator} BOOK {book_id}: Mismatch between reconstructed and published book state:\n" + "\n".join(discrepancies)
                    )

            # Always update the last snapshot
            self.last_snapshot[validator][book_id] = book.snapshot(state.timestamp)

        except Exception as ex:
            bt.logging.error(
                f"VALI {validator} BOOK {book_id}: Exception while processing state at {duration_from_timestamp(state.timestamp)} (T={state.timestamp}): {ex}\n{traceback.format_exc()}"
            )

    def __getitem__(self, validator: str) -> dict[int, L2History]:
        """
        Access the history for a specific validator.

        Args:
            validator (str): Validator identifier.

        Returns:
            dict[int, L2History]: Mapping of book IDs to their histories.
        """
        return self.history[validator]

    def __contains__(self, validator: str) -> bool:
        """
        Check if a validator has history stored.

        Args:
            validator (str): Validator identifier.

        Returns:
            bool: True if history exists, False otherwise.
        """
        return validator in self.history

    def serialize(self) -> dict[str, Any]:
        """
        Serialize the current state history into a dictionary.

        Returns:
            dict[str, Any]: Serialized representation of the state history.
        """
        return {
            "publish_interval" : self.publish_interval,
            "last_snapshot": {
                validator: {book_id: snapshot.model_dump() for book_id, snapshot in validator_snapshot.items()}
                for validator, validator_snapshot in self.last_snapshot.items()
            },
            "history": {
                validator: {
                    book_id: {
                        'snapshots': {t: snapshot.model_dump() for t, snapshot in history.snapshots.items()},
                        'trades': {t: trade.model_dump() for t, trade in history.trades.items()}
                    }
                    for book_id, history in validator_history.items()
                }
                for validator, validator_history in self.history.items()
            }
        }

    def populate(self, serialized: dict[str, Any]) -> None:
        """
        Populate the state history from a serialized dictionary.

        Args:
            serialized (dict[str, Any]): Serialized history data.
        """
        self.last_snapshot = {
            validator: {book_id: L2Snapshot.model_validate(snapshot) for book_id, snapshot in validator_snapshot.items()}
            for validator, validator_snapshot in serialized["last_snapshot"].items()
        }
        self.history = {
            validator: {
                book_id: L2History(
                    snapshots={t: L2Snapshot.model_validate(snapshot) for t, snapshot in history_data["snapshots"].items()},
                    trades={t: TradeInfo.model_validate(trade) for t, trade in history_data["trades"].items()},
                    retention_mins=self.history_retention_mins,
                    publish_interval=serialized["publish_interval"]
                )
                for book_id, history_data in validator_history.items()
            }
            for validator, validator_history in serialized["history"].items()
        }

    def _save(self) -> None:
        """
        Save the current state history to disk synchronously.
        """
        self.saving = True
        bt.logging.info("Saving history...")
        start_time = time.time()

        # Write serialized data to a temporary file
        with open(self.state_file + ".tmp", 'wb') as file:
            packed_data = msgpack.packb(self.serialize(), use_bin_type=True)
            file.write(packed_data)

        # Replace old state file with new one
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        os.rename(self.state_file + ".tmp", self.state_file)

        bt.logging.info(f"History saved to {self.state_file} ({time.time() - start_time:.2f}s)")
        self.saving = False

    def save(self) -> None:
        """
        Save the state history asynchronously in a separate thread.
        """
        if not self.saving:
            Thread(target=self._save, daemon=True, name='save_history').start()

    def load(self) -> None:
        """
        Load the state history from disk if available, otherwise initialize empty structures.
        """
        if os.path.exists(self.state_file):
            bt.logging.info(f"Loading history from {self.state_file}...")
            with open(self.state_file, 'rb') as file:
                byte_data = file.read()
            state_data = msgpack.unpackb(byte_data, use_list=False, strict_map_key=False)
            self.populate(state_data)
            bt.logging.success("Loaded history! Available data:")
            for validator, validator_history in self.history.items():
                for book_id, book_history in validator_history.items():
                    bt.logging.info(
                        f"VALI {validator} BOOK {book_id}: {duration_from_timestamp(book_history.start)} - {duration_from_timestamp(book_history.end)}"
                    )
        else:
            # No saved state found; start fresh
            bt.logging.info(f"No history file found at {self.state_file}. Initializing empty history.")
            self.last_snapshot = {}
            self.history = {}
