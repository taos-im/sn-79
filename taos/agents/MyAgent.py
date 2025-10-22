import os
import numpy as np
from collections import defaultdict, deque
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol.response import FinanceAgentResponse, OrderDirection, TimeInForce
import random

class MyAgent(FinanceSimulationAgent):
    def initialize(self):
        self.history_dirs = {}
        self.overall_window_size = 200
        self.local_window_size = 30
        self.fill_history = defaultdict(lambda: deque(maxlen=20))
        self.order_fills = defaultdict(set)
        self.order_history = defaultdict(dict)
        self.pnl_history = defaultdict(list)
        self.last_inventory_value = {}
        self.min_spread = 0.01
        self.stale_order_time = 120000
        self.volume_target = 10.0
        self.trading_time = 0
        self.trading_filename = "orders_log.csv"
    def get_validator_hotkey(self, state):
        return state.dendrite.hotkey

    def get_history_dir(self, validator_hotkey):
        if validator_hotkey not in self.history_dirs:
            path = os.path.abspath(f"history_{validator_hotkey}_{self.uid}")
            os.makedirs(path, exist_ok=True)
            self.history_dirs[validator_hotkey] = path
        return self.history_dirs[validator_hotkey]

    def get_history_file(self, validator_hotkey, book_id):
        return os.path.join(self.get_history_dir(validator_hotkey), f"price_history_{book_id}.csv")

    def append_price(self, validator_hotkey, book_id, timestamp, best_bid, best_ask):
        file_path = self.get_history_file(validator_hotkey, book_id)
        with open(file_path, "a") as f:
            f.write(f"{timestamp},{best_bid},{best_ask}\n")

    def load_windowed_history(self, validator_hotkey, book_id, window_size):
        file_path = self.get_history_file(validator_hotkey, book_id)
        if not os.path.exists(file_path):
            # Pad with default values if file doesn't exist
            return [(1.0, 1.0)] * window_size
        with open(file_path, "r") as f:
            lines = f.readlines()
            # Parse last window_size lines and extract best_bid and best_ask
            price_pairs = [
                tuple(map(float, line.strip().split(",")[1:3]))
                for line in lines[-window_size:]
            ]
        if len(price_pairs) < window_size:
            pad_pair = price_pairs[0] if price_pairs else (1.0, 1.0)
            price_pairs = [pad_pair] * (window_size - len(price_pairs)) + price_pairs
        return price_pairs[-window_size:]

    def estimate_inventory_value(self, account, mid):
        return account.bb.t * mid + account.qb.t

    def regime_detection(self, prices):
        returns = np.diff(prices)
        if len(returns) < 8:
            return 'neutral'
        up = np.sum(returns > 0)
        down = np.sum(returns < 0)
        if up > 7 or down > 7:
            return 'trend'
        else:
            return 'mean-revert'

    def update_fill_history_from_events(self, book_id, events, account):
        my_order_ids = {o.i for o in account.o}
        for event in events:
            if not hasattr(event, "makerAgentId") or not hasattr(event, "takerAgentId"):
                continue
            unique_id = getattr(event, "trade_id", None) or getattr(event, "id", None) or (getattr(event, "t", None), getattr(event, "price", None), getattr(event, "q", None))
            if unique_id in self.order_fills[book_id]:
                continue
            self.order_fills[book_id].add(unique_id)
            if (event.makerAgentId == self.uid and event.makerOrderId in my_order_ids) or (event.takerAgentId == self.uid and event.takerOrderId in my_order_ids):
                self.fill_history[book_id].append(1)

    def get_next_price(self, price, tick_size, direction):
        # direction: 'up' for sell, 'down' for buy
        if direction == 'down':
            return round(price - tick_size, 2)
        else:
            return round(price + tick_size, 2)
    
    def respond(self, state):
        validator_hotkey = self.get_validator_hotkey(state)
        response = FinanceAgentResponse(agent_id=self.uid)
        price_decimals = getattr(state.config, "priceDecimals", 2)
        vol_decimals = getattr(state.config, "volumeDecimals", 4)
        min_qty = 0.01
        if validator_hotkey == '5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF':
            self.trading_time += 1
            if self.trading_time == 8:
                self.trading_time = 0
            
        print(f"validator_hotkey: {validator_hotkey}, trading_time: {self.trading_time}")
        for book_id, book in state.books.items():
            if not book.bids or not book.asks:
                continue
            best_bid = book.bids[0].p
            best_ask = book.asks[0].p
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread = round(spread, 2)

            buy_or_sell = 1 if random.choice([0, 1]) == 1 else 0
            direction = OrderDirection.SELL if buy_or_sell == 1 else OrderDirection.BUY
            self.append_price(validator_hotkey, book_id, state.timestamp, best_bid, best_ask)
            # overall_prices = self.load_windowed_history(validator_hotkey, book_id, self.overall_window_size)
            local_prices = self.load_windowed_history(validator_hotkey, book_id, self.local_window_size)
            tick_size = 0.01
                    
            if spread < self.min_spread - 0.005:
                qty = 0.82
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.BUY,
                    quantity=qty,
                    price=best_bid,
                    timeInForce=TimeInForce.GTC
                )
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.SELL,
                    quantity=qty,
                    price=best_ask,
                    timeInForce=TimeInForce.GTC
                )
            
            elif np.abs(spread - self.min_spread) < 1e-3:
                qty = 0.27
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.BUY,
                    quantity=qty,
                    price=best_bid - 0.01,
                    timeInForce=TimeInForce.GTC
                )
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.SELL,
                    quantity=qty,
                    price=best_ask + 0.01,
                    timeInForce=TimeInForce.GTC
                )
            elif spread > self.min_spread and spread < 0.035:
                spread = max(spread, self.min_spread)
                # Simple adaptive quoting logic
                qty = 0.67
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.BUY,
                    quantity=qty,
                    price=best_bid,
                    timeInForce=TimeInForce.GTC
                )
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.SELL,
                    quantity=qty,
                    price=best_ask,
                    timeInForce=TimeInForce.GTC
                )
            else:
                spread = max(spread, self.min_spread)
                # Simple adaptive quoting logic
                qty = 1.2
                best_bid = best_bid + 0.01
                best_sell = best_ask - 0.01
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.BUY,
                    quantity=qty,
                    price=best_bid,
                    timeInForce=TimeInForce.GTC
                )
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.SELL,
                    quantity=qty,
                    price=best_sell,
                    timeInForce=TimeInForce.GTC,
                )
            # Cancel stale or far-from-market orders
            account = state.accounts[self.uid][book_id]
            stale_order_ids = []
            for order in account.o:
                # Cancel if open too long or too far from current touch/mid
                if order.s == 0 and abs(order.p - best_ask) > spread * 1.2:
                    stale_order_ids.append(order.i)
                elif order.s == 1 and abs(order.p - best_bid) > spread * 1.2:
                    stale_order_ids.append(order.i)
            if stale_order_ids:
                response.cancel_orders(book_id=book_id, order_ids=stale_order_ids)
        return response

if __name__ == "__main__":
    from taos.common.agents import launch
    launch(MyAgent)