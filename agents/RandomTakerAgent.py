# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Random market-taker agent: places market orders at random intervals.
Supports GenTRX distributed training via agent params.
"""
import bittensor as bt
from taos.common.agents import launch
from taos.im.agents import GenTRXAgent
from taos.im.protocol.models import OrderDirection, STP, LoanSettlementOption, OrderCurrency
import random
import os
import json
import time

class RandomTakerAgent(GenTRXAgent):
    """Example: takes liquidity at random intervals, useful as background flow."""
    def initialize(self):
        """
        Initialize properties, variables and quantities that will be used by the agent.
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
        self.min_leverage = self.config.min_leverage if hasattr(self.config, 'min_leverage') else 0.0
        self.max_leverage = self.config.max_leverage if hasattr(self.config, 'max_leverage') else 0.0
        self.max_fee_rate = self.config.max_fee_rate if hasattr(self.config, 'max_fee_rate') else 0.002
        self.delegate     = getattr(self.config, 'delegate', '')
        # Initialize a variable which allows to maintain the same direction of trade for a defined period
        self.direction    = {}

        # Output files. The base image runs an inotify watcher over /app/outputs and copies
        # anything written there to MinIO (5s debounce), which is what the Hangar's OUTPUT
        # FILES panel lists and serves. Writing the files IS the whole integration -- there
        # is no upload call to make. Falls back to a local ./outputs when the container path
        # is absent so the agent behaves the same when run off-container.
        self.output_dir = getattr(self.config, 'output_dir', '/app/outputs')
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError:
            self.output_dir = 'outputs'
            os.makedirs(self.output_dir, exist_ok=True)
        self.orders_csv   = os.path.join(self.output_dir, 'orders.csv')
        self.summary_json = os.path.join(self.output_dir, 'summary.json')
        self.order_counts = {}
        self.books_traded = set()
        self.started_at   = time.time()
        self._write_output(self.orders_csv, 'wall_time_utc,sim_timestamp,book,direction,quantity,leverage\n', mode='w')
        self._write_summary()

    def _write_output(self, path, text, mode='a'):
        """Best-effort write: an output file is diagnostics, and must never stop the agent trading."""
        try:
            with open(path, mode) as f:
                f.write(text)
        except OSError as e:
            bt.logging.debug(f"OUTPUT WRITE FAILED ({path}) : {e}")

    def _write_summary(self):
        """Rewrite the rolling summary. Whole-file rewrite, so the watcher always ships a valid JSON."""
        self._write_output(self.summary_json, json.dumps({
            'agent_id': os.environ.get('AGENT_ID', ''),
            'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'uptime_s': round(time.time() - self.started_at, 1),
            'orders_submitted': dict(sorted(self.order_counts.items())),
            'orders_total': sum(self.order_counts.values()),
            'books_traded': sorted(self.books_traded),
            'params': {
                'min_quantity': self.min_quantity,
                'max_quantity': self.max_quantity,
                'min_leverage': self.min_leverage,
                'max_leverage': self.max_leverage,
            },
        }, indent=2) + '\n', mode='w')

    def _record_order(self, sim_ts, book_id, direction, quantity, leverage):
        self.order_counts[direction] = self.order_counts.get(direction, 0) + 1
        self.books_traded.add(book_id)
        self._write_output(
            self.orders_csv,
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},{sim_ts},{book_id},{direction},{quantity},{leverage}\n")
        # Every 10th order, not every order: the watcher debounces 5s, so rewriting the
        # summary per order would re-upload the same small file continuously.
        if sum(self.order_counts.values()) % 10 == 0:
            self._write_summary()

    def quantity(self):
        """
        Obtains a random quantity for order placement within the bounds defined by the agent strategy parameters.
        """
        return round(random.uniform(self.min_quantity, self.max_quantity), getattr(self.simulation_config, 'volumeDecimals', 8))

    def leverage(self, response):
        """
        Obtains a random leverage value for order placement within the bounds defined by the agent strategy parameters.
        """
        # EXCHANGE MODE RUNS WITH maxLeverage=0, so a leveraged order is REFUSED at placement rather
        # than executed unleveraged. Returning 0 here rather than at each call site also fixes the
        # `quantity() * (1 + leverage())` sizing below, which would otherwise inflate an order the
        # exchange will not accept. Simulation behaviour is unchanged.
        # Taken from the RESPONSE, not the agent. One agent instance serves both validators
        # concurrently, so a flag on the agent describes whichever request last ran update(); the
        # mechanism belongs to the response being built.
        if response.exchange_mode:
            return 0.0
        return round(random.uniform(self.min_leverage, self.max_leverage), 2) if self.min_leverage != self.max_leverage else self.max_leverage

    def respond_simulation(self, state):
        """Simulation-mode branch of this agent's per-step response; the strategy itself is shared."""
        volume_decimals = getattr(self.simulation_config, 'volumeDecimals', 8)
        # Initialize a response class associated with the current miner
        response = self.make_response()
        # Iterate over the book realizations in the state message
        book_ids = list((state.books or {}).keys())

        # A FEW RANDOM BOOKS, OCCASIONALLY -- not every book, every tick.
        #
        # Trading every book on every tick is not how a miner behaves, and it breaks things downstream:
        #
        #   settlement  one signal per book per tick, and the executor prices a signal GROUP against the
        #               proxy's balance. 129 books became ~109 signals at ~1.7 mtao each -- a 0.184 tao
        #               fee against a 0.0855 tao proxy -- so every group is skipped and nothing settles.
        #   simulation  per-book volume limits mean a continuous stream of market orders is refused more
        #               often than filled (INSUFFICIENT_BASE / INSUFFICIENT_QUOTE / MINIMUM_ORDER_SIZE).
        #
        # So: 1-2 books, sampled fresh each time, with a randomised gap between batches. Sampling is
        # random rather than the first N because a fixed slice would trade the same books forever and
        # leave the rest of the market untouched.
        #
        # All three are overridable, and the gap is measured in SIM time (state.timestamp is ns), so it
        # scales with the run rather than with wall-clock.
        try:
            _max_books = max(1, int(float(getattr(self.config, 'max_books', 2))))
        except (TypeError, ValueError):
            _max_books = 2
        try:
            _iv_min = float(getattr(self.config, 'min_trade_interval_s', 10.0))
            _iv_max = float(getattr(self.config, 'max_trade_interval_s', 30.0))
        except (TypeError, ValueError):
            _iv_min, _iv_max = 10.0, 30.0
        if _iv_max < _iv_min:
            _iv_max = _iv_min

        # PER MECHANISM. One agent instance serves BOTH validators, and their clocks are
        # not comparable: simulation timestamps are sim-relative (~7.3e13 ns) while
        # exchange timestamps are wall-clock (~3.0e15 ns), about 41x larger. A single
        # shared deadline therefore breaks the simulation half completely -- the first
        # exchange response sets the next-trade time to a wall-clock instant that sim
        # time never reaches, and every later simulation query returns no instructions
        # for the life of the process: the agent goes on trading one mechanism while the other
        # receives no instructions at all.
        _ts = int(getattr(state, 'timestamp', 0) or 0)
        _slot = 'exchange' if response.exchange_mode else 'simulation'
        if not hasattr(self, '_next_trade_ts') or not isinstance(self._next_trade_ts, dict):
            self._next_trade_ts = {}
        _next = self._next_trade_ts.get(_slot)
        if _next is None:
            # Stagger the first batch so a fleet of these agents does not fire in lockstep.
            self._next_trade_ts[_slot] = _ts + int(random.uniform(0.0, _iv_max) * 1e9)
            return response
        if _ts < _next:
            return response
        self._next_trade_ts[_slot] = _ts + int(random.uniform(_iv_min, _iv_max) * 1e9)

        if len(book_ids) > _max_books:
            book_ids = random.sample(book_ids, _max_books)
        for book_id in book_ids:
            # If we have not set a trade direction for this book, or 100 simulation seconds have elapsed
            if book_id not in self.direction or state.timestamp % 100_000_000_000 == 0:
                # Randomly select a new trade direction for the agent on this book
                self.direction[book_id] = random.choice([OrderDirection.BUY, OrderDirection.SELL])

            # NEW: Maker and taker fees will be dynamically moving under the DIS fee policy
            # The below demonstrates a simple approach for reacting to the changing rates in trading logic
            previous_taker_rate = self.accounts[book_id].fees.taker_fee_rate
            # Positive rate implies a fee is to be paid, negative rate results in rebates to the trader
            # Check the rate against the configured tolerance
            if previous_taker_rate > self.max_fee_rate:
                # If taker rate is above tolerance, do not place orders this round
                continue

            # Attach a market order instruction in the current trade direction for a random quantity within bounds defined by the parameters
            bt.logging.info(f"BOOK {book_id} | QUOTE : {self.accounts[book_id].quote_balance.total} [LOAN {self.accounts[book_id].quote_loan} | COLLAT {self.accounts[book_id].quote_collateral}]")
            bt.logging.info(f"BOOK {book_id} | BASE : {self.accounts[book_id].base_balance.total} [LOAN {self.accounts[book_id].base_loan} | COLLAT {self.accounts[book_id].base_collateral}]")
            if self.direction[book_id] == OrderDirection.BUY:
                # If in the BUY regime, we place orders randomly with leverage selected from the configured range
                # Obtain a random leverage value if there is no open margin position on sell side
                leverage   = self.leverage(response) if self.accounts[book_id].base_loan == 0 else 0.0
                # If an open opposite margin position exists, repay the corresponding loans in order
                # from oldest to newest by setting LoanSettlementOption.FIFO
                settlement = LoanSettlementOption.NONE if self.accounts[book_id].base_loan == 0 else LoanSettlementOption.FIFO
                # If placing unleveraged order, increase the quantity to better match the average total size of
                # leveraged orders on the other side.  This avoids accumulating too much inventory in one currency.
                quantity   = round(self.quantity() * (1 + self.leverage(response)), volume_decimals)
                response.market_order(
                    book_id=book_id,
                    direction=self.direction[book_id],
                    quantity=quantity,
                    stp=random.choice([STP.DECREASE_CANCEL, STP.CANCEL_OLDEST]),
                    leverage=leverage,
                    settlement_option=settlement,
                    currency=OrderCurrency.BASE,
                    max_slippage=getattr(self.config, 'max_slippage', 0.01),
                )
                bt.logging.info(f"SUBMITTING BUY MARKET ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}")
                self._record_order(state.timestamp, book_id, 'BUY', quantity, leverage)
            else:
                # If in the SELL regime, we place orders randomly without leverage, but with quantity increased to match the amounts placed on buy side.
                # Obtain a random leverage value if there is no open margin position on sell side
                leverage   = self.leverage(response) if self.accounts[book_id].quote_loan == 0 else 0.0
                # If an open opposite margin position exists, repay the corresponding loans in order
                # from oldest to newest by setting LoanSettlementOption.FIFO
                settlement = LoanSettlementOption.NONE if self.accounts[book_id].quote_loan == 0 else LoanSettlementOption.FIFO
                # If placing unleveraged order, increase the quantity to better match the average total size of
                # leveraged orders on the other side.  This avoids accumulating too much inventory in one currency.
                quantity   = round(self.quantity() * (1 + self.leverage(response)), volume_decimals)
                response.market_order(
                    book_id=book_id,
                    direction=self.direction[book_id],
                    quantity=quantity,
                    stp=random.choice([STP.DECREASE_CANCEL, STP.CANCEL_OLDEST]),
                    leverage=leverage,
                    settlement_option=settlement,
                    currency=OrderCurrency.BASE,
                    max_slippage=getattr(self.config, 'max_slippage', 0.01),
                )
                bt.logging.info(f"SUBMITTING SELL MARKET ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}")
                self._record_order(state.timestamp, book_id, 'SELL', quantity, leverage)
        # Return the response with instructions appended
        # The response will be serialized and sent back to the validator for processing
        return response

if __name__ == "__main__":
    """
    Example command for local standalone testing execution using Proxy:
    python RandomTakerAgent.py --port 8888 --agent_id 0 --params min_quantity=0.25 max_quantity=1.0 min_leverage=0.0 max_leverage=1.0 max_fee_rate=0.002

    NOTE min_quantity: the engine refuses a BASE-currency order whose volume is under
    the exchange's `minOrderSize` -- 0.25 alpha in simulation_0.xml -- BEFORE it looks at
    direction, so a smaller draw is rejected outright with MINIMUM_ORDER_SIZE_VIOLATION
    and never reaches the book. 0.1 put the bottom of the sampling range below that floor,
    so a fraction of orders failed for a reason nothing in the log explains. Check the
    floor for your config: the exchange logs it at startup as
    "Exchange order floors: minOrderSize=<N> alpha".
    """
    launch(RandomTakerAgent)
