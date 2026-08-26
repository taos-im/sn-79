# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Imbalance-driven hybrid maker/taker with GenTRX distributed training.

Single signal (mean order-book imbalance over a short history window),
three regimes:

    QUOTE   flat inventory + weak signal    → inside-spread limits both sides
    ENTER   flat inventory + strong signal  → market order in signal direction
    MANAGE  non-flat inventory              → closing limit + stop-loss

The same signal drives entry and quote-skew, so the two modes don't fight.

============================================================================
THIS IS A TEMPLATE, NOT A STRATEGY.

If you deploy this unmodified across many miners, the copies will interfere
with each other: they all fire on the same imbalance, the signal vanishes
as soon as they front-run themselves, and the reward mechanism ends up
ranking them by who had lower latency that tick. Nobody wins.

To make this earn real kappa you must at least tune:
    - entry_threshold / cancel_threshold   (be selective, differently)
    - imbalance_depth / history window     (look at different horizons)
    - base_quote_size / enter_size_mult    (pick a capital profile)
    - stop_loss_bps                        (your own risk appetite)

Better still: replace the signal entirely. Imbalance is the cheapest
feature; there are many alternatives (microprice, trade-sign autocorr,
queue-position dynamics, your own regressor). The value of this file is
the scaffolding (regime gating + inventory tracking + GenTRX hook-up),
not the signal.
============================================================================

Trading is driven by the strategy above. The GenTRX model is NOT used for
trading decisions — training and inference run in parallel as data
collection / gradient production.

Usage (proxy test):
    python HybridTrainingAgent.py --port 8888 --agent_id 0 \\
        --params imbalance_depth=5 history_retention_mins=1 \\
                 entry_threshold=0.35 cancel_threshold=0.20 \\
                 stop_loss_bps=40 \\
                 base_quote_size=0.3 enter_size_mult=3.0 \\
                 max_flat_inventory=2.0 expiry_period=500000000 \\
                 gtx_training_enabled=true gtx_collect_data=false
"""

import math
import random
import time

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import StateHistoryManager
from taos.im.protocol.models import OrderDirection, STP, TimeInForce
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse

from taos.im.agents import GenTRXAgent


class HybridTrainingAgent(GenTRXAgent):
    """Regime-gated maker/taker driven by book imbalance, + GenTRX training."""

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Set up strategy state for the hybrid rule/model strategy."""
        super().initialize()
        cfg = self.config

        # --- signal ---
        self.imbalance_depth = int(getattr(cfg, "imbalance_depth", 5))
        self.history_retention_mins = float(
            getattr(cfg, "history_retention_mins", 1)
        )

        # --- thresholds (jittered per-instance so identical launches diverge) ---
        rng = random.Random()  # unseeded: each agent instance diverges naturally
        self._rng = rng
        base_entry = float(getattr(cfg, "entry_threshold", 0.35))
        base_cancel = float(getattr(cfg, "cancel_threshold", 0.20))
        self.entry_threshold = base_entry * rng.uniform(0.85, 1.15)
        self.cancel_threshold = min(
            base_cancel * rng.uniform(0.85, 1.15),
            self.entry_threshold * 0.9,
        )

        # --- risk ---
        self.stop_loss_bps = float(getattr(cfg, "stop_loss_bps", 40))

        # --- sizing ---
        self.base_quote_size = float(getattr(cfg, "base_quote_size", 0.3))
        self.enter_size_mult = float(getattr(cfg, "enter_size_mult", 3.0))
        self.max_flat_inventory = float(getattr(cfg, "max_flat_inventory", 2.0))

        # --- order plumbing ---
        self.expiry_period = int(getattr(cfg, "expiry_period", 500_000_000))
        self.max_fee_rate = float(getattr(cfg, "max_fee_rate", 0.005))

        # --- history manager for imbalance signal ---
        self.history_manager = StateHistoryManager(
            history_retention_mins=self.history_retention_mins,
            log_dir=self.log_dir,
            parallel_workers=0,
        )

        # --- per-book state (position tracked from account deltas) ---
        self._initial_base: dict[int, float] = {}
        self._entry_mid: dict[int, float] = {}

        bt.logging.info(
            f"HybridTrainingAgent init: entry={self.entry_threshold:.3f} "
            f"cancel={self.cancel_threshold:.3f} depth={self.imbalance_depth}"
        )

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------
    def _net_position(self, book_id: int) -> float:
        """Signed base units held relative to sim start. +ve long, -ve short."""
        acct = self.accounts[book_id]
        current = acct.base_balance.total - acct.base_loan
        if book_id not in self._initial_base:
            self._initial_base[book_id] = current
        return current - self._initial_base[book_id]

    def _rand_size(self, mean: float) -> float:
        """Log-normal size around `mean` — keeps volumes varied per order."""
        if mean <= 0:
            return 0.0
        sigma = 0.5
        mu = math.log(mean) - 0.5 * sigma * sigma
        return round(
            self._rng.lognormvariate(mu, sigma),
            self.simulation_config.volumeDecimals,
        )

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------
    def _signal(self, state, validator: str, book_id: int) -> float | None:
        """Mean imbalance over history + current snapshot. ~[-1, 1]."""
        hm = self.history_manager
        if (
            validator not in hm
            or book_id not in hm[validator]
            or not hm[validator][book_id].is_full()
        ):
            return None
        hist = hm[validator][book_id].imbalance(self.imbalance_depth)
        snap = state.books[book_id].snapshot(state.timestamp).imbalance(
            self.imbalance_depth
        )
        merged = hist | {state.timestamp: snap}
        return sum(merged.values()) / len(merged)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def respond(
        self, state: MarketSimulationStateUpdate
    ) -> FinanceAgentResponse:
        # GenTRX: data packaging + model poll + training trigger.
        """Produce this block's response by combining the rule signal with the model's."""
        response = super().respond(state)
        validator = state.dendrite.hotkey

        # Wait for history update to complete before reading it.
        while self.history_manager.updating:
            bt.logging.info("Waiting for history update to complete...")
            time.sleep(0.5)

        for book_id, book in state.books.items():
            if not (book.bids and book.asks):
                continue
            if self.accounts[book_id].fees.maker_fee_rate > self.max_fee_rate:
                continue

            signal = self._signal(state, validator, book_id)
            if signal is None:
                continue  # not enough history yet

            pos = self._net_position(book_id)
            flat = abs(pos) < self.max_flat_inventory * 1e-3

            if not flat:
                self._manage(response, book_id, book, pos)
            elif abs(signal) >= self.entry_threshold:
                self._enter(response, book_id, book, signal)
            elif abs(signal) < self.cancel_threshold:
                self._quote(response, book_id, book, signal)
            # dead zone between cancel_threshold and entry_threshold:
            # don't quote new, don't enter — let outstanding quotes expire.

        # Update history for next tick.
        self.history_manager.update_async(state.model_copy(deep=True))
        return response

    # ------------------------------------------------------------------
    # Regimes
    # ------------------------------------------------------------------
    def _quote(
        self,
        response: FinanceAgentResponse,
        book_id: int,
        book,
        signal: float,
    ) -> None:
        """Inside-spread two-sided limits, skewed against signal direction."""
        bid = book.bids[0].price
        ask = book.asks[0].price
        spread = ask - bid
        if spread <= 0:
            return
        mid = 0.5 * (bid + ask)

        # Jittered offset so 256 copies don't queue on the exact same tick.
        offset = self._rng.uniform(0.15, 0.35)
        # Skew away from pressure side: signal>0 (buy pressure) → tighter sell,
        # wider buy.
        skew = 0.25 * signal
        bid_px = round(
            mid - spread * (offset + skew),
            self.simulation_config.priceDecimals,
        )
        ask_px = round(
            mid + spread * (offset - skew),
            self.simulation_config.priceDecimals,
        )
        if bid_px <= 0 or bid_px >= ask_px:
            return

        size = self._rand_size(self.base_quote_size)
        if size <= 0:
            return

        acct = self.accounts[book_id]
        if acct.quote_balance.free >= size * bid_px:
            response.limit_order(
                book_id=book_id,
                direction=OrderDirection.BUY,
                quantity=size,
                price=bid_px,
                stp=STP.CANCEL_BOTH,
                timeInForce=TimeInForce.GTT,
                expiryPeriod=self.expiry_period,
            )
        if acct.base_balance.free >= size:
            response.limit_order(
                book_id=book_id,
                direction=OrderDirection.SELL,
                quantity=size,
                price=ask_px,
                stp=STP.CANCEL_BOTH,
                timeInForce=TimeInForce.GTT,
                expiryPeriod=self.expiry_period,
            )

    def _enter(
        self,
        response: FinanceAgentResponse,
        book_id: int,
        book,
        signal: float,
    ) -> None:
        """Single market order sized by |signal|, in signal direction."""
        direction = OrderDirection.BUY if signal > 0 else OrderDirection.SELL
        mag = min(abs(signal), 1.0)
        size = self._rand_size(self.base_quote_size * self.enter_size_mult * mag)
        if size <= 0:
            return

        acct = self.accounts[book_id]
        if direction == OrderDirection.BUY:
            if acct.quote_balance.free < size * book.asks[0].price:
                return
        else:
            if acct.base_balance.free < size:
                return

        response.market_order(
            book_id=book_id,
            direction=direction,
            quantity=size,
            stp=STP.CANCEL_OLDEST,
        )
        self._entry_mid[book_id] = 0.5 * (book.bids[0].price + book.asks[0].price)

    def _manage(
        self,
        response: FinanceAgentResponse,
        book_id: int,
        book,
        pos: float,
    ) -> None:
        """Non-flat inventory: stop-loss if adverse, else closing limit."""
        mid = 0.5 * (book.bids[0].price + book.asks[0].price)
        entry = self._entry_mid.get(book_id, mid)
        qty = round(abs(pos), self.simulation_config.volumeDecimals)
        if qty <= 0:
            return

        stop = self.stop_loss_bps * 1e-4
        long_loss = pos > 0 and mid < entry * (1 - stop)
        short_loss = pos < 0 and mid > entry * (1 + stop)

        if long_loss or short_loss:
            close_dir = OrderDirection.SELL if pos > 0 else OrderDirection.BUY
            response.market_order(
                book_id=book_id,
                direction=close_dir,
                quantity=qty,
                stp=STP.CANCEL_OLDEST,
            )
            self._entry_mid.pop(book_id, None)
            return

        # Otherwise close via limit at top of opposite side.
        close_dir = OrderDirection.SELL if pos > 0 else OrderDirection.BUY
        close_px = (
            book.asks[0].price if close_dir == OrderDirection.SELL
            else book.bids[0].price
        )
        response.limit_order(
            book_id=book_id,
            direction=close_dir,
            quantity=qty,
            price=close_px,
            stp=STP.CANCEL_BOTH,
            timeInForce=TimeInForce.GTT,
            expiryPeriod=self.expiry_period,
        )


if __name__ == "__main__":
    launch(HybridTrainingAgent)
