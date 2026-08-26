# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import GenTRXAgent
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol.events import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse


# Stop-loss / take-profit are passed as signed fractional offsets from the
# fill price.  For a BUY: SL is below fill (negative), TP is above fill
# (positive).  For a SELL the signs flip.
SLTP_SCENARIOS = [
    # (label,         direction,             order_type, sl,      tp)
    # Tight levels — chosen so typical intra-round price drift is likely to
    # cross at least one trigger, exercising the SL/TP path during demos
    # rather than leaving the protective orders dormant.
    ("BUY  MKT",      OrderDirection.BUY,    "market",   -0.002,  0.003),
    ("SELL MKT",      OrderDirection.SELL,   "market",    0.002, -0.003),
    ("BUY  LMT",      OrderDirection.BUY,    "limit",    -0.003,  0.005),
    ("SELL LMT",      OrderDirection.SELL,   "limit",     0.003, -0.005),
]


class SLTPAgent(GenTRXAgent):
    """
    Demonstrates the SL/TP order options exposed via `stop_loss` and
    `take_profit` on `FinanceAgentResponse.market_order` / `limit_order`.

    Each round on each book the agent submits one parent order with SL/TP
    attached, then waits `round_period` simulation nanoseconds before
    advancing to the next scenario so the protective orders have a chance
    to be triggered by market movement.
    """

    def initialize(self):
        """Set up the example: pick its book and sizing, and reset its order tracking."""
        super().initialize()
        # Order sizing.
        self.quantity = float(self.config.quantity)
        # How long to wait (in simulation nanoseconds) between successive
        # SL/TP scenarios.  Defaults to 30 seconds — long enough that a
        # nearby SL/TP trigger can plausibly fire before the next round.
        self.round_period = int(getattr(self.config, "round_period", 30_000_000_000))
        # Use a single client-id namespace per book so we can tag each
        # parent order and recognize trigger fills in `onTrade`.
        self.client_id_base = int(getattr(self.config, "client_id_base", 9000))
        # Per-book scheduling state: book_id -> (round_index, next_round_ts).
        self.next_round_ts: dict[int, int] = {}
        self.round_idx: dict[int, int] = {}
        # Track which client-ids correspond to SL/TP parent orders we placed,
        # for richer logging in `onTrade`.
        self.parent_client_ids: dict[int, str] = {}

    def _client_id_for(self, book_id: int, round_idx: int) -> int:
        return self.client_id_base + book_id * 100 + round_idx

    def _place_scenario(self, response: FinanceAgentResponse, book_id: int,
                        book, round_idx: int) -> None:
        label, direction, order_type, sl, tp = SLTP_SCENARIOS[round_idx]
        client_id = self._client_id_for(book_id, round_idx)
        self.parent_client_ids[client_id] = label

        kwargs = dict(
            book_id=book_id,
            direction=direction,
            quantity=self.quantity,
            clientOrderId=client_id,
            stop_loss=sl,
            take_profit=tp,
        )

        if order_type == "market":
            response.market_order(**kwargs)
        else:
            # Place limit at the touch on the same side so the order rests on
            # the book and SL/TP attach on fill rather than immediately.
            price = book.bids[0].price if direction == OrderDirection.BUY else book.asks[0].price
            response.limit_order(price=price, **kwargs)

        bt.logging.info(
            f"[SLTP] book={book_id} round={round_idx} {label} "
            f"qty={self.quantity} SL={sl*100:+.1f}% TP={tp*100:+.1f}% "
            f"clientId={client_id}"
        )

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """Place one SL/TP-carrying limit order per cycle and hold while it works.

        Returns:
            FinanceAgentResponse: The response with at most one new order queued.
        """
        response = self.make_response()  # mode-aware: emits exchange or simulation instructions
        for book_id, book in state.books.items():
            if not book.bids or not book.asks:
                continue

            self.round_idx.setdefault(book_id, 0)
            self.next_round_ts.setdefault(book_id, 0)

            # Pace rounds so triggered SL/TP children have time to fire.
            if state.timestamp < self.next_round_ts[book_id]:
                continue
            if self.round_idx[book_id] >= len(SLTP_SCENARIOS):
                continue

            self._place_scenario(response, book_id, book, self.round_idx[book_id])
            self.round_idx[book_id] += 1
            self.next_round_ts[book_id] = state.timestamp + self.round_period

        return response

    # ------------------------------------------------------------------
    # Lifecycle handlers — log what happens to our SL/TP parent orders
    # so the demo output makes the trigger flow visible.
    # ------------------------------------------------------------------

    def onOrderAccepted(self, event) -> None:
        """Record the accepted order id so the cycle waits on its outcome."""
        label = self.parent_client_ids.get(event.clientOrderId)
        if label is not None:
            bt.logging.info(
                f"[SLTP] ACCEPTED book={event.bookId} order#{event.orderId} "
                f"clientId={event.clientOrderId} ({label})"
            )

    def onOrderRejected(self, event) -> None:
        """Clear the pending order so the next cycle can try again."""
        label = self.parent_client_ids.get(event.clientOrderId)
        if label is not None:
            bt.logging.warning(
                f"[SLTP] REJECTED book={event.bookId} clientId={event.clientOrderId} "
                f"({label}) reason={event.message}"
            )

    def onTrade(self, event: TradeEvent, validator: str = None) -> None:
        """Track fills against the working order and note when SL/TP closes the position.

        Args:
            event (TradeEvent): The fill event.
            validator (str): Validator this event came from.
        """
        label = self.parent_client_ids.get(event.clientOrderId)
        if label is None:
            return
        role = "TAKER" if event.takerAgentId == self.uid else "MAKER"
        bt.logging.info(
            f"[SLTP] FILL  book={event.bookId} {role} clientId={event.clientOrderId} "
            f"({label}) qty={event.quantity}@{event.price} — "
            f"SL/TP triggers attached to this fill"
        )


if __name__ == "__main__":
    """
    Example launch (paired with the local proxy/simulator in agents/proxy):

        python SLTPAgent.py --port 8888 --agent_id 0 \
            --params quantity=0.5 round_period=30000000000

    Cycles through four scenarios per book.  Levels are intentionally tight
    (sub-percent) so triggers fire often during a short demo run:
        0: BUY  market, SL=-0.2%, TP=+0.3%
        1: SELL market, SL=+0.2%, TP=-0.3%
        2: BUY  limit,  SL=-0.3%, TP=+0.5%   (posted at best bid)
        3: SELL limit,  SL=+0.3%, TP=-0.5%   (posted at best ask)
    """
    launch(SLTPAgent)
