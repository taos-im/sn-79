# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Arbitrageur agent: take the "gifts", faster than the farm.

A farm can pump a few "winner" UIDs by having its own "puppet" UIDs post orders
priced *through* fair value and letting the winner take them. Those off-market
resting orders are free money for whoever crosses to them first. This agent is a
lean, market-data-only arbitrageur: each round it finds a best ask trading BELOW a
robust fair estimate (or a best bid ABOVE it) and immediately crosses with a
marketable IOC order to take it.

Why this matters: if enough well-optimised copies of this run, the farm can no
longer *reliably* hand its winners those gifts — a competing arbitrageur snaps
them up first, so the value leaks to honest miners and the pump stops paying. It
changes nothing in the background market; it just competes, which is the point of
the subnet.

SPEED IS THE STRATEGY. The validator delays each response in simulation time by an
exponential function of your wall-clock response time (min 10 ms at ~instant, up
to 1 s near the 3 s timeout). Two agents targeting the same gift are ordered by
that delay, so the faster responder takes it. The farm's winners on mainnet
respond in ~0.16 s at the fastest (~12 ms sim-delay); to beat them you must get
your *total* round-trip below ~0.15 s — realistically ~0.05–0.10 s — via minimal
per-round compute (this agent is O(books) with a handful of comparisons; keep it
that way) and optimised / co-located networking. Below ~0.3 s you already beat
~90% of their responses; the last few ms is where the fastest gifts are won.

Params (via --agent.params):
  edge        min fractional mispricing to act on (default 0.004 = 0.4%)
  fair_alpha  EMA weight on the current mid for the fair estimate (default 0.05)
  max_take    base units taken per hit (default 1.0)
  cross_frac  how far to cross past the target to guarantee the fill (default 0.001)
"""
import time

import bittensor as bt
from taos.common.agents import launch
from taos.im.agents import GenTRXAgent
from taos.im.protocol.models import OrderDirection, STP, TimeInForce


class ArbitrageAgent(GenTRXAgent):
    """Example: arbitrages the spread between the pool price and the book on one venue."""
    def initialize(self):
        """Initialise arbitrage parameters (all overridable via --agent.params)."""
        # Pure trader — no GenTRX training/collection/inference.
        for k in ("gtx_training_enabled", "gtx_collect_data"):
            if not hasattr(self.config, k):
                setattr(self.config, k, False)
        super().initialize()
        self.edge = float(getattr(self.config, "edge", 0.004))            # min mispricing to act on
        self.fair_alpha = float(getattr(self.config, "fair_alpha", 0.05))  # fair-estimate EMA speed
        self.max_take = float(getattr(self.config, "max_take", 1.0))       # base units per hit
        self.cross_frac = float(getattr(self.config, "cross_frac", 0.001)) # cross past target to guarantee fill
        self.max_books_per_round = int(getattr(self.config, "max_books_per_round", 8))
        # Optional artificial think-time. Default 0 (respond as fast as you can — that
        # is the point). Used to study speed sensitivity: your response time maps to a
        # simulation-time order delay, and the lowest-delay agent wins a contested gift.
        self.resp_delay = float(getattr(self.config, "resp_delay", 0.0))
        self.tag = str(getattr(self.config, "tag", ""))  # optional label for logs
        self._fair = {}   # book_id -> lagged fair-value estimate (EMA of mid)
        self._round = 0
        self._filled_n = 0        # cumulative own fills (confirms takes actually landed)
        self._filled_notional = 0.0
        bt.logging.info(f"ARB init | tag={self.tag or '-'} uid={self.uid} edge={self.edge} resp_delay={self.resp_delay}")

    def _own_fills(self):
        """This round's own fills (EVENT_TRADE notices) — confirms takes landed, which
        in a contested gift is the race outcome (a take that lost the race never fills)."""
        n, notl = 0, 0.0
        for ev in (self.events or []):
            et = getattr(ev, "type", None) or (isinstance(ev, dict) and (ev.get("y") or ev.get("type")))
            if et in ("EVENT_TRADE", "ET"):
                q = ev.get("q") if isinstance(ev, dict) else getattr(ev, "quantity", 0.0)
                p = ev.get("p") if isinstance(ev, dict) else getattr(ev, "price", 0.0)
                try:
                    n += 1
                    notl += float(q) * float(p)
                except (TypeError, ValueError):
                    pass
        return n, notl

    def _ensure_model_version(self):
        """No GenTRX model — skip the on-chain aggregator + S3 model scan the base
        class runs at startup (it would hang without GenTRX S3 env). Same pattern
        as MarketMakerAgent / LiquidityTakerAgent."""
        return

    def respond_simulation(self, state):
        """Simulation-mode branch of this agent's per-step response; the strategy itself is shared."""
        return self._arb(state, exchange=False)

    def respond_exchange(self, state):
        """Exchange-mode branch of this agent's per-step response; the strategy itself is shared."""
        return self._arb(state, exchange=True)

    def _top(self, levels):
        """(price, size) of the best level, or (None, 0.0) if the side is empty."""
        if not levels:
            return None, 0.0
        lvl = levels[0]
        size = getattr(lvl, "quantity", None)
        if size is None:
            size = getattr(lvl, "volume", 0.0)
        return getattr(lvl, "price", None), float(size or 0.0)

    def _arb(self, state, exchange):
        """Main logic: scan each book for an order priced through fair and take it.

        Deliberately minimal per-round work (one pass over the books, a few
        comparisons) — the edge here is latency, not cleverness. Everything below
        uses only public market data on `state`.
        """
        response = self.make_response(exchange_mode=exchange)
        self._round += 1
        if self.resp_delay > 0.0:
            time.sleep(self.resp_delay)   # inflate response time (latency study only)
        cfg = self.simulation_config
        price_dec = int(getattr(cfg, "priceDecimals", 8))
        vol_dec = int(getattr(cfg, "volumeDecimals", 8))
        min_size = float(getattr(cfg, "min_order_size", 0.0) or 0.0)
        min_qty = max(min_size, 10 ** -vol_dec)
        takes = 0

        for book_id, book in (state.books or {}).items():
            if takes >= self.max_books_per_round or book_id not in self.accounts:
                continue
            bidp, bidq = self._top(getattr(book, "bids", []) or [])
            askp, askq = self._top(getattr(book, "asks", []) or [])
            if not bidp or not askp or bidp <= 0 or askp <= 0:
                continue

            # Robust fair estimate: an EMA of the mid. Judge the CURRENT book against
            # the LAGGED estimate, so an order freshly posted through fair stands out
            # before the estimate drifts toward it; then fold today's mid in.
            mid = 0.5 * (bidp + askp)
            fair = self._fair.get(book_id, mid)
            self._fair[book_id] = (1.0 - self.fair_alpha) * fair + self.fair_alpha * mid

            acct = self.accounts[book_id]
            quote_free = float(getattr(acct.quote_balance, "free", 0.0) or 0.0)
            base_free = float(getattr(acct.base_balance, "free", 0.0) or 0.0)

            # Gift on the ASK: something is being sold below fair -> buy it cheap.
            if askp < fair * (1.0 - self.edge) and askq > 0:
                qty = round(min(askq, self.max_take, quote_free / askp), vol_dec)
                if qty >= min_qty:
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=qty,
                        price=round(askp * (1.0 + self.cross_frac), price_dec),  # cross to guarantee the take
                        stp=STP.CANCEL_OLDEST,
                        timeInForce=TimeInForce.IOC,                             # take-only, never rest
                    )
                    takes += 1
                    continue

            # Gift on the BID: something is being bought above fair -> sell into it.
            if bidp > fair * (1.0 + self.edge) and bidq > 0:
                qty = round(min(bidq, self.max_take, base_free), vol_dec)
                if qty >= min_qty:
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=qty,
                        price=round(bidp * (1.0 - self.cross_frac), price_dec),
                        stp=STP.CANCEL_OLDEST,
                        timeInForce=TimeInForce.IOC,
                    )
                    takes += 1

        fn, fnotl = self._own_fills()
        self._filled_n += fn
        self._filled_notional += fnotl
        if takes or fn:
            bt.logging.info(
                f"ARB{('['+self.tag+']') if self.tag else ''} "
                f"{'XCH' if exchange else 'SIM'} | round={self._round} | sent={takes} "
                f"filled_round={fn} filled_total={self._filled_n} filled_notional_total={self._filled_notional:.2f}"
            )
        return response


if __name__ == "__main__":
    launch(ArbitrageAgent)
