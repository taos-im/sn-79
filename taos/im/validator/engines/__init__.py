"""
validator/engines/base.py

Canonical state envelope and protocol abstraction for the unified validator.
Both SimulationEngine and ExchangeEngine produce NormalizedState; all
scoring logic in validator.py consumes only NormalizedState and never
branches on mode below handle_state().
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Canonical trade event
# ─────────────────────────────────────────────────────────────────────────────

def resolve_trade_roles(uid, signal_idx, maker_candidates, counterparty=None):
    """(maker_uid, taker_uid) for a fill, from what the ENGINE stated about it.

    The engine distinguishes the two kinds of LOB signal:

        signal_index is not None -- derived from an instruction submitted this batch, so this
                                    agent was the AGGRESSOR: it is the taker.
        signal_index is None     -- a trade-feed signal: this agent's RESTING order was filled
                                    by a crossing aggressor, so it is the maker.

    An absent side means the POOL. Settlement is always against the pool and funds never move
    agent to agent, so the pool is a real counterparty, and because exactly one side is ever
    absent the representation is unambiguous.

    This replaces two wrong assumptions. `taker_uid = uid` unconditionally recorded a resting
    fill's agent as the taker, which states the opposite of what happened. And
    `maker_uid = uid` as the no-counterparty fallback published the agent as its own
    counterparty: a live notice carried Ma = 171, Ta = 171 while the tape recorded
    maker_uid = NULL for the same trade, so two surfaces disagreed about one fill.

    `counterparty` is the OTHER side as the engine stated it (Signal.counterparty_agent_id,
    populated from the trade context's aggressingAgentId). When present it is used directly and
    the candidate list is ignored: the engine holds this fact, so deriving it here would be
    inventing an answer to a question already answered. The candidate fallback remains only for
    signals that predate the field.

    Args:
        uid: The uid the signal belongs to.
        signal_idx: The engine's signal index; None means a resting-order fill.
        maker_candidates: Uids that could be the maker.
        counterparty: The other side as the engine stated it.

    Returns:
        tuple: ``(maker_uid, taker_uid)``.
    """
    if counterparty is not None and counterparty != uid:
        return ((uid, counterparty) if signal_idx is None else (counterparty, uid))
    others = [c for c in (maker_candidates or []) if c is not None and c != uid]
    if signal_idx is None:
        # This agent was resting. The aggressor is whoever crossed it; absent that, the pool.
        return uid, (others[0] if others else None)
    return (others[0] if others else None), uid


@dataclass
class NormalizedTradeEvent:
    """
    Single canonical trade event in the format _update_trade_volumes() expects.

    Matches the simulation's ET notice dict exactly so that function needs
    zero changes. to_notice_dict() produces the {'y':'ET', 'b':..., ...}
    format directly.

    book_id  — simulation: book index  |  exchange: netuid
    side     — 0 = buy, 1 = sell
    maker_uid / taker_uid — order ROLE, as the engine determined it. taker is the aggressing
        side; maker is the resting side, or None when the fill was against the pool and there
        is no agent counterparty. Settlement is always against the pool and funds never move
        agent to agent, so an absent maker is the normal case for a marketable order, and it
        must never be filled in with the taker's own uid: doing so published the miner as its
        own counterparty (Ma = Ta = 171 on a live notice) while the tape said maker_uid NULL
        for the same trade. The engine is the source of truth for this; nothing downstream
        should re-derive it.
    """
    book_id:   int
    quantity:  float
    price:     float
    side:      int
    maker_uid: Optional[int]
    taker_uid: int
    maker_fee: float
    taker_fee: float
    timestamp:           int     = field(default_factory=lambda: int(time.time_ns()))
    order_id:            Optional[str] = field(default=None)   # truncated display id
    order_uuid:          Optional[str] = field(default=None)   # full exchange-API UUID ('xo' on notices)
    close_reason:        Optional[str] = field(default=None)   # 'SL' | 'TP' | None
    linked_order_id:     Optional[int] = field(default=None)   # originating LOB order id
    # Engine-minted trade id. Without it this event could not be matched against the
    # reconciled fill for the same trade, so the two reached the UI as separate fills
    # (one with an id, one without) and rendered twice.
    trade_id:            Optional[int] = field(default=None)
    # The two orders that met. Consumers index these directly ('Ti'/'Mi' in
    # protocol/models.py::from_json), and they are how the service and UI tie a fill back to the order it
    # filled. Zero where the producer does not know them: a pre-settlement trade event is raised before
    # reconciliation names the orders.
    taker_order_id:      int = 0
    maker_order_id:      int = 0

    def to_notice_dict(self) -> dict:
        """This trade as the compact notice dict miners receive.

        Returns:
            dict: An ET notice carrying the trade's ids, sides, amounts and fees.
        """
        d = {
            'y':  'ET',
            't':  self.timestamp,
            'a':  self.taker_uid,
            'b':  self.book_id,
            'i':  self.trade_id,
            'c':  None,
            'Ti': self.taker_order_id,
            'Mi': self.maker_order_id,
            'q':  self.quantity,
            'p':  self.price,
            's':  self.side,
            'Ma': self.maker_uid,
            'Ta': self.taker_uid,
            'Mf': self.maker_fee,
            'Tf': self.taker_fee,
        }
        if self.close_reason is not None:
            d['cr'] = self.close_reason
        if self.linked_order_id is not None:
            d['Toi'] = self.linked_order_id
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Canonical state envelope
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizedState:
    """
    Single canonical state type the validator loop operates on.

    Field mapping:
        simulation                       exchange
        ───────────────────────────────  ────────────────────────────────────
        timestamp  ← state.timestamp     block × block_time_ns
        block      ← ts // block_time_ns current chain block
        books      ← state.books         per-netuid pool state
        accounts   ← state.accounts      per-uid TAO/alpha balances from chain
        notices    ← state.notices        empty on receive; filled by execute()
        config     ← state.config        ExchangeConfig
        version    ← state.version       SPEC_VERSION

    The field names and dtypes are identical to what _update_trade_volumes(),
    _reward(), and all other validator methods already read, so those methods
    require no changes.
    """
    timestamp: int          # nanoseconds
    block:     int          # chain block number
    books:     dict         # {book_id/netuid: market state dict}
    accounts:  dict         # {uid: {book_id/netuid: balance dict}}
    notices:   dict         # {uid: [ET notice dicts]}
    config:    Any          # MarketSimulationConfig or ExchangeConfig
    version:   int
    logDir:    Any = None   # simulation log directory (preserved from raw message dict)
    pools:     dict | None = None  # {netuid: {price, tao_in, alpha_in, ...}} — exchange only


# ─────────────────────────────────────────────────────────────────────────────
# Protocol ABC
# ─────────────────────────────────────────────────────────────────────────────

class MarketEngine(ABC):
    """
    Abstracts the three mode-specific operations in the validator loop.

    The validator calls these three methods and nothing else that is
    mode-specific.  All scoring, persistence, and weight-setting is above
    this boundary and runs identically in both modes.

    receive()  — wait for the next tick; return NormalizedState
    execute()  — act on miner responses; return trade events ([] for sim)
    respond()  — write response back (IPC write for sim, no-op for exchange)
    """

    @abstractmethod
    async def receive(self) -> tuple[Any, Optional[NormalizedState], float]:
        """
        Blocking: wait for and return the next state tick.

        Returns:
            raw_message    — original message object (passed back to respond())
            normalized     — NormalizedState, or None for lifecycle-only ticks
                             (e.g. simulation END notice — skip handle_state)
            receive_start  — time.time() when message arrived
        """

    @abstractmethod
    async def execute(
        self,
        state: NormalizedState,
        miner_responses: list,
    ) -> list[NormalizedTradeEvent]:
        """
        Act on miner responses and return canonical trade events.

        Simulation: no-op. Simulator injects trade events into the next
                    tick's state.notices itself. Returns [].
        Exchange:   send instructions to LOB engine, execute on-chain,
                    convert ExecutionResult[] → NormalizedTradeEvent[].

        Args:
            state: The block's state update.
            miner_responses: The miners' responses.

        Returns:
            list: Canonical trade events (empty in simulation mode).
        """

    @abstractmethod
    def respond(self, raw_message: Any, response: dict) -> None:
        """
        Send validator response back to whoever is waiting.

        Simulation: msgpack.packb(response) → write to /taosim-res socket.
        Exchange:   no-op (on-chain execution is the response).

        Args:
            raw_message: The request being answered.
            response: The validator response.
        """

    def start(self) -> None:
        """Called once from validator.__init__() after base setup is done."""

    def stop(self) -> None:
        """Called from validator.cleanup()."""

    # ── Shared external-order polling ────────────────────────────────────────
    # UI/wallet-submitted orders are fetched from the data service on a
    # background task, not inline in receive(), so a slow or busy data service
    # never adds latency to the round and the per-order sr25519 verify stays off
    # the critical path. The data-service GET is an atomic destructive pop
    # (exactly-once), so faster polling cannot double-inject. receive() drains
    # the buffer with no await. _fetch_external_orders() is implemented per-engine.

    _EXTERNAL_POLL_INTERVAL = 1.0

    def _ensure_external_poller(self) -> None:
        """Start the background poll task on first call (requires a running loop)."""
        if getattr(self, "_external_poll_task", None) is None:
            self._external_poll_task = asyncio.ensure_future(self._external_orders_loop())

    async def _external_orders_loop(self) -> None:
        while True:
            try:
                instrs = await self._fetch_external_orders()
                if instrs:
                    self._external_buffer.extend(instrs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                import bittensor as bt

                bt.logging.debug(f"external orders poll failed: {exc}")
            await asyncio.sleep(self._EXTERNAL_POLL_INTERVAL)

    def _drain_external_orders(self) -> list:
        """Return buffered external instructions and clear the buffer.

        Sync with no await between the read and the reset, so it is atomic
        against the background poller under single-threaded asyncio.
        """
        instrs = self._external_buffer
        self._external_buffer = []
        return instrs

    def _stop_external_poller(self) -> None:
        task = getattr(self, "_external_poll_task", None)
        if task is not None:
            task.cancel()
            self._external_poll_task = None

    @property
    @abstractmethod
    def book_ids(self) -> list[int]:
        """All active book IDs. Simulation: range(book_count). Exchange: netuids."""

    @property
    def book_count(self) -> int:
        """How many books this engine runs."""
        return len(self.book_ids)

    @property
    @abstractmethod
    def mode(self) -> str:
        """'simulation' or 'exchange'."""

    # ── Lifecycle hooks (called by validator.onStart / onEnd) ────────────────

    def on_start(self, state: Any) -> None:
        """
        Called when the engine signals a new run start.
        Simulation: triggered by 'START' system notice in receive().
        Exchange:   triggered by first valid chain state block.
        Default no-op; SimulationEngine overrides with timestamp-shift logic.
        """

    def on_end(self, state: Any) -> None:
        """
        Called when the engine signals a run end.
        Simulation: triggered by 'END' system notice in receive().
        Exchange:   triggered by graceful shutdown.
        Default no-op; SimulationEngine overrides with save + update_repo.
        """

    # ── Reset detection (called by validator.process_resets) ─────────────────

    def collect_resets(self, state: NormalizedState, pending: set) -> None:
        """
        Populate `pending` with UIDs that need scoring state reset this tick.

        Simulation: scans state.notices for RDRA/ERDRA system notices.
        Exchange:   no-op — deregistrations are detected in resync_metagraph()
                    and placed directly into validator.deregistered_uids.

        Args:
            state: The block's state update.
            pending: Set of uids to extend with reset targets.
        """

    # ── New engine interface methods ─────────────────────────────────────────

    @property
    @abstractmethod
    def effective_max_uids(self) -> int:
        """Total addressable UID count. Simulation: max_uids + benchmark. Exchange: max_uids."""

    @abstractmethod
    def initialize_structures(self) -> None:
        """Initialize all validator state structures for this mode. Called from start()."""

    @abstractmethod
    def build_simulation_state(self) -> dict:
        """Return mode-specific state dict for persistence (the 'simulation' save file)."""

    @abstractmethod
    def restore_simulation_state(self, data: dict) -> None:
        """Restore mode-specific state from dict loaded from disk."""

    @abstractmethod
    def handle_deregistration(self, uid: int, old_coldkey: str | None = None) -> None:
        """Mode-specific deregistration: zero score, flag for reset, etc.

        `old_coldkey` names the coldkey that held the slot, for engines that must retire something bound to
        it (the exchange engine retires the departed miner's settlement-proxy wallet). It is part of the
        signature HERE, not only on the implementation that needs it, because the validator calls every
        engine identically and a narrower override raises TypeError out of the metagraph resync rather
        than failing locally. Optional, because the base validator
        (taos/common/neurons/validator.py:391) calls it with uid alone.

        Args:
            uid: The deregistered uid.
            old_coldkey: The coldkey that held the slot, for engines that retire state bound to it.
        """

    @abstractmethod
    def apply_resets(self, pending: set) -> None:
        """Zero all scoring state for each UID in pending."""

    def get_extended_metagraph(self) -> Any:
        """Return metagraph extended with mode-specific virtual UIDs.
        Default: return real metagraph unchanged."""
        return self.validator.metagraph

    def on_resync_metagraph(self, old_size: int, new_size: int) -> None:
        """Called when metagraph size changes during resync. Default: no-op.

        Args:
            old_size: Metagraph size before the resync.
            new_size: Metagraph size after.
        """

    def on_tick(self, state: "NormalizedState") -> None:
        """
        Called once per state tick from handle_state(), before volume injection.
        Simulation: updates simulation_timestamp, detects logDir changes, fires
                    periodic compression and update_repo.
        Exchange:   default no-op.
        """

from taos.im.validator.engines.simulation import SimulationEngine
# ExchangeEngine is an optional exchange-mode component; not part of this tree. Consumers must import it
# explicitly, under a guard:
#   from taos.im.validator.engines.exchange import ExchangeEngine, ExchangeConfig
