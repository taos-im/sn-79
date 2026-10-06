# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
from __future__ import annotations

import time
import traceback
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import bittensor as bt

from taos.im.protocol.models import TradeInfo
from taos.im.protocol.events import TradeEvent, trade_event_from_wire
from taos.im.protocol import MarketSimulationStateUpdate
from taos.im.validator.debeta import (
    accumulate_book_capture, accumulate_book_mtm, accumulate_counterparties, et_book_batches,
    flush_capture_state,
    prune_hist_2level, prune_hist_1level, shift_hist_2level, shift_hist_1level,
)

# De-beta making: half-window (in trades) for the non-lagging centered mid used for spread capture.
CAPTURE_W = 15

if TYPE_CHECKING:
    from taos.im.neurons.validator import Validator


_missing_fee_notices = 0


def missing_fee_count() -> int:
    """How many notices arrived without a fee field. Diagnostic, not control flow."""
    return _missing_fee_notices


def reset_missing_fee_count() -> None:
    """Reset the missing-fee warning counter (used between runs and in tests)."""
    global _missing_fee_notices
    _missing_fee_notices = 0


class MissingNoticeField(Exception):
    """A notice arrived without a field its producer is required to populate.

    Never substituted with a default. A fee is an input to realized PnL, and in simulation fees are
    real, so quietly reading an absent Tf as 0.0 would UNDERSTATE cost and corrupt PnL for every
    affected trade. A wrong number that looks fine is worse than a missing update, because nothing
    downstream can tell it apart from a correct one.
    """


def notice_fee(trade: dict, is_maker: bool) -> float:
    """The fee this agent paid on a trade notice. Raises rather than inventing one.

    `fee = trade['Mf'] if is_maker else trade['Tf']` raised KeyError repeatedly in live
    running, because the hand-built exchange notice omitted both fee
    fields while the canonical to_notice_dict() emits them. That exception propagated out of
    _process_uid_notices, so _process_uid_trade_volumes aborted for the whole uid and its trade
    volumes, realized PnL and roundtrip volume were left unupdated for the block.

    The fix is that producers always populate these fields (all three ET builders now do, pinned by
    tests/test_notice_contract_across_layers.py), so an absence here is a producer bug and is reported
    as one. What changed is only the blast radius: the caller skips THAT NOTICE rather than losing the
    uid's entire update, and says so loudly.
    """
    global _missing_fee_notices
    key = "Mf" if is_maker else "Tf"
    if key not in trade:
        _missing_fee_notices += 1
        raise MissingNoticeField(
            f"notice has no '{key}': every ET builder is required to populate it, so this is a "
            f"producer defect. notice={ {k: trade.get(k) for k in ('y', 'b', 'i', 'q', 'p', 's')} }"
        )
    try:
        return float(trade[key])
    except (TypeError, ValueError) as exc:
        _missing_fee_notices += 1
        raise MissingNoticeField(f"notice '{key}' is not a number: {trade[key]!r}") from exc


def _apply_pnl_delta(self: Validator, uid: int, book_id: int, delta: float) -> None:
    """Incrementally update the (uid → book → pnl) and (uid → pnl) running
    totals used by the MVTRX push payload's agent_pnl_book / agent_pnl
    fields. Called from every site in this file that mutates
    realized_pnl_history so the payload builder can read the totals
    directly instead of re-walking the whole history (O(N*T*B)) each cycle.
    Delta is the change to be added — positive for adds, negative for
    removes. Zero delta is a no-op.
    """
    if delta == 0.0:
        return
    self.agent_pnl_by_book[uid][book_id] += delta
    self.agent_pnl_total[uid] += delta


def bootstrap_pnl_totals(self: Validator) -> None:
    """Rebuild agent_pnl_by_book / agent_pnl_total from realized_pnl_history.
    Called after state load and after any bulk rebuild of the history
    (e.g. simulation restart / prune-shift). O(N*T*B) — the very cost the
    running totals are designed to avoid on the hot path — but it runs
    once at boot / restart, not per state cycle.
    """
    self.agent_pnl_by_book = defaultdict(lambda: defaultdict(float))
    self.agent_pnl_total = defaultdict(float)
    for uid, hist in self.realized_pnl_history.items():
        per_book: dict[int, float] = {}
        total = 0.0
        for ts_d in hist.values():
            for book_id, pnl in ts_d.items():
                per_book[book_id] = per_book.get(book_id, 0.0) + pnl
                total += pnl
        for book_id, v in per_book.items():
            self.agent_pnl_by_book[uid][book_id] = v
        self.agent_pnl_total[uid] = total


def match_trade_fifo(self: Validator, uid: int, book_id: int, is_buy: bool, quantity: float,
                    price: float, fee: float, timestamp: int) -> tuple[float, float]:
    """
    FIFO matching including fee accounting.
    Args:
        uid: Miner UID
        book_id: Book identifier
        is_buy: True if buying (going long), False if selling (going short)
        quantity: Trade quantity
        price: Trade price
        fee: Fee paid for this trade (positive = cost, negative = rebate)
        timestamp: Trade timestamp

    Returns:
        tuple[float, float]: (realized_pnl, roundtrip_volume)
            - realized_pnl: Realized P&L from matched trades (including fees)
            - roundtrip_volume: Total quantity that completed a round-trip
    """
    positions = self.open_positions[uid][book_id]

    if is_buy:
        shorts = positions['shorts']
        if not shorts:
            positions['longs'].append((timestamp, quantity, price, fee))
            return 0.0, 0.0
    else:
        longs = positions['longs']
        if not longs:
            positions['shorts'].append((timestamp, quantity, price, fee))
            return 0.0, 0.0

    realized_pnl = 0.0
    roundtrip_volume = 0.0
    remaining_qty = quantity

    quantity_inv = 1.0 / quantity if quantity > 0 else 0.0

    if is_buy:
        # Buying: close shorts first (FIFO), then open longs
        while remaining_qty > 0 and shorts:
            old_ts, old_qty, old_price, old_fee = shorts[0]

            if old_qty <= remaining_qty:
                # Fully close this short position
                price_pnl = (old_price - price) * old_qty
                close_fee = fee * old_qty * quantity_inv
                realized_pnl += price_pnl - old_fee - close_fee
                roundtrip_volume += old_qty
                remaining_qty -= old_qty
                shorts.popleft()
            else:
                # Partially close short position
                old_qty_inv = 1.0 / old_qty

                price_pnl = (old_price - price) * remaining_qty
                # Prorate the fill fee to the portion that closes here; earlier
                # fully-closed lots in this same fill already took their share.
                close_fee = fee * remaining_qty * quantity_inv
                open_fee = old_fee * remaining_qty * old_qty_inv
                realized_pnl += price_pnl - open_fee - close_fee
                roundtrip_volume += remaining_qty

                # Update remaining position with reduced fee
                remaining_position_fee = old_fee - open_fee
                shorts[0] = (old_ts, old_qty - remaining_qty, old_price, remaining_position_fee)
                remaining_qty = 0

        # Any remaining quantity opens new long position
        if remaining_qty > 0:
            open_fee = fee * remaining_qty * quantity_inv
            positions['longs'].append((timestamp, remaining_qty, price, open_fee))

    else:
        # Selling: close longs first (FIFO), then open shorts
        while remaining_qty > 0 and longs:
            old_ts, old_qty, old_price, old_fee = longs[0]

            if old_qty <= remaining_qty:
                # Fully close this long position
                price_pnl = (price - old_price) * old_qty
                close_fee = fee * old_qty * quantity_inv
                realized_pnl += price_pnl - old_fee - close_fee
                roundtrip_volume += old_qty
                remaining_qty -= old_qty
                longs.popleft()
            else:
                # Partially close long position
                old_qty_inv = 1.0 / old_qty

                price_pnl = (price - old_price) * remaining_qty
                # Prorate the fill fee to the portion that closes here; earlier
                # fully-closed lots in this same fill already took their share.
                close_fee = fee * remaining_qty * quantity_inv
                open_fee = old_fee * remaining_qty * old_qty_inv
                realized_pnl += price_pnl - open_fee - close_fee
                roundtrip_volume += remaining_qty

                # Update remaining position with reduced fee
                remaining_position_fee = old_fee - open_fee
                longs[0] = (old_ts, old_qty - remaining_qty, old_price, remaining_position_fee)
                remaining_qty = 0

        # Any remaining quantity opens new short position
        if remaining_qty > 0:
            open_fee = fee * remaining_qty * quantity_inv
            positions['shorts'].append((timestamp, remaining_qty, price, open_fee))

    return realized_pnl, roundtrip_volume




def _process_uid_notices(self, uid_item, notices, timestamp, sampled_timestamp, trade_volumes_uid, volume_deltas, realized_pnl_updates, roundtrip_volume_updates, uids_to_round):
    """Process this UID's trade notices: per-role (maker/taker/self) volume,
    FIFO realized-P&L + round-trip volume, and recent-trade buffers. Pure
    extraction of the notice loop from _process_uid_trade_volumes.
    """
    if uid_item in notices:
        trades = [notice for notice in notices[uid_item] if notice.get('y') in ['EVENT_TRADE', "ET"]]
        if trades:
            recent_miner_trades_uid = self.recent_miner_trades[uid_item]
            if uid_item not in volume_deltas:
                volume_deltas[uid_item] = {}

            # Settled-fill notices are DELIBERATELY redelivered on every update for
            # fill_notice_window_seconds (exchange_notices.merge_settled_fills), so the same fill
            # arrives ~75 times at a 900s window and ~12s updates. Counting each delivery inflated
            # volume, the maker/taker splits, roundtrip volume and the FIFO realized-PnL history by
            # that factor. de-beta already guards this with _debeta_seen_tids; this is the same
            # guard for the volume/PnL consumer. Keyed by (uid, book, trade id): ONE trade is
            # listed in both the maker's and the taker's notices, so a global ledger would let the
            # first uid processed consume it and starve the counterparty of its own side; and a
            # trade id names a trade only within its book.
            #
            # EXCHANGE MODE ONLY. Redelivery is an exchange mechanism; the simulation delivers each
            # fill once. In the simulation every book mints its own trade ids from 0, so the ids
            # overlap across all 128 books, and they start again at every simulation roll while a
            # ledger pruned by timestamp keeps the old ones. Keyed by (uid, trade id) and applied
            # in both modes, this ledger silently dropped a fill on book B whenever book A had
            # already used the same id (reported by a miner on the 0.6.1 testnet ratchet, # who estimated 15 to 50 percent of a heavy uid's fills lost from volume, the
            # maker/taker split, roundtrip volume and the FIFO realized-PnL leg).
            _dedup_fills = getattr(getattr(self, 'engine', None), 'mode', 'simulation') == 'exchange'
            if not hasattr(self, '_volume_seen_tids'):
                self._volume_seen_tids = {}
            _seen_tids = self._volume_seen_tids

            for trade in trades:
                _tid = trade.get('i')
                if _dedup_fills and _tid is not None:
                    _seen_key = (uid_item, trade.get('b'), _tid)
                    if _seen_key in _seen_tids:
                        continue
                    _seen_tids[_seen_key] = sampled_timestamp
                # Check the required fields up front so one malformed notice costs that notice and
                # nothing more. A missing 'Tf' raised out of this whole function costs the
                # uid its trade volumes, realized PnL and roundtrip volume for the block, and the
                # same absence separately breaks metrics publishing in report.py, where the notice is
                # rebuilt with TradeEvent.model_construct and a missing key becomes a missing
                # ATTRIBUTE. Reported as the producer defect it is, never defaulted: a fabricated zero
                # fee would understate cost and corrupt PnL wherever fees are real.
                try:
                    notice_fee(trade, trade.get('Ma') == uid_item)
                except MissingNoticeField as exc:
                    bt.logging.error(f"PD: skipping malformed trade notice for UID {uid_item}: {exc}")
                    continue

                is_maker = trade['Ma'] == uid_item
                is_taker = trade['Ta'] == uid_item
                book_id = trade['b']

                # Update recent miner trades
                recent_miner_trades_uid.setdefault(book_id, [])
                # trade_event_from_wire, not a bare model_construct: the engine's integer close reason
                # must become the model's string form here, or the copy trips the serializer on save.
                if is_maker:
                    recent_miner_trades_uid[book_id].append([trade_event_from_wire(trade), "maker"])
                if is_taker:
                    recent_miner_trades_uid[book_id].append([trade_event_from_wire(trade), "taker"])
                if len(recent_miner_trades_uid[book_id]) > 5:
                    del recent_miner_trades_uid[book_id][:-5]

                if book_id not in trade_volumes_uid:
                    trade_volumes_uid[book_id] = {'total': {sampled_timestamp: 0.0}, 'maker': {sampled_timestamp: 0.0}, 'taker': {sampled_timestamp: 0.0}, 'self': {sampled_timestamp: 0.0}}
                book_volumes = trade_volumes_uid[book_id]
                trade_value = trade['q'] * trade['p']
                if book_id not in volume_deltas[uid_item]:
                    volume_deltas[uid_item][book_id] = {'total': 0.0, 'maker': 0.0, 'taker': 0.0, 'self': 0.0, 'fee': 0.0}

                book_volumes['total'][sampled_timestamp] += trade_value
                volume_deltas[uid_item][book_id]['total'] += trade_value

                if trade['Ma'] == trade['Ta']:
                    book_volumes['self'][sampled_timestamp] += trade_value
                    volume_deltas[uid_item][book_id]['self'] += trade_value
                elif is_maker:
                    book_volumes['maker'][sampled_timestamp] += trade_value
                    volume_deltas[uid_item][book_id]['maker'] += trade_value
                elif is_taker:
                    book_volumes['taker'][sampled_timestamp] += trade_value
                    volume_deltas[uid_item][book_id]['taker'] += trade_value

                uids_to_round.add(uid_item)

                # A SELF-TRADE IS POSITION-NEUTRAL, so it must not enter FIFO at all.
                #
                # The volume split above already routes Ma == Ta to the 'self' bucket, but the FIFO
                # block below it was unconditional. With Ma == Ta both is_maker and is_taker are true,
                # so `is_buy` evaluates True for EITHER value of s, and the notice books a directional
                # leg — fabricating realized PnL and roundtrip volume from a trade in which the miner
                # was both sides and its position did not move. Both feed scoring.
                #
                # No such notice can arrive today: the sim instruction model omits STP.NO_STP from its
                # Literal so a sim self-match is always cancelled, and in exchange mode
                # resolve_trade_roles refuses to name the same uid on both sides. This is defence
                # against that invariant changing, not a fix for a live path. It is cheap and it is
                # the correct accounting either way. Raised by code review, which noted the
                # NO_STP justification in query.py leans on a protection that covers volume only.
                if trade['Ma'] is not None and trade['Ma'] == trade['Ta']:
                    continue

                # FIFO Matching: Calculate realized P&L and round-trip volume
                quantity = trade['q']
                price = trade['p']
                side = trade['s']
                is_buy = (is_taker and side == 0) or (is_maker and side == 1)
                fee = notice_fee(trade, is_maker)
                volume_deltas[uid_item][book_id]['fee'] += fee

                realized_pnl, roundtrip_volume = match_trade_fifo(
                    self, uid_item, book_id, is_buy, quantity, price, fee, timestamp
                )

                if realized_pnl != 0.0:
                    if uid_item not in realized_pnl_updates:
                        realized_pnl_updates[uid_item] = {}
                    if timestamp not in realized_pnl_updates[uid_item]:
                        realized_pnl_updates[uid_item][timestamp] = {}
                    if book_id not in realized_pnl_updates[uid_item][timestamp]:
                        realized_pnl_updates[uid_item][timestamp][book_id] = 0.0
                    realized_pnl_updates[uid_item][timestamp][book_id] += realized_pnl

                if roundtrip_volume > 0:
                    roundtrip_value = roundtrip_volume * price
                    if uid_item not in roundtrip_volume_updates:
                        roundtrip_volume_updates[uid_item] = {}
                    if sampled_timestamp not in roundtrip_volume_updates[uid_item]:
                        roundtrip_volume_updates[uid_item][sampled_timestamp] = {}
                    if book_id not in roundtrip_volume_updates[uid_item][sampled_timestamp]:
                        roundtrip_volume_updates[uid_item][sampled_timestamp][book_id] = 0.0
                    roundtrip_volume_updates[uid_item][sampled_timestamp][book_id] += roundtrip_value

            for book_id, deltas in volume_deltas[uid_item].items():
                self.volume_sums[uid_item][book_id] = self.volume_sums[uid_item].get(book_id, 0.0) + deltas['total']
                self.maker_volume_sums[uid_item][book_id] = self.maker_volume_sums[uid_item].get(book_id, 0.0) + deltas['maker']
                self.taker_volume_sums[uid_item][book_id] = self.taker_volume_sums[uid_item].get(book_id, 0.0) + deltas['taker']
                self.self_volume_sums[uid_item][book_id] = self.self_volume_sums[uid_item].get(book_id, 0.0) + deltas['self']
                self.fee_sums[uid_item][book_id] = self.fee_sums[uid_item].get(book_id, 0.0) + deltas['fee']
    # Initialize zero P&L for timestamps with no trades

def _process_uid_trade_volumes(self, uid_item, books, accounts, notices, timestamp, sampled_timestamp, should_prune, volume_prune_threshold, volume_decimals, volume_deltas, realized_pnl_updates, roundtrip_volume_updates, uids_to_round):
    """Per-UID trade-volume / FIFO-PnL / inventory processing.

    Pure extraction of the per-UID loop body of update_trade_volumes; logic
    unchanged. Shared accumulators (volume_deltas, realized_pnl_updates,
    roundtrip_volume_updates, uids_to_round) are mutated in place by reference.
    """
    from taos.im.utils.reward import get_inventory_value
    # Initialize trade volumes structure if needed
    if uid_item not in self.trade_volumes:
        self.trade_volumes[uid_item] = {
            book_id: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
            for book_id in books.keys()
        }
    trade_volumes_uid = self.trade_volumes[uid_item]

    # Prune old volumes and update sums
    if should_prune:
        for book_id, role_trades in trade_volumes_uid.items():
            for role, trades in role_trades.items():
                if not trades:
                    continue
                old_count = len(trades)
                pruned = {t: v for t, v in trades.items() if t >= volume_prune_threshold}
                if len(pruned) < old_count:
                    pruned_volume = sum(v for t, v in trades.items() if t < volume_prune_threshold)
                    if pruned_volume > 0:
                        if role == 'total':
                            self.volume_sums[uid_item][book_id] = max(0.0, self.volume_sums[uid_item][book_id] - pruned_volume)
                        elif role == 'maker':
                            self.maker_volume_sums[uid_item][book_id] = max(0.0, self.maker_volume_sums[uid_item][book_id] - pruned_volume)
                        elif role == 'taker':
                            self.taker_volume_sums[uid_item][book_id] = max(0.0, self.taker_volume_sums[uid_item][book_id] - pruned_volume)
                        elif role == 'self':
                            self.self_volume_sums[uid_item][book_id] = max(0.0, self.self_volume_sums[uid_item][book_id] - pruned_volume)
                        uids_to_round.add(uid_item)
                    trade_volumes_uid[book_id][role] = pruned

    # Initialize sampled timestamp entries
    for book_id in books.keys():
        if book_id not in trade_volumes_uid:
            trade_volumes_uid[book_id] = {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
        book_trade_volumes = trade_volumes_uid[book_id]
        if sampled_timestamp not in book_trade_volumes['total']:
            book_trade_volumes['total'][sampled_timestamp] = 0.0
            book_trade_volumes['maker'][sampled_timestamp] = 0.0
            book_trade_volumes['taker'][sampled_timestamp] = 0.0
            book_trade_volumes['self'][sampled_timestamp] = 0.0

    # Process trade notices
    _process_uid_notices(self, uid_item, notices, timestamp, sampled_timestamp, trade_volumes_uid, volume_deltas, realized_pnl_updates, roundtrip_volume_updates, uids_to_round)
    if timestamp not in self.realized_pnl_history[uid_item]:
        self.realized_pnl_history[uid_item][timestamp] = {}

    # Update inventory history
    if uid_item in accounts:
        initial_balances_uid = self.initial_balances[uid_item]
        accounts_uid = accounts[uid_item]

        for bookId, account in accounts_uid.items():
            if bookId not in initial_balances_uid:
                initial_balances_uid[bookId] = {'BASE': None, 'QUOTE': None, 'WEALTH': None}
            initial_balance_book = initial_balances_uid[bookId]
            if initial_balance_book['BASE'] is None:
                initial_balance_book['BASE'] = account.get('BASE', (account.get('bb') or {}).get('t', 0.0))
            if initial_balance_book['QUOTE'] is None:
                initial_balance_book['QUOTE'] = account.get('QUOTE', (account.get('qb') or {}).get('t', 0.0))
            if initial_balance_book['WEALTH'] is None:
                initial_balance_book['WEALTH'] = account['WEALTH'] if 'WEALTH' in account else get_inventory_value(account, books[bookId])

        current_inventory = {
            book_id: (accounts_uid[book_id]['WEALTH'] if 'WEALTH' in accounts_uid[book_id] else get_inventory_value(accounts_uid[book_id], book)) - initial_balances_uid[book_id]['WEALTH']
            for book_id, book in books.items()
            if book_id in accounts_uid
        }
        if uid_item not in self.inventory_history:
            self.inventory_history[uid_item] = {}
        hist = self.inventory_history[uid_item]
        if not hist:
            hist[timestamp] = current_inventory
        else:
            timestamps = sorted(hist.keys())
            if len(timestamps) == 1:
                hist[timestamp] = current_inventory
            else:
                first_ts = timestamps[0]
                self.inventory_history[uid_item] = {
                    first_ts: hist[first_ts],
                    timestamps[-1]: hist[timestamps[-1]],
                    timestamp: current_inventory
                }
    else:
        self.inventory_history[uid_item][timestamp] = {book_id: 0.0 for book_id in books}


def update_trade_volumes(self: Validator, state: MarketSimulationStateUpdate):
    """
    Updates and maintains all trade volume tracking and position accounting structures.

    This function processes raw trade events from the simulator state and updates
    the following per-UID per-book time series:

    **Volume Tracking:**
    • **total** — total traded notional value
    • **maker** — maker-side volume
    • **taker** — taker-side volume
    • **self** — trades where maker == taker
    • **roundtrip_volumes** — volume from completed round-trip trades (open + close)
    • **volume_sums** / **maker_volume_sums** / **taker_volume_sums** / **self_volume_sums** / **roundtrip_volume_sums**

    **Position Accounting (FIFO):**
    • **open_positions** — tracks open long/short positions with (timestamp, quantity, price, fee)
    • **realized_pnl_history** — realized profit/loss from closed positions (fee-adjusted)
    • Matches trades via FIFO to calculate realized P&L and round-trip volume

    **Inventory & History:**
    • **inventory_history** — mark-to-market inventory value changes over time
    • **recent_trades** — rolling buffer of last 25 trades per book
    • **recent_miner_trades** — rolling buffer of last 5 trades per miner per book
    • **initial_balances** — baseline balances for inventory value calculations

    **Operations:**
    • Samples volume at aligned timestamps (trade_volume_sampling_interval)
    • Prunes old volume entries outside assessment window (trade_volume_assessment_period)
    • Prunes old inventory and realized P&L history outside Kappa lookback window
    • Batch processes updates for performance (deferred rounding)
    • Ensures all nested structures are initialized dynamically

    Args:
        state (MarketSimulationStateUpdate):
            Full simulation tick state containing books, accounts, and notices.

    Returns:
        None

    Raises:
        Logs errors when UID-level processing fails but continues processing remaining UIDs.
    """
    total_start = time.time()

    books = state.books
    timestamp = state.timestamp
    accounts = state.accounts
    notices = state.notices

    volume_decimals = self.simulation.volumeDecimals

    sampled_timestamp = (timestamp // self.config.scoring.activity.trade_volume_sampling_interval) * self.config.scoring.activity.trade_volume_sampling_interval

    if not hasattr(self, '_last_prune_timestamp'):
        self._last_prune_timestamp = None

    if self._last_prune_timestamp:
        time_since_prune = timestamp - self._last_prune_timestamp
        prune_interval = 60_000_000_000
        should_prune = time_since_prune >= prune_interval
    else:
        should_prune = True
    if should_prune:
        self._last_prune_timestamp = timestamp
        bt.logging.info(f"Pruning at step {self.step} (timestamp {timestamp})")
    volume_prune_threshold = timestamp - self.config.scoring.activity.trade_volume_assessment_period

    # De-beta (P8) making + drift-strip skill inputs: accumulated over the ordered per-book fill
    # stream, carried across publish batches. ALWAYS ON since scoring.debeta.weight replaced the
    # enabled boolean: the decomposition is published at every weight (0 = legacy emissions with
    # full visibility), so the accumulators can never silently go dark behind a flag.
    _debeta_on = True
    if _debeta_on:
        # Running SUMS (what the score reads).
        for _name in ('capture_buy_sums', 'capture_sell_sums', 'realized_buy_sums', 'realized_sell_sums', 'debeta_mtm', 'debeta_invsum',
                      'debeta_heldn', 'debeta_heldinv', 'debeta_helddrift', 'debeta_notional'):
            if not hasattr(self, _name):
                setattr(self, _name, defaultdict(lambda: defaultdict(float)))
        if not hasattr(self, 'debeta_inv'):
            self.debeta_inv = defaultdict(lambda: defaultdict(float))  # STATE (carried, re-based at boundary)
        for _name in ('debeta_invn', 'debeta_pfirst', 'debeta_plast', 'debeta_drift'):
            if not hasattr(self, _name):
                setattr(self, _name, {})  # invn/drift running (per book); pfirst/plast STATE
        if not hasattr(self, 'debeta_mark_state'):
            self.debeta_mark_state = {}  # STATE (M1 rolling settlement-mark window; re-based at boundary, not persisted)
        if not hasattr(self, 'debeta_capture_mid'):
            # STATE {book: rolling print window + pending fills}: carries the capture mid ACROSS
            # state updates (median 4 prints/update live, so a batch-local mid degrades to the batch
            # mean 91% of the time). Bounded (<= W pending + 2W+1 prices per book); drained at the
            # sim boundary, not persisted.
            self.debeta_capture_mid = {}
        _debeta_cfg = getattr(self.config.scoring, 'debeta', None)
        _mark_mode = str(getattr(_debeta_cfg, 'mark_mode', 'last') or 'last')
        _mark_window = int(getattr(_debeta_cfg, 'mark_window', 0) or 0)
        # realized spread (2 Oct 2026): the horizon a fill is marked at, sim seconds; accumulated at every making_basis
        _realized_ns = int(float(getattr(_debeta_cfg, 'making_horizon_s', 20.0) or 20.0) * 1e9)
        if not hasattr(self, 'debeta_cp'):
            self.debeta_cp = {}  # {maker_uid: {taker_uid: vol}} for P11 (running sum)
            self.debeta_cpc = {}  # {class: {maker_uid: {taker_uid: vol}}}: the same rows per asset class
            self.debeta_cpc_hist = {}  # {class: {maker_uid: {taker_uid: {ts: vol}}}}
        # Timestamped HISTORIES ({...:{sampled_ts: incr}}) so every sum can be live-pruned + shifted at a
        # sim boundary exactly like trade_volumes. Invariant: running == sum(history within window).
        for _name in ('debeta_capbuy_hist', 'debeta_capsell_hist', 'debeta_realbuy_hist', 'debeta_realsell_hist', 'debeta_mtm_hist',
                      'debeta_invsum_hist', 'debeta_invn_hist', 'debeta_drift_hist', 'debeta_cp_hist',
                      'debeta_heldn_hist', 'debeta_heldinv_hist', 'debeta_helddrift_hist', 'debeta_notional_hist'):
            if not hasattr(self, _name):
                setattr(self, _name, {})

    # De-beta fill SOURCE: sim reads the book 't' events; exchange reads the ET settled-fill notices
    # (correct aggressor=taker role; AMM/pool fills reach de-beta only here), deduped by trade id and
    # built once. Stays under the _debeta_on gate, so the OFF and sim paths are byte-identical.
    _debeta_exchange = _debeta_on and getattr(getattr(self, 'engine', None), 'mode', 'simulation') == 'exchange'
    _et_batches = {}
    if _debeta_exchange:
        if not hasattr(self, '_debeta_seen_tids'):
            self._debeta_seen_tids = {}  # {trade_id: ts} carried + windowed-pruned; redelivery dedupe
        _et_batches = et_book_batches(notices, self._debeta_seen_tids, sampled_timestamp)

    for bookId, book in books.items():
        trades = [event for event in book.get('e', []) if event['y'] == 't']
        if _debeta_on:
            # sim: the book 't' events; exchange: the ET batch for this book (same accumulators). Never
            # both, so a fill is not double-counted between the parallel 't' copy and the ET notice.
            de_trades = _et_batches.get(bookId, []) if _debeta_exchange else trades
            if de_trades:
                accumulate_book_capture(
                    self.capture_buy_sums, self.capture_sell_sums, bookId, de_trades, CAPTURE_W,
                    buy_hist=self.debeta_capbuy_hist, sell_hist=self.debeta_capsell_hist,
                    ts=sampled_timestamp, mid_state=self.debeta_capture_mid,
                    real_buy_sums=self.realized_buy_sums, real_sell_sums=self.realized_sell_sums,
                    real_buy_hist=self.debeta_realbuy_hist, real_sell_hist=self.debeta_realsell_hist,
                    horizon_ns=_realized_ns,
                )
                accumulate_book_mtm(
                    self.debeta_mtm, self.debeta_invsum, self.debeta_invn, self.debeta_inv,
                    self.debeta_pfirst, self.debeta_plast, bookId, de_trades,
                    mtm_hist=self.debeta_mtm_hist, invsum_hist=self.debeta_invsum_hist,
                    invn_hist=self.debeta_invn_hist, drift=self.debeta_drift,
                    drift_hist=self.debeta_drift_hist, ts=sampled_timestamp,
                    mark_state=self.debeta_mark_state, mark_mode=_mark_mode,
                    mark_window=_mark_window,
                    heldn=self.debeta_heldn, heldinv=self.debeta_heldinv, helddrift=self.debeta_helddrift,
                    notional=self.debeta_notional, heldn_hist=self.debeta_heldn_hist,
                    heldinv_hist=self.debeta_heldinv_hist, helddrift_hist=self.debeta_helddrift_hist,
                    notional_hist=self.debeta_notional_hist,
                )
                accumulate_counterparty_rows(self, bookId, de_trades, ts=sampled_timestamp)  # P11, pooled and per class
        if trades:
            if bookId not in self.recent_trades:
                self.recent_trades[bookId] = []
            recent_trades_book = self.recent_trades[bookId]
            recent_trades_book.extend([
                TradeInfo.model_construct(
                    **{k: v for k, v in t.items() if k not in ('Ti', 'Ta', 'Mi', 'Ma', 'i', 'Tf', 'Mf')},
                    i  = t.get('i',  0),
                    Ti = t.get('Ti', 0),
                    Ta = t.get('Ta', -1),
                    Mi = t.get('Mi', 0),
                    Ma = t.get('Ma', -1),
                    Tf = t.get('Tf', None),
                    Mf = t.get('Mf', None),
                )
                for t in trades
            ])
            del recent_trades_book[:-25]

    # A book with no fills this update never reaches accumulate_book_capture, so drain its stale
    # pending capture fills here (bounded lag: a fill waits at most CAPTURE_FLUSH_NS for its
    # forward window, then finalizes truncated).
    if _debeta_on and getattr(self, 'debeta_capture_mid', None):
        flush_capture_state(self.debeta_capture_mid, self.capture_buy_sums, self.capture_sell_sums,
                            CAPTURE_W, buy_hist=self.debeta_capbuy_hist,
                            sell_hist=self.debeta_capsell_hist, ts=sampled_timestamp,
                            real_buy_sums=self.realized_buy_sums, real_sell_sums=self.realized_sell_sums,
                            real_buy_hist=self.debeta_realbuy_hist, real_sell_hist=self.debeta_realsell_hist,
                            horizon_ns=_realized_ns)

    # De-beta live prune. BOTH legs share ONE window, the kappa lookback. Pruning making on the
    # 24h volume-assessment window, each leg inheriting the retention of the legacy leg it
    # replaced, splits them; de-beta replaces both, so that split is an artifact of implementation. It also
    # pointed the wrong way: making alignment against two-sidedness falls monotonically as the
    # window lengthens (measured 23 windows, all sliding positions: +0.175 at 2h to +0.067 at 24h),
    # because over a long window a directional miner accumulates offsetting flow across time and
    # presents as balanced. Subtracts pruned mass from the running sums so
    # running == sum(history within window).
    if _debeta_on and should_prune:
        _debeta_prune_threshold = timestamp - self.config.scoring.kappa.lookback
        prune_hist_2level(self.debeta_capbuy_hist, self.capture_buy_sums, _debeta_prune_threshold)
        prune_hist_2level(self.debeta_capsell_hist, self.capture_sell_sums, _debeta_prune_threshold)
        prune_hist_2level(self.debeta_realbuy_hist, self.realized_buy_sums, _debeta_prune_threshold)
        prune_hist_2level(self.debeta_realsell_hist, self.realized_sell_sums, _debeta_prune_threshold)
        prune_hist_2level(self.debeta_cp_hist, self.debeta_cp, _debeta_prune_threshold)
        for _c, _h in (getattr(self, 'debeta_cpc_hist', None) or {}).items():
            prune_hist_2level(_h, self.debeta_cpc.setdefault(_c, {}), _debeta_prune_threshold)
        prune_hist_2level(self.debeta_mtm_hist, self.debeta_mtm, _debeta_prune_threshold)
        prune_hist_2level(self.debeta_invsum_hist, self.debeta_invsum, _debeta_prune_threshold)
        prune_hist_2level(self.debeta_heldn_hist, self.debeta_heldn, _debeta_prune_threshold)
        prune_hist_2level(self.debeta_heldinv_hist, self.debeta_heldinv, _debeta_prune_threshold)
        prune_hist_2level(self.debeta_helddrift_hist, self.debeta_helddrift, _debeta_prune_threshold)
        prune_hist_2level(self.debeta_notional_hist, self.debeta_notional, _debeta_prune_threshold)
        prune_hist_1level(self.debeta_invn_hist, self.debeta_invn, _debeta_prune_threshold)
        prune_hist_1level(self.debeta_drift_hist, self.debeta_drift, _debeta_prune_threshold)
        if hasattr(self, '_debeta_seen_tids'):  # bound the exchange dedup ledger to the same window
            self._debeta_seen_tids = {k: v for k, v in self._debeta_seen_tids.items()
                                      if v >= _debeta_prune_threshold}
        if hasattr(self, '_volume_seen_tids'):  # same, for the volume/PnL dedup ledger
            self._volume_seen_tids = {k: v for k, v in self._volume_seen_tids.items()
                                      if v >= volume_prune_threshold}

    volume_deltas = {}
    realized_pnl_updates = {}
    roundtrip_volume_updates = {}
    uids_to_round = set()

    uid_count = 0
    for uid_item in range(self.effective_max_uids):
        uid_count += 1
        try:
            _process_uid_trade_volumes(
                self, uid_item, books, accounts, notices, timestamp, sampled_timestamp,
                should_prune, volume_prune_threshold, volume_decimals, volume_deltas,
                realized_pnl_updates, roundtrip_volume_updates, uids_to_round,
            )
        except Exception as ex:
            self.pagerduty_alert(f"Failed to update trade data for UID {uid_item}: {ex}", details={"trace": traceback.format_exc()})

    if should_prune:
        lookback_time = self.config.scoring.kappa.lookback
        lookback_threshold = timestamp - lookback_time
        # Both realized_pnl_history[uid] and roundtrip_volumes[uid][book] are
        # keyed by simulation timestamp inserted monotonically — exactly one
        # append per round (lines below), value-updates to an existing ts leave
        # dict insertion order unchanged, and msgpack save/restore preserves it.
        # So expired entries (ts < threshold) are always a contiguous HEAD; we
        # delete that head in place and break at the first retained ts, making
        # each prune O(expired) instead of O(total). The old form did two full
        # scans + a whole-dict rebuild per uid (~3M ops/prune at mainnet
        # cardinality) every 60s while holding _reward_lock on the round path.
        for uid_item in self.realized_pnl_history:
            pnl_hist = self.realized_pnl_history[uid_item]
            if not pnl_hist:
                continue
            # Sum pnl in the expired head so we can subtract the equivalent from
            # the running totals — otherwise the push builder's agent_pnl_book
            # would include already-pruned data. Batch the subtraction per
            # (uid, book_id).
            book_deltas = {}
            uid_total_delta = 0.0
            expired_ts = []
            for ts in pnl_hist:
                if ts >= lookback_threshold:
                    break
                for book_id, pnl in pnl_hist[ts].items():
                    book_deltas[book_id] = book_deltas.get(book_id, 0.0) - pnl
                    uid_total_delta -= pnl
                expired_ts.append(ts)
            if book_deltas:
                per_book = self.agent_pnl_by_book[uid_item]
                for book_id, delta in book_deltas.items():
                    per_book[book_id] = per_book.get(book_id, 0.0) + delta
                self.agent_pnl_total[uid_item] = self.agent_pnl_total.get(uid_item, 0.0) + uid_total_delta
            for ts in expired_ts:
                del pnl_hist[ts]
        for uid_item in self.roundtrip_volumes:
            roundtrip_volumes_uid = self.roundtrip_volumes[uid_item]

            for book_id, rt_volumes in roundtrip_volumes_uid.items():
                if not rt_volumes:
                    continue
                pruned_rt_volume = 0.0
                expired_ts = []
                for t in rt_volumes:
                    if t >= volume_prune_threshold:
                        break
                    pruned_rt_volume += rt_volumes[t]
                    expired_ts.append(t)
                if expired_ts:
                    if pruned_rt_volume > 0:
                        current = self.roundtrip_volume_sums[uid_item][book_id]
                        self.roundtrip_volume_sums[uid_item][book_id] = max(0.0, current - pruned_rt_volume)
                        uids_to_round.add(uid_item)
                    for t in expired_ts:
                        del rt_volumes[t]

    for uid_item, timestamps in realized_pnl_updates.items():
        if uid_item not in self.realized_pnl_history:
            self.realized_pnl_history[uid_item] = {}
        for ts, books in timestamps.items():
            if ts not in self.realized_pnl_history[uid_item]:
                self.realized_pnl_history[uid_item][ts] = {}
            ts_pnl = self.realized_pnl_history[uid_item][ts]
            for book_id, pnl in books.items():
                rounded_pnl = round(pnl, volume_decimals)
                if rounded_pnl == 0.0:
                    continue
                current = ts_pnl.get(book_id, 0.0)
                new_value = round(current + rounded_pnl, volume_decimals)
                if new_value != 0.0:
                    ts_pnl[book_id] = new_value
                    # Running-total maintenance: delta = new - current so the
                    # rounded-value drift stays consistent with what's stored.
                    _apply_pnl_delta(self, uid_item, book_id, new_value - current)
                elif book_id in ts_pnl:
                    # Explicit removal: subtract the entry we're deleting from
                    # the running total so agent_pnl_book stays in sync.
                    del ts_pnl[book_id]
                    _apply_pnl_delta(self, uid_item, book_id, -current)
    for uid_item, timestamps in roundtrip_volume_updates.items():
        for ts, books in timestamps.items():
            for book_id, rt_vol in books.items():
                if uid_item not in self.roundtrip_volumes:
                    self.roundtrip_volumes[uid_item] = defaultdict(lambda: defaultdict(float))
                if book_id not in self.roundtrip_volumes[uid_item]:
                    self.roundtrip_volumes[uid_item][book_id] = defaultdict(float)
                if ts not in self.roundtrip_volumes[uid_item][book_id]:
                    self.roundtrip_volumes[uid_item][book_id][ts] = 0.0
                self.roundtrip_volumes[uid_item][book_id][ts] += rt_vol
                self.roundtrip_volume_sums[uid_item][book_id] = self.roundtrip_volume_sums[uid_item].get(book_id, 0.0) + rt_vol
                uids_to_round.add(uid_item)
    for uid_item in uids_to_round:
        changed_books = set(volume_deltas.get(uid_item, {}).keys())

        if uid_item in roundtrip_volume_updates:
            for ts_books in roundtrip_volume_updates[uid_item].values():
                changed_books.update(ts_books.keys())
        if not changed_books:
            changed_books = books.keys()

        for book_id in changed_books:
            if uid_item in self.trade_volumes and book_id in self.trade_volumes[uid_item]:
                book_vols = self.trade_volumes[uid_item][book_id]
                for role in ['total', 'maker', 'taker', 'self']:
                    if sampled_timestamp in book_vols[role]:
                        book_vols[role][sampled_timestamp] = round(book_vols[role][sampled_timestamp], volume_decimals)

            if book_id in self.volume_sums[uid_item]:
                self.volume_sums[uid_item][book_id] = round(self.volume_sums[uid_item][book_id], volume_decimals)
            if book_id in self.maker_volume_sums[uid_item]:
                self.maker_volume_sums[uid_item][book_id] = round(self.maker_volume_sums[uid_item][book_id], volume_decimals)
            if book_id in self.taker_volume_sums[uid_item]:
                self.taker_volume_sums[uid_item][book_id] = round(self.taker_volume_sums[uid_item][book_id], volume_decimals)
            if book_id in self.self_volume_sums[uid_item]:
                self.self_volume_sums[uid_item][book_id] = round(self.self_volume_sums[uid_item][book_id], volume_decimals)
            if book_id in self.roundtrip_volume_sums[uid_item]:
                self.roundtrip_volume_sums[uid_item][book_id] = round(self.roundtrip_volume_sums[uid_item][book_id], volume_decimals)

            if uid_item in realized_pnl_updates:
                for ts in realized_pnl_updates[uid_item]:
                    if book_id in books and ts in self.realized_pnl_history[uid_item]:
                        if book_id in self.realized_pnl_history[uid_item][ts]:
                            self.realized_pnl_history[uid_item][ts][book_id] = round(
                                self.realized_pnl_history[uid_item][ts][book_id],
                                volume_decimals
                            )
    total_time = time.time() - total_start
    if should_prune:
        bt.logging.debug(f"[UPDATE_VOLUMES] Total: {total_time:.4f}s (pruned, {uid_count} UIDs)")
    else:
        bt.logging.debug(f"[UPDATE_VOLUMES] Total: {total_time:.4f}s ({uid_count} UIDs)")


def shift_simulation_histories(
    self, old_ts: int, new_ts: int, *,
    book_count: int, volume_decimals: int, lookback: int,
    volume_assessment_period: int, miner_wealth, effective_max_uids: int,
    log=None, book_ids=None,
):
    """Shift every history structure from the old simulation's time base to the
    new one on a simulation restart, pruning entries that fall outside the
    volume-assessment / kappa-lookback windows and adjusting the running sums.

    Extracted verbatim from SimulationEngine.on_start so main AND the shadow
    scoring service run the SAME transition (the shadow receives a
    ("sim_start", (old_ts, new_ts)) frame and calls this on its own container).
    Deterministic in (structures, old_ts, new_ts, knobs) — no wall clock.

    NOTE: simulation-restart only. It is NEVER invoked in exchange mode: the
    exchange engine inherits the no-op base on_start and a live chain has no sim
    restarts, so the shadow's "sim_start" frame never fires there either.
    Book iteration runs over `book_ids` when given (multi-asset: sparse canonical
    ids) and falls back to the dense 0-based range that is correct for
    single-market simulation. Neither is the exchange's root-excluded [1..128]
    set — this is never invoked in exchange mode.

    Args:
        old_ts: The previous simulation's time base.
        new_ts: The new simulation's time base.
    """
    _log = log or (lambda m: None)
    book_ids = list(book_ids) if book_ids is not None else list(range(book_count))
    new_threshold = new_ts - lookback
    new_volume_threshold = new_ts - volume_assessment_period

    pruned_total = defaultdict(lambda: defaultdict(float))
    pruned_maker = defaultdict(lambda: defaultdict(float))
    pruned_taker = defaultdict(lambda: defaultdict(float))
    pruned_self = defaultdict(lambda: defaultdict(float))
    pruned_roundtrip = defaultdict(lambda: defaultdict(float))

    _log("Shifting trade volume timestamps...")
    shifted_trade_volumes = {}
    for uid in range(effective_max_uids):
        if uid in self.trade_volumes:
            shifted_trade_volumes[uid] = {}
            for bookId in book_ids:
                if bookId in self.trade_volumes[uid]:
                    shifted_trade_volumes[uid][bookId] = {}
                    for role in ['total', 'maker', 'taker', 'self']:
                        if role in self.trade_volumes[uid][bookId]:
                            shifted_times = {}
                            for prev_time, volume in self.trade_volumes[uid][bookId][role].items():
                                new_time = new_ts - (old_ts - prev_time)
                                if new_time >= new_volume_threshold:
                                    shifted_times[new_time] = volume
                                else:
                                    if role == 'total':
                                        pruned_total[uid][bookId] += volume
                                    elif role == 'maker':
                                        pruned_maker[uid][bookId] += volume
                                    elif role == 'taker':
                                        pruned_taker[uid][bookId] += volume
                                    elif role == 'self':
                                        pruned_self[uid][bookId] += volume
                            if shifted_times:
                                shifted_trade_volumes[uid][bookId][role] = shifted_times

    self.trade_volumes = {
        uid: {
            bookId: {
                role: shifted_trade_volumes.get(uid, {}).get(bookId, {}).get(role, {})
                for role in ['total', 'maker', 'taker', 'self']
            }
            for bookId in book_ids
        }
        for uid in range(effective_max_uids)
    }

    _log("Adjusting volume sums for pruned data...")
    for pruned, sums in (
        (pruned_total, self.volume_sums),
        (pruned_maker, self.maker_volume_sums),
        (pruned_taker, self.taker_volume_sums),
        (pruned_self, self.self_volume_sums),
    ):
        for uid in pruned:
            for bookId in pruned[uid]:
                sums[uid][bookId] = max(0.0, sums[uid][bookId] - pruned[uid][bookId])
                sums[uid][bookId] = round(sums[uid][bookId], volume_decimals)

    _log("Shifting inventory history timestamps...")
    shifted_inventory = {}
    for uid in range(effective_max_uids):
        if uid in self.inventory_history and self.inventory_history[uid]:
            hist = self.inventory_history[uid]
            if len(hist) > 3:
                timestamps_to_keep = sorted(hist.keys())[-3:]
                hist = {ts: hist[ts] for ts in timestamps_to_keep}
            shifted_inventory[uid] = {}
            for prev_time, values in hist.items():
                shifted_inventory[uid][new_ts - (old_ts - prev_time)] = values
    self.inventory_history = {
        uid: shifted_inventory.get(uid, {}) for uid in range(effective_max_uids)
    }

    _log("Shifting realized P&L history timestamps...")
    shifted_pnl_history = {}
    self._last_prune_timestamp = None
    for uid in range(effective_max_uids):
        if uid in self.realized_pnl_history and self.realized_pnl_history[uid]:
            hist = self.realized_pnl_history[uid]
            shifted_pnl_history[uid] = {}
            for prev_time, books in hist.items():
                new_time = new_ts - (old_ts - prev_time)
                if new_time >= new_threshold:
                    shifted_pnl_history[uid][new_time] = books
    self.realized_pnl_history = defaultdict(lambda: defaultdict(dict))
    for uid, timestamps_data in shifted_pnl_history.items():
        for ts, books in timestamps_data.items():
            # Preserve EVERY timestamp, including empty (zero-PnL / breakeven) book-dicts. The prior
            # nested `for book_id, pnl in books.items()` never created the ts key when books was {},
            # silently dropping breakeven-round timestamps. Those timestamps count toward the kappa
            # assessment span in-sim, so dropping them at the crossover could collapse a miner's span
            # below min_lookback -> kappa None -> score 0 (established miners then get their EMA
            # dragged down). Keep the crossover a pure time-rebase of the exact in-sim history.
            self.realized_pnl_history[uid][ts] = dict(books)
    bootstrap_pnl_totals(self)
    _log(f"Shifted realized P&L history: {len(shifted_pnl_history)} UIDs with data")

    _log("Shifting round-trip volume timestamps...")
    shifted_rt_volumes = {}
    for uid in range(effective_max_uids):
        if uid in self.roundtrip_volumes:
            shifted_rt_volumes[uid] = {}
            for bookId in book_ids:
                if bookId in self.roundtrip_volumes[uid]:
                    shifted_times = {}
                    for prev_time, volume in self.roundtrip_volumes[uid][bookId].items():
                        new_time = new_ts - (old_ts - prev_time)
                        if new_time >= new_volume_threshold:
                            shifted_times[new_time] = volume
                        else:
                            pruned_roundtrip[uid][bookId] += volume
                    if shifted_times:
                        shifted_rt_volumes[uid][bookId] = shifted_times
    self.roundtrip_volumes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for uid, books in shifted_rt_volumes.items():
        for book_id, volumes in books.items():
            for ts, volume in volumes.items():
                self.roundtrip_volumes[uid][book_id][ts] = volume
    _log(f"Shifted round-trip volumes: {len(shifted_rt_volumes)} UIDs with data")

    for uid in pruned_roundtrip:
        for bookId in pruned_roundtrip[uid]:
            self.roundtrip_volume_sums[uid][bookId] = max(
                0.0, self.roundtrip_volume_sums[uid][bookId] - pruned_roundtrip[uid][bookId]
            )
            self.roundtrip_volume_sums[uid][bookId] = round(
                self.roundtrip_volume_sums[uid][bookId], volume_decimals
            )

    # De-beta transition (only present when enabled): shift+prune the additive histories onto the new
    # clock so the assessment window spans the boundary continuously (identical to the volume/kappa
    # histories), then re-base the reconstructed-inventory + last-price STATE so the boundary price jump
    # (openB-closeA) never enters the dp/drift accumulator. BOTH legs use the kappa lookback, matching
    # the live prune: see the retention note there for why making no longer uses the 24h window.
    # Shared verbatim with the shadow. See SCORING_PATH_ARCHITECTURE.md §7.
    if getattr(self, 'debeta_capbuy_hist', None) is not None:
        _log("Shifting de-beta histories...")
        # Drain pending capture fills at the OLD clock first (their increments then shift with the
        # histories); the mid window must never span the boundary price jump, so the stream resets.
        if getattr(self, 'debeta_capture_mid', None):
            flush_capture_state(self.debeta_capture_mid, self.capture_buy_sums, self.capture_sell_sums,
                                CAPTURE_W, buy_hist=self.debeta_capbuy_hist,
                                sell_hist=self.debeta_capsell_hist, ts=old_ts, force=True,
                                real_buy_sums=getattr(self, 'realized_buy_sums', None), real_sell_sums=getattr(self, 'realized_sell_sums', None),
                                real_buy_hist=getattr(self, 'debeta_realbuy_hist', None), real_sell_hist=getattr(self, 'debeta_realsell_hist', None))
            self.debeta_capture_mid = {}
        shift_hist_2level(self.debeta_capbuy_hist, self.capture_buy_sums, old_ts, new_ts, new_threshold)
        shift_hist_2level(self.debeta_capsell_hist, self.capture_sell_sums, old_ts, new_ts, new_threshold)
        if getattr(self, 'debeta_realbuy_hist', None) is not None:
            shift_hist_2level(self.debeta_realbuy_hist, self.realized_buy_sums, old_ts, new_ts, new_threshold)
            shift_hist_2level(self.debeta_realsell_hist, self.realized_sell_sums, old_ts, new_ts, new_threshold)
        shift_hist_2level(self.debeta_cp_hist, self.debeta_cp, old_ts, new_ts, new_threshold)
        for _c, _h in (getattr(self, 'debeta_cpc_hist', None) or {}).items():
            shift_hist_2level(_h, self.debeta_cpc.setdefault(_c, {}), old_ts, new_ts, new_threshold)
        shift_hist_2level(self.debeta_mtm_hist, self.debeta_mtm, old_ts, new_ts, new_threshold)
        shift_hist_2level(self.debeta_invsum_hist, self.debeta_invsum, old_ts, new_ts, new_threshold)
        shift_hist_2level(self.debeta_heldn_hist, self.debeta_heldn, old_ts, new_ts, new_threshold)
        shift_hist_2level(self.debeta_heldinv_hist, self.debeta_heldinv, old_ts, new_ts, new_threshold)
        shift_hist_2level(self.debeta_helddrift_hist, self.debeta_helddrift, old_ts, new_ts, new_threshold)
        shift_hist_2level(self.debeta_notional_hist, self.debeta_notional, old_ts, new_ts, new_threshold)
        shift_hist_1level(self.debeta_invn_hist, self.debeta_invn, old_ts, new_ts, new_threshold)
        shift_hist_1level(self.debeta_drift_hist, self.debeta_drift, old_ts, new_ts, new_threshold)
        # STATE re-base: fresh flat market, fresh price reference (new sim starts everyone flat).
        self.debeta_inv = defaultdict(lambda: defaultdict(float))
        self.debeta_pfirst = {}
        self.debeta_plast = {}
        self.debeta_mark_state = {}

    _log("Clearing open positions...")
    self.open_positions = defaultdict(lambda: defaultdict(lambda: {
        'longs': deque(), 'shorts': deque()
    }))
    self.initial_balances = {
        uid: {
            bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': miner_wealth}
            for bookId in book_ids
        } for uid in range(effective_max_uids)
    }
    self.recent_trades = {bookId: [] for bookId in book_ids}
    self.recent_miner_trades = {
        uid: {bookId: [] for bookId in book_ids}
        for uid in range(effective_max_uids)
    }


def rebase_entries_beyond_clock(self, now: int, *, lookback: int, log=None, inventory_correction=None) -> int:
    """Repair a state whose histories carry timestamps ahead of the clock.

    The rebase at a run change (shift_simulation_histories) runs from the engine's start event. A
    validator that restarts after the engine has already opened the new run never receives that event,
    so the previous run's last window stays in every history under the previous clock. On the new
    clock those stamps are in the future: they sort after everything the new run writes, no window
    ever prunes them, and every windowed sum counts them. This guard runs at load: any stamp ahead of
    the clock is treated as the previous run's, moved onto the new time base with the previous run's
    end mapped to the new run's start (so it sits at or below zero, where each structure's own prune
    drops it), the de-beta windows are pruned here with their running sums rebuilt, and the inventory
    history drops them outright: its entries are differences against initial balances the new run
    reset, and nothing prunes that history by time. Entries on the clock are not touched.

    The fill-stream reconstruction state (per-book inventory, last print, mark state) is carried too:
    the run change resets it because the new engine opens every account flat, so a carried position
    is a phantom that every later price step is charged against. With `inventory_correction`
    ({book: {uid: qty}}, the positions held at the previous run's end) those phantoms are subtracted
    and the positions built since the run opened are kept; without it the state is reset as the run
    change would have done, which drops the positions built since the run opened.

    Args:
        now: The simulation clock the state was saved under.
        lookback: The de-beta assessment window in simulation ns.
        log: Optional logger for the summary line.
        inventory_correction: Optional ``{book: {uid: qty}}`` to subtract from the fill-stream inventory.

    Returns:
        int: Entries moved; 0 when the state was already on the clock.
    """
    from taos.im.validator.debeta import sum_hist_1level, sum_hist_2level
    tolerance = 600_000_000_000
    limit = int(now) + tolerance
    two_level = ("debeta_mtm_hist", "debeta_invsum_hist", "debeta_capbuy_hist", "debeta_capsell_hist",
                 "debeta_realbuy_hist", "debeta_realsell_hist", "debeta_cp_hist", "debeta_heldn_hist", "debeta_heldinv_hist", "debeta_helddrift_hist",
                 "debeta_notional_hist")
    one_level = ("debeta_invn_hist", "debeta_drift_hist")
    other = ("realized_pnl_history", "inventory_history", "trade_volumes", "roundtrip_volumes")

    def stamps(d):
        if not isinstance(d, dict):
            return
        keys = [k for k in d if isinstance(k, int) and k > 10**11]
        if keys:
            yield from keys
            return
        for v in d.values():
            yield from stamps(v)

    _cpc_hist = getattr(self, "debeta_cpc_hist", None) or {}
    ahead = [ts for name in two_level + one_level + other for ts in stamps(getattr(self, name, None)) if ts > limit]
    ahead += [ts for _h in _cpc_hist.values() for ts in stamps(_h) if ts > limit]
    if not ahead:
        return 0
    old_end = max(ahead)

    def shift(d):
        if not isinstance(d, dict):
            return 0
        keys = [k for k in d if isinstance(k, int) and k > 10**11]
        if keys:
            moved = 0
            for k in keys:
                if k > limit:
                    d[k - old_end] = d.pop(k)
                    moved += 1
            return moved
        return sum(shift(v) for v in d.values())

    moved = sum(shift(getattr(self, name, None)) for name in two_level + one_level + other)
    moved += sum(shift(_h) for _h in _cpc_hist.values())
    # inventory entries are differences against initial balances the new run reset, and nothing prunes
    # this history by time, so a moved entry would become an account's first entry and the previous
    # run's final value its baseline
    inv = getattr(self, "inventory_history", None) or {}
    for uid in list(inv):
        for ts in [t for t in inv[uid] if isinstance(t, int) and t <= 0]:
            del inv[uid][ts]
    threshold = int(now) - int(lookback)
    for name in two_level:
        hist = getattr(self, name, None) or {}
        for k1 in list(hist):
            for k2 in list(hist[k1]):
                kept = {ts: v for ts, v in hist[k1][k2].items() if ts >= threshold}
                if kept:
                    hist[k1][k2] = kept
                else:
                    del hist[k1][k2]
            if not hist[k1]:
                del hist[k1]
    for name in one_level:
        hist = getattr(self, name, None) or {}
        for k in list(hist):
            kept = {ts: v for ts, v in hist[k].items() if ts >= threshold}
            if kept:
                hist[k] = kept
            else:
                del hist[k]
    running = (("debeta_mtm_hist", "debeta_mtm"), ("debeta_invsum_hist", "debeta_invsum"),
               ("debeta_capbuy_hist", "capture_buy_sums"), ("debeta_capsell_hist", "capture_sell_sums"),
               ("debeta_realbuy_hist", "realized_buy_sums"), ("debeta_realsell_hist", "realized_sell_sums"),
               ("debeta_cp_hist", "debeta_cp"), ("debeta_heldn_hist", "debeta_heldn"),
               ("debeta_heldinv_hist", "debeta_heldinv"), ("debeta_helddrift_hist", "debeta_helddrift"),
               ("debeta_notional_hist", "debeta_notional"))
    for hist_name, sum_name in running:
        sums = sum_hist_2level(getattr(self, hist_name, None) or {})
        setattr(self, sum_name, defaultdict(lambda: defaultdict(float), {u: defaultdict(float, b) for u, b in sums.items()}))
    for _c, _h in list(_cpc_hist.items()):
        for k1 in list(_h):
            for k2 in list(_h[k1]):
                kept = {ts: v for ts, v in _h[k1][k2].items() if ts >= threshold}
                if kept:
                    _h[k1][k2] = kept
                else:
                    del _h[k1][k2]
            if not _h[k1]:
                del _h[k1]
        self.debeta_cpc[_c] = sum_hist_2level(_h)
    for hist_name, sum_name in (("debeta_invn_hist", "debeta_invn"), ("debeta_drift_hist", "debeta_drift")):
        setattr(self, sum_name, defaultdict(float, sum_hist_1level(getattr(self, hist_name, None) or {})))
    inv = getattr(self, "debeta_inv", None)
    inv_note = ""
    if inv is not None:
        if inventory_correction:
            # {"positions": {book: {uid: qty}}, "hotkeys": {uid: hotkey}} gates each uid on the hotkey that held
            # the slot at the previous run's end: a slot that changed hands since had its inventory reset
            # already and carries nothing to subtract. A bare {book: {uid: qty}} applies to every uid.
            positions = inventory_correction.get("positions", inventory_correction)
            gate = inventory_correction.get("hotkeys") if isinstance(inventory_correction.get("positions"), dict) else None
            current = list(getattr(self, "hotkeys", None) or [])
            fixed = skipped = 0
            for b, by_uid in positions.items():
                for u, q in (by_uid or {}).items():
                    if gate is not None:
                        held = current[int(u)] if int(u) < len(current) else None
                        was = gate.get(str(u), gate.get(int(u)))
                        # a uid outside the metagraph on both sides (the validator's own agent) cannot have
                        # changed hands; a uid with a hotkey on either side is corrected only when it matches
                        if (held is not None or was is not None) and was != held:
                            skipped += 1
                            continue
                    inv[int(b)][int(u)] -= float(q)
                    fixed += 1
            inv_note = (f"; fill-stream inventory corrected on {fixed} (book, uid) positions"
                        + (f", {skipped} skipped on slots that changed hands" if skipped else ""))
        else:
            self.debeta_inv = defaultdict(lambda: defaultdict(float))
            self.debeta_pfirst = {}
            self.debeta_plast = {}
            self.debeta_mark_state = {}
            inv_note = "; fill-stream inventory, last print and mark state reset as the run change would have (no correction supplied)"
    if log:
        log(f"HISTORIES AHEAD OF THE CLOCK: {moved} entries stamped up to {old_end} moved onto the clock at {now} "
            f"(a run change the process did not see); de-beta windows pruned below {threshold} and sums rebuilt{inv_note}")
    return moved


def missed_run_change(previous_ts, incoming_ts, *, fresh_threshold) -> bool:
    """A state update whose clock is behind the one the process holds, on a run that has only just opened, is the
    first update of a run whose start event this process never received: the process went down while the old
    run's end handling was starting the new engine, and came up after the event had been delivered. A checkpoint
    resume can also step the clock back, but it fires the start event itself and never lands near zero."""
    try:
        previous_ts = int(previous_ts or 0)
        incoming_ts = int(incoming_ts or 0)
    except (TypeError, ValueError):
        return False
    return incoming_ts < previous_ts and incoming_ts <= int(fresh_threshold)


def history_clock_health(self) -> dict:
    """What a dashboard needs to see a run change that was not applied: how far the furthest de-beta stamp sits
    ahead of the clock (0 when clean; every book's history carries the same stamps, so the per-book inventory
    history is the sentinel), and the counters of the two repairs."""
    now = int(getattr(self, "simulation_timestamp", 0) or 0)
    ahead = 0
    for tsd in (getattr(self, "debeta_invn_hist", None) or {}).values():
        if tsd:
            gap = max(tsd) - now
            if gap > ahead:
                ahead = gap
    return {
        "history_stamp_ahead_ns": int(ahead),
        "run_changes_recovered": int(getattr(self, "_run_changes_recovered", 0) or 0),
        "histories_rebased_at_load": int(getattr(self, "_histories_rebased_at_load", 0) or 0),
    }


def reset_agent_histories(self, uid: int, book_ids: list) -> None:
    """Zero one UID's history/scoring structures (deregistration reset).

    Extracted from SimulationEngine.apply_resets so main AND the shadow scoring
    service run the SAME zeroing (the shadow receives a ("resets", uids) frame).
    Main-only bookkeeping (miner_stats, deregistered_uids, publish flags,
    unnormalized_scores) stays in apply_resets.

    Args:
        uid: The uid whose structures are zeroed.
        book_ids: The books to zero across.
    """
    self.kappa_values[uid] = {
        'books': {bookId: None for bookId in book_ids},
        'books_weighted': {bookId: None for bookId in book_ids},
        'total': None, 'average': None, 'median': None,
        'normalized_average': 0.0, 'normalized_median': 0.0,
        'normalized_total': 0.0,
        'activity_weighted_normalized_median': 0.0,
        'penalty': 0.0, 'score': 0.0,
    }
    # Evict the kappa fingerprint-cache entry for the reused slot so a stale (old-occupant) entry can
    # never be served after the reset (belt-and-suspenders alongside the dereg-first guard in kappa_3).
    if hasattr(self, 'kappa_cache'):
        self.kappa_cache.pop(uid, None)
    self.activity_factors[uid] = {bookId: 0.0 for bookId in book_ids}
    self.pnl_factors[uid] = {bookId: 1.0 for bookId in book_ids}
    self.inventory_history[uid] = {}
    self.trade_volumes[uid] = {
        bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
        for bookId in book_ids
    }
    for book_id in book_ids:
        self.volume_sums[uid][book_id] = 0.0
        self.maker_volume_sums[uid][book_id] = 0.0
        self.taker_volume_sums[uid][book_id] = 0.0
        self.self_volume_sums[uid][book_id] = 0.0
    self.roundtrip_volumes[uid] = defaultdict(lambda: defaultdict(float))
    for book_id in book_ids:
        self.roundtrip_volume_sums[uid][book_id] = 0.0
    # De-beta (P8/E5/P11): a reused UID must NOT inherit the deregistered miner's making/skill/
    # counterparty accumulation. Clear this UID from every de-beta accumulator (only present when
    # de-beta is enabled; getattr guards the off/shadow case).
    # uid-keyed running sums + their timestamped histories (invn/drift are book-keyed, not per-uid).
    for _n in ('capture_buy_sums', 'capture_sell_sums', 'realized_buy_sums', 'realized_sell_sums', 'debeta_mtm', 'debeta_invsum',
               'debeta_heldn', 'debeta_heldinv', 'debeta_helddrift', 'debeta_notional',
               'debeta_capbuy_hist', 'debeta_capsell_hist', 'debeta_realbuy_hist', 'debeta_realsell_hist', 'debeta_mtm_hist', 'debeta_invsum_hist',
               'debeta_heldn_hist', 'debeta_heldinv_hist', 'debeta_helddrift_hist', 'debeta_notional_hist'):
        _d = getattr(self, _n, None)
        if _d is not None:
            _d.pop(uid, None)
    _inv = getattr(self, 'debeta_inv', None)
    if _inv is not None:
        for _b in list(_inv.keys()):
            _inv[_b].pop(uid, None)
    clear_counterparty_rows(self, uid)    # UID as a maker AND as a counterparty of other makers, pooled and per class
    self.realized_pnl_history[uid] = {}
    if hasattr(self, 'agent_pnl_by_book'):
        self.agent_pnl_by_book.pop(uid, None)
        self.agent_pnl_total.pop(uid, None)
    self.open_positions[uid] = defaultdict(lambda: {
        'longs': deque(), 'shorts': deque()
    })
    self.initial_balances[uid] = {
        bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
        for bookId in book_ids
    }
    self.recent_miner_trades[uid] = {bookId: [] for bookId in book_ids}


_RESET_NOTICE_TYPES = frozenset({
    'RDRA', 'RESPONSE_DISTRIBUTED_RESET_AGENT',
    'ERDRA', 'ERROR_RESPONSE_DISTRIBUTED_RESET_AGENT',
})


def collect_reset_uids(state, validator_uid: int):
    """Scan the validator's own notices in `state` for agent-reset results.

    Returns (pending_uids, failed_resets). Shared by main's collect_resets and
    the scoring service: BOTH sides derive resets from the same teed state and
    apply them at the same position (right after that round's volume update),
    which makes the reset transition deterministic by construction — a reset
    delivered via a separate control frame raced the round stream and left the
    two sides one round apart on the reset uid's history.
    """
    pending, failed = set(), []
    notices = state.notices.get(validator_uid, []) if isinstance(state.notices, dict) else []
    for notice in notices:
        if notice.get('y') in _RESET_NOTICE_TYPES:
            for reset in notice.get('r', []):
                if reset.get('u'):
                    pending.add(reset['a'])
                else:
                    failed.append(reset)
    return pending, failed


def class_of_book(self, book_id):
    """The asset class of a canonical book id on this layout, from the class map the scorer uses; None on a
    single market. Cached against the simulation object, so the per-fill lookup costs a dict read."""
    sim = getattr(self, 'simulation', None)
    sc = getattr(self, 'simulation_config', None)
    key = (id(sim), id(sc))
    cache = getattr(self, '_debeta_class_map_cache', None)
    if not cache or cache[0] != key:
        from taos.im.validator.reward import class_map
        try:
            cmap = class_map(self)
        except Exception:
            cmap = {}
        cache = (key, cmap)
        self._debeta_class_map_cache = cache
    return cache[1].get(int(book_id))


def accumulate_counterparty_rows(self, book_id, trades, *, ts):
    """One book's trade batch into the pooled counterparty rows (P11 as deployed on a single market, and the
    tether everywhere) and into the book's class rows (P11 within the class under a multi-asset layout)."""
    accumulate_counterparties(self.debeta_cp, book_id, trades, cp_hist=self.debeta_cp_hist, ts=ts)
    cls = class_of_book(self, book_id)
    if cls is None:
        return
    if not isinstance(getattr(self, 'debeta_cpc', None), dict):
        self.debeta_cpc = {}
    if not isinstance(getattr(self, 'debeta_cpc_hist', None), dict):
        self.debeta_cpc_hist = {}
    accumulate_counterparties(self.debeta_cpc.setdefault(cls, {}), book_id, trades,
                              cp_hist=self.debeta_cpc_hist.setdefault(cls, {}), ts=ts)


def clear_counterparty_rows(self, uid):
    """Drop a uid from the counterparty rows as a maker and as a counterparty of every other maker, in the pooled
    rows and in every class's rows, so a slot's new occupant inherits no concentration."""
    for _n in ('debeta_cp', 'debeta_cp_hist'):
        _cp = getattr(self, _n, None)
        if _cp is not None:
            _cp.pop(uid, None)
            for _m in list(_cp.keys()):
                _cp[_m].pop(uid, None)
    for _n in ('debeta_cpc', 'debeta_cpc_hist'):
        _by = getattr(self, _n, None)
        if isinstance(_by, dict):
            for _cp in _by.values():
                _cp.pop(uid, None)
                for _m in list(_cp.keys()):
                    _cp[_m].pop(uid, None)
