# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
from __future__ import annotations

import time
import traceback
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import bittensor as bt

from taos.im.protocol.models import TradeInfo
from taos.im.protocol.events import TradeEvent
from taos.im.protocol import MarketSimulationStateUpdate

if TYPE_CHECKING:
    from taos.im.neurons.validator import Validator


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

            for trade in trades:
                is_maker = trade['Ma'] == uid_item
                is_taker = trade['Ta'] == uid_item
                book_id = trade['b']

                # Update recent miner trades
                recent_miner_trades_uid.setdefault(book_id, [])
                if is_maker:
                    recent_miner_trades_uid[book_id].append([TradeEvent.model_construct(**trade), "maker"])
                if is_taker:
                    recent_miner_trades_uid[book_id].append([TradeEvent.model_construct(**trade), "taker"])
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

                # FIFO Matching: Calculate realized P&L and round-trip volume
                quantity = trade['q']
                price = trade['p']
                side = trade['s']
                is_buy = (is_taker and side == 0) or (is_maker and side == 1)
                fee = trade['Mf'] if is_maker else trade['Tf']
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

    for bookId, book in books.items():
        trades = [event for event in book.get('e', []) if event['y'] == 't']
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
    log=None,
):
    """Shift every history structure from the old simulation's time base to the
    new one on a simulation restart, pruning entries that fall outside the
    volume-assessment / kappa-lookback windows and adjusting the running sums.

    Extracted verbatim from SimulationEngine.on_start so main AND the shadow
    scoring service run the SAME transition (the shadow receives a
    ("sim_start", (old_ts, new_ts)) frame and calls this on its own container).
    Deterministic in (structures, old_ts, new_ts, knobs) — no wall clock.
    """
    _log = log or (lambda m: None)
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
            for bookId in range(book_count):
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
            for bookId in range(book_count)
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
            for bookId in range(book_count):
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

    _log("Clearing open positions...")
    self.open_positions = defaultdict(lambda: defaultdict(lambda: {
        'longs': deque(), 'shorts': deque()
    }))
    self.initial_balances = {
        uid: {
            bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': miner_wealth}
            for bookId in range(book_count)
        } for uid in range(effective_max_uids)
    }
    self.recent_trades = {bookId: [] for bookId in range(book_count)}
    self.recent_miner_trades = {
        uid: {bookId: [] for bookId in range(book_count)}
        for uid in range(effective_max_uids)
    }


def reset_agent_histories(self, uid: int, book_ids: list) -> None:
    """Zero one UID's history/scoring structures (deregistration reset).

    Extracted from SimulationEngine.apply_resets so main AND the shadow scoring
    service run the SAME zeroing (the shadow receives a ("resets", uids) frame).
    Main-only bookkeeping (miner_stats, deregistered_uids, publish flags,
    unnormalized_scores) stays in apply_resets.
    """
    self.kappa_values[uid] = {
        'books': {bookId: None for bookId in book_ids},
        'books_weighted': {bookId: 0.0 for bookId in book_ids},
        'total': None, 'average': None, 'median': None,
        'normalized_average': 0.0, 'normalized_median': 0.0,
        'normalized_total': 0.0,
        'activity_weighted_normalized_median': 0.0,
        'penalty': 0.0, 'score': 0.0,
    }
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
