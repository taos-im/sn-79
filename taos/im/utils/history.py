# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Order-book history reconstruction: replays order, trade, and cancel events
onto an LOB snapshot to produce a timestamped sequence of book states.
"""
from loky.backend.context import set_start_method
set_start_method('forkserver', force=True)
from loky import get_reusable_executor, as_completed
from loky.process_executor import BrokenProcessPool, TerminatedWorkerError
import logging
import time
import copy

logger = logging.getLogger(__name__)


def _add(side, price, delta, volume_decimals):
    """A new side mapping with one level moved by `delta`: the level is a NEW dict, every other level and the
    previous side mapping are untouched, so the snapshot that holds them reads as it did. A level at zero leaves."""
    new = dict(side)
    level = new.get(price)
    q = round((level['q'] if level is not None else 0.0) + delta, volume_decimals)
    if q == 0.0 and delta < 0.0:
        new.pop(price, None)
    else:
        new[price] = {'p': price, 'q': q, 'o': level['o'] if level is not None else None}
    return new


def _take(side, price, amount, volume_decimals):
    """_add for a removal that touches only a level that exists (a trade or a cancellation at a price not resting
    changes nothing)."""
    if price not in side:
        return side
    return _add(side, price, -amount, volume_decimals)


def history(snapshot, events, volume_decimals):
    """
    Replay a sequence of order-book events onto an initial snapshot.

    Applies order placements, trades, and cancellations in chronological order,
    recording the book state at each event timestamp. Each recorded snapshot is
    its own top-level dict; the side an event did not touch and every level it
    did not move are the previous snapshot's own objects, and only the moved
    level is new. A full copy of the book per event would make a worker's
    memory grow with events times levels, enough to kill workers under memory
    pressure. Readers treat the snapshots as read-only;
    the agent library validates each into its own model.

    Args:
        snapshot (dict): Initial LOB state with 'bids', 'asks', and 'timestamp' keys.
        events (list[dict]): List of event dicts with 'y' (type), 't' (timestamp),
            'p' (price), 'q' (quantity), and 's' (side) fields.
        volume_decimals (int): Decimal precision for rounding quantity accumulation.

    Returns:
        Tuple[dict, dict]: (history_dict, trades_dict) where history_dict maps
            timestamp to book snapshot and trades_dict maps timestamp to trade event.
    """
    current = copy.deepcopy(snapshot)
    history = {snapshot['timestamp']: current}
    trades = {}
    # The engine rests a leveraged order at quantity x (1 + leverage) and removes the same on its cancellation, so the
    # replay keeps each order's leverage by id for the cancellation that follows (an order rested before the snapshot
    # is unknown here and its cancellation is taken at its own quantity)
    leverage_by_id = {}
    # Apply events in chronological order
    for event in sorted(events, key=lambda x: x['t']):
        nxt = dict(current)
        match event:
            case o if event['y'] == 'o':
                # Place new order
                lev = float(o.get('l') or 0.0)
                if lev > 0.0 and o.get('i') is not None:
                    leverage_by_id[o['i']] = lev
                rested = o['q'] * (1.0 + lev)
                side = 'bids' if o['s'] == 0 else 'asks'
                nxt[side] = _add(current[side], o['p'], rested, volume_decimals)

            case t if event['y'] == 't':
                # Record trade
                trades[t['t']] = t
                side = 'asks' if t['s'] == 0 else 'bids'
                nxt[side] = _take(current[side], t['p'], t['q'], volume_decimals)

            case c if event['y'] == 'c':
                # Cancel existing order: on the side the event names when it carries one, else the guess from the
                # best ask (for an engine that publishes no side)
                side_flag = c.get('s')
                if side_flag is not None:
                    on_ask = int(side_flag) == 1
                else:
                    on_ask = bool(current['asks']) and c['p'] >= min(current['asks'].keys())
                c_lev = c.get('l')
                removed = c['q'] * (1.0 + (float(c_lev) if c_lev is not None else leverage_by_id.get(c.get('i'), 0.0)))
                side = 'asks' if on_ask else 'bids'
                nxt[side] = _take(current[side], c['p'], removed, volume_decimals)

        # Add snapshot to history after each update
        nxt['timestamp'] = event['t']
        history[event['t']] = nxt
        current = nxt
    return history, trades

def decimals_for(volume_decimals, book_id):
    """The volume decimals of one book: a mapping book id to decimals under a multi-asset layout (each class has
    its own grid), or the one integer that applies to every book."""
    if isinstance(volume_decimals, dict):
        return volume_decimals[book_id]
    return volume_decimals


def history_batch(snapshots, events, volume_decimals):
    """
    Compute order-book histories for a batch of books sequentially.

    Args:
        snapshots (dict): Mapping of book_id to initial LOB snapshot.
        events (dict): Mapping of book_id to list of event dicts.
        volume_decimals (int | dict): Decimal precision for volume rounding, one integer or a mapping per book.

    Returns:
        dict: Mapping of book_id to the (history_dict, trades_dict) tuple from `history`.
    """
    start = time.time()
    result = {book_id : history(snapshot, events[book_id], decimals_for(volume_decimals, book_id)) for book_id, snapshot in snapshots.items()}
    print(f"Calculated histories for books {list(result.keys())[0]}-{list(result.keys())[-1]} ({time.time() - start}s)")
    return result

def batch_history(snapshots, events, batches, volume_decimals):
    """
    Compute order-book histories for all books in parallel using loky workers.

    Args:
        snapshots (dict): Mapping of book_id to initial LOB snapshot.
        events (dict): Mapping of book_id to list of event dicts.
        batches (list[list[int]]): Partition of book IDs into worker batches.
        volume_decimals (int): Decimal precision for volume rounding.

    Returns:
        dict: Mapping of book_id (int) to (history_dict, trades_dict) tuples.
    """
    history_batches= []
    pool = get_reusable_executor(max_workers=len(batches))
    tasks = {}
    for batch in batches:
        task = pool.submit(history_batch, {book_id : snapshots[book_id] for book_id in batch}, {book_id : events[book_id] for book_id in batch}, volume_decimals)
        tasks[task] = batch
    for task in as_completed(list(tasks)):
        try:
            result = task.result()
        except (TerminatedWorkerError, BrokenProcessPool, EOFError) as ex:
            # A worker killed under memory pressure would take the whole update down with it: its batch is
            # rebuilt here instead, so every book comes back and the agent keeps its history.
            batch = tasks[task]
            logger.warning(f"book rebuild worker died for books {batch} ({type(ex).__name__}: {ex}); rebuilt in-process")
            result = history_batch({book_id : snapshots[book_id] for book_id in batch}, {book_id : events[book_id] for book_id in batch}, volume_decimals)
        history_batches.append(result)
    return {int(k): v for d in history_batches for k, v in d.items()}