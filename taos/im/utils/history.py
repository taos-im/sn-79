# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Order-book history reconstruction: replays order, trade, and cancel events
onto an LOB snapshot to produce a timestamped sequence of book states.
"""
from loky.backend.context import set_start_method
set_start_method('forkserver', force=True)
from loky import get_reusable_executor, as_completed
import time
import copy

def history(snapshot, events, volume_decimals):
    """
    Replay a sequence of order-book events onto an initial snapshot.

    Applies order placements, trades, and cancellations in chronological order,
    recording a deep copy of the book state at each event timestamp.

    Args:
        snapshot (dict): Initial LOB state with 'bids', 'asks', and 'timestamp' keys.
        events (list[dict]): List of event dicts with 'y' (type), 't' (timestamp),
            'p' (price), 'q' (quantity), and 's' (side) fields.
        volume_decimals (int): Decimal precision for rounding quantity accumulation.

    Returns:
        Tuple[dict, dict]: (history_dict, trades_dict) where history_dict maps
            timestamp to book snapshot and trades_dict maps timestamp to trade event.
    """
    history = {snapshot['timestamp']: copy.deepcopy(snapshot)}
    trades = {}
    # The engine rests a leveraged order at quantity x (1 + leverage) and removes the same on its cancellation, so the
    # replay keeps each order's leverage by id for the cancellation that follows (an order rested before the snapshot
    # is unknown here and its cancellation is taken at its own quantity)
    leverage_by_id = {}
    # Apply events in chronological order
    for event in sorted(events, key=lambda x: x['t']):
        match event:
            case o if event['y'] == 'o':
                # Place new order
                lev = float(o.get('l') or 0.0)
                if lev > 0.0 and o.get('i') is not None:
                    leverage_by_id[o['i']] = lev
                rested = o['q'] * (1.0 + lev)
                if o['s'] == 0:
                    if o['p'] not in snapshot['bids']:
                        snapshot['bids'][o['p']] = {'p':o['p'], 'q':0.0, 'o':None}
                    snapshot['bids'][o['p']]['q'] = round(
                        snapshot['bids'][o['p']]['q'] + rested,
                        volume_decimals
                    )
                else:
                    if o['p'] not in snapshot['asks']:
                        snapshot['asks'][o['p']] =  {'p':o['p'], 'q':0.0, 'o':None}
                    snapshot['asks'][o['p']]['q'] = round(
                        snapshot['asks'][o['p']]['q'] + rested,
                        volume_decimals
                    )

            case t if event['y'] == 't':
                # Record trade
                trades[t['t']] = t
                if t['s'] == 0:
                    if t['p'] in snapshot['asks']:
                        snapshot['asks'][t['p']]['q'] = round(
                            snapshot['asks'][t['p']]['q'] - t['q'],
                            volume_decimals
                        )
                        if snapshot['asks'][t['p']]['q'] == 0.0:
                            del snapshot['asks'][t['p']]
                else:
                    if t['p'] in snapshot['bids']:
                        snapshot['bids'][t['p']]['q'] = round(
                            snapshot['bids'][t['p']]['q'] - t['q'],
                            volume_decimals
                        )
                        if snapshot['bids'][t['p']]['q'] == 0.0:
                            del snapshot['bids'][t['p']]

            case c if event['y'] == 'c':
                # Cancel existing order: on the side the event names when it carries one, else the guess from the
                # best ask (an engine that publishes no side behaves as before)
                side = c.get('s')
                if side is not None:
                    on_ask = int(side) == 1
                else:
                    on_ask = bool(snapshot['asks']) and c['p'] >= min(snapshot['asks'].keys())
                c_lev = c.get('l')
                removed = c['q'] * (1.0 + (float(c_lev) if c_lev is not None else leverage_by_id.get(c.get('i'), 0.0)))
                if on_ask:
                    if c['p'] in snapshot['asks']:
                        snapshot['asks'][c['p']]['q'] = round(
                            snapshot['asks'][c['p']]['q'] - removed,
                            volume_decimals
                        )
                        if snapshot['asks'][c['p']]['q'] == 0.0:
                            del snapshot['asks'][c['p']]
                else:
                    if c['p'] in snapshot['bids']:
                        snapshot['bids'][c['p']]['q'] = round(
                            snapshot['bids'][c['p']]['q'] - removed,
                            volume_decimals
                        )
                        if snapshot['bids'][c['p']]['q'] == 0.0:
                            del snapshot['bids'][c['p']]

        # Add snapshot to history after each update
        snapshot['timestamp'] = event['t']
        history[event['t']] = copy.deepcopy(snapshot)
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
    tasks = [pool.submit(history_batch, {book_id : snapshots[book_id] for book_id in batch}, {book_id : events[book_id] for book_id in batch}, volume_decimals) for batch in batches]
    for task in as_completed(tasks):
        result = task.result()        
        history_batches.append(result)
    return {int(k): v for d in history_batches for k, v in d.items()}