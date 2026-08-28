# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
WebSocket stream helpers for Coinbase and Binance: connection management,
health monitoring, and optional periodic trade sampling.
"""

import json
import time
import pandas as pd
import bittensor as bt

from typing import Callable, Tuple
from threading import Thread
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient as BinanceClient
from taos.im.utils.coinbase import CoinbaseClient
from coinbase.websocket import WSClientConnectionClosedException, WSClientException

def connect_coinbase(symbols: list[str], on_trade: Callable[[dict], None]) -> Tuple[CoinbaseClient | None, Exception | None]:
    """
    Establishes a WebSocket connection to Coinbase and subscribes to trade events for the given symbols.

    Args:
        symbols (list[str]): List of product IDs (symbols) to subscribe to.
        on_trade (Callable[[dict], None]): Callback function invoked for each trade received.

    Returns:
        Tuple[CoinbaseClient | None, Exception | None]: The connected client and None
            on success, or (None, exception) if the connection fails.
    """
    try:
        def on_coinbase_message(message: str) -> None:
            """
            Internal callback function to handle incoming Coinbase messages.
            Filters trade updates and passes parsed trade data to the user-defined on_trade callback.
            """
            try:
                message_dict = json.loads(message)
                if 'channel' in message_dict and message_dict['channel'] == 'market_trades':
                    for event in message_dict['events']:
                        if event['type'] == 'update' and 'trades' in event:
                            trade = {
                                "product_id": event['trades'][0]['product_id'],
                                "price": float(event['trades'][0]['price']),
                                "time": event['trades'][0]['time'],
                                "timestamp": pd.Timestamp(event['trades'][0]['time']).timestamp(),
                                "received": time.time(),
                            }
                            on_trade(trade)
                            return
            except Exception as ex:
                bt.logging.error(f"Exception getting Coinbase seed value : Message={message} | Error={ex}")

        client = CoinbaseClient(on_message=on_coinbase_message)
        bt.logging.info("Attempting to connect to Coinbase Trades Stream...")
        client.open()
        client.subscribe(product_ids=symbols, channels=["market_trades"])
        bt.logging.success(f"Subscribed to Coinbase Trades Stream! {symbols}")
        return client, None
    except Exception as ex:
        bt.logging.warning(f"Unable to connect to Coinbase Trades Stream! {ex}.")
        return None, ex

def maintain_coinbase(client: CoinbaseClient, reconnect: Callable[[], None], check: Callable[[], bool], interval=10):
    """
    Periodically checks the health of the Coinbase WebSocket connection, reconnecting if necessary.

    Args:
        client (CoinbaseClient): The WebSocket client to monitor.
        reconnect (Callable[[], None]): Function to reconnect the client.
        check (Callable[[], bool]): Function that returns True if connection is healthy, otherwise False.
        interval (int): Time in seconds between checks.
    """
    try:
        client.sleep_with_exception_check(interval)
    except WSClientException as e:
        bt.logging.warning(f"Error in Coinbase websocket : {e}")
    except WSClientConnectionClosedException:
        bt.logging.error(f"Coinbase connection closed! Sleeping for {interval} seconds before reconnecting...")
        time.sleep(interval)
    if not client._is_websocket_open() or not check():
        reconnect()

def subscribe_coinbase_trades(
    symbol,
    on_trade: Callable[[dict], None],
    inactivity_threshold_secs: int,
    sampling_period: int | None = None,
    on_sampled: Callable[[dict], None] | None = None
):
    """
    Subscribes to Coinbase trade stream with inactivity monitoring and optional periodic sampling.

    Args:
        symbol (str): Product ID to subscribe to.
        on_trade (Callable[[dict], None]): Callback for each raw trade received.
        inactivity_threshold_secs (int): Time in seconds to wait before declaring stream inactive.
        sampling_period (int | None): Optional time interval (in seconds) for sampling trade data.
        on_sampled (Callable[[dict], None] | None): Callback for each sampled trade.
    """
    def _on_trade(trade: dict):
        # Handle each trade, possibly marking for sampling.
        global last_trade, next_sampled
        # next_external_sampling_time is a module global initialised lazily by the
        # sampling handler below; the `in globals()` guard makes the read safe.
        if 'next_external_sampling_time' in globals() and trade['received'] <= next_external_sampling_time:  # noqa: F821
            next_sampled = trade
        last_trade = trade
        on_trade(trade)

    def _on_sampled(trade: dict):
        # Handle each sampled trade.
        global last_sampled
        last_sampled = trade
        on_sampled(trade)

    def monitor_stream():
        """Watch the stream for silence and reconnect when it goes quiet."""
        def check():
            """
            Validates stream activity and triggers sampling callback if needed.
            """
            global last_trade, last_sampled, next_external_sampling_time
            if 'last_trade' in globals() and last_trade['received'] < time.time() - inactivity_threshold_secs:
                bt.logging.warning(f"No new trades from Coinbase {symbol} stream in last {inactivity_threshold_secs} seconds!  Restarting connection.")
                if client._is_websocket_open():
                    client.close()
                return False

            current_time = time.time()

            if 'next_external_sampling_time' not in globals() or 'next_sampled' not in globals():
                # Calculate the next sampling time aligned to the nearest interval.
                seconds_since_start_of_day = current_time % 86400
                start_of_day = current_time - seconds_since_start_of_day
                next_external_sampling_time = start_of_day + seconds_since_start_of_day + (sampling_period - (seconds_since_start_of_day % sampling_period))

            if current_time >= next_external_sampling_time:
                if 'next_sampled' in globals():
                    sampled = next_sampled
                    sampled['received'] = next_external_sampling_time
                    next_external_sampling_time += sampling_period
                    _on_sampled(sampled)
                else:
                    return True

            return True

        def connect():
            """Open (or re-open) the upstream connection."""
            global client
            while True:
                client, ex = connect_coinbase([symbol], _on_trade)
                if not client:
                    bt.logging.error(f"Failed to connect to coinbase trades stream : {ex}")
                else:
                    break

        connect()
        # Continuously monitor the stream in a background thread.
        while True:
            try:
                maintain_coinbase(client, connect, check, interval=10 if not sampling_period else 1)
            except Exception as ex:
                bt.logging.error(f"Exception in trade subscription loop : {ex}")

    # Start monitoring in a background thread.
    Thread(target=monitor_stream, args=(), daemon=True, name=f'trades_{symbol}').start()

def connect_binance(symbols, on_trade: Callable[[dict], None]) -> Tuple[BinanceClient | None, Exception | None]:
    """
    Connects to Binance WebSocket stream for real-time trades on given symbols.

    Args:
        symbols (list[str]): List of trading symbols to subscribe to.
        on_trade (Callable[[dict], None]): Callback invoked for each trade message.

    Returns:
        Tuple[BinanceClient | None, Exception | None]: The connected client and None
            on success, or (None, exception) if the connection fails.
    """
    try:
        def on_binance_message(_, message: str) -> None:
            """
            Internal message handler for Binance trade stream.
            Parses trade data and invokes user-defined callback.
            """
            try:
                message_dict = json.loads(message)
                if 'e' in message_dict and message_dict['e'] == 'trade':
                    trade = {
                        "product_id": message_dict['s'].lower(),
                        "price": float(message_dict['p']),
                        "time": pd.to_datetime(message_dict['T']*1000000).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        "timestamp": message_dict['T'],
                        "received": time.time()
                    }
                    on_trade(trade)
            except Exception as ex:
                bt.logging.error(f"Exception getting Binance seed value : Message={message} | Error={ex}")

        bt.logging.info("Attempting to connect to Binance Trades Stream...")
        client = BinanceClient(on_message=on_binance_message)
        for symbol in symbols:
            client.trade(symbol=symbol)
        bt.logging.success(f"Subscribed to Binance Trades Stream! {symbols}")
        return client, None
    except Exception as ex:
        bt.logging.warning(f"Unable to connect to Binance Trades Stream! {ex}.")
        return None, ex

def maintain_binance(client: BinanceClient, reconnect: Callable[[], None], check: Callable[[], bool], interval=10):
    """
    Monitors the Binance client for inactivity and reconnects if necessary.

    Args:
        client (BinanceClient): Binance WebSocket client.
        reconnect (Callable[[], None]): Function to call for reconnecting the client.
        check (Callable[[], bool]): Health check function that returns False if stream is inactive.
    """
    time.sleep(interval)
    if not check():
        reconnect()