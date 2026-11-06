# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
External price collection: the external market's trades from Coinbase (Binance when Coinbase is unavailable),
written to CSV files in the run's directory for the simulator. The fundamental's step seed is not taken here: the
validator attaches a drand round to its batch (fundamental_seed.py).
"""
import os
import argparse
import signal
import threading
import traceback
import time

_current_log_dir = None
# mtime of the handoff file when we last read it. The signal is an OPTIMISATION, not the contract:
# see check_log_dir_change.
_log_dir_mtime = None
_log_dir_changed = False


def _flag_sigusr1(signum, frame):
    global _log_dir_changed
    _log_dir_changed = True


# Installed before the slow imports below: the validator signals about two seconds after spawning this
# process, and SIGUSR1's default disposition kills it. Only the main thread
# may install a handler, and an importer on another thread (the proxy's seed thread) does not need it.
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGUSR1, _flag_sigusr1)

import bittensor as bt


from taos.im.utils.streams import *


def _ensure_log_dir(path: str) -> None:
    """Create the directory an external-price CSV is about to be appended to.

    The run directory can be absent for two ordinary reasons: it lives under /tmp on some deployments
    and is cleared by systemd, or this service learned of it before the simulator created it. Neither
    is exceptional, but `open(..., 'a')` raises FileNotFoundError for both, and record_external is called
    per trade, so an absent directory would produce several "Exception in external price handling: [Errno 2]"
    lines a second for as long as it persists, with every row in between lost.

    Creating it is the correct repair once the path itself is trustworthy, which the mtime refresh in
    check_log_dir_change now makes it: a stale path is corrected within one seed rather than written
    to forever.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except OSError as ex:
        bt.logging.warning(f"Could not create log directory for {path}: {ex}")

def _handle_sigusr1(signum, frame):
    """Signal handler for log directory updates."""
    global _log_dir_changed
    _log_dir_changed = True
    bt.logging.info("Received SIGUSR1: log directory update pending")

def run_seed_service(config):
    """
    Standalone seed service entry point that runs in its own process.

    Loops forever, restarting `seed()` on any exception. Completely isolated
    from the validator to prevent blocking.

    Args:
        config: Configuration object with seed symbols and parameters.
    """
    global _current_log_dir
    
    bt.logging.info("Seed service starting...")
    signal.signal(signal.SIGUSR1, _handle_sigusr1)
    try:
        os.nice(5)
    except PermissionError:
        bt.logging.warning("Cannot set process priority")
    while True:
        try:
            seed(config)
        except KeyboardInterrupt:
            bt.logging.info("Seed service shutting down...")
            break
        except Exception as ex:
            bt.logging.error(f"Seed service crashed: {ex}")
            bt.logging.error(traceback.format_exc())
            time.sleep(10)  # Prevent rapid restart loops

def seed(config):
    """
    Main external price collection loop (runs in a separate process).

    Opens a WebSocket connection to Coinbase (Binance when Coinbase is unavailable), writes the external symbol's
    trades to CSV files in the run's directory, and follows the run directory through the handoff file (SIGUSR1
    makes a change seen at once). The fundamental symbol is subscribed too; its trades are not recorded.

    Args:
        config: Configuration object with seed symbols and data directory path.
    """
    global _current_log_dir, _log_dir_changed
    
    external_count = 0
    sampled_external_count = 0
    external_filename = None
    external_file = None
    sampled_external_filename = None
    sampled_external_file = None
    last_external = None
    last_sampled_external = None
    next_sampled_external = None
    next_external_sampling_time = None
    pending_external_data = ''
    seed_exchange = 'coinbase'
    
    def check_log_dir_change():
        """Check if log directory has changed via signal or environment."""
        global _current_log_dir, _log_dir_changed, _log_dir_mtime
        nonlocal external_filename, external_file, sampled_external_filename
        nonlocal sampled_external_file, external_count, last_external, pending_external_data
        nonlocal sampled_external_count

        # STAT THE FILE; DO NOT TRUST THE SIGNAL TO ARRIVE. Refreshing only on SIGUSR1 latches after
        # the first successful read, because the flag below clears only when the directory did NOT
        # change. The signal is delivered to
        # `v.seed_process`, a handle that goes stale the moment this service outlives the validator
        # that spawned it: the validator then writes the file, finds its handle dead, logs "cannot
        # notify" and returns, while this process keeps writing to a directory from a previous run.
        # The handoff file can hold the correct run directory the whole time while this service keeps
        # writing to a directory that no longer exists, several errors a second. An mtime check is one
        # stat per seed and makes a lost signal cost latency, not correctness.
        log_dir_file = os.environ.get('TAOS_VALIDATOR_LOG_DIR_FILE', '/tmp/validator_log_dir.txt')
        try:
            _mtime = os.stat(log_dir_file).st_mtime
        except OSError:
            _mtime = None
        if not _log_dir_changed and _current_log_dir and _mtime == _log_dir_mtime:
            return
        _log_dir_mtime = _mtime
        new_log_dir = None
        if _mtime is not None:
            try:
                with open(log_dir_file, 'r') as f:
                    new_log_dir = f.read().strip() or None
            except OSError as _ex:
                bt.logging.warning(f"Could not read log dir handoff {log_dir_file}: {_ex}")
                return
        
        if new_log_dir != _current_log_dir:
            bt.logging.info(f"Log directory changed: {_current_log_dir} -> {new_log_dir}")
            _current_log_dir = new_log_dir
            
            # Close existing files
            if external_file:
                try:
                    external_file.close()
                except Exception:
                    pass
                external_file = None
            if sampled_external_file:
                try:
                    sampled_external_file.close()
                except Exception:
                    pass
                sampled_external_file = None
            
            # Reset state
            external_filename = None
            sampled_external_filename = None
            last_external = None
            external_count = 0
            sampled_external_count = 0
            pending_external_data = ''
        # Cleared on BOTH paths: clearing it only here leaves a run that DID change directory re-reading
        # the file on every seed thereafter.
        _log_dir_changed = False
    
    def on_coinbase_trade(trade: dict):
        """Handle one Coinbase trade message: an external symbol's trade is recorded, any other is not.

        Args:
            trade (dict): The exchange's trade payload.
        """
        nonlocal last_external, external_count, pending_external_data
        nonlocal next_sampled_external

        check_log_dir_change()
        
        match trade['product_id']:
            case config.seed.external.symbol.coinbase:
                if next_external_sampling_time and trade['received'] <= next_external_sampling_time:
                    next_sampled_external = trade
                record_external(trade)

    def on_binance_trade(trade: dict):
        """Handle one Binance trade message: an external symbol's trade is recorded, any other is not.

        Args:
            trade (dict): The exchange's trade payload.
        """
        nonlocal last_external, external_count, pending_external_data

        check_log_dir_change()
        
        match trade['product_id']:
            case config.seed.external.symbol.binance:
                record_external(trade)


    def record_external(trade: dict) -> None:
        """Record an external-venue trade, the simulation's external price.

        Args:
            trade (dict): Normalised trade with price, size and side.
        """
        nonlocal external_count, pending_external_data, last_external
        nonlocal external_filename, external_file, sampled_external_filename
        nonlocal sampled_external_file, sampled_external_count
        
        try:
            if not last_external or last_external != trade:
                if not _current_log_dir:
                    external_count += 1
                    pending_external_data += f"{external_count},{trade['price']},{trade['time']}\n"
                    if len(pending_external_data.split("\n")) > 10000:
                        pending_external_data = "\n".join(pending_external_data.split("\n")[-10000:])
                else:
                    if external_filename != os.path.join(_current_log_dir, "external_seed.csv"):
                        last_external = None
                        external_count = 0
                        pending_external_data = ''
                        if external_file:
                            external_file.close()
                            external_file = None
                        if sampled_external_file:
                            sampled_external_file.close()
                            sampled_external_file = None
                    if not last_external:
                        external_filename = os.path.join(_current_log_dir, "external_seed.csv")
                        if os.path.exists(external_filename) and os.stat(external_filename).st_size > 0:
                            with open(external_filename) as f:
                                for _line in f:
                                    external_count += 1
                        _ensure_log_dir(external_filename)
                        external_file = open(external_filename, 'a')
                        external_file.write(pending_external_data)
                        pending_external_data = ''
                        
                        sampled_external_filename = os.path.join(_current_log_dir, "external_seed_sampled.csv")
                        if os.path.exists(sampled_external_filename) and os.stat(sampled_external_filename).st_size > 0:
                            with open(sampled_external_filename) as f:
                                for _line in f:
                                    sampled_external_count += 1
                        _ensure_log_dir(sampled_external_filename)
                        sampled_external_file = open(sampled_external_filename, 'a')
                    external_count += 1
                    external_file.write(f"{external_count},{trade['price']},{trade['time']}\n")
                    external_file.flush()
                    last_external = trade
                    
        except Exception as ex:
            bt.logging.error(f"Exception in external price handling: trade={trade} | Error={ex}")
    
    last_reconnect_time = 0
    reconnect_cooldown = 5.0
    seed_client = None
    
    def connect() -> None:
        """Open (or re-open) the upstream websocket and subscribe to the configured symbols."""
        nonlocal last_reconnect_time, seed_exchange, seed_client
        time_since_reconnect = time.time() - last_reconnect_time
        if time_since_reconnect < reconnect_cooldown:
            bt.logging.debug(
                f"Reconnect on cooldown ({time_since_reconnect:.1f}s < {reconnect_cooldown}s), waiting..."
            )
            time.sleep(reconnect_cooldown - time_since_reconnect)
        
        attempts = 0
        while True:
            attempts += 1
            seed_exchange = 'coinbase'
            seed_client, ex = connect_coinbase(
                [config.seed.fundamental.symbol.coinbase, config.seed.external.symbol.coinbase],
                on_coinbase_trade
            )
            if not seed_client:
                bt.logging.warning(f"Unable to connect to Coinbase Trades Stream! {ex}. Trying Binance.")
                seed_exchange = 'binance'
                seed_client, ex = connect_binance(
                    [config.seed.fundamental.symbol.binance, config.seed.external.symbol.binance],
                    on_binance_trade
                )
                if not seed_client:
                    bt.logging.error(f"Unable to connect to Binance Trades Stream: {ex}.")
                    if attempts >= 3:
                        bt.logging.error(f"Failed connecting to seed streams after {attempts} attempts")
                    time.sleep(min(attempts * 2, 30))
                else:
                    last_reconnect_time = time.time()
                    bt.logging.info("Connected to Binance seed stream")
                    break
            else:
                last_reconnect_time = time.time()
                bt.logging.info("Connected to Coinbase seed stream")
                break
    
    def check_seeds():
        """Verify the external price is still arriving and reconnect the stream when it has gone quiet."""
        nonlocal last_external, external_count
        nonlocal next_external_sampling_time, next_sampled_external, last_sampled_external
        nonlocal sampled_external_count, external_file, sampled_external_file
        
        reconnect = False
        current_time = time.time()
        
        if last_external:
            if external_file:
                external_file.flush()
            time_since_external = current_time - last_external['received']
            if time_since_external > 120:
                bt.logging.warning(f"No new external price in last {time_since_external:.1f}s! Will reconnect.")
                if seed_exchange == 'coinbase' and seed_client._is_websocket_open():
                    try:
                        seed_client.close()
                    except Exception:
                        pass
                reconnect = True
                last_external = None
                external_count = 0
                
            sampling_period = config.seed.external.sampling_seconds
            if not next_external_sampling_time or not next_sampled_external:
                seconds_since_start_of_day = current_time % 86400
                start_of_day = current_time - seconds_since_start_of_day
                next_external_sampling_time = start_of_day + seconds_since_start_of_day + (
                    sampling_period - (seconds_since_start_of_day % sampling_period)
                )
            if current_time >= next_external_sampling_time and next_sampled_external:
                sampled_external_count += 1
                next_sampled_external['received'] = next_external_sampling_time
                if sampled_external_file:
                    sampled_external_file.write(
                        f"{sampled_external_count},{next_sampled_external['price']}\n"
                    )
                    sampled_external_file.flush()
                last_sampled_external = next_sampled_external
                last_sampled_external['received'] = next_external_sampling_time
                next_external_sampling_time = next_external_sampling_time + sampling_period
        
        return not reconnect

    connect()
    
    while True:
        try:
            if seed_exchange == 'coinbase':
                maintain_coinbase(seed_client, connect, check_seeds, 1)
            if seed_exchange == 'binance':
                maintain_binance(seed_client, connect, check_seeds, 1)
        except Exception as ex:
            bt.logging.error(f"Exception in seed loop: {ex}")
            bt.logging.debug(traceback.format_exc())
            time.sleep(2)            
            
def seed_thread(self) -> None:
        """
        Record the external market's price to the simulation's log directory, where the simulator reads it; the
        fundamental symbol is subscribed too and its trades are not recorded. Runs in a thread beside the FastAPI
        server and restarts on any failure.
        """
        while True:
            try:
                self.external_count = 0
                self.sampled_external_count = 0
                self.external_filename = None
                self.sampled_external_filename = None
                self.last_external_count = self.external_count
                self.last_external = None
                self.last_sampled_external = None
                self.next_sampled_external = None
                self.next_external_sampling_time = None
                self.pending_external_data = ''
                self.seed_exchange = 'coinbase'
                
                def on_coinbase_trade(trade : dict):
                    match trade['product_id']:
                        case self.config.simulation.seeding.external.symbol.coinbase:
                            if self.next_external_sampling_time and trade['received'] <= self.next_external_sampling_time:
                                self.next_sampled_external = trade
                            record_external(trade)

                def on_binance_trade(trade : dict):
                    match trade['product_id']:
                        case self.config.simulation.seeding.external.symbol.binance:
                            record_external(trade)


                def record_external(trade : dict) -> None:
                    try:
                        if not self.last_external or self.last_external != trade:
                            if not self.simulation.logDir:
                                self.external_count += 1
                                self.pending_external_data += f"{self.external_count},{trade['price']},{trade['time']}\n"
                                if len(self.pending_external_data.split("\n")) > 10000:
                                    self.pending_external_data = "\n".join(self.pending_external_data.split("\n")[-10000:])
                            else:
                                if self.external_filename != os.path.join(self.simulation.logDir,"external_seed.csv"):
                                    self.last_external = None
                                    self.external_count = 0
                                    self.pending_external_data = ''
                                if not self.last_external:
                                    self.external_filename = os.path.join(self.simulation.logDir,"external_seed.csv")
                                    if os.path.exists(self.external_filename) and os.stat(self.external_filename).st_size > 0:
                                        with open(self.external_filename) as f:
                                            for _line in f:
                                                self.external_count += 1
                                    _ensure_log_dir(self.external_filename)
                                    self.external_file = open(self.external_filename,'a')
                                    self.external_file.write(self.pending_external_data)
                                    self.pending_external_data = ''
                                    
                                    self.sampled_external_filename = os.path.join(self.simulation.logDir,"external_seed_sampled.csv")
                                    if os.path.exists(self.sampled_external_filename) and os.stat(self.sampled_external_filename).st_size > 0:
                                        with open(self.sampled_external_filename) as f:
                                            for _line in f:
                                                self.sampled_external_count += 1
                                    _ensure_log_dir(self.sampled_external_filename)
                                    self.sampled_external_file = open(self.sampled_external_filename,'a')
                                self.external_count += 1
                                self.external_file.write(f"{self.external_count},{trade['price']},{trade['time']}\n")
                                self.external_file.flush()
                                self.last_external = trade
                                
                    except Exception as ex:
                        bt.logging.error(f"Exception in external price handling : trade={trade} | Error={ex}")
                    
                last_reconnect_time = 0
                reconnect_cooldown = 5.0
                
                def connect() -> None:
                    nonlocal last_reconnect_time
                    
                    # Rate-limit reconnects
                    time_since_reconnect = time.time() - last_reconnect_time
                    if time_since_reconnect < reconnect_cooldown:
                        bt.logging.debug(
                            f"Reconnect on cooldown ({time_since_reconnect:.1f}s < {reconnect_cooldown}s), waiting..."
                        )
                        time.sleep(reconnect_cooldown - time_since_reconnect)
                    
                    attempts = 0
                    while True:
                        attempts += 1
                        self.seed_exchange='coinbase'
                        self.seed_client, ex = connect_coinbase([self.config.simulation.seeding.fundamental.symbol.coinbase, self.config.simulation.seeding.external.symbol.coinbase], on_coinbase_trade)
                        if not self.seed_client:
                            bt.logging.warning(f"Unable to connect to Coinbase Trades Stream! {ex}. Trying Binance.")
                            self.seed_exchange='binance'
                            self.seed_client, ex = connect_binance([self.config.simulation.seeding.fundamental.symbol.binance,self.config.simulation.seeding.external.symbol.binance], on_binance_trade)
                            if not self.seed_client:
                                bt.logging.error(f"Unable to connect to Binance Trades Stream : {ex}.")
                                if attempts >= 3:
                                    self.pagerduty_alert(f"Failed connecting to seed streams after {attempts} attempts")
                                time.sleep(min(attempts * 2, 30))  # Exponential backoff capped at 30s
                            else:
                                last_reconnect_time = time.time()
                                bt.logging.info("Connected to Binance seed stream")
                                break
                        else:
                            last_reconnect_time = time.time()
                            bt.logging.info("Connected to Coinbase seed stream")
                            break
                
                def check_seeds():
                    reconnect = False
                    current_time = time.time()
                    
                    if self.last_external:
                        self.external_file.flush()
                        time_since_external = current_time - self.last_external['received']
                        if time_since_external > 120:
                            bt.logging.warning(f"No new external price in last {time_since_external:.1f}s! Will reconnect.")
                            if self.seed_exchange=='coinbase' and self.seed_client._is_websocket_open():
                                try:
                                    self.seed_client.close()
                                except Exception:
                                    pass
                            reconnect = True
                            self.last_external = None
                            self.external_count = 0
                        sampling_period = self.config.simulation.seeding.external.sampling_seconds
                        if not self.next_external_sampling_time or not self.next_sampled_external:
                            seconds_since_start_of_day = current_time % 86400
                            start_of_day = current_time - seconds_since_start_of_day
                            self.next_external_sampling_time = start_of_day + seconds_since_start_of_day + (sampling_period - (seconds_since_start_of_day % sampling_period))
                        if current_time >= self.next_external_sampling_time and self.next_sampled_external:
                            self.sampled_external_count += 1
                            self.next_sampled_external['received'] = self.next_external_sampling_time
                            self.sampled_external_file.write(f"{self.sampled_external_count},{self.next_sampled_external['price']}\n")
                            self.sampled_external_file.flush()
                            self.last_sampled_external = self.next_sampled_external
                            self.last_sampled_external['received'] = self.next_external_sampling_time
                            self.next_external_sampling_time = self.next_external_sampling_time + sampling_period
                    return not reconnect

                connect()
                while True:
                    try:
                        if self.seed_exchange=='coinbase':
                            maintain_coinbase(self.seed_client, connect, check_seeds, 1)
                        if self.seed_exchange=='binance':
                            maintain_binance(self.seed_client, connect, check_seeds, 1)
                    except Exception as ex:
                        bt.logging.error(f"Exception in seed loop : {ex}")
                        bt.logging.debug(traceback.format_exc())
                        time.sleep(2)
                        
            except Exception as ex:
                bt.logging.error(f"Fatal error in seeding process: {ex}")
                bt.logging.error(traceback.format_exc())
                self.pagerduty_alert(
                    f"Seeding process crashed, restarting in 10s: {ex}", 
                    details={"traceback": traceback.format_exc()}
                )
                time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.logging.set_info()
    parser.add_argument('--seed.fundamental.symbol.coinbase', type=str, required=True)
    parser.add_argument('--seed.fundamental.symbol.binance', type=str, required=True)
    parser.add_argument('--seed.external.symbol.coinbase', type=str, required=True)
    parser.add_argument('--seed.external.symbol.binance', type=str, required=True)
    parser.add_argument('--seed.external.sampling_seconds', type=int, required=True)
    parser.add_argument('--logging.level', type=str, default='INFO')

    config = bt.Config(parser)
    bt.logging(config=config)

    run_seed_service(config)