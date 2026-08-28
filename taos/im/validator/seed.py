# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Market data seed collection: real-time price feed from Coinbase/Binance,
written to CSV files accessible by the simulator.
"""
import os
import argparse
import signal
import traceback
import time
import bittensor as bt


from taos.im.utils.streams import *

_current_log_dir = None
_log_dir_changed = False

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
    Main seed data collection loop (runs in a separate process).

    Opens WebSocket connections to Coinbase and Binance, writes incoming
    price data to CSV files, and rotates log directories on SIGUSR1.

    Args:
        config: Configuration object with seed symbols and data directory path.
    """
    global _current_log_dir, _log_dir_changed
    
    seed_count = 0
    seed_filename = None
    seed_file = None
    last_seed = None
    pending_seed_data = ''
    
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
        global _current_log_dir, _log_dir_changed
        nonlocal seed_filename, seed_file, seed_count, last_seed, pending_seed_data
        nonlocal external_filename, external_file, sampled_external_filename
        nonlocal sampled_external_file, external_count, last_external, pending_external_data
        nonlocal sampled_external_count
        
        if not _log_dir_changed and _current_log_dir:
            return
        new_log_dir = None
        log_dir_file = '/tmp/validator_log_dir.txt'
        if os.path.exists(log_dir_file):
            with open(log_dir_file, 'r') as f:
                new_log_dir = f.read().strip()
        
        if new_log_dir != _current_log_dir:
            bt.logging.info(f"Log directory changed: {_current_log_dir} -> {new_log_dir}")
            _current_log_dir = new_log_dir
            
            # Close existing files
            if seed_file:
                try:
                    seed_file.close()
                except Exception:
                    pass
                seed_file = None
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
            seed_filename = None
            last_seed = None
            seed_count = 0
            pending_seed_data = ''
            
            external_filename = None
            sampled_external_filename = None
            last_external = None
            external_count = 0
            sampled_external_count = 0
            pending_external_data = ''
        else:            
            _log_dir_changed = False
    
    def on_coinbase_trade(trade: dict):
        """Handle one Coinbase trade message and record it as seed flow.

        Args:
            trade (dict): The exchange's trade payload.
        """
        nonlocal last_seed, seed_count, pending_seed_data
        nonlocal last_external, external_count, pending_external_data
        nonlocal next_sampled_external

        check_log_dir_change()
        
        match trade['product_id']:
            case config.seed.fundamental.symbol.coinbase:
                record_seed(trade)
            case config.seed.external.symbol.coinbase:
                if next_external_sampling_time and trade['received'] <= next_external_sampling_time:
                    next_sampled_external = trade
                record_external(trade)

    def on_binance_trade(trade: dict):
        """Handle one Binance trade message and record it as seed flow.

        Args:
            trade (dict): The exchange's trade payload.
        """
        nonlocal last_seed, seed_count, pending_seed_data
        nonlocal last_external, external_count, pending_external_data

        check_log_dir_change()
        
        match trade['product_id']:
            case config.seed.fundamental.symbol.binance:
                record_seed(trade)
            case config.seed.external.symbol.binance:
                record_external(trade)

    def record_seed(trade: dict) -> None:
        """Record a fundamental-seed trade for the simulation to consume.

        Args:
            trade (dict): Normalised trade with price, size and side.
        """
        nonlocal seed_count, pending_seed_data, last_seed, seed_filename, seed_file
        
        try:
            seed = trade['price']
            if not last_seed or last_seed['price'] != seed:
                if not _current_log_dir:
                    seed_count += 1
                    pending_seed_data += f"{seed_count},{seed}\n"
                    if len(pending_seed_data.split("\n")) > 10000:
                        pending_seed_data = "\n".join(pending_seed_data.split("\n")[-10000:])
                else:
                    if seed_filename != os.path.join(_current_log_dir, "fundamental_seed.csv"):
                        last_seed = None
                        seed_count = 0
                        pending_seed_data = ''
                        if seed_file:
                            seed_file.close()
                            seed_file = None
                    if not last_seed:
                        seed_filename = os.path.join(_current_log_dir, "fundamental_seed.csv")
                        if os.path.exists(seed_filename) and os.stat(seed_filename).st_size > 0:
                            with open(seed_filename) as f:
                                for _line in f:
                                    seed_count += 1
                        seed_file = open(seed_filename, 'a')
                        seed_file.write(pending_seed_data)
                        pending_seed_data = ''
                    seed_count += 1
                    seed_file.write(f"{seed_count},{seed}\n")
                    seed_file.flush()
                    last_seed = trade
        except Exception as ex:
            bt.logging.error(f"Exception in seed handling: Seed={seed} | Error={ex}")
                    
    def record_external(trade: dict) -> None:
        """Record an external-venue trade used for calibration rather than seeding.

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
                        external_file = open(external_filename, 'a')
                        external_file.write(pending_external_data)
                        pending_external_data = ''
                        
                        sampled_external_filename = os.path.join(_current_log_dir, "external_seed_sampled.csv")
                        if os.path.exists(sampled_external_filename) and os.stat(sampled_external_filename).st_size > 0:
                            with open(sampled_external_filename) as f:
                                for _line in f:
                                    sampled_external_count += 1
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
        """Verify seed flow is still arriving and reconnect the sources that have gone quiet."""
        nonlocal last_seed, seed_count, last_external, external_count
        nonlocal next_external_sampling_time, next_sampled_external, last_sampled_external
        nonlocal sampled_external_count, seed_file, external_file, sampled_external_file
        
        reconnect = False
        current_time = time.time()
        
        if last_seed:
            if seed_file:
                seed_file.flush()
            time_since_seed = current_time - last_seed['received']
            if time_since_seed > 10:
                bt.logging.warning(f"No new seed in last {time_since_seed:.1f}s! Will reconnect.")
                if seed_exchange == 'coinbase' and seed_client._is_websocket_open():
                    try:
                        seed_client.close()
                    except Exception:
                        pass
                reconnect = True
                last_seed = None
                seed_count = 0
            
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
        Retrieve data for use as simulation fundamental price and external seed, and record to simulator-accessible location.
        This process is run in a separate thread parallel to the FastAPI server.
        """
        while True:
            try:
                self.seed_count = 0
                self.seed_filename = None
                self.last_seed_count = self.seed_count
                self.last_seed = None
                self.pending_seed_data = ''
                
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
                        case self.config.simulation.seeding.fundamental.symbol.coinbase:
                            record_seed(trade)
                        case self.config.simulation.seeding.external.symbol.coinbase:
                            if self.next_external_sampling_time and trade['received'] <= self.next_external_sampling_time:
                                self.next_sampled_external = trade
                            record_external(trade)

                def on_binance_trade(trade : dict):
                    match trade['product_id']:
                        case self.config.simulation.seeding.fundamental.symbol.binance:
                            record_seed(trade)
                        case self.config.simulation.seeding.external.symbol.binance:
                            record_external(trade)

                def record_seed(trade : dict) -> None:
                    try:
                        seed = trade['price']
                        if not self.last_seed or self.last_seed['price'] != seed:
                            if not self.simulation.logDir:
                                self.seed_count += 1
                                self.pending_seed_data += f"{self.seed_count},{seed}\n"
                                if len(self.pending_seed_data.split("\n")) > 10000:
                                    self.pending_seed_data = "\n".join(self.pending_seed_data.split("\n")[-10000:])
                            else:
                                if self.seed_filename != os.path.join(self.simulation.logDir,"fundamental_seed.csv"):
                                    self.last_seed = None
                                    self.seed_count = 0
                                    self.pending_seed_data = ''
                                if not self.last_seed:
                                    self.seed_filename = os.path.join(self.simulation.logDir,"fundamental_seed.csv")
                                    if os.path.exists(self.seed_filename) and os.stat(self.seed_filename).st_size > 0:
                                        with open(self.seed_filename) as f:
                                            for _line in f:
                                                self.seed_count += 1
                                    self.seed_file = open(self.seed_filename,'a')
                                    self.seed_file.write(self.pending_seed_data)
                                    self.pending_seed_data = ''
                                self.seed_count += 1
                                self.seed_file.write(f"{self.seed_count},{seed}\n")
                                self.seed_file.flush()
                                self.last_seed = trade
                    except Exception as ex:
                        bt.logging.error(f"Exception in seed handling : Seed={seed} | Error={ex}")
                        
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
                                    self.external_file = open(self.external_filename,'a')
                                    self.external_file.write(self.pending_external_data)
                                    self.pending_external_data = ''
                                    
                                    self.sampled_external_filename = os.path.join(self.simulation.logDir,"external_seed_sampled.csv")
                                    if os.path.exists(self.sampled_external_filename) and os.stat(self.sampled_external_filename).st_size > 0:
                                        with open(self.sampled_external_filename) as f:
                                            for _line in f:
                                                self.sampled_external_count += 1
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
                    
                    if self.last_seed:
                        self.seed_file.flush()
                        time_since_seed = current_time - self.last_seed['received']
                        if time_since_seed > 10:
                            bt.logging.warning(f"No new seed in last {time_since_seed:.1f}s! Will reconnect.")
                            if self.seed_exchange=='coinbase' and self.seed_client._is_websocket_open():
                                try:
                                    self.seed_client.close()
                                except Exception:
                                    pass
                            reconnect = True
                            self.last_seed = None
                            self.seed_count = 0
                        self.last_seed_count = self.seed_count
                        
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