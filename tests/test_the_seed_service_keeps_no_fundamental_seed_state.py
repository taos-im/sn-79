# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The seed service records the external market's price only, and keeps no state for a fundamental seed.

The fundamental's step seed is a drand round the validator attaches to its batch (fundamental_seed.py). The seed
service still subscribes to the fundamental symbol, whose options the launchers pass, and records none of its trades,
so it holds no seed file, count or last seed, and has no watchdog on a seed that never arrives. Both the service's
process and the proxy's seed thread follow that, and the options' help says what they do."""
import inspect

from taos.im.validator import seed


def test_neither_the_service_nor_the_thread_keeps_seed_state():
    src = inspect.getsource(seed)
    for name in ("seed_count", "seed_filename", "seed_file", "last_seed", "pending_seed_data", "No new seed"):
        assert name not in src.replace("external", ""), name


def test_the_external_price_is_still_recorded_and_watched():
    for fn in (seed.seed, seed.seed_thread):
        src = inspect.getsource(fn)
        assert '"external_seed.csv"' in src and "No new external price" in src, fn.__name__


def test_the_fundamental_options_say_their_trades_are_not_recorded():
    import argparse
    from taos.im.config.simulation import add_simulation_args
    parser = argparse.ArgumentParser()
    add_simulation_args(None, parser)
    helps = {a.dest: a.help for a in parser._actions}
    for venue in ("coinbase", "binance"):
        text = helps[f"simulation.seeding.fundamental.symbol.{venue}"]
        assert "seed simulation price" not in text and "not recorded" in text, text
