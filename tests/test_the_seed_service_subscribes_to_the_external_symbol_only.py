# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The seed service subscribes to the external market's symbol only.

The fundamental's step seed is a drand round the validator attaches to its batch, so the seed service has no use for
the fundamental symbol's trades: neither the service's process nor the proxy's seed thread subscribes to it, and the
validator no longer passes it when it starts the service. The service still accepts the old options and ignores them,
so a validator started before this change can still start an updated service; the validator's own options are kept
and ignored for the same reason, since a launcher may still pass them."""
import argparse
import ast
import inspect
from pathlib import Path

from taos.im.validator import seed

ENGINE = Path(__file__).resolve().parents[1] / "taos" / "im" / "validator" / "engines" / "simulation.py"


def test_the_service_and_the_thread_subscribe_to_the_external_symbol_only():
    for fn in (seed.seed, seed.seed_thread):
        src = inspect.getsource(fn)
        assert "fundamental.symbol" not in src, fn.__name__
        assert "external.symbol.coinbase" in src and "external.symbol.binance" in src, fn.__name__


def test_the_validator_starts_the_service_without_the_fundamental_symbol():
    tree = ast.parse(ENGINE.read_text())
    starts = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_start_seed_service"]
    assert len(starts) == 1
    src = ast.get_source_segment(ENGINE.read_text(), starts[0])
    assert "--seed.fundamental" not in src
    assert "--seed.external.symbol.coinbase" in src


def test_the_service_accepts_the_old_options_and_does_not_need_them():
    parser = argparse.ArgumentParser()
    seed.add_seed_service_args(parser)
    external = ["--seed.external.symbol.coinbase", "BTC-USD", "--seed.external.symbol.binance", "btcusdt",
                "--seed.external.sampling_seconds", "60"]
    parser.parse_args(external)
    old = parser.parse_args(external + ["--seed.fundamental.symbol.coinbase", "BTC-USD",
                                        "--seed.fundamental.symbol.binance", "btcusdt"])
    assert getattr(old, "seed.external.symbol.coinbase") == "BTC-USD"
