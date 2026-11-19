# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The proxy fills in the account entries an engine under dirtyAccountsOnly left out, as the validator does.

The proxy stands in for the validator on local runs, and a scorer built on it subclasses it. An engine on account deltas sends an unchanged entry with its balances, orders and loans nil; read
as it came, the agents and the scorer see an account with nothing in it. The proxy keeps the last full entry of
each agent and book (kept_accounts) and fills the rest in; a state it cannot fill yet, on a proxy that has just
started, is answered with no instructions and a request for every account in full.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

from taos.im.validator.kept_accounts import OMITTED

PROXY = Path(__file__).resolve().parents[1] / "agents" / "proxy" / "proxy.py"
FULL = {"bb": {"t": 3.5}, "qb": {"t": 1000.0}, "bl": 0.0, "ql": 0.0, "bc": 0.0, "qc": 0.0, "o": [{"i": 9}], "l": {}}


def _state(entry, deltas=True):
    s = {"accounts": {7: {0: {"i": 7, "b": 0, **entry, "f": {"m": 0.0001}}}}}
    if deltas:
        s["dirtyAccountsOnly"] = True
    return s


def test_a_left_out_entry_is_filled_from_the_last_full_one():
    import agents.proxy.proxy as proxy
    p = SimpleNamespace()
    assert proxy.Proxy._complete_accounts(p, _state(FULL))
    thin = _state(dict.fromkeys(OMITTED))
    assert proxy.Proxy._complete_accounts(p, thin)
    entry = thin["accounts"][7][0]
    assert entry["bb"] == {"t": 3.5} and entry["o"] == [{"i": 9}] and entry["f"] == {"m": 0.0001}


def test_a_proxy_with_no_full_copy_skips_the_state():
    import agents.proxy.proxy as proxy
    assert not proxy.Proxy._complete_accounts(SimpleNamespace(), _state(dict.fromkeys(OMITTED)))


def test_an_engine_that_sends_every_account_is_read_as_it_came():
    import agents.proxy.proxy as proxy
    p = SimpleNamespace()
    assert proxy.Proxy._complete_accounts(p, _state(dict.fromkeys(OMITTED), deltas=False))
    assert getattr(p, "_kept_accounts", None) is None


def test_the_listener_fills_in_before_it_parses_and_asks_for_every_account_on_a_skip():
    tree = ast.parse(PROXY.read_text())
    listen = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_listen")
    src = ast.get_source_segment(PROXY.read_text(), listen)
    fill, parse = src.find("self._complete_accounts(message)"), src.find("MarketSimulationStateUpdate.parse_dict(message)")
    assert 0 < fill < parse, "the accounts are filled in before the state is parsed"
    assert '"fullAccounts": True' in src[fill:parse], "a skipped state asks the engine for every account in full"
