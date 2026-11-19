# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""An engine under dirtyAccountsOnly sends an account's entry for a book in full only when it changed there, and says
so with "dirtyAccountsOnly" in the state. Main and the scoring process each fill the rest in from the last full copy,
so everything downstream sees complete accounts; a state that cannot be completed (a validator that has just started)
is skipped and the next answer asks the engine for every account in full; the scoring process's INIT carries main's
copy, so a child rebuilt mid-run fills from the same point main applied."""
from types import SimpleNamespace

import msgpack

import test_scoring_shadow as tss
import taos.im.validator.trade as trade
from taos.im.validator.engines.simulation import SimulationEngine
from taos.im.validator.kept_accounts import OMITTED, fill_accounts
from taos.im.validator.scoring_shadow import apply_state_bytes

FEES = {"v": None, "m": 0.001, "t": 0.002}


def entry(agent, book, total=None):
    """A full entry when `total` is given (the base balance's total), else one the engine left out."""
    if total is None:
        return {"i": agent, "b": book, **{k: None for k in OMITTED}, "f": FEES}
    balance = {"c": "BASE", "t": total, "f": total, "r": 0.0, "i": total}
    return {"i": agent, "b": book, "bb": balance, "qb": dict(balance, c="QUOTE"), "bl": 0.0, "ql": 0.0,
            "bc": 0.0, "qc": 0.0, "o": [], "l": {}, "f": FEES}


def accounts(**totals):
    """{agent: {book: entry}} for agents 0 and 1 on books 0 and 1; totals["a0b1"] makes that entry full."""
    return {a: {b: entry(a, b, totals.get(f"a{a}b{b}")) for b in (0, 1)} for a in (0, 1)}


def full_accounts(total):
    return accounts(**{f"a{a}b{b}": total for a in (0, 1) for b in (0, 1)})


def test_entries_left_out_are_filled_from_the_last_full_ones():
    kept = {}
    assert fill_accounts(kept, full_accounts(5.0)) == 0
    later = accounts(a0b0=7.0)
    assert fill_accounts(kept, later) == 0
    assert later[0][0]["bb"]["t"] == 7.0, "a full entry is taken as sent"
    assert later[1][1]["bb"]["t"] == 5.0, "a left-out entry gets the last full values"
    assert all(later[a][b][k] is not None for a in (0, 1) for b in (0, 1) for k in OMITTED)
    then = accounts()
    fill_accounts(kept, then)
    assert then[0][0]["bb"]["t"] == 7.0, "the newer full entry replaced the older in the copy"


def test_entries_with_no_full_copy_are_counted():
    assert fill_accounts({}, accounts(a0b0=1.0)) == 3


def _engine():
    engine = SimulationEngine.__new__(SimulationEngine)
    engine.validator = SimpleNamespace()
    engine.sent = []
    engine._send_bytes = engine.sent.append
    return engine


def test_main_fills_a_state_the_engine_marks_and_leaves_any_other_alone():
    engine = _engine()
    assert engine._complete_accounts({"dirtyAccountsOnly": True, "accounts": full_accounts(5.0)})
    state = {"dirtyAccountsOnly": True, "accounts": accounts(a1b0=6.0)}
    assert engine._complete_accounts(state)
    assert state["accounts"][0][1]["bb"]["t"] == 5.0
    assert engine.dirty_accounts_only

    other = _engine()
    plain = {"accounts": accounts()}
    assert other._complete_accounts(plain), "an engine that does not mark its state sends every entry in full"
    assert plain["accounts"][0][0]["bb"] is None and not other.dirty_accounts_only


def test_a_state_main_cannot_complete_is_skipped_and_the_next_answer_asks_for_everything_once():
    engine = _engine()
    assert not engine._complete_accounts({"dirtyAccountsOnly": True, "accounts": accounts(a0b0=1.0)})

    engine.respond(None, {"responses": []})
    engine.respond(None, {"responses": []})
    first, second = (msgpack.unpackb(b, raw=False) for b in engine.sent)
    assert first == {"responses": [], "fullAccounts": True}
    assert second == {"responses": []}


def _shadow_applying(monkeypatch, shadow, state):
    seen = {}
    monkeypatch.setattr(trade, "update_trade_volumes", lambda s, st: seen.setdefault("accounts", st.accounts))
    raw = msgpack.packb({"timestamp": tss._S, "books": {}, "notices": {}, **state}, use_bin_type=True)
    apply_state_bytes(shadow, raw)
    return seen["accounts"]


def test_the_scoring_process_fills_from_the_copy_its_init_carried(monkeypatch, capsys):
    main = tss._fresh_main()
    main.kept_accounts = full_accounts(5.0)
    shadow = tss._snapshot_to_shadow(main)
    assert shadow.kept_accounts == main.kept_accounts, "the INIT carries main's last complete accounts"

    shadow.validator_uid = None
    filled = _shadow_applying(monkeypatch, shadow, {"dirtyAccountsOnly": True, "accounts": accounts(a0b0=8.0)})
    assert filled[0][0]["bb"]["t"] == 8.0
    assert filled[1][1]["bb"]["t"] == 5.0
    assert "no full copy" not in capsys.readouterr().out


def test_an_init_without_accounts_ships_nothing_for_them():
    main = tss._fresh_main()
    shadow = tss._snapshot_to_shadow(main)
    assert shadow.kept_accounts == {}


def test_a_state_answered_without_being_read_asks_for_everything_once():
    # The engine counts any answer as delivered and stops listing what that state carried, so an answer the listener
    # sends after failing to read a state must ask for every account, or main's copy misses those changes for good.
    engine = _engine()
    assert engine._complete_accounts({"dirtyAccountsOnly": True, "accounts": full_accounts(5.0)})
    engine.ask_full_accounts()

    engine.respond(None, {"responses": []})
    engine.respond(None, {"responses": []})
    first, second = (msgpack.unpackb(b, raw=False) for b in engine.sent)
    assert first == {"responses": [], "fullAccounts": True}
    assert second == {"responses": []}


def test_an_engine_that_sends_every_account_is_never_asked_for_them():
    engine = _engine()
    assert engine._complete_accounts({"accounts": accounts()})
    engine.ask_full_accounts()

    engine.respond(None, {"responses": []})
    assert msgpack.unpackb(engine.sent[0], raw=False) == {"responses": []}


def test_the_listener_asks_for_every_account_when_it_could_not_read_a_state():
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parents[1] / "taos/im/neurons/validator.py").read_text()
    receive = src.index("await self.engine.receive()")
    handler = src.index("except Exception as ex:", receive)
    answer = src.index("self.engine.respond(raw_message, response)", handler)
    assert re.search(r"self\.engine\.ask_full_accounts\(\)", src[handler:answer]), \
        "the listener's error path answers the state; it must ask for every account before that answer"
