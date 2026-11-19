# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The account deltas are a round-time lever like the others (levers.py): turned off on a live validator, every answer
asks the engine for every account in full, so the next state and each one after it lists them all, as from an engine
without dirtyAccountsOnly, with no engine restart. On, only an answer that has to (a state it could not complete or
read) asks, once. A validator whose engine sends every account anyway never asks."""
import importlib
import os

import msgpack
import pytest
from test_account_entries_left_out_are_filled_in import _engine

from taos.im.validator import levers as lv


@pytest.fixture
def levers(tmp_path, monkeypatch):
    path = tmp_path / "levers_off"
    monkeypatch.setenv("TAOS_LEVERS_FILE", str(path))
    monkeypatch.delenv("TAOS_LEVERS_OFF", raising=False)
    mod = importlib.reload(lv)
    monkeypatch.setattr(mod, "CHECK_S", 0.0)
    yield mod, path
    monkeypatch.delenv("TAOS_LEVERS_FILE", raising=False)
    importlib.reload(lv)


def _answers(engine, n=2):
    engine.sent.clear()
    for _ in range(n):
        engine.respond(None, {"responses": []})
    return [msgpack.unpackb(b, raw=False) for b in engine.sent]


def _write(path, text, ns):
    path.write_text(text)
    os.utime(path, ns=(ns, ns))


def test_account_deltas_is_a_lever(levers):
    mod, _ = levers
    assert "account_deltas" in mod.LEVERS and mod.states()["account_deltas"] is True


def test_off_every_answer_asks_for_every_account_and_on_again_none_does(levers):
    mod, path = levers
    engine = _engine()
    engine.dirty_accounts_only = True
    assert _answers(engine) == [{"responses": []}] * 2

    _write(path, "account_deltas\n", 1)
    assert _answers(engine) == [{"responses": [], "fullAccounts": True}] * 2

    _write(path, "", 2)
    assert _answers(engine) == [{"responses": []}] * 2


def test_an_engine_that_sends_every_account_is_never_asked(levers):
    _, path = levers
    engine = _engine()
    _write(path, "account_deltas\n", 1)
    assert _answers(engine) == [{"responses": []}] * 2
