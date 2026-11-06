# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A levers file that cannot be read turns no lever off, and nothing a lever's reader or the query's hook raises
reaches the forward.

levers.enabled() runs after the query is sent and before the wait for the miners' answers (on_query_sent), and in
the report's pack. A levers file that is not UTF-8 reads as no file, so every lever stays on; the hook the forward
calls once the query is with the query process cannot fail the forward; and a lever turned off through
TAOS_LEVERS_OFF is logged once, as a change in the file is."""
import importlib
import inspect
import os

import pytest

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


def test_a_file_that_is_not_utf8_leaves_every_lever_on(levers):
    mod, path = levers
    path.write_bytes(b"push_in_wait\n\xff\xfe\x80 not text\n")
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    assert mod.enabled("push_in_wait") and mod.enabled("pack_shared")
    assert mod.states() == {n: True for n in mod.LEVERS}


def test_a_lever_turned_off_through_the_environment_is_logged_once(tmp_path, monkeypatch):
    monkeypatch.setenv("TAOS_LEVERS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("TAOS_LEVERS_OFF", "pack_shared,pack_sharde")
    try:
        mod = importlib.reload(lv)
        said = []
        monkeypatch.setattr(mod, "_say", lambda msg: said.append(msg))
        assert not mod.enabled("pack_shared")
        assert mod.enabled("push_in_wait")
        mod.states()
        assert len(said) == 1, said
        assert "TAOS_LEVERS_OFF" in said[0] and "pack_shared off" in said[0] and "pack_sharde" in said[0], said
    finally:
        monkeypatch.delenv("TAOS_LEVERS_OFF")
        monkeypatch.delenv("TAOS_LEVERS_FILE")
        importlib.reload(lv)


def test_nothing_the_query_sent_hook_raises_reaches_the_forward():
    from taos.im.validator import forward as fwd
    src = inspect.getsource(fwd.forward)
    call = src.index("_on_sent(synapse)")
    before, after = src[:call], src[call:]
    guard = before.rindex("try:")
    assert "\n" in before[guard:] and before[guard:].count("\n") == 1, "the call is the try block's first line"
    handler = after[:after.index("max_wait")]
    assert "except Exception" in handler and "bt.logging.exception(" in handler, handler
