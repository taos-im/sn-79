# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Every pm2 start of the engine in update.py registers a 60 s kill timeout.

pm2 SIGKILLs 1600 ms after the stop signal by default, before the engine reaches a barrier and
writes its shutdown checkpoint. The entry registered here is what pm2 re-runs for the rest of the
host's life, so a start without the timeout turns every later stop into a crash.
"""
import ast
from pathlib import Path

UPDATE = Path(__file__).resolve().parents[1] / "taos" / "im" / "validator" / "update.py"


def _engine_starts():
    tree = ast.parse(UPDATE.read_text())
    starts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        words = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if words[:2] == ["pm2", "start"] and "--name=simulator" in words:
            starts.append((node.lineno, words))
    return starts


def test_update_starts_the_engine_somewhere():
    assert len(_engine_starts()) >= 3, "the resume and new-simulation starts were not found"


def test_every_engine_start_carries_the_kill_timeout():
    missing = []
    for lineno, words in _engine_starts():
        i = words.index("--kill-timeout") if "--kill-timeout" in words else -1
        if i < 0 or words[i + 1:i + 2] != ["60000"]:
            missing.append(lineno)
    assert not missing, f"pm2 start of the engine without --kill-timeout 60000 at update.py lines {missing}"


def test_no_engine_start_is_a_shell_string():
    assert "pm2 start" not in UPDATE.read_text(), "a pm2 start built as a string escapes this check"
