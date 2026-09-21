# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A state push scheduled with a bare create_task can be collected before it sends.

`asyncio.create_task` returns the only strong reference to the task; the running loop keeps a weak
one. Drop that reference and an UNFINISHED task becomes eligible for garbage collection, at which
point the push never happens and nothing raises. This repo's standard says so in as many words:
"Background tasks: always keep a strong ref. Pattern in service/main.py:_spawn_bg (set +
add_done_callback(set.discard))".

The simulation state push was the one place that mattered most, because
`_build_sim_push_payload` spends 7-11s in an executor before anything is sent -- so the task sat
alive, unreferenced and collectable for seconds on every tick, not microseconds.

The damage is silent and lands on the tape: agent_fills holds the fill (written per notice, down an
independent path) while trades never receives the block. The shape is a small percentage of fills
missing on EVERY book regardless of its activity, while every ingest POST that arrived returned 200.
A uniform rate across books is whole pushes disappearing, not a per-book cap.

These are static guards: reproducing a GC race is flaky by construction, whereas "is the reference
held" is exactly what the standard requires and is deterministic.
"""

import re
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1] / "taos" / "im" / "neurons" / "validator.py").read_text()


def test_no_push_is_scheduled_without_a_strong_reference():
    # A create_task whose result is discarded on the same line is the bug. Assignment to something
    # (self._gentrx_task = ...) is fine, because that IS a strong reference.
    for _m in re.finditer(r"^\s*asyncio\.create_task\(", _SRC, re.M):
        _line = _SRC[_m.start(): _SRC.index("\n", _m.start())]
        raise AssertionError(f"push scheduled with no strong reference: {_line.strip()}")


def test_the_spawn_helper_retains_and_releases():
    assert "_PUSH_TASKS" in _SRC, "the retaining set is gone"
    _fn = _SRC[_SRC.index("def _spawn_push("):]
    _fn = _fn[: _fn.index("\n\n")]
    assert "_PUSH_TASKS.add(" in _fn, "the task is not retained"
    assert "add_done_callback(_PUSH_TASKS.discard)" in _fn, \
        "a finished task must be released, or the set leaks one entry per tick"


def test_a_failed_push_has_its_exception_retrieved():
    # Otherwise a push that raises surfaces later as "Task exception was never retrieved" attached to
    # whatever unrelated task happens to be running.
    _fn = _SRC[_SRC.index("def _spawn_push("):]
    _fn = _fn[: _fn.index("\n\n")]
    assert "exception()" in _fn, "a failed push must have its exception retrieved"


def test_the_sim_state_push_uses_the_helper():
    assert "_spawn_push(self._push_sim_state_bg(" in _SRC, \
        "the simulation state push must hold a reference; it is the longest-lived of them"
