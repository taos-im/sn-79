# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The scoring shadow's side buffer must be bounded, like every other queue it keeps.

The shadow child holds two queues of raw state frames. `buffered` collects them before INIT and is
capped at 64 with an explicit drop-oldest and a log line. `side_buffer` collects them while a score
is awaited and was capped at nothing at all -- so a scoring round that stalls accumulates whole
rounds of raw state, each of them megabytes, until something gives.

That is the shape of the validator-restart memory spike: a spawned child reaching many gigabytes
while the metagraph path it was blamed on costs about 125MB in total.

A drop here is NOT free, and differs from the pre-INIT queue's drop. State frames are applied
cumulatively, so a dropped frame leaves the shadow permanently short of a round and it diverges from
main thereafter. The divergence is caught rather than hidden: the parent's parity check sees the
mismatch and calls request_reinit(), and the shadow rebuilds. That self-heal is what makes a drop
admissible at all, it is expensive, and enough repeats trip the reinit storm breaker. So the cap is
set to fire only in genuine pathology -- but a cap that fires expensively still beats a queue that
grows until the box gives out. Main scoring is unaffected either way.

A source-level guard, because reaching the branch needs a live child, a socketpair, an INIT and a
stalled scoring round -- and the property worth pinning is that the cap exists at all.
"""

import re
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parents[1] / "taos" / "im" / "validator" / "scoring_shadow.py"
_SRC = _SRC_PATH.read_text(encoding="utf-8")


def _child_body() -> str:
    """The child function's body: the two queues live there and nowhere else."""
    i = _SRC.index("side_buffer = []")
    return _SRC[i:i + 12000]


def test_the_side_buffer_has_a_cap():
    body = _child_body()
    assert re.search(r"len\(side_buffer\)\s*>", body), (
        "side_buffer grows without limit; a stalled scoring round accumulates whole rounds of raw "
        "state until the box runs out"
    )


def test_it_drops_the_oldest_rather_than_refusing_the_newest():
    """The newest frame is the one closest to the score being awaited, so it is the one worth keeping."""
    body = _child_body()
    assert "side_buffer.pop(0)" in body, "the cap must drop the oldest frame, not discard new ones"


def test_the_drop_is_announced():
    """A silent drop turns a parity gap into an unexplained mismatch later."""
    body = _child_body()
    i = body.index("side_buffer.pop(0)")
    assert "[SHADOW]" in body[max(0, i - 400):i], "a dropped frame must say so, as the pre-INIT queue does"


def test_both_queues_are_bounded():
    """buffered was always capped; the asymmetry between the two is what let this through."""
    body = _child_body()
    assert re.search(r"len\(buffered\)\s*>", body), "the pre-INIT queue lost its cap"
