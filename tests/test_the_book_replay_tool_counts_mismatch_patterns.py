# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""tools/replay_book_reconstruction.py replays a captured state stream through the miner library's book rebuild and
counts the mismatches by pattern. It is the instrument that shows what a miner's book check sees on a real stream,
so its counting is held to known streams: a cancellation on the published side rebuilds exactly, one the engine
publishes without a side and the guess gets wrong counts as a side mismatch, and a publisher's extra precision
counts as tolerance only under the exact comparison the library used before."""
import importlib.util
import sys
from pathlib import Path

import msgpack
import pytest

from taos.im.validator import state_capture

ROOT = Path(__file__).resolve().parents[1]
_TOOL = ROOT / "tools" / "replay_book_reconstruction.py"
if not _TOOL.exists():
    # The public tree ships tests/ without tools/; the instrument under test exists in the dev tree only.
    pytest.skip("tools/replay_book_reconstruction.py is not part of this tree", allow_module_level=True)
spec = importlib.util.spec_from_file_location("replay_book_reconstruction", _TOOL)
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)


def _update(ts, bids, asks, events):
    book = {"i": 0, "r": None, "b": [{"p": p, "q": q} for p, q in bids.items()], "a": [{"p": p, "q": q} for p, q in asks.items()],
            "e": events}
    return msgpack.packb({"timestamp": int(ts), "books": {0: book}, "accounts": {}, "notices": {}}, use_bin_type=True)


def _capture(tmp_path, updates):
    env = {state_capture.ENV_DIR: str(tmp_path)}
    for ts, raw in updates:
        assert state_capture.capture_state(raw, ts, env=env)
    return str(tmp_path)


def test_classify_names_tolerance_side_and_other():
    lines = ["BID : RECON 1.00004@100.0 vs. TARGET 1.0@100.0",
             "BID : RECON 0.0@99.0 vs. TARGET 2.0@99.0", "ASK : RECON 2.0@99.0 vs. TARGET 0.0@99.0",
             "ASK : RECON 5.0@101.0 vs. TARGET 3.0@101.0"]
    out = tool.classify(lines, 0.00005, cancel_prices=[99.0])
    assert out == {"tolerance": 1, "side": 1, "other": 1}, out
    assert tool.classify(lines, 0.00005) == {"tolerance": 1, "other": 3}


def test_a_cancellation_on_the_published_side_rebuilds_the_next_update_exactly(tmp_path):
    # a bid at 99 is cancelled while the best ask sits below it: the guess from the best ask would take it for an ask
    first = _update(1_000, {99.0: 2.0}, {98.5: 1.0}, [])
    second = _update(2_000, {}, {98.5: 1.0}, [{"y": "c", "i": 7, "t": 1_500, "p": 99.0, "q": 2.0, "s": 0}])
    out = tool.replay(_capture(tmp_path, [(1_000, first), (2_000, second)]), lambda b: 4)
    assert out["steps"] == 1 and out["totals"]["books_matched"] == 1, out
    assert out["books_with_side_mismatch"] == []


def test_a_cancellation_without_a_side_the_guess_gets_wrong_counts_as_a_side_mismatch(tmp_path):
    first = _update(1_000, {99.0: 2.0}, {98.5: 1.0}, [])
    second = _update(2_000, {}, {98.5: 1.0}, [{"y": "c", "i": 7, "t": 1_500, "p": 99.0, "q": 2.0}])
    out = tool.replay(_capture(tmp_path, [(1_000, first), (2_000, second)]), lambda b: 4)
    assert out["totals"]["books_matched"] == 0 and out["totals"].get("side", 0) >= 1, out
    assert out["books_with_side_mismatch"] == [0]


def test_extra_precision_counts_as_tolerance_only_under_the_exact_comparison(tmp_path):
    first = _update(1_000, {99.0: 557.0339}, {}, [])
    second = _update(2_000, {99.0: 557.03392}, {}, [])
    captured = _capture(tmp_path, [(1_000, first), (2_000, second)])
    tolerant = tool.replay(captured, lambda b: 4)
    assert tolerant["totals"]["books_matched"] == 1 and "tolerance" not in tolerant["totals"], tolerant
    exact = tool.replay(captured, lambda b: 4, exact=True)
    assert exact["totals"]["books_matched"] == 0 and exact["totals"]["tolerance"] == 1, exact
