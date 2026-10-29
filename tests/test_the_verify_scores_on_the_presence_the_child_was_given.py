# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The cutover VERIFY recomputes a boundary in-process and compares it with the child's result, so both must score
on the same inputs. The child scores on the absent set and presence shares main pinned when it teed the boundary
state, before that state's query; main's verify read its live query windows, which the listener keeps updating
with the outcomes of the boundary's own query and of every query that lands while the verify runs. A uid that
crosses the presence gate in between (one success after a window of failures) is scorable on one side only, every
other uid's share shifts with it, and the verify reads a MISMATCH and re-initialises a child that had not diverged.
The verify now pins the shipped absent set and shares like the other inputs, and main's de-beta scorer takes them."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

import taos.im.validator.reward as reward

from test_debeta_presence_gate import _fake_validator

DEV = Path(__file__).resolve().parents[1]
VALIDATOR = (DEV / "taos/im/neurons/validator.py").read_text()


def test_a_pinned_absent_set_outranks_the_live_windows():
    # uid 2's live window holds one success, so live it is present; the pin says it was absent at tee time
    live = {2: deque([False] * 49 + [True], maxlen=50), 1: deque([True] * 50, maxlen=50)}
    assert 2 in reward.compute_debeta_scores(_fake_validator(presence=live)), "present on the live windows"
    v = _fake_validator(presence=live)
    pinned = reward.compute_debeta_scores(v, absent=[2], presence={1: 1.0, 2: 0.0, 3: 1.0})
    assert 2 not in pinned and 1 in pinned and 3 in pinned
    assert v.debeta_detail[2]["present"] is False
    assert not hasattr(v, "debeta_absent"), "the pin is scoped to the call, never left on main"


def test_no_pin_reads_the_live_windows():
    dark = {2: deque([False] * 50, maxlen=50)}
    assert 2 not in reward.compute_debeta_scores(_fake_validator(presence=dark), absent=None, presence=None)


class _Stop(Exception):
    pass


def _capture(monkeypatch):
    seen = {}

    def _debeta(self, absent=None, presence=None):
        seen.update(absent=absent, presence=presence)
        raise _Stop

    monkeypatch.setattr(reward, "build_scoring_config", lambda self: {})
    monkeypatch.setattr(reward, "build_simulation_config_dict", lambda self: {})
    monkeypatch.setattr(reward, "compute_debeta_scores", _debeta)
    v = SimpleNamespace(roundtrip_volumes={}, realized_pnl_history={}, effective_max_uids=4, kappa_values={},
                        kappa_cache={}, activity_factors={}, pnl_factors={}, simulation_timestamp=0,
                        deregistered_uids=[], device="cpu", miner_presence={})
    return seen, v


def test_get_rewards_hands_the_pinned_presence_to_the_scorer(monkeypatch):
    seen, v = _capture(monkeypatch)
    with pytest.raises(_Stop):
        reward.get_rewards(v, pinned_inputs={"simulation_timestamp": 5, "absent": [3], "presence": {1: 0.5}})
    assert seen == {"absent": [3], "presence": {1: 0.5}}


def test_get_rewards_without_a_pin_scores_on_the_live_windows(monkeypatch):
    seen, v = _capture(monkeypatch)
    with pytest.raises(_Stop):
        reward.get_rewards(v)
    assert seen == {"absent": None, "presence": None}


def test_the_verify_pins_the_presence_the_child_was_given():
    block = VALIDATOR[VALIDATOR.index("Verify must compute with EXACTLY the inputs the child"):]
    block = block[:block.index("} if _verify else None")]
    assert "'absent': _absent" in block and "'presence': _presence" in block, (
        "the verify must pin the absent set and the shares shipped with the boundary, not read the live windows"
    )
