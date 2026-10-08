# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The scoring acceptance's replay harness certifies the pay a validator produces from the state it receives, so
the validator must be able to record that state: the raw bytes of every update, one file per update named by its
timestamp, only when STATE_CAPTURE_DIR is set, within a byte budget, and never at the cost of the listener loop."""
import os
import re
from pathlib import Path

from taos.im.validator import state_capture as sc

VALIDATOR = Path(__file__).resolve().parents[1] / "taos" / "im" / "neurons" / "validator.py"


def test_capture_is_off_without_the_directory(tmp_path):
    assert sc.capture_state(b"abc", 1_000, env={}) is None
    assert list(tmp_path.iterdir()) == []


def test_a_captured_update_reads_back_byte_for_byte_in_timestamp_order(tmp_path):
    env = {sc.ENV_DIR: str(tmp_path)}
    raw_a, raw_b = os.urandom(5000), b"\x82\xa9timestamp\xcf" + os.urandom(2000)
    assert sc.capture_state(raw_b, 2_000_000_000, env=env)
    assert sc.capture_state(raw_a, 1_000_000_000, env=env)
    assert [ts for ts, _ in sc.captured_files(str(tmp_path))] == [1_000_000_000, 2_000_000_000]
    assert [(ts, raw) for ts, raw in sc.iter_captured(str(tmp_path))] == [(1_000_000_000, raw_a), (2_000_000_000, raw_b)]
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir()), "no partial file is left behind"


def test_the_byte_budget_drops_the_oldest_updates_first(tmp_path):
    env = {sc.ENV_DIR: str(tmp_path), sc.ENV_MAX_BYTES: "1"}
    for ts in (1, 2, 3):
        sc.capture_state(os.urandom(3000), ts * 1_000_000_000, env=env)
    kept = [ts for ts, _ in sc.captured_files(str(tmp_path))]
    assert kept == [3_000_000_000], f"a one-byte budget keeps only the newest update, kept {kept}"


def test_a_failing_write_returns_none_instead_of_raising(tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    assert sc.capture_state(b"abc", 5, env={sc.ENV_DIR: str(blocker)}) is None


def test_the_validator_captures_every_received_state_before_anything_else_reads_it():
    src = VALIDATOR.read_text()
    receive = src.index("raw_message, normalized_state, receive_start = await self.engine.receive()")
    call = re.search(r"capture_state\(raw_message, normalized_state\.timestamp\)", src)
    assert call, "the listener loop does not call capture_state on the received bytes"
    tee = src.index("self._scoring_shadow.tee(raw_message, normalized_state.timestamp)")
    assert receive < call.start() < tee, "the capture sits between the receive and the shadow tee, so it sees every update"
    assert re.search(r"^\s*from taos\.im\.validator\.state_capture import capture_state", src, flags=re.M)
