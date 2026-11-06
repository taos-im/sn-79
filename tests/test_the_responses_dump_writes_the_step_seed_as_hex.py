# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The engine's RESPONSES_ERROR_REPORT dumps the last batch sent, including a batch that carries the step seed.

The batch that crosses a seed boundary carries "seed": [round, randomness] with the randomness as 32 raw bytes, the
form the engine reads. JSON has no bytes, so the dump writes them as hex; a dump that raised there would skip the
error counter, the alert and the reply that tells the engine whether to continue."""
import inspect
import json

import pytest


def _validator():
    return pytest.importorskip("taos.im.neurons.validator")


def test_a_batch_with_the_seed_is_dumped_with_its_randomness_as_hex(tmp_path):
    validator = _validator()
    path = tmp_path / "x.responses.json"
    randomness = bytes(range(32))
    validator._dump_responses(str(path), {"responses": [{"agentId": 3}], "seed": [123, randomness]})
    dumped = json.loads(path.read_text())
    assert dumped["seed"] == [123, randomness.hex()]
    assert dumped["responses"] == [{"agentId": 3}]


def test_no_batch_sent_yet_dumps_an_empty_batch(tmp_path):
    validator = _validator()
    path = tmp_path / "none.responses.json"
    validator._dump_responses(str(path), None)
    assert json.loads(path.read_text()) == {"responses": []}


def test_the_error_report_dumps_through_the_helper():
    validator = _validator()
    src = inspect.getsource(validator)
    block = src[src.index("elif message['type'] == 'RESPONSES_ERROR_REPORT':"):]
    block = block[:block.index("self.msgpack_error_counter +=")]
    assert "_dump_responses(dump_file, self.last_response)" in block, block
