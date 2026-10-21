# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator names the engine's four IPC objects with TAOSIM_IPC_SUFFIX, as the engine does.

The engine and the proxy append the suffix to taosim-req, taosim-res, state and responses, so several
engines can share one host. A validator that ignored it would sit on the default names whatever the
engine beside it used, and lose its engine to any unsuffixed one started on the same host. With the
suffix unset or empty the names are the ones every existing deployment uses.

These tests never open a real queue or segment: a fault here must not reach an engine running on the
same host.
"""
import ast
import sys
import types
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[1] / "taos" / "im" / "validator" / "engines" / "simulation.py"
BASES = {"taosim-req", "taosim-res", "state", "responses"}


@pytest.fixture
def sim():
    import taos.im.validator.engines.simulation as m
    return m


@pytest.mark.parametrize("base", sorted(BASES))
def test_unset_suffix_keeps_the_default_names(sim, monkeypatch, base):
    monkeypatch.delenv("TAOSIM_IPC_SUFFIX", raising=False)
    assert sim._ipc_name(base) == f"/{base}"
    monkeypatch.setenv("TAOSIM_IPC_SUFFIX", "")
    assert sim._ipc_name(base) == f"/{base}"


@pytest.mark.parametrize("base", sorted(BASES))
def test_the_suffix_is_appended(sim, monkeypatch, base):
    monkeypatch.setenv("TAOSIM_IPC_SUFFIX", "_arm7")
    assert sim._ipc_name(base) == f"/{base}_arm7"


def _ipc_calls():
    tree = ast.parse(SIM.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("MessageQueue", "SharedMemory"):
                yield node


def test_every_engine_queue_and_segment_goes_through_the_helper():
    seen, bare = set(), []
    for call in _ipc_calls():
        arg = call.args[0] if call.args else None
        if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "_ipc_name"
                and isinstance(arg.args[0], ast.Constant)):
            seen.add(arg.args[0].value)
        else:
            bare.append(call.lineno)
    assert not bare, f"IPC objects named without _ipc_name at simulation.py lines {bare}"
    assert seen == BASES, f"helper used for {sorted(seen)}, expected {sorted(BASES)}"


class _FakeIpc(types.ModuleType):
    O_CREAT = 0o100

    def __init__(self):
        super().__init__("posix_ipc")
        self.opened = []
        outer = self

        class MessageQueue:
            def __init__(self, name, flags=0, max_messages=0, max_message_size=0):
                outer.opened.append(("mq", name))

            def send(self, data):
                pass

            def close(self):
                pass

        class SharedMemory:
            def __init__(self, name, flags=0, size=0):
                outer.opened.append(("shm", name))
                self.fd = -1

            def close_fd(self):
                pass

        self.MessageQueue = MessageQueue
        self.SharedMemory = SharedMemory


class _FakeMap:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, data):
        pass


def test_the_response_is_written_to_the_suffixed_objects(sim, monkeypatch):
    fake = _FakeIpc()
    fake_mmap = types.ModuleType("mmap")
    fake_mmap.mmap = _FakeMap
    fake_mmap.MAP_SHARED = fake_mmap.PROT_WRITE = fake_mmap.PROT_READ = 0
    monkeypatch.setitem(sys.modules, "posix_ipc", fake)
    monkeypatch.setitem(sys.modules, "mmap", fake_mmap)
    monkeypatch.setenv("TAOSIM_IPC_SUFFIX", "_arm7")
    sim.SimulationEngine._send_bytes(object(), b"\x00" * 16)
    assert fake.opened == [("mq", "/taosim-res_arm7"), ("shm", "/responses_arm7")]
