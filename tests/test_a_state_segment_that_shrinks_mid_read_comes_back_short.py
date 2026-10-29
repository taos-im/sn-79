"""A /state segment the engine shrinks between the size check and the read comes back short, not as SIGBUS.

A validator died with "Fatal Python error: Bus error" in its state reader
(engines/simulation.py receive -> _recv_bytes, mm.read). The reader waited for fstat to reach the announced size, then
mapped that many bytes and read them; the engine had shrunk the segment in between, and reading a shared mapping past
the end of its object is SIGBUS, which no handler in the process can survive. pread past the end returns a short read,
which the unpack retry already handles by reading the segment again.
"""

import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import msgpack
import pytest

posix_ipc = pytest.importorskip("posix_ipc")

from taos.im.validator.engines.simulation import _read_segment

SRC = Path(__file__).resolve().parents[1] / "taos" / "im" / "validator" / "engines" / "simulation.py"


@pytest.fixture
def segment():
    name = f"/test_state_{uuid.uuid4().hex[:8]}"
    shm = posix_ipc.SharedMemory(name, flags=posix_ipc.O_CREX, size=0)
    try:
        yield shm
    finally:
        shm.close_fd()
        shm.unlink()


def _write(shm, data):
    os.ftruncate(shm.fd, len(data))
    os.pwrite(shm.fd, data, 0)


def test_the_whole_segment_is_read(segment):
    data = msgpack.packb({"timestamp": 7, "books": {str(i): [i] * 50 for i in range(200)}})
    _write(segment, data)
    assert _read_segment(segment.fd, len(data)) == data


def test_a_segment_shrunk_after_the_size_check_comes_back_short(segment):
    data = msgpack.packb({"timestamp": 7, "books": {str(i): [i] * 50 for i in range(200)}})
    _write(segment, data)
    announced = len(data)
    os.ftruncate(segment.fd, announced // 2)
    got = _read_segment(segment.fd, announced)
    assert len(got) == announced // 2 and got == data[: announced // 2]
    with pytest.raises((ValueError, msgpack.UnpackValueError)):
        msgpack.unpackb(got, raw=False)  # what the unpack retry catches before it reads again


def test_the_mapping_it_replaces_dies_on_the_same_shrink():
    """The hazard itself, in a child process: a shared mapping read past the object's end is SIGBUS."""
    code = textwrap.dedent("""
        import mmap, os, uuid, posix_ipc
        shm = posix_ipc.SharedMemory("/test_bus_" + uuid.uuid4().hex[:8], flags=posix_ipc.O_CREX, size=1 << 20)
        try:
            m = mmap.mmap(shm.fd, 1 << 20, mmap.MAP_SHARED, mmap.PROT_READ)
            os.ftruncate(shm.fd, 1 << 10)
            m.read(1 << 20)
        finally:
            shm.unlink()
    """)
    rc = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30).returncode
    assert rc == -7, f"expected SIGBUS (-7), got {rc}"


def test_both_readers_read_the_segment_with_pread():
    src = SRC.read_text()
    for fn in ("    async def _recv_bytes(self", "    async def _recv_bytes_shm_only(self"):
        start = src.index(fn)
        body = src[start:src.index("\n    def ", start + 10) if "\n    def " in src[start + 10:] else None]
        body = body[: body.find("\n    async def ", 10)] if body.find("\n    async def ", 10) > 0 else body
        assert "_read_segment(" in body, f"{fn.strip()} reads with _read_segment"
        assert "mmap.mmap(" not in body, f"{fn.strip()} still maps the segment"
