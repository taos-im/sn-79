"""Two aggregators must never share one pending-row buffer.

The staging path was `checkpoint_path.parent / "pending_rows.msgpack"`, and checkpoint_path is
`.resolve()`d (gradient_server.py:328). The harness gave the exchange aggregator its own state dir but
linked best.pt back into the simulation dir, so the resolve collapsed the parent and BOTH aggregators
wrote one buffer. The exchange server logged
"Restored pending buffer: 128 books, 165419 rows" -- the simulation's rows -- exchange rows landed in
simulation books, and the exchange mechanism never accumulated a page of its own.

Scoping the filename by mode makes the isolation hold even when two servers resolve to the same
directory, which is the failure the directory split alone could not prevent.
"""
from pathlib import Path


def _staging_name(mode: str) -> str:
    return f"pending_rows.{mode}.msgpack" if mode else "pending_rows.msgpack"


def test_two_modes_never_collide_even_in_one_directory():
    shared = Path("/tmp/ckpt")
    sim = shared / _staging_name("simulation")
    xch = shared / _staging_name("exchange")
    assert sim != xch, "one directory must still yield two distinct buffers"


def test_mode_scoped_name_is_used_for_each_mode():
    assert _staging_name("simulation") == "pending_rows.simulation.msgpack"
    assert _staging_name("exchange") == "pending_rows.exchange.msgpack"


def test_unknown_mode_keeps_the_legacy_name():
    """No mode (older invocations) must not silently orphan an existing buffer."""
    assert _staging_name("") == "pending_rows.msgpack"


def test_server_uses_a_mode_scoped_staging_path():
    """The real attribute on the server, not just the helper above."""
    import inspect

    from GenTRX.src import gradient_server

    src = inspect.getsource(gradient_server)
    assert "_pending_staging_path" in src
    idx = src.index("_pending_staging_path")
    window = src[idx: idx + 400]
    assert "_mode" in window, "staging path is not scoped by mode; the buffers can still collide"
