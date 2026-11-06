# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Python's cyclic garbage collector paused around one allocation-heavy step.

The collector runs on allocation counts, and a full collection walks every tracked object while it holds the
interpreter, so a step that allocates millions of objects (decoding a large msgpack message, publishing a report
cycle) triggers collections that walk the process's whole heap several times. Reference counting still frees what the
step drops; the collector resumes after it, also when it raises.
"""
import contextlib
import gc


@contextlib.contextmanager
def cyclic_gc_paused():
    """Pause the collector for the block; a caller that had already paused it keeps it paused."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
