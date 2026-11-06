# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The reporting child decodes and publishes each cycle with Python's cyclic garbage collector paused.

A cycle allocates millions of tuples and dicts, and the collector, triggered by allocation counts, would run many young
and several full collections a cycle, each full one walking every tracked object. Reference counting still frees what
a cycle drops (the snapshots are plain dicts of tuples and floats, with no reference cycles), and the collector resumes
after the cycle, also when the cycle raises."""
import gc
import inspect

import pytest

from taos.im.validator import report


def test_the_collector_is_paused_inside_and_resumed_after():
    assert gc.isenabled()
    with report.cyclic_gc_paused():
        assert not gc.isenabled()
    assert gc.isenabled()


def test_the_collector_resumes_when_the_cycle_raises():
    with pytest.raises(RuntimeError):
        with report.cyclic_gc_paused():
            raise RuntimeError("cycle failed")
    assert gc.isenabled()


def test_a_collector_already_paused_by_the_caller_stays_paused():
    gc.disable()
    try:
        with report.cyclic_gc_paused():
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_the_service_decodes_and_publishes_inside_the_pause():
    src = inspect.getsource(report.ReportingService.run)
    body = src[src.index("with cyclic_gc_paused():"):]
    assert body.index("msgpack.unpackb(") < body.index("await self.publish_metrics(data)")
