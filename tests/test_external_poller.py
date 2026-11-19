# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Orchestration tests for MarketEngine's background external-order poller.

The poller decouples the data-service fetch+verify from receive(): a background
task fills _external_buffer via _fetch_external_orders(), and receive() drains it
with no await. These tests cover only that NEW mechanism (buffer fill, atomic
drain, idempotent start, cancel-on-stop, survives a fetch error) with a stub
_fetch_external_orders, so they are deterministic and offline. The wire fetch and
downstream injection are unchanged and validated on the live validator.
"""
import asyncio

from taos.im.validator.engines import MarketEngine


class _StubEngine(MarketEngine):
    """Minimal concrete MarketEngine: abstract methods stubbed, _fetch scripted."""

    def __init__(self, batches):
        # batches: lists returned by successive _fetch_external_orders() calls
        self._external_instructions = []
        self._external_buffer = []
        self._external_poll_task = None
        self._batches = list(batches)
        self.fetch_calls = 0

    async def _fetch_external_orders(self):
        self.fetch_calls += 1
        return self._batches.pop(0) if self._batches else []

    # ── abstract stubs (unused by the poller) ──
    async def receive(self):
        return (None, None, 0.0)

    async def execute(self, state, miner_responses):
        return []

    def respond(self, raw_message, response):
        pass

    @property
    def book_ids(self):
        return []

    @property
    def mode(self):
        return "test"

    @property
    def effective_max_uids(self):
        return 0

    def initialize_structures(self):
        pass

    def build_simulation_state(self):
        return {}

    def restore_simulation_state(self, data):
        pass

    def handle_deregistration(self, uid):
        pass

    def apply_resets(self, pending):
        pass


async def _drain_after_stop(eng):
    task = eng._external_poll_task
    eng._stop_external_poller()
    assert eng._external_poll_task is None
    if task is not None:
        try:
            await task
        except asyncio.CancelledError:
            pass
    return eng._drain_external_orders()


def test_drain_returns_and_clears_buffer():
    eng = _StubEngine([])
    eng._external_buffer = [{"a": 1}, {"a": 2}]
    drained = eng._drain_external_orders()
    # each order leaves the drain at the delay of its receipt: 0 without the service's stamp
    assert drained == [{"a": 1, "delay": 0}, {"a": 2, "delay": 0}]
    assert eng._external_buffer == []
    # a second drain yields nothing — exactly-once hand-off to the round
    assert eng._drain_external_orders() == []


def test_poll_loop_fills_buffer_and_ensure_is_idempotent():
    async def go():
        eng = _StubEngine([[{"o": 1}], [{"o": 2}, {"o": 3}]])
        eng._EXTERNAL_POLL_INTERVAL = 0.01
        eng._ensure_external_poller()
        assert eng._external_poll_task is not None
        # second ensure must NOT spawn a second task (strong ref preserved)
        first = eng._external_poll_task
        eng._ensure_external_poller()
        assert eng._external_poll_task is first
        await asyncio.sleep(0.05)
        drained = await _drain_after_stop(eng)
        # both batches accumulated across polls; one drain hands them all off
        assert {"o": 1, "delay": 0} in drained and {"o": 2, "delay": 0} in drained and {"o": 3, "delay": 0} in drained
        assert eng.fetch_calls >= 2

    asyncio.run(go())


def test_poll_loop_survives_fetch_exception():
    async def go():
        eng = _StubEngine([])
        eng._EXTERNAL_POLL_INTERVAL = 0.01
        calls = {"n": 0}

        async def boom():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("data service down")
            return [{"ok": calls["n"]}]

        eng._fetch_external_orders = boom
        eng._ensure_external_poller()
        await asyncio.sleep(0.05)
        drained = await _drain_after_stop(eng)
        # first poll raised; the loop kept going and later polls populated the buffer
        assert any(d.get("ok") for d in drained)
        assert calls["n"] >= 2

    asyncio.run(go())
