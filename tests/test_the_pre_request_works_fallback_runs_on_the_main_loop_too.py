# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""handle_state runs on the listener's loop and, after the forward, runs the pre-request work for a state whose
query was not sent (observing, a failed hand-over), as the exchange engine runs it before its forward. Those two
calls ran the work on the listener's thread, where it could still meet the reward's prune on the main loop whenever
the loop had not yet run on_query_sent's callback. Now every path hands the work to the main loop and awaits it:
_pre_request_on_main_loop schedules the work there and returns once it ran, the done flag is written on the main
loop alone, so the work runs once per state and on one thread, and a work that raises raises to the caller as before.
"""
import asyncio
import inspect
import threading
import types

from taos.im.neurons import validator


def _stub():
    calls, queued = [], []
    stub = types.SimpleNamespace(engine=types.SimpleNamespace(mode="simulation"), miner_stats={}, calls=calls,
                                 queued=queued)
    stub.main_loop = types.SimpleNamespace(call_soon_threadsafe=lambda fn, *a: queued.append((fn, a)))
    stub.process_resets = lambda state: calls.append(("resets", state.timestamp))
    stub._gentrx = types.SimpleNamespace(push_state=lambda state: calls.append(("gentrx", state.timestamp)))
    stub._pre_request_work = types.MethodType(validator.Validator._pre_request_work, stub)
    stub._pre_request_on_main_loop = types.MethodType(validator.Validator._pre_request_on_main_loop, stub)
    return stub


def _state(ts=5_000_000_000):
    return types.SimpleNamespace(timestamp=ts, notices={3: [{"y": "RDPOL", "b": 0}]}, books={0: {"b": [], "a": []}},
                                 accounts={3: {0: {"bb": 1.0}}})


def _drain_later(stub, delay=0.05):
    """Another thread plays the main loop: it runs the queued callbacks after a moment."""
    def run():
        import time
        time.sleep(delay)
        while stub.queued:
            fn, args = stub.queued.pop(0)
            fn(*args)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_the_await_returns_only_once_the_main_loop_ran_the_work():
    stub, state = _stub(), _state()

    async def go():
        t = _drain_later(stub)
        assert stub.calls == [], "nothing before the loop runs it"
        await stub._pre_request_on_main_loop(state)
        t.join(1.0)
        return list(stub.calls)
    calls = asyncio.run(go())
    assert ("resets", state.timestamp) in calls and state._pre_request_done is True


def test_a_state_already_done_is_not_handed_over_again():
    stub, state = _stub(), _state()
    state._pre_request_done = True
    asyncio.run(stub._pre_request_on_main_loop(state))
    assert stub.queued == [] and stub.calls == []


def test_a_work_that_raises_raises_to_the_caller():
    stub, state = _stub(), _state()

    def boom(state):
        raise RuntimeError("reset scan failed")
    stub.process_resets = boom

    async def go():
        _drain_later(stub)
        await stub._pre_request_on_main_loop(state)
    try:
        asyncio.run(go())
    except RuntimeError as e:
        assert "reset scan failed" in str(e)
    else:
        raise AssertionError("the failure reached nobody")


def test_handle_state_hands_both_calls_to_the_main_loop():
    src = inspect.getsource(validator.Validator.handle_state)
    assert src.count("await self._pre_request_on_main_loop(state)") == 2, "the exchange's call and the fallback"
    assert "self._pre_request_work(state)" not in src, "no direct call on the listener's thread remains"
    hook = inspect.getsource(validator.Validator.on_query_sent)
    assert hook.count("self._pre_request_work(state)") == 1, "the wait's callback alone calls the work directly"
