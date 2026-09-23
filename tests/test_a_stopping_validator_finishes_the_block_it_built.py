# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A validator that is asked to stop must not discard a block it has already built.

The per-block state push is detached: the validator builds a payload, hands it to a task, and keeps
stepping. On shutdown `cleanup_event_loop` takes every pending task and cancels it. That is right for
the long-lived loops -- the query service, the reporting service, the metagraph sync all exist to be
stopped -- and wrong for a POST that is already built and on its way. Cancel it and nothing will ever
send that block again, so its trades never reach `trades`, while `agent_fills` -- written per fill,
down a separate path in the service -- already holds them. The acceptance verification layer reports
that asymmetry as "no agent fill without a matching tape entry", at zero tolerance, about fills that
were real.

The losses arrive in bursts: zero for long stretches, then a large count inside a single minute. A
rate shaped like that is whole pushes dying at a moment rather than anything per-fill, and the minutes
that carry it are the ones containing a validator restart.

The drain is asserted through its two observable properties -- pushes finish, everything else is still
cancelled -- rather than by driving a real shutdown, which would need a validator, a loop and a socket
to prove a decision that is entirely local.
"""

import asyncio

import taos.im.validator.cleanup as C


def test_the_two_modules_agree_on_the_task_name():
    # The name is the whole mechanism: cleanup finds the pushes by it rather than importing the
    # validator module, so a name that drifts on one side silently disables the drain and nothing
    # fails until fills go missing again.
    import taos.im.neurons.validator as V

    assert getattr(V, "PUSH_TASK_NAME", None) == C.PUSH_TASK_NAME


def test_the_drain_is_bounded():
    # One block of state is worth a few seconds of shutdown and no more. An unbounded wait turns a
    # wedged socket into a validator that never stops.
    assert 0 < C.PUSH_DRAIN_TIMEOUT_S <= 30


def test_an_in_flight_push_finishes_and_the_loops_are_still_cancelled():
    loop = asyncio.new_event_loop()
    try:
        sent = []

        async def _push():
            await asyncio.sleep(0.05)
            sent.append("block")

        async def _forever():
            while True:
                await asyncio.sleep(3600)

        async def _arm():
            p = asyncio.ensure_future(_push())
            p.set_name(C.PUSH_TASK_NAME)
            f = asyncio.ensure_future(_forever())
            f.set_name("query-service")
            await asyncio.sleep(0)
            return p, f

        push, forever = loop.run_until_complete(_arm())

        class _V:
            main_loop = loop

        C.cleanup_event_loop(_V())

        # The block was sent, not thrown away.
        assert sent == ["block"], "the in-flight push was discarded"
        assert push.done() and not push.cancelled()
        # And the long-lived loop was still cancelled, which is what shutdown is for.
        assert forever.cancelled() or forever.done()
    finally:
        if not loop.is_closed():
            loop.close()


def test_a_push_that_never_finishes_does_not_hold_shutdown_open(monkeypatch):
    # The bound has to bind. A push wedged on a socket must cost the drain timeout and then be
    # cancelled with the rest, not stop the process from exiting.
    monkeypatch.setattr(C, "PUSH_DRAIN_TIMEOUT_S", 0.2)
    loop = asyncio.new_event_loop()
    try:

        async def _wedged():
            await asyncio.sleep(3600)

        async def _arm():
            p = asyncio.ensure_future(_wedged())
            p.set_name(C.PUSH_TASK_NAME)
            await asyncio.sleep(0)
            return p

        push = loop.run_until_complete(_arm())

        class _V:
            main_loop = loop

        C.cleanup_event_loop(_V())
        assert push.cancelled() or push.done(), "a wedged push was left running"
    finally:
        if not loop.is_closed():
            loop.close()


def test_the_drain_works_against_a_loop_running_in_another_thread():
    """THE SHAPE THE VALIDATOR ACTUALLY HAS, which the assertions above do not exercise.

    main_loop runs in its own daemon thread and cleanup is called from a different one, so
    `loop.run_until_complete(...)` raises "this event loop is already running" before it waits for
    anything. The assertions above do not catch it: the drain logs "Could not drain in-flight state
    pushes" and cancels the block regardless, which is the behaviour it was written to prevent.

    So this drives a real loop on a real thread and asserts the push COMPLETED, rather than
    asserting the intent of the code that is supposed to make it complete.
    """
    import asyncio
    import threading

    from taos.im.validator.cleanup import PUSH_TASK_NAME, cleanup_event_loop

    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True, name="main")
    t.start()
    ready.wait(5)

    landed = []

    async def _slow_push():
        await asyncio.sleep(0.4)
        landed.append("sent")

    async def _spawn():
        task = asyncio.ensure_future(_slow_push())
        task.set_name(PUSH_TASK_NAME)
        return task

    asyncio.run_coroutine_threadsafe(_spawn(), loop).result(5)

    class _V:
        pass

    v = _V()
    v.main_loop = loop
    # cleanup_event_loop now stops and closes the loop itself, so this only waits for the thread.
    # It used to have to stop the loop by hand, which was the bug wearing a teardown as a disguise.
    cleanup_event_loop(v)
    t.join(timeout=10)

    assert landed == ["sent"], (
        "the in-flight push was not allowed to finish; its block's trades are gone and nothing resends them"
    )


def test_shutdown_actually_stops_and_closes_the_loop():
    """Cancelling is only half of it: the cancellation has to be awaited, and the loop closed.

    The same thread boundary defeats the block that follows the drain. `task.cancel()` is called
    from the wrong thread, the `run_until_complete(gather(...))` after it raises on the running
    loop, and the two statements that stop and close the loop never run at all -- every shutdown
    ends in "Error shutting down main event loop" with the loop still open and the cancelled tasks
    never given a chance to run their cleanup. It survives only because the loop's thread is a
    daemon and the process is exiting anyway, which is not the same as shutting down.
    """
    import asyncio
    import threading

    from taos.im.validator.cleanup import cleanup_event_loop

    loop = asyncio.new_event_loop()
    ready = threading.Event()
    cancelled = []

    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True, name="main")
    t.start()
    ready.wait(5)

    async def _long_lived():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append("cleaned up")
            raise

    asyncio.run_coroutine_threadsafe(_spawn_named(_long_lived()), loop).result(5)

    class _V:
        pass

    v = _V()
    v.main_loop = loop
    cleanup_event_loop(v)
    t.join(timeout=10)

    assert cancelled == ["cleaned up"], "the cancelled task never ran its cleanup"
    assert not loop.is_running(), "the loop was left running"
    assert loop.is_closed(), "the loop was left open"


async def _spawn_named(coro):
    return asyncio.ensure_future(coro)
