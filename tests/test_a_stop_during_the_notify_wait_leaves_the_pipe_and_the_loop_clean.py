# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The wait on the query process's notify pipe ends cleanly however it ends.

A stop cancels the main loop's wait for the query process's notification. The reader on the pipe comes off the loop
with it, so a later byte sets no result on the cancelled future and stays in the pipe for the next wait. The same
holds for a wait that times out.
"""
import asyncio
import os

import pytest

from taos.im.validator.forward import read_notify_byte


@pytest.fixture
def pipe():
    r, w = os.pipe()
    os.set_blocking(r, False)
    yield r, w
    os.close(r)
    os.close(w)


def _run(coro_fn):
    loop = asyncio.new_event_loop()
    errors = []
    loop.set_exception_handler(lambda _loop, ctx: errors.append(ctx))
    try:
        return loop.run_until_complete(coro_fn(loop)), errors
    finally:
        loop.close()


def test_the_byte_written_is_the_byte_read(pipe):
    r, w = pipe

    async def go(loop):
        loop.call_later(0.05, os.write, w, b"1")
        return await read_notify_byte(r, 1.0)
    got, errors = _run(go)
    assert got == b"1" and not errors


def test_a_timed_out_wait_takes_its_reader_off_the_loop(pipe):
    r, w = pipe

    async def go(loop):
        with pytest.raises(asyncio.TimeoutError):
            await read_notify_byte(r, 0.05)
        os.write(w, b"1")
        await asyncio.sleep(0.05)
        return loop.remove_reader(r)
    still_registered, errors = _run(go)
    assert still_registered is False and not errors
    assert os.read(r, 1) == b"1"


def test_a_cancelled_wait_raises_nothing_later_and_leaves_the_byte_for_the_next(pipe):
    r, w = pipe

    async def go(loop):
        task = loop.create_task(read_notify_byte(r, 5.0))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        os.write(w, b"1")
        await asyncio.sleep(0.05)
        return loop.remove_reader(r)
    still_registered, errors = _run(go)
    assert still_registered is False
    assert not errors, errors
    assert os.read(r, 1) == b"1"
