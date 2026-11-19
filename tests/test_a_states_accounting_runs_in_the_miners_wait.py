# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A simulation state's reward starts once its query is with the query process: its accounting pass runs in the
miners' wait, and its boundary scoring waits for the forward's end, which it reads.

Spawned after the forward, the reward's accounting pass (the checkpoint snapshot, update_trade_volumes, the resets,
the parity digest; 'Agent Scores Data Updated'), synchronous on the main loop, can run on into the next round and
hold up that round's pre-request work. It reads the state and the validator's accumulators, nothing of the miners'
answers, so it starts in the wait (on_query_sent). The boundary scoring reads this round's answers (miner_presence,
updated in forward), so a boundary state's reward holds the lock and waits for the forward's end, signalled after the
response's build and before it goes out, where the reward is spawned with the lever off, and by the listener whatever
handle_state's exit was. A wait that is never signalled times out rather than hold the lock for good. The lock is
first come, first served, so the rewards keep their order and every earlier reward is done before a state's
checkpoint snapshot, as before. The exchange engine settles after the forward, so its reward keeps its place."""
import asyncio
import inspect
import types

import pytest

validator = pytest.importorskip("taos.im.neurons.validator")


def _src(name):
    return inspect.getsource(getattr(validator.Validator, name))


def test_the_reward_starts_in_the_wait_with_a_forward_done_signal():
    hook = _src("on_query_sent")
    assert hook.index("return") < hook.index("state._forward_done = asyncio.Event()"), "simulation only"
    assert hook.index("self._pre_request_work(state)") < hook.index("state._forward_done = asyncio.Event()") \
        < hook.index("self.reward(state)"), "the resets are stashed before the accounting reads them"


def test_only_a_boundary_states_scoring_waits_for_the_forward():
    src = _src("_reward")
    accounted = src.index("self._update_trade_volumes(state)")
    non_boundary = src.index("if timestamp % self.config.scoring.interval != 0:")
    wait = src.index("await self._forward_done_wait(state)")
    scoring = src.index('bt.logging.info("Starting reward calculation...")')
    assert accounted < non_boundary < wait < scoring


def test_handle_state_signals_the_forwards_end_or_spawns_the_reward_as_before():
    src = _src("handle_state")
    at = src.index("self.maintain()")
    tail = src[at:at + 600]
    assert "_forward_done.set()" in tail and "self.reward(state)" in tail
    assert tail.index("_forward_done.set()") < tail.index("self.save_state()")


def test_the_listener_signals_the_forwards_end_whatever_handle_states_exit():
    src = _src("_listen")
    assert "normalized_state = None" in src
    fin = src.index("self.engine.respond(raw_message, response)")
    assert "_forward_done" in src[fin:fin + 600]


def _stub(grace=None):
    return types.SimpleNamespace(config=types.SimpleNamespace(neuron=types.SimpleNamespace(global_query_timeout=0.05)),
                                 FORWARD_DONE_GRACE_S=validator.Validator.FORWARD_DONE_GRACE_S if grace is None else grace)


def test_the_wait_returns_once_signalled_and_at_once_without_a_signal():
    stub = _stub()

    async def run():
        plain = types.SimpleNamespace(timestamp=5)
        await validator.Validator._forward_done_wait(stub, plain)
        state = types.SimpleNamespace(timestamp=5, _forward_done=asyncio.Event())
        waiter = asyncio.ensure_future(validator.Validator._forward_done_wait(stub, state))
        await asyncio.sleep(0.01)
        assert not waiter.done()
        state._forward_done.set()
        await asyncio.wait_for(waiter, 1.0)
    asyncio.run(run())


def test_a_wait_never_signalled_times_out_instead_of_holding_the_lock():
    stub = _stub(grace=0.05)

    async def run():
        state = types.SimpleNamespace(timestamp=5, _forward_done=asyncio.Event())
        await asyncio.wait_for(validator.Validator._forward_done_wait(stub, state), 2.0)
    asyncio.run(run())


class _ScoringReached(Exception):
    pass


def _reward_stub(events):
    import threading
    stub = types.SimpleNamespace(
        _rewarding_lock=threading.Lock(), _pending_reward_tasks=0, step=7, reward_gate_waits=[],
        shared_state_rewarding=False, _scoring_shadow=None, engine=types.SimpleNamespace(apply_resets=lambda r: None),
        config=types.SimpleNamespace(scoring=types.SimpleNamespace(interval=5_000_000_000),
                                     neuron=types.SimpleNamespace(global_query_timeout=1.0)),
        FORWARD_DONE_GRACE_S=5.0, pagerduty_alert=lambda *a, **k: events.append("alert"))
    stub._update_trade_volumes = lambda state: events.append(("accounted", state.timestamp))
    stub._forward_done_wait = types.MethodType(validator.Validator._forward_done_wait, stub)
    return stub


def test_the_accounting_runs_at_once_and_a_boundary_scoring_waits_for_the_forward(monkeypatch):
    """The real _reward on a stub: a boundary state's accounting runs as soon as the task runs, its scoring only once
    the forward's end is signalled, holding the reward lock in between; a state off the boundary returns after its
    accounting without waiting."""
    events = []
    real_info = validator.bt.logging.info

    def info(msg, *a, **k):
        if msg == "Starting reward calculation...":
            events.append("scoring")
            raise _ScoringReached()
        return real_info(msg, *a, **k)
    monkeypatch.setattr(validator.bt.logging, "info", info)
    stub = _reward_stub(events)

    async def run():
        stub._reward_lock = asyncio.Lock()
        boundary = types.SimpleNamespace(timestamp=10_000_000_000, _forward_done=asyncio.Event())
        task = asyncio.ensure_future(validator.Validator._reward(stub, boundary))
        await asyncio.sleep(0.02)
        assert events == [("accounted", boundary.timestamp)] and stub._reward_lock.locked() and not task.done()
        boundary._forward_done.set()
        await asyncio.wait_for(task, 1.0)
        assert events[1] == "scoring", events
        events.clear()
        off = types.SimpleNamespace(timestamp=11_000_000_000, _forward_done=asyncio.Event())
        await asyncio.wait_for(validator.Validator._reward(stub, off), 1.0)
        assert events == [("accounted", off.timestamp)] and not stub._reward_lock.locked()
    asyncio.run(run())
