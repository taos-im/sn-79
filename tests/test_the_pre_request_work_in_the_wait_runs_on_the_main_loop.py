# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The pre-request work that prereq_in_wait moves into the miners' wait writes the accounting's structures (the reset
stash, the placement counts, the order-to-trade histories) that the reward prunes and reads on the main loop. Run on
the listener's thread, which announces the query, it touched those dicts while the reward iterated them: a
RuntimeError in the prune, and a count written into a dict the prune had just replaced. So on_query_sent hands the
work to the main loop, where it still runs inside the wait (the loop is idle then) but serialised with the reward;
reward_in_wait, when on, starts the accounting from that same callback once the work is done, and a work that fails
leaves the reward to its place after the forward.
"""
import inspect
import types

from taos.im.neurons import validator
from taos.im.validator import levers


def _stub(on=("prereq_in_wait",)):
    calls, queued = [], []
    stub = types.SimpleNamespace(engine=types.SimpleNamespace(mode="simulation"), miner_stats={}, calls=calls,
                                 queued=queued, main_loop=types.SimpleNamespace(call_soon_threadsafe=lambda fn, *a: queued.append((fn, a))))
    stub.process_resets = lambda state: calls.append(("resets", state.timestamp))
    stub._gentrx = types.SimpleNamespace(push_state=lambda state: calls.append(("gentrx", state.timestamp)))
    stub._pre_request_work = types.MethodType(validator.Validator._pre_request_work, stub)
    stub._sim_push_url = lambda: None
    stub.rewarded = []
    stub.reward = lambda state: stub.rewarded.append(state)
    return stub


def _state(ts=5_000_000_000):
    return types.SimpleNamespace(timestamp=ts, notices={3: [{"y": "RDPOL", "b": 0}]}, books={0: {"b": [], "a": []}},
                                 accounts={3: {0: {"bb": 1.0}}})


def test_the_work_is_handed_to_the_main_loop_not_run_on_the_callers_thread(monkeypatch):
    monkeypatch.setattr(levers, "enabled", lambda name: name == "prereq_in_wait")
    stub, state = _stub(), _state()
    validator.Validator.on_query_sent(stub, state)
    assert stub.calls == [] and not getattr(state, "_pre_request_ok", False), "nothing ran on the caller's thread"
    assert len(stub.queued) == 1, "one callback for the loop"
    fn, args = stub.queued[0]
    fn(*args)
    assert ("resets", state.timestamp) in stub.calls and state._pre_request_ok is True
    assert stub.rewarded == [], "reward_in_wait off: the accounting keeps its place after the forward"


def test_reward_in_wait_starts_from_the_same_callback_after_the_work(monkeypatch):
    monkeypatch.setattr(levers, "enabled", lambda name: name in ("prereq_in_wait", "reward_in_wait"))
    stub, state = _stub(), _state()
    validator.Validator.on_query_sent(stub, state)
    assert stub.rewarded == [], "not before the work"
    fn, args = stub.queued[0]
    fn(*args)
    assert stub.rewarded == [state] and getattr(state, "_forward_done", None) is not None
    assert stub.calls.index(("resets", state.timestamp)) >= 0


def test_a_work_that_fails_on_the_loop_is_logged_and_starts_no_reward(monkeypatch):
    monkeypatch.setattr(levers, "enabled", lambda name: name in ("prereq_in_wait", "reward_in_wait"))
    stub, state = _stub(), _state()

    def boom(state):
        raise RuntimeError("reset scan failed")
    stub.process_resets = boom
    validator.Validator.on_query_sent(stub, state)
    fn, args = stub.queued[0]
    fn(*args)
    assert stub.rewarded == [] and not getattr(state, "_pre_request_ok", False)


def test_the_lever_off_queues_nothing(monkeypatch):
    monkeypatch.setattr(levers, "enabled", lambda name: False)
    stub = _stub()
    validator.Validator.on_query_sent(stub, _state())
    assert stub.queued == [] and stub.calls == []


def test_no_accounting_write_remains_on_the_listeners_path():
    hook = inspect.getsource(validator.Validator.on_query_sent)
    body = hook[hook.index("levers.enabled('prereq_in_wait')"):]
    assert "self.main_loop.call_soon_threadsafe(_prereq_on_loop)" in body
    direct = body.index("self._pre_request_work(state)")
    assert body.index("def _prereq_on_loop") < direct, "the only direct call is inside the loop's callback"
    assert body.count("self._pre_request_work(state)") == 1
