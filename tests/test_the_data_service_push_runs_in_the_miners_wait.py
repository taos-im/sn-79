# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A simulation state's data-service push starts once its query is with the query process, so its payload build and
its encode run while the miners answer.

The push reads the state and the validator's own maps, nothing from the miners' answers, so it starts in the miners'
wait. A state whose query is not sent (an observing validator, a failed hand-off) is pushed after the forward, once."""
import inspect

from taos.im.validator import forward as fwd


def test_the_forward_announces_the_query_once_it_is_with_the_query_process():
    src = inspect.getsource(fwd.forward)
    sent = src.index("self.request_queue.send(b'query')")
    hook = src.index("on_query_sent", sent)
    wait = src.index("read_notify_byte(", sent)
    assert sent < hook < wait


def test_the_push_starts_in_the_wait_and_after_the_forward_only_when_it_did_not():
    from taos.im.neurons import validator
    src = inspect.getsource(validator)
    hook = inspect.getsource(validator.Validator.on_query_sent)
    assert "self._sim_push_ts = state.timestamp" in hook and "_spawn_push(self._push_sim_state_bg(state," in hook
    assert "getattr(self, '_sim_push_ts', None) != state.timestamp" in src
    assert src.count("_spawn_push(self._push_sim_state_bg(state,") == 2
