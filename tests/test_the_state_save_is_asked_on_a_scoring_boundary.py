# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The periodic state save is asked for on a scoring boundary. The scoring service writes the save from its replica
once it has applied the last state teed, and it does nothing else while it writes. Main tees a boundary's state and
its score inputs on receipt, before it asks for the save, so a save asked on the boundary reaches the service after
them: the service scores the boundary, then writes, and the save's snapshot point is the boundary the light fields
describe. Asked on the state before a boundary, the save would make the boundary's score wait the whole write."""
import time
from types import SimpleNamespace

from taos.im.validator import persistence

S = 10 ** 9


def _fired(monkeypatch, step_s, seconds=600, busy=()):
    monkeypatch.setattr(persistence, "_SAVE_EVERY_INTERVALS", 12)
    monkeypatch.setattr(persistence.asyncio, "get_event_loop", lambda: None)
    calls = []
    loop = SimpleNamespace(call_soon_threadsafe=lambda f: calls.append(1), create_task=lambda c: None)
    v = SimpleNamespace(config=SimpleNamespace(scoring=SimpleNamespace(interval=5 * S)), querying=False,
                        main_loop=loop, _cleanup_done=False)
    fired = []
    for t in range(0, seconds * S, step_s * S):
        v.last_state, v.step, v.shared_state_saving = SimpleNamespace(timestamp=t), t // S, (t // S) in busy
        before = len(calls)
        persistence.schedule_save(v)
        if len(calls) > before:
            fired.append(t // S)
    return fired


def test_a_one_second_state_save_is_asked_on_the_minutes_scoring_boundary(monkeypatch):
    fired = _fired(monkeypatch, 1)
    assert fired == [60 * k for k in range(1, 10)]
    assert all(t % 5 == 0 for t in fired), "every save is asked on a scoring boundary"


def test_a_save_asked_on_the_boundary_is_written_after_the_boundary_is_scored(tmp_path):
    from test_scoring_shadow import _S, _fresh_main, _light_fields, _raw_state, _state

    from taos.im.validator.scoring_shadow import ScoringShadow
    from taos.im.validator.trade import update_trade_volumes

    main = _fresh_main()
    update_trade_volumes(main, _state(1 * _S, [(1, 2, 0, 3.0, 100.0, 0)]))
    main._shadow_applied_ts = 1 * _S
    shadow = ScoringShadow(parity_ns=5 * _S)
    shadow.start()
    try:
        shadow.send_init(main).result(timeout=60)
        for ts, trades in [(2 * _S, [(2, 1, 0, 1.5, 101.0, 1)]), (3 * _S, []), (4 * _S, [])]:
            update_trade_volumes(main, _state(ts, trades))
            shadow.tee(_raw_state(ts, trades), ts)
        boundary = (5 * _S, [(1, 3, 1, 2.0, 56.0, 1)])
        update_trade_volumes(main, _state(*boundary))
        # as the listener does on receipt: the boundary's state, then its score inputs
        shadow.tee(_raw_state(*boundary), 5 * _S)
        shadow.tee_score_inputs(5 * _S, 5 * _S, [], {}, {})
        light = dict(_light_fields(), simulation_timestamp=5 * _S)
        req = shadow.request_save(str(tmp_path / "validator_state.mp"), light)
        assert req is not None
        expect_ts, fut = req
        assert expect_ts == light["simulation_timestamp"] == 5 * _S, "the snapshot is the state the light fields describe"
        scored_first = []
        fut.add_done_callback(lambda f: scored_first.append(5 * _S in shadow._score_results))
        assert fut.result(timeout=60) is not None, "child save failed"
        deadline = time.monotonic() + 5
        while not scored_first and time.monotonic() < deadline:
            time.sleep(0.01)
        assert scored_first == [True], "the boundary's score reached main before the save was written"
    finally:
        shadow.stop()
