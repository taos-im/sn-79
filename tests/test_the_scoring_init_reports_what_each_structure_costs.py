# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The first report after a validator start sends the scoring child's INIT snapshot under the reward lock, which at 256
books under load can run past the stop save's budget. Before the send is moved off the lock or its structures packed
differently, the INIT says what each structure costs: send_init times every structure's plainify, pickle and send and
its bytes, keeps them on the handle (last_init_timings) and logs the total with the slowest structures
(init_timings_line), so the next bed or testnet run names the structures that make it long."""
from taos.im.validator import scoring_shadow as ss


def _t(p, k, s, b):
    return {"plainify_s": p, "pickle_s": k, "send_s": s, "bytes": b}


def test_the_line_totals_the_structures_and_names_the_slowest_with_their_slowest_step():
    timings = {"a": _t(0.1, 0.2, 0.05, 2_000_000), "b": _t(1.5, 0.3, 0.2, 90_000_000), "c": _t(0.0, 0.0, 0.01, 1000),
               "d": _t(0.2, 0.9, 0.1, 30_000_000)}
    line = ss.init_timings_line(timings, top=2)
    assert line.startswith("4 structures 3.56s 122.0 MB; slowest: ")
    assert "b 2.00s (90.0 MB, plainify), d 1.20s (30.0 MB, pickle)" in line
    assert "a " not in line.split("slowest:")[1]
    assert ss.init_timings_line({}) == "no structures"


def test_send_init_logs_the_line_and_keeps_every_structures_timing_on_the_handle():
    import inspect
    src = inspect.getsource(ss.ScoringShadow.send_init)
    assert "self.last_init_timings = timings" in src and "init_timings_line(timings)" in src
    for key in ('"plainify_s"', '"pickle_s"', '"send_s"', '"bytes"'):
        assert key in src


def test_a_real_init_times_every_structure():
    from test_scoring_shadow import _S, _fresh_main, _state
    from taos.im.validator.trade import update_trade_volumes

    main = _fresh_main()
    update_trade_volumes(main, _state(1 * _S, [(1, 2, 0, 3.0, 100.0, 0)]))
    main._shadow_applied_ts = 1 * _S
    shadow = ss.ScoringShadow(parity_ns=5 * _S)
    shadow.start()
    try:
        shadow.send_init(main).result(timeout=60)
        timings = shadow.last_init_timings
    finally:
        shadow.stop()
    assert set(timings) == set(ss._STRUCT_NAMES)
    for name, t in timings.items():
        assert t["bytes"] > 0 and t["plainify_s"] >= 0 and t["pickle_s"] >= 0 and t["send_s"] >= 0, name
    line = ss.init_timings_line(timings)
    assert line.startswith(f"{len(ss._STRUCT_NAMES)} structures ") and "slowest: " in line
