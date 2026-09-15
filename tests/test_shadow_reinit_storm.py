# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""0.6.1 testnet: 23 parity re-INITs in 45 minutes. Each one pickled 365 MB under the
reward lock (a 10 s stall, the "Waiting for rewarding to catch up" backlog, GenTRX health flaps,
skipped report cycles) and none converged. The self-heal had no stop: a child that cannot be brought
into parity was re-initialised forever, and the log could not say why (tee drops are silent, the
child's applied count was not in the MISMATCH line).

Pinned here: the storm breaker suspends the scoring service after N re-INITs inside a window and makes
every main-facing method inert so main scores and saves in-process; drops are logged; the MISMATCH
line names the child's applied count; the counters reach the report data and the validator gauges.
"""
from pathlib import Path

import bittensor as bt

import taos.im.validator.scoring_shadow as ss

DEV = Path(__file__).resolve().parents[1]
VALIDATOR = (DEV / "taos/im/neurons/validator.py").read_text()
REPORT = (DEV / "taos/im/validator/report.py").read_text()


class _Rec:
    def __init__(self):
        self.msgs = []

    def __call__(self, msg, *a, **k):
        self.msgs.append(str(msg))


def _shadow(monkeypatch, n=3, window=900.0):
    monkeypatch.setattr(ss, "SHADOW_REINIT_STORM_N", n)
    monkeypatch.setattr(ss, "SHADOW_REINIT_STORM_S", window)
    monkeypatch.setattr(bt.logging, "warning", _Rec())
    monkeypatch.setattr(bt.logging, "error", _Rec())
    return ss.ScoringShadow(child_fn=lambda *a: None)


def test_storm_detector_counts_inside_the_window_only():
    storm = ss._ReinitStorm(3, 900.0)
    assert not storm.hit(0.0) and not storm.hit(10.0)
    assert storm.hit(20.0), "third re-INIT inside the window trips it"
    storm = ss._ReinitStorm(3, 900.0)
    assert not storm.hit(0.0) and not storm.hit(10.0)
    assert not storm.hit(1000.0), "the first one has aged out of the window"
    assert ss._ReinitStorm(0, 900.0).hit(0.0) is False, "0 disables"


def test_repeated_reinits_suspend_the_cutover(monkeypatch):
    sh = _shadow(monkeypatch, n=3)
    sh.request_reinit()
    sh.request_reinit()
    assert not sh.cutover_suspended and sh.reinits == 2
    sh.request_reinit()
    assert sh.cutover_suspended and sh.reinits == 3
    assert not sh.initialized
    errors = bt.logging.error.msgs
    assert any("suspend" in m.lower() and "re-INIT" in m for m in errors), errors


def test_suspended_shadow_is_inert_for_main(monkeypatch):
    sh = _shadow(monkeypatch, n=1)
    sh.request_reinit()
    assert sh.cutover_suspended
    monkeypatch.setattr(sh, "is_alive", lambda: True)
    sh.tee(b"round", 1)
    assert sh._pending == 0 and sh._last_teed_ts == 0, "no frames reach a suspended child"
    sh.tee_score_inputs(5, 5, [], {}, {})
    assert sh.eager_inputs_for(5) is None
    assert sh.request_scores(5, 5, []) is None, "main must fall back to in-process scoring"
    fut = sh.send_init(object())
    assert fut.result(timeout=1) is None, "send_init must stay awaitable and do nothing"
    assert not sh.initialized
    sh.record_main_digest(10, {"a": "1"})
    assert not sh._ring
    sh._last_teed_ts = 10
    assert sh.request_save("/nonexistent/path", {}) is None, "main must save in-process"
    h = sh.health()
    assert h["cutover_suspended"] == 1 and h["reinits"] == 1


def test_mismatches_after_suspension_do_not_rearm(monkeypatch):
    sh = _shadow(monkeypatch, n=1)
    sh.request_reinit()
    before = sh.reinits
    sh._compare_parity(20, {"vol": "a"}, ({"vol": "b"}, {}, 5))
    sh._compare_parity(30, {"vol": "a"}, ({"vol": "b"}, {}, 6))
    assert sh.reinits == before and sh.cutover_suspended
    msg = bt.logging.error.msgs[-1]
    assert "applied=6" in msg and "dropped=" in msg, msg


def test_mismatch_line_names_applied_and_dropped(monkeypatch):
    sh = _shadow(monkeypatch, n=0)
    sh._dropped = 4
    sh._compare_parity(20, {"vol": "a", "n_pos": "x"}, ({"vol": "b", "n_pos": "x"}, {}, 17))
    msg = bt.logging.error.msgs[-1]
    assert "MISMATCH" in msg and "['vol']" in msg and "applied=17" in msg and "dropped=4" in msg, msg
    sh._compare_parity(30, {"vol": "a"}, ({"vol": "a"}, {}, 18))
    assert not sh.cutover_suspended


def test_tee_drop_is_logged_once_per_window(monkeypatch):
    sh = _shadow(monkeypatch, n=0)
    monkeypatch.setattr(sh, "is_alive", lambda: True)
    sh._pending = 8
    sh.tee(b"x", 1)
    sh.tee(b"x", 2)
    warnings = bt.logging.warning.msgs
    assert sh._dropped == 2
    assert len(warnings) == 1 and "dropped" in warnings[0] and "ts=1" in warnings[0], warnings
    sh._last_drop_log -= 1000.0
    sh.tee(b"x", 3)
    assert len(warnings) == 2 and "3 dropped" in warnings[1], warnings
    assert sh.health()["tee_dropped"] == 3


def test_validator_gates_the_cutover_on_suspension():
    block = VALIDATOR[VALIDATOR.index("Cutover: the scoring service computes; main adopts"):]
    block = block[:block.index("request_scores(")]
    assert "cutover_suspended" in block, "a suspended shadow must not be asked for scores"
    init_block = VALIDATOR[VALIDATOR.index("Applied-ts + one-time INIT"):]
    init_block = init_block[:init_block.index("scoring.interval != 0")]
    assert "cutover_suspended" in init_block, "a suspended shadow must not be re-INIT'd or digest-compared"


def test_shadow_health_reaches_the_report_gauges():
    assert "'scoring_shadow':" in VALIDATOR and ".health()" in VALIDATOR, "report data must carry the shadow health"
    assert "scoring_shadow_health" in REPORT
    for gauge in ("scoring_shadow_parity_mismatches", "scoring_shadow_reinits",
                  "scoring_shadow_tee_dropped", "scoring_shadow_suspended"):
        assert f'"{gauge}"' in REPORT, gauge
