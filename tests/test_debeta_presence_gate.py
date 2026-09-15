# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner that has stopped answering the validator is not scored on the de-beta path while it is absent.

Without the gate a material share of the paying board can go to uids whose last request window held
no successful response, and such a uid can lead it. The skill leg is built from fills, so the resting
orders of a dead miner earn timing credit for the whole lookback window; under 0.6.0 kappa needed
realized round trips and the same miner scored nothing. The gate: the validator keeps the last
`presence_window` query outcomes per uid (forward.update_stats); a uid whose window is full and holds no success is
absent and is dropped from the de-beta map for that cycle (not scorable, score 0), its accumulators untouched. A uid
with fewer outcomes than the window is present; the windows are saved with the validator state, so a restart neither
empties the board nor reopens the gate for a miner that went dark before it. Dial --scoring.debeta.presence_gate (1 default,
0 off) and --scoring.debeta.presence_window (50). The scoring child receives the absent set with its score inputs.
"""
import argparse
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from taos.im.validator import forward
from taos.im.validator.debeta import absent_uids

DEV = Path(__file__).resolve().parents[1]


def test_absent_means_a_full_window_with_no_success():
    presence = {1: deque([False] * 50, maxlen=50), 2: deque([False] * 49 + [True], maxlen=50),
                3: deque([False] * 10, maxlen=50), 4: deque([True] * 50, maxlen=50)}
    assert absent_uids(presence, 50) == {1}, "only the full all-failure window is absent"
    assert absent_uids(presence, 10) == {1, 3}, "a shorter window sees uid 3 as absent too"
    assert absent_uids({}, 50) == set()


def test_lists_from_the_wire_work_like_deques():
    assert absent_uids({7: [0] * 50, 8: [0] * 49 + [1]}, 50) == {7}


def _synapse(status):
    return SimpleNamespace(is_timeout=status == 408, is_failure=status not in (200, 408, 403), is_blacklist=status == 403,
                           response=object() if status == 200 else None, dendrite=SimpleNamespace(process_time=0.5 if status == 200 else None))


def test_update_stats_records_presence_outcomes():
    v = SimpleNamespace(miner_stats={}, config=SimpleNamespace(scoring=SimpleNamespace(debeta=SimpleNamespace(presence_window=3))))
    forward.update_stats(v, {1: _synapse(200), 2: _synapse(503)})
    forward.update_stats(v, {1: _synapse(200), 2: _synapse(503)})
    forward.update_stats(v, {1: _synapse(408), 2: _synapse(503)})
    assert list(v.miner_presence[1]) == [True, True, False]
    assert list(v.miner_presence[2]) == [False, False, False]
    assert absent_uids(v.miner_presence, 3) == {2}
    forward.update_stats(v, {2: _synapse(200)})
    assert absent_uids(v.miner_presence, 3) == set(), "one success and the uid is present again"
    assert v.miner_stats[2]["failures"] == 3, "the published counters are unchanged"


def _fake_validator(presence=None, gate=1, window=50, absent=None):
    from taos.im.validator.debeta import accumulate_book_mtm

    cfg = SimpleNamespace(enabled=True, w_make=0.30, centered_window=15, floor_scale=0.0, min_books=1, p11_strength=0.0,
                          making_floor_scale=0.0, skill_rank_scope="positives", making_rank_scope="positives",
                          presence_gate=gate, presence_window=window)
    v = SimpleNamespace(config=SimpleNamespace(scoring=SimpleNamespace(debeta=cfg)))
    mtm, invsum, invn, inv = defaultdict(dict), defaultdict(dict), {}, defaultdict(dict)
    pf, pl, drift = {}, {}, {}
    # three miners buy on six books before a rise: uid 1 largest, 2 middle, 3 smallest; all positive alpha, distinct shapes
    for b in range(6):
        trades = []
        for u, q in ((1, 3.0), (2, 2.0), (3, 1.0)):
            trades.append({"p": 100.0 + 0.1 * b, "q": q, "s": 0, "Ma": 7, "Ta": u})
        trades += [{"p": 100.0 + 0.1 * b + 0.5 * k + 0.05 * (k % 2), "q": 1.0, "s": 0, "Ma": 8, "Ta": 9} for k in range(1, 6)]
        accumulate_book_mtm(mtm, invsum, invn, inv, pf, pl, b, trades, drift=drift)
    v.debeta_mtm, v.debeta_invsum, v.debeta_invn, v.debeta_drift = mtm, invsum, invn, drift
    v.debeta_pfirst, v.debeta_plast = pf, pl
    cb = defaultdict(lambda: defaultdict(float)); cs = defaultdict(lambda: defaultdict(float))
    for b in range(6):
        cb[1][b] = 3.0 + 0.1 * b; cs[1][b] = 3.0
        cb[2][b] = 1.0; cs[2][b] = 1.0
    v.capture_buy_sums, v.capture_sell_sums = cb, cs
    v.miner_presence = presence if presence is not None else {}
    if absent is not None:
        v.debeta_absent = absent
    return v


def test_an_absent_uid_is_dropped_from_the_map_and_its_detail_says_so():
    from taos.im.validator.reward import compute_debeta_scores

    live = compute_debeta_scores(_fake_validator())
    assert live and all(u in live for u in (1, 2, 3)), live
    v = _fake_validator(presence={2: deque([False] * 50, maxlen=50), 1: deque([True] * 50, maxlen=50)})
    gated = compute_debeta_scores(v)
    assert 2 not in gated, "the absent uid is not scorable this cycle"
    assert gated[1] == live[1] and gated[3] == live[3], "the others' scores are untouched (no renormalisation)"
    assert v.debeta_detail[2]["present"] is False
    assert "skill_raw" in v.debeta_detail[2] and "making_rank" in v.debeta_detail[2], "the legs stay published for the absent uid"
    assert v.debeta_detail[1]["present"] is True
    assert v.debeta_mtm[2], "the accumulators keep running while the uid is absent"


def test_a_partial_window_after_a_restart_is_present():
    from taos.im.validator.reward import compute_debeta_scores

    v = _fake_validator(presence={2: deque([False] * 49, maxlen=50)})
    assert 2 in compute_debeta_scores(v)


def test_the_gate_can_be_switched_off():
    from taos.im.validator.reward import compute_debeta_scores

    v = _fake_validator(presence={2: deque([False] * 50, maxlen=50)}, gate=0)
    scores = compute_debeta_scores(v)
    assert 2 in scores and v.debeta_detail[2]["present"] is True


def test_a_shipped_absent_set_takes_precedence_for_the_scoring_child():
    from taos.im.validator.reward import compute_debeta_scores

    v = _fake_validator(presence={}, absent=[3])
    scores = compute_debeta_scores(v)
    assert 3 not in scores and 1 in scores and 2 in scores


def test_the_dials_exist_and_default_on():
    from taos.im.config import add_im_validator_args

    parser = argparse.ArgumentParser()
    add_im_validator_args(object, parser)
    args, _ = parser.parse_known_args([])
    assert getattr(args, "scoring.debeta.presence_gate") == 1
    assert getattr(args, "scoring.debeta.presence_window") == 50
    args, _ = parser.parse_known_args(["--scoring.debeta.presence_gate", "0", "--scoring.debeta.presence_window", "20"])
    assert getattr(args, "scoring.debeta.presence_gate") == 0 and getattr(args, "scoring.debeta.presence_window") == 20


def test_the_child_config_duck_and_the_frames_carry_the_gate():
    shadow_src = (DEV / "taos/im/validator/scoring_shadow.py").read_text()
    assert "presence_gate=_num(getattr(_d, 'presence_gate', None), 1, int)" in shadow_src
    assert "presence_window=_num(getattr(_d, 'presence_window', None), 50, int)" in shadow_src
    assert "absent" in shadow_src and "debeta_absent" in shadow_src, "the child scores on the absent set main ships"
    validator_src = (DEV / "taos/im/neurons/validator.py").read_text()
    assert validator_src.count("absent_now(self)") == 2, "main computes the absent set for both the eager and the pull path"
    report_src = (DEV / "taos/im/validator/report.py").read_text()
    assert "('present', 'debeta_present')" in report_src, "the dashboards get a presence gauge"


def test_presence_windows_survive_a_restart_through_the_saved_state():
    import msgpack
    from taos.im.validator.persistence import snapshot_miner_presence, restore_miner_presence

    v = SimpleNamespace(effective_max_uids=256,
                        miner_presence={1: deque([False] * 50, maxlen=50), 2: deque([True] * 3, maxlen=50)})
    snap = msgpack.unpackb(msgpack.packb(snapshot_miner_presence(v), use_bin_type=True), strict_map_key=False)
    fresh = SimpleNamespace(effective_max_uids=256)
    assert restore_miner_presence(fresh, snap, 50) == 2
    assert absent_uids(fresh.miner_presence, 50) == {1}, "the dark uid is absent from the first cycle after the restart"
    assert list(fresh.miner_presence[2]) == [True, True, True], "a partial window stays partial"
    assert fresh.miner_presence[1].maxlen == 50
    assert restore_miner_presence(fresh, {"bad": [1], 999: [1], 3: "x"}, 50) == 0, "malformed and out-of-range entries are skipped"
    assert restore_miner_presence(fresh, {4: [0] * 80}, 50) == 1 and len(fresh.miner_presence[4]) == 50, "a longer saved window is trimmed to the dial"


def test_all_three_state_writers_carry_the_presence_windows():
    # Three writers produce the validator state file: main's in-process save (build_validator_state), the
    # light fields main ships to the scoring child (build_save_light_fields), and the child's composer that
    # writes the file from its replica plus those light fields (child_save_validator_state, the writer that
    # actually runs at the final rung with the scoring service live). A key missing from any one of them
    # is silently absent from the file: the first deploy carried the windows in the two persistence.py
    # dicts and not in the child's composer, and the second restart presumed every uid present again.
    src = (DEV / "taos/im/validator/persistence.py").read_text()
    assert src.count('"miner_presence": snapshot_miner_presence(self)') == 2, "the main save and the offloaded light fields both persist presence"
    assert "restore_miner_presence(self, _loaded_presence" in src, "the load path restores them"
    shadow_src = (DEV / "taos/im/validator/scoring_shadow.py").read_text()
    assert shadow_src.count('"miner_presence": light.get("miner_presence", {})') == 1, "the child's composer writes the windows main shipped"
