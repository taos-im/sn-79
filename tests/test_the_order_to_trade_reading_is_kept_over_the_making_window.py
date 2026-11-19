"""The order-to-trade discount reads each uid's messages per fill over the making window: the
validator keeps a placement-and-cancel count and a fill count per uid as timestamped histories, pruned and shifted
with the de-beta histories, saved with the state and handed to the scoring service child, and the scorer reads
their ratio when the dial is on."""
import inspect
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taos.im.validator import forward, persistence, reward, scoring_shadow, trade  # noqa: E402


def test_placements_cancels_and_fills_land_in_timestamped_histories():
    v = SimpleNamespace(miner_stats={}, engine=SimpleNamespace(mode="simulation"))
    trade.init_otr_state(v)
    notices = {3: [{"y": "RDPOL", "b": 0, "i": 1}, {"y": "RDPOL", "b": 0, "i": 2},
                   {"y": "ET", "b": 0, "i": 1, "t": 9}]}
    forward.update_placement_stats(v, notices, ts=100)
    forward.update_placement_stats(v, {3: [{"y": "RDPOM", "b": 0, "i": 3}]}, ts=160)
    assert v.debeta_msg_hist[3] == {100: 2, 160: 1} and v.debeta_msgs[3] == 3
    assert v.debeta_fill_hist[3] == {100: 1} and v.debeta_fills[3] == 1
    assert forward.messages_per_fill(v) == {3: 3.0}


def test_a_cancel_the_engine_applied_counts_as_a_message():
    v = SimpleNamespace(miner_stats={}, engine=SimpleNamespace(mode="simulation"))
    trade.init_otr_state(v)
    forward.update_placement_stats(v, {4: [{"y": "RDCO", "b": 0, "c": [{"o": 7}]}]}, ts=100)
    assert v.debeta_msgs[4] == 1 and forward.messages_per_fill(v) == {4: float("inf")}


def test_the_histories_are_pruned_shifted_saved_and_handed_to_the_child():
    src = inspect.getsource(trade)
    for name in ("debeta_msg_hist", "debeta_fill_hist"):
        assert src.count(name) >= 4, f"{name}: init, prune, shift and restore"
    assert "debeta_msg_hist" in persistence._DEBETA_HIST2 and "debeta_fill_hist" in persistence._DEBETA_HIST2
    assert "debeta_msgs" in scoring_shadow._DEBETA_PLAIN and "debeta_fill_hist" in scoring_shadow._DEBETA_PLAIN


def test_the_scorer_reads_the_ratio_when_the_dial_is_on(monkeypatch):
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("pg", pathlib.Path(__file__).with_name("test_debeta_presence_gate.py"))
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    v = pg._fake_validator()
    trade.init_otr_state(v)
    v.debeta_msgs = {1: 500, 2: 50}
    v.debeta_fills = {1: 1, 2: 1}
    seen = {}

    def fake_scores(*a, **kw):
        seen.update(kw)
        return {}
    monkeypatch.setattr(reward, "debeta_scores", fake_scores)
    monkeypatch.setattr(reward, "debeta_dial", lambda cfg, name, _d=reward.debeta_dial: (
        200.0 if name == "making_otr_threshold" else 0.5 if name == "making_otr_power" else _d(cfg, name)))
    reward.compute_debeta_scores(v)
    assert seen.get("otr") == {1: 500.0, 2: 50.0}
    assert seen.get("otr_threshold") == 200.0 and seen.get("otr_power") == 0.5


def test_a_replay_counts_a_states_messages_at_once():
    from taos.im.validator.block_stats import note_otr

    v = SimpleNamespace()
    note_otr(v, 5, "msg", 100, 40)
    note_otr(v, 5, "fill", 100, 2)
    note_otr(v, 5, "msg", 160)
    note_otr(v, 5, "msg", 160, 0)
    assert v.debeta_msg_hist[5] == {100: 40, 160: 1} and v.debeta_msgs[5] == 41
    assert v.debeta_fill_hist[5] == {100: 2} and v.debeta_fills[5] == 2
    assert forward.messages_per_fill(v) == {5: 20.5}
