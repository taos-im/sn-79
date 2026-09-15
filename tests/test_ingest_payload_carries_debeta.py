# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The ingest push carries the de-beta decomposition, from one builder, in one contract.

Three pushers built the per-agent scoring maps by hand and had drifted: the
reporting child sent the kappa SCORE under agent_kappa where the
validator's two paths sent the TOTAL, both landing in agent_snapshots.kappa_raw; and none of the three
carried the de-beta fields, so the UI could only show a kappa that emissions no longer follow. Every
pusher now composes its scoring keys from taos.im.validator.ingest_payload, uids whose de-beta values are
None are omitted rather than sent as nulls (the sinks coerce with float(... or 0), which would rank an
unscored miner last), and the scalar dials travel as scoring_params.
"""
from pathlib import Path
from types import SimpleNamespace

from taos.im.validator.ingest_payload import agent_scoring_maps, debeta_maps, kappa_maps, scoring_params

DEV = Path(__file__).resolve().parents[1]

SCORED = {
    "total": 1.7, "normalized_total": 0.62, "penalty": 0.9, "books": {3: 1.1, 7: 2.3},
    "books_weighted": {3: 0.5}, "activity_weighted_normalized_median": 0.55,
    "debeta_score": 0.81, "making_rank": 0.4, "skill_rank": 0.9, "making_raw": 0.012, "skill_raw": 1.9,
    "p11_factor": 1.0, "num_scored_books": 6, "scorable": True,
    "debeta_weight": 0.5, "debeta_w_make": 0.3, "debeta_floor": 0.0021,
}
WARMING = {  # a uid de-beta did not apply to this cycle: every decomposition field is None
    "total": 0.4, "normalized_total": 0.3, "penalty": 1.0, "books": {}, "books_weighted": {},
    "activity_weighted_normalized_median": 0.2,
    "debeta_score": None, "making_rank": None, "skill_rank": None, "making_raw": None, "skill_raw": None,
    "p11_factor": None, "debeta_weight": 0.5, "debeta_w_make": None, "debeta_floor": None,
}


def test_debeta_fields_are_carried_per_uid_and_absent_when_unknown():
    maps = debeta_maps({1: SCORED, 2: WARMING, 3: None})
    assert maps["agent_debeta_score"] == {"1": 0.81}
    assert maps["agent_making_rank"] == {"1": 0.4} and maps["agent_skill_rank"] == {"1": 0.9}
    assert maps["agent_making_raw"] == {"1": 0.012} and maps["agent_skill_raw"] == {"1": 1.9}
    assert maps["agent_p11_factor"] == {"1": 1.0}
    assert maps["agent_num_scored_books"] == {"1": 6} and maps["agent_scorable"] == {"1": True}
    for key, per_uid in maps.items():
        assert "2" not in per_uid and "3" not in per_uid, f"{key} must omit a uid with no value, not send null"


def test_kappa_maps_send_the_total_under_agent_kappa():
    maps = kappa_maps({1: SCORED, 2: WARMING})
    assert maps["agent_kappa"] == {"1": 1.7, "2": 0.4}
    assert maps["agent_kappa_score"] == {"1": 0.62, "2": 0.3}
    assert maps["agent_kappa_penalty"] == {"1": 0.9, "2": 1.0}
    assert maps["agent_kappa_books"] == {"1": {"3": 1.1, "7": 2.3}} and maps["agent_kappa_books_w"] == {"1": {"3": 0.5}}
    assert maps["agent_median_kappa"] == {"1": 0.55, "2": 0.2}


def _config(kappa=0.395, pnl=0.105, debeta=0.5, w_make=0.3):
    return SimpleNamespace(scoring=SimpleNamespace(
        kappa=SimpleNamespace(weight=kappa), pnl=SimpleNamespace(weight=pnl),
        debeta=SimpleNamespace(weight=debeta, w_make=w_make)))


def test_scoring_params_carry_the_dials_and_this_cycles_floor():
    params = scoring_params(_config(), {1: SCORED, 2: WARMING})
    assert params == {"kappa_weight": 0.395, "pnl_weight": 0.105, "debeta_weight": 0.5,
                      "debeta_w_make": 0.3, "debeta_floor": 0.0021, "debeta_enabled": True}
    warming_only = scoring_params(_config(), {2: WARMING})
    assert warming_only["debeta_floor"] is None and warming_only["debeta_weight"] == 0.5
    assert scoring_params(_config(debeta=0.0), {})["debeta_enabled"] is False
    assert scoring_params(None, {1: SCORED})["debeta_weight"] == 0.5, "no config: the carrier's dial is used"


class _Validator:
    def __init__(self):
        self.config = _config()
        self.scores = [0.0, 0.7, 0.2]
        self.kappa_values = {1: SCORED, 2: WARMING}
        self.volume_sums = {1: {3: 10.0, 7: 5.0}, 2: {}}
        self.maker_volume_sums = {1: {3: 4.0}}
        self.taker_volume_sums = {1: {3: 6.0, 7: 5.0}}
        self.fee_sums = {1: {3: 0.01}}
        self.roundtrip_volume_sums = {1: {3: 8.0}}
        self.activity_factors = {1: {3: 1.0, 7: 0.5}}
        self.agent_pnl_by_book = {1: {3: 0.5, 7: -0.25}}
        self.agent_pnl_total = {1: 0.25, 2: 0.0}


def test_the_builder_carries_legacy_and_debeta_keys_together():
    maps = agent_scoring_maps(_Validator())
    assert maps["agent_scores"] == {"0": 0.0, "1": 0.7, "2": 0.2}
    assert maps["agent_kappa"] == {"1": 1.7, "2": 0.4}
    assert maps["agent_volume"] == {"1": 15.0} and maps["agent_maker_volume"] == {"1": 4.0}
    assert maps["agent_pnl"] == {"1": 0.25}, "a zero total is omitted, as before"
    assert maps["agent_pnl_book"] == {"1": {"3": 0.5, "7": -0.25}}
    assert maps["agent_volume_book"] == {"1": {"3": 10.0, "7": 5.0}}
    assert maps["agent_fee_book"] == {"1": {"3": 0.01}}
    assert maps["agent_roundtrip_volume"] == {"1": 8.0} and maps["agent_activity_factor"] == {"1": 0.75}
    assert maps["agent_debeta_score"] == {"1": 0.81} and maps["agent_scorable"] == {"1": True}
    assert maps["scoring_params"]["debeta_weight"] == 0.5 and maps["scoring_params"]["debeta_floor"] == 0.0021


def test_the_builder_survives_a_bare_validator():
    maps = agent_scoring_maps(SimpleNamespace())
    assert maps["agent_scores"] == {} and maps["agent_debeta_score"] == {}
    assert maps["scoring_params"]["debeta_enabled"] is False


def test_every_pusher_uses_the_shared_builder():
    validator_src = (DEV / "taos/im/neurons/validator.py").read_text()
    report_src = (DEV / "taos/im/validator/report.py").read_text()
    assert "_kappa_raw = {}" not in validator_src and "_kappa_raw   = {}" not in validator_src, (
        "a hand-built kappa map is back in validator.py; both push paths must use agent_scoring_maps"
    )
    assert validator_src.count("self._build_agent_scoring_maps()") >= 2, "both the sim and exchange pushes"
    assert "return agent_scoring_maps(self)" in validator_src
    assert "kappa_maps(self.kappa_values)" in report_src and "debeta_maps(self.kappa_values)" in report_src
    assert '"agent_kappa":           {uid: float((kv or {}).get(\'score\'' not in report_src, (
        "the report push sent the kappa SCORE as agent_kappa"
    )
