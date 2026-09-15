# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""In cutover mode main ADOPTS the child's kappa_values for publishing (validator.py: updated_data =
adopted['factors']; self.kappa_values = updated_data['kappa_values']). The per-uid publisher in
reward.py reads the de-beta decomposition from validator_data['debeta_detail'] / 'debeta_floor' /
'debeta_w_make', which compute_debeta_scores stashes on the scorer object; main's get_rewards ships
them, the child's shadow_score did not. Result on the 0.6.1 testnet deploy: debeta_score published,
every rank, raw leg, CP factor, w_make and floor silently None, num_scored_books 0 for every miner.
The child must ship exactly what main ships.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_scoring_shadow import _S, _fresh_main, _snapshot_to_shadow, _state  # noqa: E402

from taos.im.validator import reward as reward_mod  # noqa: E402
from taos.im.validator.scoring_shadow import shadow_score  # noqa: E402
from taos.im.validator.trade import update_trade_volumes  # noqa: E402

DETAIL = {
    1: {"making_raw": 4.0, "making_rank": 1.0, "skill_raw": 0.9, "skill_rank": 0.5, "p11_factor": 0.8, "skill_books": 6},
    2: {"making_raw": 0.0, "making_rank": 0.0, "skill_raw": 1.7, "skill_rank": 1.0, "p11_factor": 1.0, "skill_books": 9},
}
SCORES = {1: 0.3 * 1.0 + 0.7 * 0.5, 2: 0.7}


def _stub_compute_debeta_scores(scorer):
    scorer.debeta_detail = dict(DETAIL)
    scorer.debeta_floor = 0.123
    scorer.debeta_w_make = 0.30
    return dict(SCORES)


def test_shadow_score_ships_the_decomposition_main_ships(monkeypatch):
    monkeypatch.setattr(reward_mod, "compute_debeta_scores", _stub_compute_debeta_scores)
    main = _fresh_main()
    update_trade_volumes(main, _state(1 * _S, [(1, 2, 0, 3.0, 100.0, 0)]))
    shadow = _snapshot_to_shadow(main)
    update_trade_volumes(shadow, _state(2 * _S, [(2, 1, 0, 1.5, 101.0, 1)]))

    kv = shadow_score(shadow, 5 * _S, [])["factors"]["kappa_values"]

    for uid, d in DETAIL.items():
        got = kv[uid]
        assert got["debeta_score"] == SCORES[uid]
        assert got["making_rank"] == d["making_rank"] and got["skill_rank"] == d["skill_rank"]
        assert got["making_raw"] == d["making_raw"] and got["skill_raw"] == d["skill_raw"]
        assert got["p11_factor"] == d["p11_factor"]
        assert got["num_scored_books"] == d["skill_books"]
        assert got["scorable"] is True
        assert got["debeta_w_make"] == 0.30
        assert got["debeta_floor"] == 0.123
