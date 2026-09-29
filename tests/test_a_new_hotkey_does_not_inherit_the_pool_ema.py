# SPDX-License-Identifier: MIT
"""A slot that changes hands starts the proportional pool from nothing. The pool smooths each uid's making term
and making share with the track-record EMA (`_debeta_pool_ema`, keyed "term" and "share"). The deregistration
handler cleared the track-record EMA itself but not these two, so they were cleared only if a scoring boundary
happened to run while the uid sat in `deregistered_uids`, about ten seconds. When none did, the new occupant was
paid the departed miner's smoothed making share until it made something itself.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taos.im.neurons.validator as vmod  # noqa: E402


def _validator():
    return SimpleNamespace(
        engine=SimpleNamespace(handle_deregistration=lambda uid, old_coldkey=None: None),
        gentrx_scores={5: 0.4, 6: 0.1},
        _gentrx_ema={5: 0.3},
        _trading_score_ema={5: 0.9, 6: 0.8},
        _trading_score_ema_n={5: 12, 6: 30},
        _debeta_pool_ema={"term": {5: 0.31, 6: 0.22}, "share": {5: 0.04, 6: 0.02}},
        _gentrx=None,
    )


def test_the_departed_miners_pool_term_and_share_are_cleared():
    v = _validator()
    vmod.Validator.handle_deregistration(v, 5)
    assert 5 not in v._debeta_pool_ema["term"] and 5 not in v._debeta_pool_ema["share"]
    assert 5 not in v._trading_score_ema and 5 not in v._trading_score_ema_n


def test_other_uids_keep_theirs():
    v = _validator()
    vmod.Validator.handle_deregistration(v, 5)
    assert v._debeta_pool_ema["term"][6] == 0.22 and v._debeta_pool_ema["share"][6] == 0.02
    assert v._trading_score_ema[6] == 0.8


def test_a_validator_without_a_pool_ema_is_unaffected():
    v = _validator()
    del v._debeta_pool_ema
    vmod.Validator.handle_deregistration(v, 5)
    assert 5 not in v._trading_score_ema
