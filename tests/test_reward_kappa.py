# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Contract tests for the reward core.

The incentive mechanism had zero test coverage. These pin load-bearing
invariants of the Kappa-3 risk-adjusted-return maths (`kappa_3`) and the
score orchestrator's early-exit, exercising the real functions (no stubs).
"""
import pytest

from taos.im.utils.kappa import kappa_3
from taos.im.validator.reward import calculate_kappa_score

# kappa_3 gating knobs kept permissive so the maths, not the gates, is under test.
_K = dict(
    tau=0.0,
    lookback=10800_000_000_000,  # permissive: >> any synthetic series span
                                 # (kappa_3 now applies lookback as an explicit window)
    norm_min=0.0,
    norm_max=2.0,
    min_lookback=1,
    min_realized_observations=3,
    grace_period=0,
    deregistered_uids=[],
    book_count=1,
)


def _series(pnls, step=1_000_000):
    """Build {timestamp: {book0: pnl}} from a list of per-period P&Ls."""
    return {i * step: {0: p} for i, p in enumerate(pnls)}


def test_kappa3_none_on_no_data():
    assert kappa_3(1, {}, **_K) is None


def test_kappa3_none_on_deregistered():
    k = dict(_K)
    k["deregistered_uids"] = [1]
    assert kappa_3(1, _series([1.0, 2.0, 3.0, 4.0]), **k) is None


def test_kappa3_deregistered_ignores_matching_cache():
    """Regression: on a UID replacement the reused slot's realized_pnl_history stays byte-identical to
    the old occupant's until reset lands, so its fingerprint matches the cache. The dereg guard must
    still win — the cache must NOT hand the still-deregistered slot the old miner's kappa."""
    from taos.im.utils.kappa import _get_pnl_fingerprint
    hist = _series([1.0, 2.0, 3.0, 4.0])
    cache = {1: (_get_pnl_fingerprint(hist), {"sentinel": "OLD_OCCUPANT"})}
    k = dict(_K)
    k["deregistered_uids"] = [1]
    assert kappa_3(1, hist, cache=cache, **k) is None
    # a NON-deregistered uid with a matching cache is still served (the cache itself is not broken)
    cache2 = {2: (_get_pnl_fingerprint(hist), {"sentinel": "VALID"})}
    assert kappa_3(2, hist, cache=cache2, **_K) == {"sentinel": "VALID"}


def test_kappa3_returns_book_structure_for_sufficient_data():
    out = kappa_3(1, _series([1.0, 2.0, 1.5, 2.5, 1.0]), **_K)
    assert out is not None
    assert "books" in out and 0 in out["books"]


def test_kappa3_rewards_steady_over_volatile_for_same_mean():
    """Risk adjustment: a steady positive series must out-score a volatile one
    with the same mean (lower downside → higher Kappa-3)."""
    steady = kappa_3(1, _series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]), **_K)["books"][0]
    volatile = kappa_3(1, _series([8.0, -4.0, 8.0, -4.0, 8.0, -4.0]), **_K)["books"][0]
    assert steady is not None and volatile is not None
    assert steady > volatile


def test_calculate_kappa_score_full_path_regression():
    """Locks the orchestrator's full path (active + idle + None-kappa books, decay
    and P&L weighting enabled) to a known value. Guards the helper extraction in
    calculate_kappa_score against any change in calculation outcome."""
    S = 1_000_000_000
    ts = 50_000 * S
    uid = 7
    cfg = {
        "interval": 5_000_000_000,
        "max_inactive_books_ratio": 0.375,
        "kappa": {"normalization_min": -2.5, "normalization_max": 2.5,
                  "lookback": 10800_000_000_000, "pnl": {"impact": 0.5}},
        "activity": {"capital_turnover_cap": 10.0, "trade_volume_sampling_interval": 600_000_000_000,
                     "decay_grace_period": 600_000_000_000, "impact": 0.33, "decay_rate": 1.0},
    }
    score = calculate_kappa_score(
        uid=uid,
        kappa_values={uid: {"books": {0: 1.8, 1: -0.5, 2: None, 3: 0.9}}},
        activity_factors={uid: {0: 1.0, 1: 0.8, 2: 1.2, 3: 0.5}},
        pnl_factors={uid: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}},
        roundtrip_volumes={uid: {0: {ts - 100 * S: 3000.0}, 1: {ts - 30000 * S: 50.0}, 3: {ts - 700 * S: 1200.0}}},
        realized_pnl_history={uid: {ts - 200 * S: {0: 300.0, 1: -150.0, 3: 80.0}}},
        config=cfg,
        simulation_config={"miner_wealth": 50000.0, "publish_interval": 5 * S, "volumeDecimals": 4},
        simulation_timestamp=ts,
    )
    assert float(score) == pytest.approx(0.6848940067839999, abs=1e-12)


def test_calculate_kappa_score_unknown_uid_is_zero():
    """Score orchestrator early-exits to 0.0 for a miner with no kappa data."""
    score = calculate_kappa_score(
        uid=42,
        kappa_values={},
        activity_factors={},
        pnl_factors={},
        roundtrip_volumes={},
        realized_pnl_history={},
        config={"kappa": {"normalization_min": 0.0, "normalization_max": 2.0, "lookback": 10}},
        simulation_config={"miner_wealth": 1000.0, "publish_interval": 60, "volumeDecimals": 2},
        simulation_timestamp=1_000_000_000,
    )
    assert score == 0.0


def test_kappa_cache_flag_default_off_with_optin(monkeypatch):
    """The fingerprint cache defaults OFF (all UIDs -> loky); KAPPA_CACHE=1 or a
    .kappa_cache sentinel re-enables it (sentinel wins over the env)."""
    import taos.im.utils.kappa as k

    monkeypatch.setattr(k.os.path, "exists", lambda p: False)
    monkeypatch.delenv("KAPPA_CACHE", raising=False)
    assert k._kappa_cache_enabled() is False

    monkeypatch.setenv("KAPPA_CACHE", "1")
    assert k._kappa_cache_enabled() is True

    monkeypatch.setenv("KAPPA_CACHE", "0")
    assert k._kappa_cache_enabled() is False

    # .kappa_cache sentinel is an override that wins over env=0.
    monkeypatch.setattr(k.os.path, "exists", lambda p: str(p).endswith(".kappa_cache"))
    monkeypatch.setenv("KAPPA_CACHE", "0")
    assert k._kappa_cache_enabled() is True


def test_kappa_3_batch_skips_cache_updates_when_disabled():
    """Cache-off path: build_cache_updates=False returns the SAME kappa results
    but an EMPTY cache_updates dict — the redundant (fingerprint, kappa_values)
    copy that dominated the parent's collect is not built/pickled. Results must
    stay identical so scoring is unaffected."""
    from taos.im.utils.kappa import kappa_3_batch

    pnl = {
        1: _series([1.0, 2.0, 1.5, 2.5, 1.0]),
        2: _series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    }
    res_on, upd_on = kappa_3_batch(pnl, **_K, build_cache_updates=True)
    res_off, upd_off = kappa_3_batch(pnl, **_K, build_cache_updates=False)

    assert res_on == res_off              # scoring output unchanged
    assert upd_off == {}                  # nothing built when disabled
    assert set(upd_on) == {1, 2}          # built when enabled
    for uid in (1, 2):
        _fp, kv = upd_on[uid]
        assert kv == res_on[uid]          # cache copy mirrors the result exactly


def test_kappa3_lookback_excludes_old_observations():
    """kappa_3 applies `lookback` as an explicit window: observations older than
    the last `lookback` ns are dropped. Here the old obs would satisfy
    min_realized_observations but the windowed recent obs alone do not, so book0
    resolves to None — proving the old observations were excluded."""
    k = dict(_K)
    k["lookback"] = 5_000_000          # 5e6 ns window
    k["min_realized_observations"] = 3
    series = {
        0: {0: 1.0}, 1_000_000: {0: 1.0}, 2_000_000: {0: 1.0}, 3_000_000: {0: 1.0},  # 4 old
        100_000_000: {0: 2.0}, 101_000_000: {0: 2.0},                                # 2 recent (in window)
    }
    out = kappa_3(1, series, **k)
    assert out is None or out.get("books", {}).get(0) is None
    # sanity: with a permissive window (all 6 obs) book0 IS valid — confirms the
    # None above is caused by windowing, not by the data itself.
    k_wide = dict(k)
    k_wide["lookback"] = 10800_000_000_000
    out_wide = kappa_3(1, series, **k_wide)
    assert out_wide is not None and out_wide["books"][0] is not None
