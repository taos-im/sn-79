# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Q3 Stage A tests: the shadow must replicate the history accounting exactly.

The core property (in-process, no subprocess): a ShadowState built from a
plainified+pickled+rebuilt snapshot of a 'main' state, fed the SAME rounds via
the SAME trade.update_trade_volumes, produces the SAME parity digest. Plus the
plumbing: digest determinism/sensitivity/order-invariance, plainify round-trip
of lambda-defaultdicts + deques, and frame IO over a socketpair.
"""
import pickle
import socket
import struct
from collections import defaultdict, deque
from types import SimpleNamespace

from taos.im.validator.engines import NormalizedTradeEvent
from taos.im.validator.scoring_shadow import (
    ShadowState,
    compute_parity_components,
    _plainify,
    _rebuild_structs,
    _recv_frame,
    _send_frame,
    _STRUCT_NAMES,
    compute_parity_digest,
)
from taos.im.validator.trade import update_trade_volumes

_BOOKS = 2
_UIDS = 4
_S = 1_000_000_000


def _scoring_cfg():
    """Full scoring/rewarding dict as build_scoring_config produces (serial
    kappa so tests stay loky-free)."""
    return {
        'scoring': {
            'kappa': {'weight': 0.7, 'normalization_min': -2.5, 'normalization_max': 2.5,
                      'min_lookback': 1, 'lookback': 10800 * _S, 'min_realized_observations': 3,
                      'parallel_workers': 0, 'reward_cores': [], 'tau': 0.0, 'pnl_impact': 0.5},
            'pnl': {'weight': 0.3, 'lookback': 10800 * _S,
                    'normalization': {'min_daily_return': -0.05, 'max_daily_return': 0.05}},
            'gentrx': {'simulation_share': 0.0, 'ema_alpha': 0.1},
            'activity': {'capital_turnover_cap': 10.0, 'trade_volume_sampling_interval': 600 * _S,
                         'trade_volume_assessment_period': 3600 * _S, 'decay_grace_period': 600 * _S,
                         'impact': 0.33, 'decay_rate': 1.0},
            'max_inactive_books_ratio': 0.375,
            'interval': 5 * _S,
        },
        'rewarding': {'seed': 42, 'pareto': {'shape': 1.0, 'scale': 1.0}},
    }


def _sim_cfg(books=_BOOKS):
    return {'miner_wealth': 50000.0, 'publish_interval': _S, 'volumeDecimals': 4,
            'grace_period': 600 * _S, 'book_count': books}


def _fresh_main(uids=_UIDS, books=_BOOKS):
    """Minimal validator-shaped container with main's exact shell shapes."""
    m = SimpleNamespace()
    m.trade_volumes = {
        u: {b: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}} for b in range(books)}
        for u in range(uids)
    }
    for name in ("volume_sums", "maker_volume_sums", "taker_volume_sums",
                 "self_volume_sums", "fee_sums", "roundtrip_volume_sums", "agent_pnl_by_book"):
        setattr(m, name, defaultdict(lambda: defaultdict(float)))
    m.agent_pnl_total = defaultdict(float)
    m.roundtrip_volumes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    m.realized_pnl_history = defaultdict(lambda: defaultdict(dict))
    m.open_positions = defaultdict(lambda: defaultdict(lambda: {'longs': deque(), 'shorts': deque()}))
    m.inventory_history = {u: {} for u in range(uids)}
    m.initial_balances = {
        u: {b: {'BASE': 100.0, 'QUOTE': 10000.0, 'WEALTH': 50000.0} for b in range(books)}
        for u in range(uids)
    }
    m.recent_trades = {b: [] for b in range(books)}
    m.recent_miner_trades = {u: {b: [] for b in range(books)} for u in range(uids)}
    m.kappa_values = {u: {} for u in range(uids)}
    m.activity_factors = {u: {b: 0.0 for b in range(books)} for u in range(uids)}
    m.pnl_factors = {u: {b: 1.0 for b in range(books)} for u in range(uids)}
    m._last_prune_timestamp = None
    m.step = 0
    m.effective_max_uids = uids
    m.reward_cores = []
    # Full config surface (values mirror _scoring_cfg) so build_scoring_config
    # on this duck produces exactly _scoring_cfg — send_init reads it in the E2E.
    m.config = SimpleNamespace(
        scoring=SimpleNamespace(
            kappa=SimpleNamespace(
                weight=0.7, normalization_min=-2.5, normalization_max=2.5,
                min_lookback=1, lookback=10800 * _S, min_realized_observations=3,
                parallel_workers=0, tau=0.0, pnl=SimpleNamespace(impact=0.5),
            ),
            pnl=SimpleNamespace(
                weight=0.3, lookback=10800 * _S,
                normalization=SimpleNamespace(min_daily_return=-0.05, max_daily_return=0.05),
            ),
            gentrx=SimpleNamespace(simulation_share=0.0, ema_alpha=0.1),
            activity=SimpleNamespace(
                capital_turnover_cap=10.0,
                trade_volume_sampling_interval=600 * _S,
                trade_volume_assessment_period=3600 * _S,
                decay_grace_period=600 * _S, impact=0.33, decay_rate=1.0,
            ),
            max_inactive_books=0.375,
            interval=5 * _S,
        ),
        rewarding=SimpleNamespace(seed=42, pareto=SimpleNamespace(shape=1.0, scale=1.0)),
    )
    m.simulation = SimpleNamespace(
        volumeDecimals=4, miner_wealth=50000.0, publish_interval=_S,
        grace_period=600 * _S, book_count=books,
    )
    m.pagerduty_alert = lambda *a, **k: None
    return m


def _state(ts, trades):
    """Duck-typed state: .timestamp/.notices/.books/.accounts as trade.py reads them.
    trades: list of (taker_uid, maker_uid, book, qty, price, side)."""
    notices = defaultdict(list)
    for taker, maker, book, qty, price, side in trades:
        ev = NormalizedTradeEvent(
            book_id=book, quantity=qty, price=price, side=side,
            maker_uid=maker, taker_uid=taker, maker_fee=qty * price * 0.001,
            taker_fee=qty * price * 0.002, timestamp=ts,
        ).to_notice_dict()
        notices[taker].append(ev)
        if maker != taker:
            notices[maker].append(ev)
    return SimpleNamespace(timestamp=ts, notices=dict(notices), books={}, accounts={})


_ROUNDS = [
    _state(1 * _S, [(1, 2, 0, 3.0, 100.0, 0)]),
    _state(2 * _S, [(2, 1, 0, 1.5, 101.0, 1), (3, 1, 1, 2.0, 55.0, 0)]),
    _state(3 * _S, []),
    _state(4 * _S, [(1, 3, 1, 2.0, 56.0, 1)]),
]


def _snapshot_to_shadow(main):
    """The exact INIT path: plainify -> pickle -> unpickle -> rebuild -> ShadowState."""
    parts = {}
    for name in _STRUCT_NAMES:
        parts[name] = pickle.loads(pickle.dumps(_plainify(getattr(main, name)), protocol=5))
    knobs = {
        "_last_prune_timestamp": main._last_prune_timestamp,
        "step": main.step,
        "effective_max_uids": main.effective_max_uids,
        "config": main.config,
        "simulation": main.simulation,
        "scoring_config": pickle.loads(pickle.dumps(_scoring_cfg())),
        "simulation_config": _sim_cfg(),
        "scoring_interval": 5 * _S,
        "validator_uid": 0,
    }
    return ShadowState(_rebuild_structs(parts), knobs)


def _main_score(main, sim_ts, deregs=()):
    """Replica of get_rewards' trading path (score_uids + Pareto) on `main` —
    what the validator computes and records for the shadow comparison."""
    from taos.im.validator.reward import distribute_rewards, score_uids

    all_uids = list(range(main.effective_max_uids))
    vd = {
        'kappa_values': main.kappa_values,
        'kappa_cache': {},
        'activity_factors': main.activity_factors,
        'pnl_factors': main.pnl_factors,
        'roundtrip_volumes': main.roundtrip_volumes,
        'realized_pnl_history': main.realized_pnl_history,
        'config': _scoring_cfg(),
        'simulation_config': _sim_cfg(),
        'simulation_timestamp': sim_ts,
        'uids': all_uids,
        'deregistered_uids': list(deregs),
        'device': 'cpu',
        'gentrx_scores': {},
        'gentrx_ema': {},
    }
    trading, _g = score_uids(vd)
    return [float(x) for x in distribute_rewards([trading[u] for u in all_uids], _scoring_cfg()).tolist()]


def test_shadow_replicates_main_exactly():
    """Snapshot mid-stream, then apply identical rounds to both — digests match
    at every step (the Stage-A replication property)."""
    main = _fresh_main()
    update_trade_volumes(main, _ROUNDS[0])  # pre-snapshot history

    shadow = _snapshot_to_shadow(main)
    assert compute_parity_digest(shadow) == compute_parity_digest(main)

    for st in _ROUNDS[1:]:
        update_trade_volumes(main, st)
        update_trade_volumes(shadow, st)
        assert compute_parity_digest(shadow) == compute_parity_digest(main), f"diverged at ts={st.timestamp}"


def test_digest_sensitivity_and_order_invariance():
    a, b = _fresh_main(), _fresh_main()
    update_trade_volumes(a, _ROUNDS[0])
    update_trade_volumes(b, _ROUNDS[0])
    assert compute_parity_digest(a) == compute_parity_digest(b)

    # sensitivity: one extra trade changes the digest
    update_trade_volumes(b, _ROUNDS[1])
    assert compute_parity_digest(a) != compute_parity_digest(b)

    # insertion-order invariance of the projection
    c = _fresh_main()
    c.agent_pnl_total[3] = 1.5
    c.agent_pnl_total[1] = 2.5
    d = _fresh_main()
    d.agent_pnl_total[1] = 2.5
    d.agent_pnl_total[3] = 1.5
    assert compute_parity_digest(c) == compute_parity_digest(d)


def test_plainify_rebuild_preserves_defaultdict_behaviour():
    main = _fresh_main()
    update_trade_volumes(main, _ROUNDS[0])
    shadow = _snapshot_to_shadow(main)
    # values equal
    assert dict(shadow.agent_pnl_total) == dict(main.agent_pnl_total)
    # auto-create shells work (load-bearing defaultdict behaviour)
    shadow.volume_sums[99][7] += 1.0
    assert shadow.volume_sums[99][7] == 1.0
    pos = shadow.open_positions[98][1]
    assert isinstance(pos['longs'], deque)
    shadow.roundtrip_volumes[97][0][123] += 2.0
    assert shadow.roundtrip_volumes[97][0][123] == 2.0


def test_shadow_score_matches_main_scoring():
    """Shadow-reward parity in-process: snapshot -> identical rounds -> the
    shadow's shadow_score equals main's score_uids+Pareto trading list, and the
    in-place factor mutations keep BOTH history digests and a SECOND scoring
    round in lockstep."""
    from taos.im.validator.scoring_shadow import shadow_score

    main = _fresh_main()
    update_trade_volumes(main, _ROUNDS[0])
    shadow = _snapshot_to_shadow(main)

    for st in _ROUNDS[1:]:
        update_trade_volumes(main, st)
        update_trade_volumes(shadow, st)

    sim_ts = 5 * _S
    main_scores = _main_score(main, sim_ts)
    shadow_scores = shadow_score(shadow, sim_ts, [])['trading']
    assert shadow_scores == main_scores

    # factors mutated in place on both sides -> still in lockstep afterwards
    assert compute_parity_digest(shadow) == compute_parity_digest(main)
    st = _state(6 * _S, [(2, 3, 0, 1.0, 99.0, 0)])
    update_trade_volumes(main, st)
    update_trade_volumes(shadow, st)
    assert shadow_score(shadow, 10 * _S, [])['trading'] == _main_score(main, 10 * _S)


def test_frame_io_roundtrip():
    a, b = socket.socketpair()
    try:
        _send_frame(a, ("state", (123, b"\x00\x01payload")))
        kind, payload = _recv_frame(b)
        assert kind == "state" and payload == (123, b"\x00\x01payload")
        # oversized-safe framing: 1MB payload
        big = b"x" * (1 << 20)
        _send_frame(b, ("init_part", ("trade_volumes", big)))
        kind, (name, blob) = _recv_frame(a)
        assert kind == "init_part" and name == "trade_volumes" and blob == big
    finally:
        a.close()
        b.close()


def _raw_state(ts, trades):
    """msgpack bytes shaped exactly as parse_dict consumes (lenient schema)."""
    import msgpack
    st = _state(ts, trades)
    return msgpack.packb(
        {"timestamp": ts, "books": {}, "accounts": {}, "notices": st.notices},
        use_bin_type=True,
    )


def test_two_sided_ring_order_independence():
    """The child normally computes AHEAD of main (applies on tee; main's reward
    runs seconds later behind the lock) — a parity/scores frame arriving before
    main records must be stashed and compared when main's side lands, and vice
    versa. This was a live bug: the reader popped an empty ring and dropped
    every result at DEBUG."""
    from taos.im.validator.scoring_shadow import ScoringShadow

    sh = ScoringShadow(parity_ns=10)
    # child first, main second
    sh._stash(sh._pending_child_parity, 100, "abc", 32)
    sh.record_main_digest(100, "abc")
    assert sh.matches == 1 and sh.mismatches == 0
    # main first, child second (reader path simulated directly)
    sh.record_main_digest(200, "xyz")
    mine = sh._ring.pop(200, None)
    assert mine == "xyz"
    sh._compare_parity(200, mine, "xyz")
    assert sh.matches == 2
    # mismatch is loud
    sh._stash(sh._pending_child_parity, 300, "aaa", 32)
    sh.record_main_digest(300, "bbb")
    assert sh.mismatches == 1
    # scores: child-first order
    sh._stash(sh._pending_child_scores, 400, [1.0, 2.0], 8)
    sh.on_main_scored(400, 400, [], [1.0, 2.0])
    assert sh.score_matches == 1 and sh.score_mismatches == 0


def test_sim_start_shift_parity():
    """Simulation restart: main and shadow run the SAME shared shift
    (trade.shift_simulation_histories) and stay digest-identical across the
    time-base change, including after post-restart rounds."""
    from taos.im.validator.trade import shift_simulation_histories

    def _shift(s):
        shift_simulation_histories(
            s, old_ts=4 * _S, new_ts=0,
            book_count=_BOOKS, volume_decimals=4, lookback=10800 * _S,
            volume_assessment_period=3600 * _S, miner_wealth=50000.0,
            effective_max_uids=_UIDS,
        )

    main = _fresh_main()
    for st in _ROUNDS:
        update_trade_volumes(main, st)
    shadow = _snapshot_to_shadow(main)

    _shift(main)
    _shift(shadow)
    assert compute_parity_digest(shadow) == compute_parity_digest(main)

    # post-restart rounds (new time base) still in lockstep
    st = _state(1 * _S, [(1, 2, 0, 2.0, 100.0, 0)])
    update_trade_volumes(main, st)
    update_trade_volumes(shadow, st)
    assert compute_parity_digest(shadow) == compute_parity_digest(main)


def test_reset_agent_histories_parity():
    """Dereg reset: shared zeroing keeps both sides digest-identical, and the
    reset uid's structures still work for subsequent rounds."""
    from taos.im.validator.trade import reset_agent_histories

    main = _fresh_main()
    for st in _ROUNDS:
        update_trade_volumes(main, st)
    shadow = _snapshot_to_shadow(main)

    book_ids = list(range(_BOOKS))
    reset_agent_histories(main, 1, book_ids)
    reset_agent_histories(shadow, 1, book_ids)
    assert compute_parity_digest(shadow) == compute_parity_digest(main)

    st = _state(6 * _S, [(1, 3, 0, 1.0, 100.0, 0)])
    update_trade_volumes(main, st)
    update_trade_volumes(shadow, st)
    assert compute_parity_digest(shadow) == compute_parity_digest(main)


def test_cutover_default_on_with_optouts(monkeypatch):
    """Cutover is the DEFAULT; SCORING_PROC=0 or a .no_scoring_proc sentinel
    disables it (rollback switches); SCORING_PROC=1 forces on over the sentinel."""
    import taos.im.validator.scoring_shadow as ss

    monkeypatch.delenv("SCORING_PROC", raising=False)
    monkeypatch.setattr(ss.Path, "exists", lambda self: False)
    assert ss.cutover_enabled() is True  # default ON

    monkeypatch.setenv("SCORING_PROC", "0")
    assert ss.cutover_enabled() is False  # env rollback

    monkeypatch.delenv("SCORING_PROC", raising=False)
    monkeypatch.setattr(ss.Path, "exists", lambda self: str(self).endswith(".no_scoring_proc"))
    assert ss.cutover_enabled() is False  # sentinel rollback

    monkeypatch.setenv("SCORING_PROC", "1")
    assert ss.cutover_enabled() is True  # env force-on beats sentinel


def test_cutover_request_scores_roundtrip():
    """Cutover flow with the REAL child: INIT -> states -> request_scores at the
    boundary returns the full result (trading == main's replica computation,
    factors + gentrx + ema present) and the child continues applying afterwards."""
    import time as _time

    from taos.im.validator.scoring_shadow import ScoringShadow

    main = _fresh_main()
    update_trade_volumes(main, _state(1 * _S, [(1, 2, 0, 3.0, 100.0, 0)]))
    main._shadow_applied_ts = 1 * _S

    shadow = ScoringShadow(parity_ns=5 * _S)
    shadow.start()
    try:
        shadow.send_init(main).result(timeout=60)
        for ts, trades in [
            (2 * _S, [(2, 1, 0, 1.5, 101.0, 1)]),
            (3 * _S, [(3, 1, 1, 2.0, 55.0, 0)]),
            (4 * _S, []),
            (5 * _S, [(1, 3, 1, 2.0, 56.0, 1)]),  # boundary
        ]:
            update_trade_volumes(main, _state(ts, trades))
            shadow.tee(_raw_state(ts, trades), ts)

        result = shadow.request_scores(5 * _S, 5 * _S, [], {}, {}, timeout=60)
        assert result is not None, "child did not return scores"
        assert result['trading'] == _main_score(main, sim_ts=5 * _S)
        assert result['gentrx'] == [0.0] * _UIDS  # no gentrx inputs shipped
        assert set(result['factors']) == {'kappa_values', 'activity_factors', 'pnl_factors'}
        assert result['gentrx_ema'] == {}

        # child continues applying after the boundary release
        update_trade_volumes(main, _state(6 * _S, [(2, 3, 0, 1.0, 99.0, 0)]))
        shadow.record_main_digest(10 * _S, "")  # placeholder slot, not compared
        shadow.tee(_raw_state(6 * _S, [(2, 3, 0, 1.0, 99.0, 0)]), 6 * _S)
        _time.sleep(1.0)
        assert shadow.is_alive()
    finally:
        shadow.stop()


def _dead_child(sock, cores, parity_ns):
    sock.close()


def test_request_scores_dead_child_returns_none_and_respawns():
    from taos.im.validator.scoring_shadow import ScoringShadow

    sh = ScoringShadow(child_fn=_dead_child)
    sh.start()
    import time as _time
    deadline = _time.monotonic() + 10
    while sh.is_alive() and _time.monotonic() < deadline:
        _time.sleep(0.05)
    assert sh.request_scores(1, 1, []) is None  # dead -> None + respawn armed
    assert sh.initialized is False
    sh.stop()


def test_eager_scoring_computes_at_boundary_without_score_at():
    """Eager path: inputs shipped at tee time -> the child scores the boundary
    immediately (no score_at needed) and the result matches the pull path's."""
    import time as _time

    from taos.im.validator.scoring_shadow import ScoringShadow

    main = _fresh_main()
    update_trade_volumes(main, _state(1 * _S, [(1, 2, 0, 3.0, 100.0, 0)]))
    main._shadow_applied_ts = 1 * _S

    shadow = ScoringShadow(parity_ns=5 * _S)
    shadow.start()
    try:
        shadow.send_init(main).result(timeout=60)
        rounds = [
            (2 * _S, [(2, 1, 0, 1.5, 101.0, 1)]),
            (3 * _S, [(3, 1, 1, 2.0, 55.0, 0)]),
            (4 * _S, []),
        ]
        for ts, trades in rounds:
            update_trade_volumes(main, _state(ts, trades))
            shadow.tee(_raw_state(ts, trades), ts)

        # boundary round: ship inputs at tee time, THEN the state
        boundary = (5 * _S, [(1, 3, 1, 2.0, 56.0, 1)])
        update_trade_volumes(main, _state(*boundary))
        shadow.tee(_raw_state(*boundary), 5 * _S)
        shadow.tee_score_inputs(5 * _S, 5 * _S, [], {}, {})

        assert shadow.eager_inputs_for(5 * _S) is not None
        # give the child a moment to compute proactively, then collect
        deadline = _time.monotonic() + 60
        result = None
        while result is None and _time.monotonic() < deadline:
            result = shadow.request_scores(5 * _S, 5 * _S, [], eager=True, timeout=10)
        assert result is not None
        assert result['trading'] == _main_score(main, sim_ts=5 * _S)
    finally:
        shadow.stop()


def test_reset_derived_from_state_parity():
    """Reset ordering fix: both sides derive RDRA resets from the state and
    apply them right after that round's update — identical position, no frame.
    Exercises the REAL child apply path (apply_state_bytes) against main's
    _reward-order replica; digests must match through and after the reset."""
    import msgpack

    from taos.im.validator.scoring_shadow import apply_state_bytes
    from taos.im.validator.trade import collect_reset_uids, reset_agent_histories

    VUID = 0
    main = _fresh_main()
    update_trade_volumes(main, _ROUNDS[0])
    update_trade_volumes(main, _ROUNDS[1])
    shadow = _snapshot_to_shadow(main)

    # round with trades for uid 1 AND an RDRA reset of uid 1 in the same state
    st = _state(5 * _S, [(1, 2, 0, 1.0, 100.0, 0)])
    st.notices[VUID] = [{'y': 'RDRA', 'r': [{'u': True, 'a': 1}]}]
    raw = msgpack.packb(
        {"timestamp": 5 * _S, "books": {}, "accounts": {}, "notices": st.notices},
        use_bin_type=True,
    )

    # main: _reward order = update THEN reset (from the same state)
    update_trade_volumes(main, st)
    pending, failed = collect_reset_uids(st, VUID)
    assert pending == {1} and not failed
    for uid in sorted(pending):
        reset_agent_histories(main, uid, list(range(_BOOKS)))

    # child: the real apply path derives + applies the same reset internally
    apply_state_bytes(shadow, raw)

    assert compute_parity_digest(shadow) == compute_parity_digest(main)

    # post-reset rounds remain in lockstep (uid 1 trading again)
    st2 = _state(6 * _S, [(1, 3, 0, 2.0, 101.0, 1)])
    update_trade_volumes(main, st2)
    raw2 = msgpack.packb(
        {"timestamp": 6 * _S, "books": {}, "accounts": {}, "notices": st2.notices},
        use_bin_type=True,
    )
    apply_state_bytes(shadow, raw2)
    assert compute_parity_digest(shadow) == compute_parity_digest(main)


def test_e2e_spawned_child_parity_match():
    """Full path with the REAL child process: INIT snapshot -> teed raw state
    frames -> child applies via the real update_trade_volumes -> parity digest
    returned -> owner compares against main's own digest = MATCH. This is the
    exact production flow minus the live simulator."""
    import time as _time

    from taos.im.validator.scoring_shadow import ScoringShadow

    P = 5 * _S  # parity every 5 sim-s
    main = _fresh_main()
    update_trade_volumes(main, _state(1 * _S, [(1, 2, 0, 3.0, 100.0, 0)]))
    main._shadow_applied_ts = 1 * _S

    shadow = ScoringShadow(parity_ns=P)
    shadow.start()
    try:
        shadow.send_init(main).result(timeout=60)

        rounds = [
            (2 * _S, [(2, 1, 0, 1.5, 101.0, 1)]),
            (3 * _S, [(3, 1, 1, 2.0, 55.0, 0)]),
            (4 * _S, []),
            (5 * _S, [(1, 3, 1, 2.0, 56.0, 1)]),  # parity + scoring boundary (5e9)
            (6 * _S, [(2, 3, 0, 1.0, 99.0, 0)]),  # applied only after score_at releases
        ]
        for ts, trades in rounds:
            update_trade_volumes(main, _state(ts, trades))
            if ts % P == 0:
                shadow.record_main_digest(ts, compute_parity_components(main))
            shadow.tee(_raw_state(ts, trades), ts)
            if ts % (5 * _S) == 0:
                # main scores at the boundary BEFORE applying later rounds
                # (production does this under _reward_lock) and releases the
                # child, which is holding applies at the same boundary.
                shadow.on_main_scored(ts, ts, [], _main_score(main, sim_ts=ts))

        deadline = _time.monotonic() + 90
        while _time.monotonic() < deadline and (
            (shadow.matches + shadow.mismatches) == 0
            or (shadow.score_matches + shadow.score_mismatches) == 0
        ):
            _time.sleep(0.2)
        assert shadow.mismatches == 0, "history digests diverged"
        assert shadow.matches >= 1, "no parity result within timeout (child dead?)"
        assert shadow.score_mismatches == 0, "trading scores diverged"
        assert shadow.score_matches >= 1, "no scores result within timeout"
    finally:
        shadow.stop()

def _light_fields():
    """Main-only save fields as persistence.build_save_light_fields ships them."""
    return {
        "step": 7,
        "simulation_timestamp": 4 * _S,
        "hotkeys": [f"hk{u}" for u in range(_UIDS)],
        "scores": [0.1, 0.2, 0.3, 0.4],
        "gentrx_scores": [0.0] * _UIDS,
        "unnormalized_scores": {u: 0.0 for u in range(_UIDS)},
        "deregistered_uids": [],
        "trading_score_ema": {u: 0.1 * (u + 1) for u in range(_UIDS)},
        "trading_score_ema_ts": 3 * _S,
        "trading_score_ema_n": {u: u + 1 for u in range(_UIDS)},
        "miner_stats": {1: {"requests": 3, "timeouts": 0, "failures": 0,
                            "rejections": 0, "call_time": [0.5]}},
    }


def _mp_norm(obj):
    """Normalize through a msgpack roundtrip (int keys, tuples->lists) so both
    sides of a comparison have identical shapes."""
    import msgpack

    return msgpack.unpackb(msgpack.packb(obj, use_bin_type=True), raw=False, strict_map_key=False)


def test_child_save_validator_state_roundtrip(tmp_path):
    """Save offload: the file the child writes must be exactly what main's own
    save path would produce — same keys in the same order, heavy subtrees equal
    to the persistence snapshot functions applied to the same state, light
    fields passed through verbatim."""
    import os

    import msgpack

    from taos.im.validator import persistence as p
    from taos.im.validator.scoring_shadow import child_save_validator_state

    main = _fresh_main()
    for st in _ROUNDS:
        update_trade_volumes(main, st)
    shadow = _snapshot_to_shadow(main)
    assert compute_parity_digest(shadow) == compute_parity_digest(main)

    light = _light_fields()
    path = str(tmp_path / "validator_state.mp")
    nbytes = child_save_validator_state(shadow, light, path)
    assert nbytes == os.path.getsize(path)
    assert not os.path.exists(f"{path}.tmp.shadow")

    saved = msgpack.unpackb(open(path, "rb").read(), raw=False, strict_map_key=False)
    assert list(saved.keys()) == [
        "step", "simulation_timestamp", "hotkeys", "scores", "gentrx_scores",
        "activity_factors", "pnl_factors", "inventory_history", "kappa_values",
        "realized_pnl_history", "open_positions", "unnormalized_scores",
        "deregistered_uids", "trading_score_ema", "trading_score_ema_ts",
        "trading_score_ema_n", "trade_volumes", "roundtrip_volumes",
        "volume_sums", "maker_volume_sums", "taker_volume_sums",
        "self_volume_sums", "roundtrip_volume_sums", "miner_stats",
    ]

    for key in ("step", "simulation_timestamp", "hotkeys", "scores",
                "gentrx_scores", "deregistered_uids",
                "trading_score_ema", "trading_score_ema_ts", "trading_score_ema_n"):
        assert saved[key] == _mp_norm(light[key])
    assert saved["unnormalized_scores"] == _mp_norm(light["unnormalized_scores"])
    assert saved["miner_stats"] == _mp_norm(light["miner_stats"])

    vols = p.snapshot_volume_sums(main)
    heavy = {
        "inventory_history": p.snapshot_inventory_history(main),
        "realized_pnl_history": p.snapshot_realized_pnl_history(main),
        "open_positions": p.snapshot_open_positions(main),
        "trade_volumes": p.snapshot_trade_volumes(main),
        "roundtrip_volumes": p.snapshot_roundtrip_volumes(main),
        "kappa_values": main.kappa_values,
        "activity_factors": main.activity_factors,
        "pnl_factors": main.pnl_factors,
        **vols,
    }
    for key, expected in heavy.items():
        assert saved[key] == _mp_norm(expected), f"subtree {key} diverged"


def test_e2e_spawned_child_save(tmp_path):
    """Full save-offload path with the REAL child: INIT -> teed states ->
    request_save -> child writes the file from its replica -> save_done resolves
    the future -> file content matches main's snapshot of the same rounds."""
    import msgpack

    from taos.im.validator import persistence as p
    from taos.im.validator.scoring_shadow import ScoringShadow

    main = _fresh_main()
    update_trade_volumes(main, _state(1 * _S, [(1, 2, 0, 3.0, 100.0, 0)]))
    main._shadow_applied_ts = 1 * _S

    shadow = ScoringShadow(parity_ns=5 * _S)
    shadow.start()
    try:
        shadow.send_init(main).result(timeout=60)
        # non-boundary rounds only (no score pause in play)
        for ts, trades in [(2 * _S, [(2, 1, 0, 1.5, 101.0, 1)]),
                           (3 * _S, [(3, 1, 1, 2.0, 55.0, 0)]),
                           (4 * _S, [(1, 3, 1, 2.0, 56.0, 1)])]:
            update_trade_volumes(main, _state(ts, trades))
            shadow.tee(_raw_state(ts, trades), ts)

        path = str(tmp_path / "validator_state.mp")
        req = shadow.request_save(path, _light_fields())
        assert req is not None
        expect_ts, fut = req
        assert expect_ts == 4 * _S
        result = fut.result(timeout=60)
        assert result is not None, "child save failed"
        nbytes, _io_secs = result
        assert nbytes == (tmp_path / "validator_state.mp").stat().st_size

        saved = msgpack.unpackb(open(path, "rb").read(), raw=False, strict_map_key=False)
        assert saved["step"] == 7
        assert saved["realized_pnl_history"] == _mp_norm(p.snapshot_realized_pnl_history(main))
        assert saved["trade_volumes"] == _mp_norm(p.snapshot_trade_volumes(main))
        assert saved["open_positions"] == _mp_norm(p.snapshot_open_positions(main))
    finally:
        shadow.stop()


def test_request_save_unavailable_returns_none():
    """Dead/uninitialized child or nothing teed -> None (caller falls back)."""
    from taos.im.validator.scoring_shadow import ScoringShadow

    sh = ScoringShadow.__new__(ScoringShadow)
    sh._proc = None
    sh.initialized = False
    sh._last_teed_ts = 0
    assert sh.request_save("/tmp/x", {}) is None
