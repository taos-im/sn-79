# SPDX-License-Identifier: MIT
"""A slot that changes hands starts from nothing in every structure the pay path reads.

The per-uid resets were written as lists of names, so each new piece of state had to be added to them by hand. The
pool's smoothed making term and share were not, and a new occupant inherited the departed miner's making share (fixed
28 Sep 2026). This test does not trust the lists. It discovers every validator attribute the pay path touches, from
the source of the pay-path modules, and requires each to be classified here:

- CLEARED: per-uid state. The test fills it for two uids, runs the full reset path for one (the validator's
  handle_deregistration, then trade.reset_agent_histories as apply_resets runs it), and asserts that nothing of the
  reset uid survives anywhere in the structure, including as a counterparty key, while the other uid is untouched.
- EXEMPT: not per-uid state carried across rounds, with the reason.

A new attribute in the pay path that is in neither fails the test, which is the point: whoever adds state decides,
in review, whether a reset must clear it.
"""
import collections
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taos.im.neurons.validator as vmod  # noqa: E402
import taos.im.validator.trade as trade  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAY_PATH = ["taos/im/validator/reward.py", "taos/im/validator/debeta.py", "taos/im/validator/trade.py",
            "taos/im/validator/forward.py"]
S = 12345.0
BOOKS = [0, 1]
GONE, KEPT = 5, 6

# per-uid state, keyed by uid at the top level ({uid: ...}); counterparty maps also key the uid one level down
CLEARED = {
    # scorer inputs (the scoring child's replicated state)
    "trade_volumes", "volume_sums", "maker_volume_sums", "taker_volume_sums", "self_volume_sums",
    "roundtrip_volumes", "roundtrip_volume_sums", "realized_pnl_history", "agent_pnl_by_book", "agent_pnl_total",
    "open_positions", "inventory_history", "initial_balances", "recent_miner_trades",
    "kappa_values", "kappa_cache", "activity_factors", "pnl_factors",
    "capture_buy_sums", "capture_sell_sums", "realized_buy_sums", "realized_sell_sums", "debeta_mtm", "debeta_invsum", "debeta_heldn", "debeta_heldinv",
    "debeta_helddrift", "debeta_notional", "debeta_capbuy_hist", "debeta_capsell_hist", "debeta_realbuy_hist", "debeta_realsell_hist", "debeta_mtm_hist",
    "debeta_invsum_hist", "debeta_heldn_hist", "debeta_heldinv_hist", "debeta_helddrift_hist",
    "debeta_notional_hist", "debeta_cp", "debeta_cp_hist",
    # main-side smoothing and presence carried across rounds
    "_trading_score_ema", "_trading_score_ema_n", "_debeta_pool_ema", "_gentrx_ema", "gentrx_scores",
    "miner_presence", "_trading_score_ema_pre",
}
# book-first per-uid state ({book: {uid: ...}})
CLEARED_BOOK_FIRST = {"debeta_inv"}
# class-first per-uid state ({asset class: {maker uid: {taker uid: ...}}}): the counterparty rows kept per class
CLEARED_CLASS_FIRST = {"debeta_cpc", "debeta_cpc_hist"}
EXEMPT = {
    # configuration, handles and scalars
    "config": "configuration", "engine": "engine handle", "device": "torch device", "simulation": "config",
    "simulation_config": "the scoring child's copy of the simulation config (skill_bar reads its book count)",
    "simulation_timestamp": "clock", "effective_max_uids": "scalar", "reward_cores": "worker count",
    "subtensor": "chain handle", "metagraph": "chain view", "wallet": "wallet", "uid": "own uid",
    "deregistered_uids": "the reset queue itself",
    "step": "counter", "block": "chain height",
    "scores": "cleared by apply_resets (main-only bookkeeping); the moving average of pay",
    "unnormalized_scores": "cleared by apply_resets", "miner_stats": "cleared by apply_resets",
    # per-round outputs, rebuilt from the state above every round
    "debeta_detail": "per-round output", "debeta_alphas_by_book": "per-round output", "_debeta_last": "per-round",
    "debeta_floor": "per-round output", "debeta_w_make": "per-round output", "debeta_absent": "per-round, from presence",
    "_debeta_class_shares": "per-round output (the class-mixed pool shares, rebuilt from the accumulators)",
    "_debeta_class_summary": "per-round output (the per-class summary the report publishes)",
    "_debeta_class_details": "per-round output (the per-class per-uid detail the report publishes as agent_class_gauges)",
    "_debeta_class_map_cache": "layout cache (book to asset class), keyed by the simulation object, not by uid",
    "debeta_presence_shares": "per-round, from presence", "_trading_score_ema_ts": "a timestamp",
    # keyed by book or by trade id, not by uid
    "debeta_invn": "book-keyed", "debeta_invn_hist": "book-keyed", "debeta_drift": "book-keyed",
    "debeta_drift_hist": "book-keyed", "debeta_pfirst": "book-keyed", "debeta_plast": "book-keyed",
    "debeta_mark_state": "book-keyed", "debeta_capture_mid": "book-keyed",
    "_debeta_seen_tids": "trade-id ledger", "_volume_seen_tids": "trade-id ledger",
    "fee_sums": "not read by the pay path's scoring", "recent_trades": "book-keyed trade tape",
    "_gentrx": "object; its _scores are cleared by handle_deregistration",
    # runtime handles and flags of the query service, the reporter and load bookkeeping
    "_histories_rebased_at_load": "load counter", "_last_prune_timestamp": "clock", "_run_changes_recovered": "counter",
    "_pending_reward_tasks": "task queue", "_start_query_service": "method", "dendrite": "network handle",
    "get_extended_metagraph": "method", "hotkeys": "chain view", "miner_net_lock": "lock",
    "pagerduty_alert": "method", "query_ipc_executor": "executor", "query_notify_read": "IPC handle",
    "query_process": "process handle", "querying": "flag", "request_mem": "IPC handle", "request_queue": "IPC handle",
    "response_mem": "IPC handle", "response_queue": "IPC handle", "should_block_queries": "flag",
}
# counterparty maps: the uid is also a key one level down ({maker: {taker: ...}})
COUNTERPARTY = {"debeta_cp", "debeta_cp_hist"}


def _discovered():
    names = set()
    pat = re.compile(r"(?:\bself\.|getattr\((?:self|validator|v), *['\"])([A-Za-z_][A-Za-z0-9_]*)")
    for rel in PAY_PATH:
        src = open(os.path.join(ROOT, rel)).read()
        names |= set(pat.findall(src))
    return names


def _deep():
    return collections.defaultdict(_deep)


def _filled(counterparty=False):
    d = _deep()
    for u in (GONE, KEPT):
        for b in BOOKS:
            d[u][b] = S
    if counterparty:
        d[7][GONE] = S       # the reset uid as a counterparty of another maker
        d[7][KEPT] = S
    return d


def _book_first():
    d = _deep()
    for b in BOOKS:
        d[b][GONE] = S
        d[b][KEPT] = S
    return d


def _class_first():
    return {0: _filled(True), 1: _filled(True)}


def _has(name, o, uid):
    if isinstance(o, tuple):
        return any(_has(name, x, uid) for x in o if isinstance(x, dict))
    if not isinstance(o, dict):
        return False
    if name == "_debeta_pool_ema":
        return any(_has("", x, uid) for x in o.values())
    if name in CLEARED_BOOK_FIRST:
        return any(isinstance(x, dict) and uid in x and _has_sentinel(x[uid]) for x in o.values())
    if name in CLEARED_CLASS_FIRST:
        return any(_has("debeta_cp", x, uid) for x in o.values() if isinstance(x, dict))
    if uid in o and _has_sentinel(o[uid]):
        return True
    if name in COUNTERPARTY:
        return any(isinstance(x, dict) and uid in x and _has_sentinel(x[uid]) for x in o.values())
    return False


def _has_sentinel(v):
    if isinstance(v, float):
        return v == S
    if isinstance(v, dict):
        return any(_has_sentinel(x) for x in v.values())
    if isinstance(v, (list, tuple, collections.deque)):
        return any(_has_sentinel(x) for x in v)
    return False


def _validator():
    v = SimpleNamespace()
    for n in CLEARED:
        setattr(v, n, _filled(n in COUNTERPARTY))
    for n in CLEARED_BOOK_FIRST:
        setattr(v, n, _book_first())
    for n in CLEARED_CLASS_FIRST:
        setattr(v, n, _class_first())
    v._debeta_pool_ema = {"term": _filled(), "share": _filled()}
    v.miner_presence = {GONE: collections.deque([S, S]), KEPT: collections.deque([S, S])}
    v._trading_score_ema_pre = (dict(_filled()), dict(_filled()), None)
    v.engine = SimpleNamespace(handle_deregistration=lambda uid, old_coldkey=None: None)
    v._gentrx = None
    return v


def test_every_attribute_the_pay_path_touches_is_classified():
    unclassified = sorted(_discovered() - CLEARED - CLEARED_BOOK_FIRST - CLEARED_CLASS_FIRST - set(EXEMPT))
    unclassified = [n for n in unclassified if not n.startswith("__") and n not in {"prometheus", "logger"}]
    assert not unclassified, (
        f"pay-path attributes with no reset classification: {unclassified}. Add each to CLEARED (a reset must "
        f"clear it, and this test will check that it does) or to EXEMPT with the reason it is not per-uid state.")


def test_the_reset_path_leaves_nothing_of_the_departed_uid():
    v = _validator()
    vmod.Validator.handle_deregistration(v, GONE)
    trade.reset_agent_histories(v, GONE, BOOKS)
    survived = sorted(n for n in CLEARED | CLEARED_BOOK_FIRST | CLEARED_CLASS_FIRST if _has(n, getattr(v, n), GONE))
    assert not survived, f"the departed uid survives a reset in: {survived}"


def test_the_reset_leaves_other_uids_alone():
    v = _validator()
    vmod.Validator.handle_deregistration(v, GONE)
    trade.reset_agent_histories(v, GONE, BOOKS)
    lost = sorted(n for n in CLEARED | CLEARED_BOOK_FIRST | CLEARED_CLASS_FIRST if not _has(n, getattr(v, n), KEPT))
    assert not lost, f"a reset of one uid cleared another uid's state in: {lost}"
