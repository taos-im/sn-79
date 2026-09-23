"""G-I: de-beta exchange-mode fill coverage. Exchange AMM/pool fills reach de-beta only via the ET
settled-fill notices (pool fills carry Ma or Ta = None, aggressor = taker). They must feed the SAME
accumulators without crashing, deduped by trade id across redelivery + maker/taker duplication, ordered
by trade id. Sim mode is unaffected (no ET notices)."""
import sys
from collections import defaultdict

_REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)
from taos.im.validator.debeta import (  # noqa: E402
    accumulate_book_mtm,
    accumulate_counterparties,
    accumulate_book_capture,
    et_book_batches,
    balanced_reward,
)


def _mtm_state():
    return (
        defaultdict(lambda: defaultdict(float)),  # mtm
        defaultdict(lambda: defaultdict(float)),  # invsum
        {},  # invn
        defaultdict(lambda: defaultdict(float)),  # inv
        {},  # p_first
        {},  # p_last
    )


def test_counterparties_none_uid_does_not_crash():
    # pool fill: taker=agent 5, maker=None (the pool has no miner uid). int(None) must not raise.
    cp = {}
    accumulate_counterparties(cp, 0, [{"p": 100.0, "q": 3.0, "s": 0, "Ma": None, "Ta": 5}])
    assert cp == {}  # the None (pool) side is skipped: no miner-miner counterparty pair


def test_mtm_none_uid_does_not_crash_and_matches_minus_one():
    # a None maker-side must behave identically to -1 (both are "not a miner")
    trades_none = [{"p": 100.0, "q": 3.0, "s": 0, "Ma": None, "Ta": 5},
                   {"p": 101.0, "q": 3.0, "s": 0, "Ma": None, "Ta": 5}]
    trades_m1 = [{"p": 100.0, "q": 3.0, "s": 0, "Ma": -1, "Ta": 5},
                 {"p": 101.0, "q": 3.0, "s": 0, "Ma": -1, "Ta": 5}]
    a = _mtm_state()
    b = _mtm_state()
    accumulate_book_mtm(*a[:6], 0, trades_none)  # must not raise
    accumulate_book_mtm(*b[:6], 0, trades_m1)
    assert dict(a[3][0]) == dict(b[3][0])  # inv identical
    assert 5 in a[3][0] and -1 not in a[3][0] and None not in a[3][0]


def test_et_batches_dedup_group_and_order():
    seen = {}
    notices = {
        5: [{"y": "ET", "b": 0, "i": 2, "p": 101.0, "q": 3.0, "s": 0, "Ta": 5, "Ma": None},
            {"y": "ET", "b": 0, "i": 1, "p": 100.0, "q": 3.0, "s": 0, "Ta": 5, "Ma": None}],
        9: [{"y": "ET", "b": 0, "i": 1, "p": 100.0, "q": 3.0, "s": 0, "Ta": 5, "Ma": None},  # redelivered dup
            {"y": "OTHER", "b": 0, "i": 99}],  # non-ET ignored
    }
    batches = et_book_batches(notices, seen, ts=1000)
    assert [b["i"] for b in batches[0]] == [1, 2]  # deduped (i=1 once) + ordered by trade id
    assert et_book_batches(notices, seen, ts=2000) == {}  # all tids already seen -> nothing new


def test_pool_taker_earns_no_making():
    # a one-sided pool-fill taker (buys only) must earn ~0 balanced (two-sided) making
    seen = {}
    notices = {5: [{"y": "ET", "b": 0, "i": i, "p": 100.0 + i, "q": 3.0, "s": 0, "Ta": 5, "Ma": None}
                   for i in range(40)]}
    batches = et_book_batches(notices, seen, ts=1000)
    cb = defaultdict(lambda: defaultdict(float))
    cs = defaultdict(lambda: defaultdict(float))
    accumulate_book_capture(cb, cs, 0, batches[0], 15)
    r = balanced_reward({5: [sum(cb[5].values()), sum(cs[5].values())]})
    assert r.get(5, 0.0) <= 1e-6
