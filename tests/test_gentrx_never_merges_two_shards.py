"""A training window must never mix data from two shards.

_maybe_train merges every queued assignment into ONE window: it concatenates `data` and `books`
across assignments and uploads a single gradient. That merge was written for several validators
serving the SAME mechanism ("same-round multi-validator consolidation"). Under dual-mechanism SN79
the two validators serve DIFFERENT shards, so the merge trains one model on simulation AND exchange
data and publishes it to whichever shard happened to be learned last.

With both mechanisms live:

    assignments: 2 validator(s), round=5237 books=['20','83','35','5','10'] data=5 files total
    aggregator shard learned from assignment: gentrx/localnet/simulation/
    aggregator shard learned from assignment: gentrx/localnet/exchange/
    window 1 STARTED | 5 files from 2 validator(s) | books=['20','83','35','5','10']
    gradient uploaded to S3 (round=5237)

Books 20/83/35 are simulation, 5/10 are exchange, and the gradient landed under
gentrx/localnet/simulation/gradients/4/00005237.grad. The stage's isolation assertion passed that run
because it only asks whether SOME shard received a gradient.

Grouping by bucket_prefix keeps each window shard-pure. Assignments for the other shard are requeued
rather than dropped, and selection alternates so a slower mechanism is not starved by a faster one.
"""
import pytest

from taos.im.agents.gentrx import group_assignments_by_shard, pick_shard_group


SIM = "gentrx/localnet/simulation/"
XCH = "gentrx/localnet/exchange/"


def _a(shard, books, data, rnd=5237):
    return {"bucket_prefix": shard, "books": books, "data": data, "round": rnd}


def test_two_shards_are_grouped_apart():
    a = [_a(SIM, ["20", "83"], ["s1", "s2"]), _a(XCH, ["5", "10"], ["x1"])]
    g = group_assignments_by_shard(a)
    assert set(g) == {SIM, XCH}
    assert [x["books"] for x in g[SIM]] == [["20", "83"]]
    assert [x["books"] for x in g[XCH]] == [["5", "10"]]


def test_one_shard_is_unchanged():
    """The same-mechanism multi-validator case must still consolidate as before."""
    a = [_a(SIM, ["20"], ["s1"]), _a(SIM, ["83"], ["s2"])]
    g = group_assignments_by_shard(a)
    assert list(g) == [SIM]
    assert len(g[SIM]) == 2


def test_chosen_group_is_shard_pure_and_rest_requeued():
    a = [_a(SIM, ["20", "83"], ["s1", "s2"]), _a(XCH, ["5", "10"], ["x1"])]
    chosen, requeue = pick_shard_group(group_assignments_by_shard(a), last_shard=None)
    shards = {x["bucket_prefix"] for x in chosen}
    assert len(shards) == 1, "a training window carried two shards"
    assert requeue, "the other shard's assignment was dropped instead of requeued"
    assert {x["bucket_prefix"] for x in requeue} == ({SIM, XCH} - shards)


def test_selection_alternates_so_neither_shard_starves():
    groups = group_assignments_by_shard(
        [_a(SIM, ["20"], ["s1"]), _a(XCH, ["5"], ["x1"])]
    )
    first, _ = pick_shard_group(groups, last_shard=None)
    served = first[0]["bucket_prefix"]
    second, _ = pick_shard_group(groups, last_shard=served)
    assert second[0]["bucket_prefix"] != served, "the same shard was served twice in a row"


def test_empty_input_is_safe():
    assert group_assignments_by_shard([]) == {}
    assert pick_shard_group({}, last_shard=None) == ([], [])
