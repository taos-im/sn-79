"""A round dispatches its requests in a fresh permutation of the miners. The recorded response time starts before a
request is prepared and runs while the later requests of the round are still being prepared on the same loop, so a
dispatch in uid order would make the recorded time follow the uid. A fresh permutation each round removes any
standing edge by uid."""
import random

from taos.im.validator.query import round_order


def test_each_round_is_a_permutation_that_keeps_every_uid_with_its_axon_index():
    uids = [3, 7, 11, 200, 201, 250]
    order = round_order(uids, random.Random(1))
    assert sorted(order) == list(enumerate(uids)), "every (index, uid) pair once, so each uid keeps its axon"


def test_two_rounds_do_not_share_an_order():
    uids = list(range(256))
    a = round_order(uids, random.Random(1))
    b = round_order(uids, random.Random(2))
    assert a != b and a != list(enumerate(uids)) and b != list(enumerate(uids))


def test_both_fan_out_paths_dispatch_in_the_round_order():
    import inspect
    from taos.im.validator import query
    src = inspect.getsource(query)
    assert src.count("for index, uid in round_order(uid_list)") == 2, "the threaded path and the fanout path"
    assert "for index, uid in enumerate(uid_list)\n            if uid not in deregistered_uids" not in src
