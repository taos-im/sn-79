# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The scoring child's INIT sent each structure as pickle.dumps(_plainify(structure)): the plain copy exists because
lambda-factory defaultdicts do not pickle, and on a 256-book save the copy took about half of the INIT's time under
the reward lock. dump_plain pickles
the structure through a pickler whose override writes every defaultdict as a plain dict with the same items, so the
bytes load to what the copy loaded to, without the copy; the timing line records the whole as the pickle step."""
import pickle
from collections import defaultdict, deque

from taos.im.validator import scoring_shadow as ss


def _nested():
    dd = defaultdict(lambda: defaultdict(float))
    dd[1][0] = 2.5
    dd[1][3] = -1.0
    dd[7][0] = 0.0
    inner = defaultdict(list)
    inner["a"].append((1, 2.0))
    return {"sums": dd, "hist": {5: deque([(1, {"q": 1.0}), (2, defaultdict(int, {"n": 3}))], maxlen=10)},
            "lists": [defaultdict(float, {0: 1.0}), [1, 2, (3, 4)]], "scalar": 3, "inner": inner, "plain": {"k": "v"}}


def _has_defaultdict(o):
    if isinstance(o, defaultdict):
        return True
    if isinstance(o, dict):
        return any(_has_defaultdict(v) for v in o.values())
    if isinstance(o, (list, tuple, deque)):
        return any(_has_defaultdict(v) for v in o)
    return False


def test_the_bytes_load_to_what_the_plain_copy_loaded_to_and_carry_no_defaultdict():
    obj = _nested()
    loaded = pickle.loads(ss.dump_plain(obj))
    assert loaded == pickle.loads(pickle.dumps(ss._plainify(obj), protocol=5))
    assert not _has_defaultdict(loaded)
    assert loaded["sums"][1][0] == 2.5 and loaded["hist"][5][1][1] == {"n": 3} and loaded["lists"][0] == {0: 1.0}
    assert isinstance(loaded["hist"][5], deque) and loaded["hist"][5].maxlen == 10
    assert _has_defaultdict(obj), "the source is untouched"


def test_a_lambda_factory_defaultdict_pickles_through_the_override_where_plain_pickle_fails():
    dd = defaultdict(lambda: defaultdict(lambda: 0.0))
    dd[2][1] = 1.5
    try:
        pickle.dumps(dd, protocol=5)
        raise AssertionError("a lambda factory should not pickle")
    except (pickle.PicklingError, AttributeError):
        pass
    assert pickle.loads(ss.dump_plain(dd)) == {2: {1: 1.5}}


def test_send_init_uses_the_one_pass_dump():
    import inspect
    src = inspect.getsource(ss.ScoringShadow.send_init)
    assert "blob = dump_plain(getattr(validator, name, {}))" in src
    assert "_plainify(getattr(validator, name" not in src
    assert '"plainify_s": 0.0' in src
