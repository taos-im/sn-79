# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A SHADOW-PARITY mismatch on debeta_mtm names the diverging UIDs. Since 1 October 2026 the local validator's shadow
diverged on debeta_mtm within one round of every INIT and suspended itself, and the log carried only two hashes, so the
cause could not be read. The parity frame now carries a uid-level vector for debeta_mtm next to the n_pnl one (books
held and summed mark-to-market per uid), both sides build it through one function, and the mismatch line lists the
uids that differ, ordered by the size of the difference, with uids present on one side only called out."""
import types

from taos.im.validator import scoring_shadow as ss


def test_the_mtm_vector_carries_books_and_sum_per_uid_and_the_diff_orders_by_size():
    s = types.SimpleNamespace(debeta_mtm={1: {0: 1.5, 2: -0.5}, 2: {}, 3: {1: 10.0}}, realized_pnl_history={1: [1, 2], 3: []},
                              debeta_inv={0: {1: 4.0}, 2: {1: -1.0}, 1: {3: 7.5}})
    v = ss.mtm_vector(s)
    assert v == {1: (2, 1.0, 3.0), 2: (0, 0.0, 0.0), 3: (1, 10.0, 7.5)}, "books, summed mark-to-market, summed inventory on those books"
    pv = ss.parity_vectors(s)
    assert set(pv) == {"n_pnl", "debeta_mtm", "mtm_books"} and pv["n_pnl"] == {1: 2} and pv["debeta_mtm"] == v
    assert pv["mtm_books"]["books"][1] == {0: (1.5, 4.0), 2: (-0.5, -1.0)}
    mine = {1: (2, 1.0), 2: (0, 0.0), 3: (1, 10.0), 4: (1, 0.25)}
    theirs = {1: (2, 1.0), 3: (1, 7.0), 4: (1, 0.25), 5: (3, 2.0)}
    d = ss.uid_level_diff(mine, theirs, limit=8)
    assert list(d["uids"]) == [3, 5, 2], "largest sum difference first, then the one-sided uids"
    assert d["uids"][3] == ((1, 10.0), (1, 7.0)) and d["uids"][2] == ((0, 0.0), None) and d["uids"][5] == (None, (3, 2.0))
    assert d["only_main"] == [2] and d["only_shadow"] == [5] and d["n_diff"] == 3


def test_both_ends_of_the_frame_use_the_shared_vectors_and_the_mismatch_names_mtm_uids():
    import inspect
    import pathlib

    src = inspect.getsource(ss)
    assert "parity_vectors(shadow)" in src, "the child ships the vectors dict"
    assert "uid_level_diff(" in inspect.getsource(ss.ScoringShadow._compare_parity) and "'debeta_mtm' in _diff" in inspect.getsource(ss.ScoringShadow._compare_parity)
    val = (pathlib.Path(ss.__file__).resolve().parents[1] / "neurons/validator.py").read_text()
    assert "parity_vectors" in val and "pnl_len_vector" not in val, "main binds the same vectors function"
