# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Equivalence tests for the O(1) fundamental-CSV tail-read.

load_fundamental must not readlines() the entire ever-growing fundamental CSV to
take just the first line (book ids) and last non-empty line (latest prices) —
O(file) under both the GIL and _reward_lock, ~9s/scoring round after 20h of sim.
_read_first_and_last_nonempty replaces the scan; these tests pin that it returns
exactly what the old full-scan logic derived, across file shapes and chunk
boundaries.
"""
import pytest

from taos.im.validator.engines.simulation import _read_first_and_last_nonempty


def _old_logic(path):
    """The original readlines() derivation of (first line, last non-empty line)."""
    first = None
    fp_line = None
    for line in open(path, 'r').readlines():
        if first is None:
            first = line
        if line.strip() != '':
            fp_line = line
    return first, fp_line


def _check_equiv(path):
    first_new, last_new = _read_first_and_last_nonempty(path, tail_chunk=32)
    first_old, last_old = _old_logic(path)
    if first_old is None:
        assert first_new == ''
    else:
        assert first_new == first_old
    if last_old is None:
        assert last_new is None
    else:
        # old keeps the trailing newline; parsing uses .strip(), so compare stripped
        assert last_new.strip() == last_old.strip()


HEADER = "8,9,10,11,Timestamp\n"


def test_normal_multiline(tmp_path):
    p = tmp_path / "f.csv"
    rows = [f"{100+i},{200+i},{300+i},{400+i},{i}000000000\n" for i in range(50)]
    p.write_text(HEADER + "".join(rows))
    _check_equiv(str(p))
    first, last = _read_first_and_last_nonempty(str(p), tail_chunk=32)
    assert first == HEADER
    assert last.strip() == rows[-1].strip()


def test_no_trailing_newline(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text(HEADER + "1.0,2.0,3.0,4.0,123\n" + "5.0,6.0,7.0,8.0,456")
    _check_equiv(str(p))


def test_trailing_blank_lines(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text(HEADER + "1.0,2.0,3.0,4.0,123\n\n\n")
    _check_equiv(str(p))
    _, last = _read_first_and_last_nonempty(str(p), tail_chunk=32)
    assert last.strip() == "1.0,2.0,3.0,4.0,123"


def test_header_only(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text(HEADER)
    _check_equiv(str(p))
    first, last = _read_first_and_last_nonempty(str(p), tail_chunk=32)
    assert last.strip() == HEADER.strip()  # old code also fell back to the header


def test_empty_file(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("")
    first, last = _read_first_and_last_nonempty(str(p))
    assert first == '' and last is None


def test_last_line_longer_than_chunk(tmp_path):
    # forces the backwards multi-chunk path: line start outside the first tail chunk
    p = tmp_path / "f.csv"
    long_row = ",".join(str(i) for i in range(200)) + ",999\n"
    assert len(long_row) > 64
    p.write_text(HEADER + "1,2,3\n" + long_row)
    first, last = _read_first_and_last_nonempty(str(p), tail_chunk=16)
    assert first == HEADER
    assert last.strip() == long_row.strip()
    _check_equiv(str(p))


@pytest.mark.parametrize("n_rows", [1, 2, 3, 7, 33])
def test_chunk_boundary_sweep(tmp_path, n_rows):
    p = tmp_path / "f.csv"
    rows = [f"{i}.5,{i}.6,{i}\n" for i in range(n_rows)]
    p.write_text(HEADER + "".join(rows))
    for chunk in (8, 16, 31, 32, 33, 1024):
        first, last = _read_first_and_last_nonempty(str(p), tail_chunk=chunk)
        assert first == HEADER
        assert last.strip() == rows[-1].strip()
