# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator lists each realization's fundamental files once a run, not once a round.

load_fundamental runs at every state update, on the path to the miners' query, and a realization's directory holds
every book's logs, a listing that grows through the run. Each realization writes its fundamental files once, at the
run's start, so the listing of fundamental.*.csv is kept per directory and read again only when it is empty (the engine
has not written them yet), when a listed file is gone, or when the run's directory changes. The prices it returns are
the ones a fresh listing gives.
"""
import glob as globmod

from test_every_book_carries_its_fundamental_in_the_state import RUN, S, _engine, _state, _write

NEW_RUN = "20000102_000000"  # a later run than RUN: run names sort by their start stamp


def _count_globs(monkeypatch):
    calls = []
    real = globmod.glob

    def counting(pattern, *a, **k):
        calls.append(pattern)
        return real(pattern, *a, **k)
    monkeypatch.setattr(globmod, "glob", counting)
    return calls


def _run(tmp_path, name=RUN, prices=((310.5, 290.25), (50.5, 49.75))):
    run = tmp_path / name
    _write(run / "simulation_0-0" / "fundamental.0-1.csv", [0, 1], [[*prices[0], 30 * S]])
    _write(run / "simulation_1-0" / "fundamental.2-3.csv", [2, 3], [[*prices[1], 30 * S]])
    return run


def test_each_realization_is_listed_once_and_its_new_rows_are_still_read(tmp_path, monkeypatch):
    run = _run(tmp_path)
    e, v = _engine(run, multiasset=True)
    calls = _count_globs(monkeypatch)
    for k in range(5):
        e.on_tick(_state(run, 5 * 60 * S + (8 + k) * S, [0, 1, 2, 3]))
    assert len(calls) == 2, f"two realizations listed once each, not every round: {len(calls)} globs"
    with open(run / "simulation_0-0" / "fundamental.0-1.csv", "a") as f:
        f.write("\n311.0,289.0,60000000000")
    state = _state(run, 5 * 60 * S + 20 * S, [0, 1, 2, 3])
    e.on_tick(state)
    assert {b: book["f"] for b, book in state.books.items()} == {0: 311.0, 1: 289.0, 2: 50.5, 3: 49.75}
    assert len(calls) == 2


def test_a_realization_without_files_yet_is_listed_again(tmp_path, monkeypatch):
    run = tmp_path / RUN
    (run / "simulation_0-0").mkdir(parents=True)
    (run / "simulation_1-0").mkdir(parents=True)
    e, v = _engine(run, multiasset=True)
    calls = _count_globs(monkeypatch)
    e.on_tick(_state(run, 5 * 60 * S + 8 * S, [0, 1, 2, 3]))
    _run(tmp_path)
    state = _state(run, 5 * 60 * S + 9 * S, [0, 1, 2, 3])
    e.on_tick(state)
    assert state.books[0]["f"] == 310.5 and state.books[2]["f"] == 50.5, "files the engine wrote later are found"
    assert len(calls) == 4


def test_a_new_run_directory_or_a_vanished_file_lists_again(tmp_path, monkeypatch):
    run = _run(tmp_path)
    e, v = _engine(run, multiasset=True)
    calls = _count_globs(monkeypatch)
    e.on_tick(_state(run, 5 * 60 * S + 8 * S, [0, 1, 2, 3]))
    assert len(calls) == 2
    other = _run(tmp_path, name=NEW_RUN, prices=((10.0, 11.0), (12.0, 13.0)))
    v.simulation.logDir = str(other)
    state = _state(other, 5 * 60 * S + 9 * S, [0, 1, 2, 3])
    e.on_tick(state)
    assert {b: book["f"] for b, book in state.books.items()} == {0: 10.0, 1: 11.0, 2: 12.0, 3: 13.0}
    assert len(calls) == 4, "a new run directory is listed afresh"
    (other / "simulation_1-0" / "fundamental.2-3.csv").rename(other / "simulation_1-0" / "fundamental.2-3.csv.gone")
    _write(other / "simulation_1-0" / "fundamental.2-3b.csv", [2, 3], [[14.0, 15.0, 30 * S]])
    state = _state(other, 5 * 60 * S + 10 * S, [0, 1, 2, 3])
    e.on_tick(state)
    assert state.books[2]["f"] == 14.0, "a listed file that is gone makes its directory be listed again"
