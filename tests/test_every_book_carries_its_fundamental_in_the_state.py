# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Every book in the simulation state the validator sends to miners carries that book's latest fundamental price,
and the metrics publish the value the state carried, so a miner reading the state and a reader of the metrics see
one value per step.

The value is loaded from the simulator's fundamental files at each state update, written into each book (key `f`,
read as `book.fundamental`), and read back from the reported state's books by the reporting path. A miner on the
previous release parses a state that carries the field: books travel as dicts inside the compressed payload and the
book model ignores a key it does not declare; the last test runs that parse against the previous release's own
protocol package. A state without the field (the exchange mechanism, or an older validator) reads as None."""
import io
import json
import os
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

import taos.im.validator.engines.simulation as sim
from taos.im.validator.fundamental import carried, reported, stamp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREVIOUS_RELEASE = "0.6.3"
S = 1_000_000_000
RUN = "20000101_000000"


def _book(book_id, bid=300.0, ask=300.5):
    return {
        "i": book_id,
        "r": 0.4,
        "b": [{"p": bid, "q": 1.5, "o": None}],
        "a": [{"p": ask, "q": 2.0, "o": None}],
        "e": [{"y": "o", "i": 1, "c": None, "t": 1, "q": 1.5, "s": 1, "p": bid, "l": 0.0}],
    }


def _engine(run_dir, *, multiasset):
    v = SimpleNamespace()
    v.simulation_timestamp = 5 * 60 * S + 7 * S
    v.simulation = SimpleNamespace(
        logDir=str(run_dir), simulation_id=RUN[:13], grace_period=0, log_window=3600 * S, publish_interval=S,
        book_count=4, block_count=2, books_per_block=2, book_ids=[0, 1, 2, 3],
        backgrounds=["a", "b"] if multiasset else None,
        realizations=lambda: [(0, "simulation_0", 0, None), (1, "simulation_1", 0, None)],
    )
    v.update_repo = lambda end=False: None
    e = sim.SimulationEngine.__new__(sim.SimulationEngine)
    e.validator = v
    e._notify_seed_log_dir_change = lambda d: None
    e.compress_outputs = lambda start=False: None
    return e, v


def _write(path, header_ids, rows, *, end="\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ",".join(str(i) for i in header_ids) + ",Timestamp\n"
    text += "".join(",".join(str(x) for x in row) + "\n" for row in rows)
    path.write_text(text.rstrip("\n") + end if rows else text)


def _state(run_dir, ts, book_ids):
    return SimpleNamespace(timestamp=ts, logDir=str(run_dir), books={b: _book(b) for b in book_ids})


def _single_market(tmp_path):
    run = tmp_path / RUN
    _write(run / "fundamental.0-1.csv", [0, 1], [[300.0, 300.0, 0], [301.25, 299.5, 30 * S]])
    _write(run / "fundamental.2-3.csv", [2, 3], [[300.0, 300.0, 0], [302.0, 298.75, 30 * S]])
    return run


def test_each_book_in_the_state_carries_its_latest_fundamental(tmp_path):
    run = _single_market(tmp_path)
    e, v = _engine(run, multiasset=False)
    state = _state(run, 5 * 60 * S + 8 * S, [0, 1, 2, 3])
    e.on_tick(state)
    assert {b: book["f"] for b, book in state.books.items()} == {0: 301.25, 1: 299.5, 2: 302.0, 3: 298.75}
    assert v.fundamental_price == {0: 301.25, 1: 299.5, 2: 302.0, 3: 298.75}


def test_a_multiasset_state_reads_each_realizations_files(tmp_path):
    run = tmp_path / RUN
    _write(run / "simulation_0-0" / "fundamental.0-1.csv", [0, 1], [[310.5, 290.25, 30 * S]])
    _write(run / "simulation_1-0" / "fundamental.2-3.csv", [2, 3], [[50.5, 49.75, 30 * S]])
    e, _v = _engine(run, multiasset=True)
    state = _state(run, 5 * 60 * S + 8 * S, [0, 1, 2, 3])
    e.on_tick(state)
    assert {b: book["f"] for b, book in state.books.items()} == {0: 310.5, 1: 290.25, 2: 50.5, 3: 49.75}


def test_a_book_without_a_known_fundamental_carries_none(tmp_path):
    run = _single_market(tmp_path)
    e, _v = _engine(run, multiasset=False)
    state = _state(run, 5 * 60 * S + 8 * S, [0, 1, 9])
    e.on_tick(state)
    assert state.books[9]["f"] is None
    assert "f" in state.books[9], "the key is present, so a reader sees an explicit absence"


def test_a_header_only_file_gives_no_fundamental(tmp_path):
    run = tmp_path / RUN
    _write(run / "fundamental.0-1.csv", [0, 1], [])
    _write(run / "fundamental.2-3.csv", [2, 3], [])
    e, _v = _engine(run, multiasset=False)
    state = _state(run, 5 * 60 * S + 8 * S, [0, 1, 2, 3])
    e.on_tick(state)
    assert all(book["f"] is None for book in state.books.values()), "the header's book ids are not prices"


def test_a_partly_written_row_keeps_the_last_complete_one(tmp_path):
    run = _single_market(tmp_path)
    e, _v = _engine(run, multiasset=False)
    e.on_tick(_state(run, 5 * 60 * S + 8 * S, [0, 1, 2, 3]))
    with open(run / "fundamental.0-1.csv", "a") as f:
        f.write("303.5,29")
    state = _state(run, 5 * 60 * S + 9 * S, [0, 1, 2, 3])
    e.on_tick(state)
    assert (state.books[0]["f"], state.books[1]["f"]) == (301.25, 299.5)
    assert (state.books[2]["f"], state.books[3]["f"]) == (302.0, 298.75)


def test_the_metrics_publish_the_value_the_state_carried(tmp_path):
    run = _single_market(tmp_path)
    e, v = _engine(run, multiasset=False)
    state = _state(run, 5 * 60 * S + 8 * S, [0, 1, 2, 3])
    e.on_tick(state)
    sent = {b: book["f"] for b, book in state.books.items()}
    with open(run / "fundamental.0-1.csv", "a") as f:
        f.write("305.0,295.0,60000000000\n")
    e.load_fundamental()
    assert v.fundamental_price[0] == 305.0, "the files moved on after the state went out"
    assert reported(state.books, v.fundamental_price) == sent


def test_a_state_whose_books_carry_no_fundamental_reports_the_held_values():
    books = {7: _book(7), 8: _book(8)}
    assert carried(books) is None
    assert reported(books, {7: None, 8: None}) == {7: None, 8: None}


def test_stamping_reads_book_ids_of_either_key_type():
    books = {"0": _book(0), 1: _book(1)}
    stamp(books, {0: 1.5, 1: 2.5})
    assert (books["0"]["f"], books[1]["f"]) == (1.5, 2.5)
    assert carried(books) == {0: 1.5, 1: 2.5}


def test_the_reporting_path_reads_the_reported_states_books():
    import ast

    src = open(os.path.join(REPO, "taos/im/neurons/validator.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_prepare_reporting_data")
    body = ast.get_source_segment(src, fn)
    assert "load_fundamental" not in body, "the metrics value is the state's, not a later read of the files"
    assert "reported_fundamental(" in body and "last_state" in body


@pytest.mark.parametrize("rel", ["taos/im/neurons/validator.py", "taos/im/validator/engines/simulation.py",
                                 "taos/im/validator/report.py", "taos/im/validator/fundamental.py"])
def test_the_state_and_reporting_paths_name_nothing_undefined(rel):
    """The reporting preparation runs once per report cycle, so a name left undefined there fails only in a live
    validator; ruff's F821 reads every path without running it."""
    try:
        out = subprocess.run(["ruff", "check", "--select", "F821", "--no-cache", os.path.join(REPO, rel)],
                             capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        pytest.skip("ruff is not installed")
    assert not [ln for ln in out.stdout.splitlines() if "F821" in ln], out.stdout


def _wire(books_by_id, fundamentals):
    """The state as it leaves the validator: books compressed once, the per-miner payload around them."""
    from taos import __spec_version__
    from taos.im.protocol import MarketSimulationStateUpdate
    from taos.im.utils.compress import compress

    books = {b: _book(b, bid=p) for b, p in books_by_id.items()}
    if fundamentals is not None:
        stamp(books, fundamentals)
    compressed_books = compress(books, level=1, engine="lz4", version=__spec_version__)
    synapse = MarketSimulationStateUpdate.parse_dict({"timestamp": 8 * S, "books": books})
    object.__setattr__(synapse, "accounts", {5: {}})
    object.__setattr__(synapse, "notices", {5: []})
    object.__setattr__(synapse, "config", None)
    synapse.version = __spec_version__
    out = synapse.compress(level=1, engine="lz4", compressed_books=compressed_books)
    return json.loads(out.model_dump_json())


@pytest.mark.parametrize("lazy", [False, True])
def test_a_miner_reads_each_books_fundamental(lazy):
    from taos.im.protocol import MarketSimulationStateUpdate

    received = MarketSimulationStateUpdate(**_wire({0: 300.0, 1: 299.0}, {0: 301.25, 1: None}))
    assert received.decompress(lazy=lazy) is received
    assert received.books[0].fundamental == 301.25
    assert received.books[1].fundamental is None
    if lazy:
        assert received.books[0].parse().fundamental == 301.25


@pytest.mark.parametrize("lazy", [False, True])
def test_a_state_without_the_field_reads_as_none(lazy):
    from taos.im.protocol import MarketSimulationStateUpdate

    received = MarketSimulationStateUpdate(**_wire({0: 300.0}, None))
    received.decompress(lazy=lazy)
    assert received.books[0].fundamental is None


def test_the_exchange_book_model_reads_none_without_the_field():
    from taos.im.protocol.exchange.models import Book as ExchangeBook
    from taos.im.protocol.exchange.models import LazyBooks as ExchangeLazyBooks

    raw = {"i": 3, "r": None, "b": [], "a": [], "e": []}
    assert ExchangeBook.model_validate(raw).fundamental is None
    assert ExchangeLazyBooks({3: raw})[3].fundamental is None
    assert ExchangeLazyBooks({3: raw})[3].parse().fundamental is None


_PREVIOUS_RELEASE_MINER = r"""
import json, sys
root, body_path = sys.argv[1], sys.argv[2]
import taos
assert taos.__file__.startswith(root), taos.__file__
from taos.im.protocol import MarketSimulationStateUpdate
body = json.load(open(body_path))
out = {}
for lazy in (False, True):
    state = MarketSimulationStateUpdate(**body)
    assert state.decompress(lazy=lazy) is state, "decompress failed"
    out["lazy" if lazy else "eager"] = {
        str(book_id): {"type": type(book).__name__, "id": book.id, "bid": book.bids[0].price,
                       "events": len(book.events)}
        for book_id, book in state.books.items()
    }
print("RESULT" + json.dumps(out))
"""


def test_a_miner_on_the_previous_release_parses_a_state_that_carries_the_field(tmp_path):
    archive = subprocess.run(["git", "-C", REPO, "archive", PREVIOUS_RELEASE, "taos"], capture_output=True, timeout=120)
    if archive.returncode != 0:
        pytest.skip(f"the {PREVIOUS_RELEASE} tree is not available here")
    root = tmp_path / "previous"
    root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(root)
    body = tmp_path / "state.json"
    body.write_text(json.dumps(_wire({0: 300.0, 1: 299.0}, {0: 301.25, 1: 299.75})))
    script = tmp_path / "previous_miner.py"
    script.write_text(_PREVIOUS_RELEASE_MINER)
    env = {k: val for k, val in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(root)
    run = subprocess.run([sys.executable, str(script), str(root), str(body)], capture_output=True, text=True,
                         timeout=300, cwd=str(tmp_path), env=env)
    line = next((ln for ln in run.stdout.splitlines() if ln.startswith("RESULT")), None)
    assert run.returncode == 0 and line, f"the previous release could not parse the state:\n{run.stderr[-3000:]}"
    result = json.loads(line[len("RESULT"):])
    for mode in ("eager", "lazy"):
        books = result[mode]
        assert {k: (b["id"], b["bid"], b["events"]) for k, b in books.items()} == {
            "0": (0, 300.0, 1), "1": (1, 299.0, 1)}, mode
    assert {b["type"] for b in result["eager"].values()} == {"Book"}
