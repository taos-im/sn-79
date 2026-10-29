"""Tests for GenTRXService — validator-driven scheduling, assignment creation, delivery.

The service is HTTP-only; these tests stand up an in-process HTTP server
that mocks the gradient server endpoints (data-status, round, scores).
No chain, no S3.

Run: pytest GenTRX/tests/test_service.py -v
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock

import pytest

from GenTRX.src.service import GenTRXService
from GenTRX.src.state_packager import StatePackager


# ---------------------------------------------------------------------------
# Mock gradient server
# ---------------------------------------------------------------------------


class MockGradientServer:
    """In-memory gradient server mock — serves data-status, accepts rounds, serves scores."""

    def __init__(self):
        self.version = 1
        self.max_ts = 600_000_000_000  # 10 min of sim data
        self.books = {
            "0": {"parquets": ["00000000-00000300.parquet"], "max_ts": 300_000_000_000},
            "1": {"parquets": ["00000000-00000300.parquet"], "max_ts": 300_000_000_000},
            "2": {"parquets": ["00000000-00000300.parquet"], "max_ts": 300_000_000_000},
            "3": {"parquets": ["00000000-00000300.parquet"], "max_ts": 300_000_000_000},
            "4": {"parquets": ["00000000-00000300.parquet"], "max_ts": 300_000_000_000},
        }
        self.received_rounds: list[dict] = []
        self.received_states: list[bytes] = []
        self.state_fail_remaining: int = 0
        self.latest_scores: dict | None = None

    def data_status(self) -> dict:
        return {
            "max_ts": self.max_ts,
            "version": self.version,
            "round": len(self.received_rounds),
            "books": self.books,
        }


@pytest.fixture
def mock_server():
    return MockGradientServer()


@pytest.fixture
def http_server(mock_server):
    """Start a real HTTP server backed by mock_server."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/data-status"):
                self._json_response(mock_server.data_status())
                return
            if self.path.startswith("/scores"):
                if mock_server.latest_scores is None:
                    self.send_response(204)
                    self.end_headers()
                    return
                self._json_response(mock_server.latest_scores)
                return
            if self.path.startswith("/version"):
                self._json_response({"version": mock_server.version})
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            if self.path.startswith("/state"):
                if mock_server.state_fail_remaining > 0:
                    mock_server.state_fail_remaining -= 1
                    self.send_response(500)
                    self.end_headers()
                    return
                mock_server.received_states.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
                return
            if self.path.startswith("/round"):
                payload = json.loads(body)
                mock_server.received_rounds.append(payload)
                self._json_response({
                    "status": "ok",
                    "round": payload.get("round", 0),
                    "n_assignments": len(payload.get("assignments", {})),
                })
                return
            self.send_response(404)
            self.end_headers()

        def _json_response(self, data):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, *a, **kw):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _service(server_url: str, **kwargs) -> GenTRXService:
    """Helper — pass a packager + URL by default."""
    kwargs.setdefault("packager", StatePackager())
    return GenTRXService(gradient_server_url=server_url, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_service_requires_gradient_server_url():
    """Constructing a service without a URL is a programmer error."""
    with pytest.raises(ValueError):
        GenTRXService(packager=StatePackager(), gradient_server_url="")


def test_fetch_data_status(mock_server, http_server):
    """Service can fetch data-status from gradient server."""
    import asyncio
    s = _service(http_server)
    status = asyncio.run(s._fetch_data_status())
    assert status is not None
    assert status["version"] == 1
    assert "0" in status["books"]


def test_create_assignments_from_available_data(mock_server, http_server):
    """Service creates assignments with books and data keys from data-status."""
    import asyncio
    s = _service(http_server, miner_uids=[2, 3], poll_interval=0)
    status = asyncio.run(s._fetch_data_status())
    s._current_round = 1
    assignments = s._create_assignments(status)
    assert set(assignments.keys()) == {2, 3}
    for _uid, a in assignments.items():
        assert a["round"] == 1
        assert len(a["books"]) > 0
        assert len(a["data"]) > 0
        assert a["model_version"] == 1


def test_create_assignments_returns_empty_when_no_pages(mock_server, http_server):
    """Page-based: no flushed pages yet (books known but empty) → no assignments."""
    import asyncio
    for b in mock_server.books.values():
        b["parquets"] = []
    s = _service(http_server, miner_uids=[0], poll_interval=0)
    status = asyncio.run(s._fetch_data_status())
    assignments = s._create_assignments(status)
    assert assignments == {}


def test_create_assignments_iid_shared_window(mock_server, http_server):
    """Flavor B: per-miner IID book sample, but ONE shared [ts_start, ts_end]
    window across all miners (so held-out scoring uses the same baseline)."""
    import asyncio
    # Distinct per-book spans so a shared window is observable.
    for i, b in enumerate(mock_server.books.values()):
        b["parquets"] = [[f"page{i}.parquet", i * 100, i * 100 + 100]]
    s = _service(http_server, miner_uids=[0, 1, 2, 3], poll_interval=0)
    status = asyncio.run(s._fetch_data_status())
    s._current_round = 7
    a = s._create_assignments(status)
    assert set(a.keys()) == {0, 1, 2, 3}
    starts = {x["ts_start"] for x in a.values()}
    ends = {x["ts_end"] for x in a.values()}
    assert len(starts) == 1 and len(ends) == 1  # one shared window for all miners
    for x in a.values():
        assert len(x["books"]) == len(set(x["books"]))  # a sample, no repeats
    assert s._create_assignments(status) == a  # deterministic per round


def test_push_round_includes_disjoint_val_books(mock_server, http_server):
    """The round payload carries the rotating held-out split, disjoint from
    every miner's assigned (training) books — the server's single source of
    truth for what to score held-out."""
    import asyncio
    s = _service(http_server, miner_uids=[0, 1, 2], poll_interval=0)
    status = asyncio.run(s._fetch_data_status())
    s._current_round = 5
    a = s._create_assignments(status)
    assert asyncio.run(s._push_round(5, a)) is True
    pushed = mock_server.received_rounds[-1]
    val_books = set(pushed["val_books"])
    assert val_books  # non-empty split
    for asg in a.values():
        assert not (set(asg["books"]) & val_books)  # never train on val books


def test_create_assignments_flavor_a_identical_books(mock_server, http_server):
    """Flavor A toggle: every miner gets the identical book sample."""
    import asyncio
    for i, b in enumerate(mock_server.books.values()):
        b["parquets"] = [[f"page{i}.parquet", i * 100, i * 100 + 100]]
    s = _service(http_server, miner_uids=[0, 1, 2], poll_interval=0)
    s._iid_shared_books = True
    status = asyncio.run(s._fetch_data_status())
    s._current_round = 3
    a = s._create_assignments(status)
    book_sets = {tuple(sorted(x["books"])) for x in a.values()}
    assert len(book_sets) == 1  # identical across miners


def test_split_keyed_on_round_id_not_current_round(mock_server, http_server):
    """The val/train split (and per-miner book sample) is seeded on the round
    being ASSIGNED, not self._current_round.

    Regression: seeds used self._current_round, which only advances after a
    successful push. In block mode new_round can jump by >1 (missed blocks), so
    two validators with different push histories would derive a DIFFERENT split
    for the same round_id. Passing round_id must fully determine the split
    regardless of _current_round.
    """
    import asyncio
    s = _service(http_server, miner_uids=[0, 1, 2], poll_interval=0)
    status = asyncio.run(s._fetch_data_status())

    s._current_round = 0
    a1 = s._create_assignments(status, round_id=7)
    val1 = set(s._val_books)
    books1 = {uid: tuple(sorted(x["books"])) for uid, x in a1.items()}

    # Same round_id, a wildly different _current_round (validator that missed
    # blocks): the split and per-miner samples must be identical.
    s._current_round = 99
    a2 = s._create_assignments(status, round_id=7)
    val2 = set(s._val_books)
    books2 = {uid: tuple(sorted(x["books"])) for uid, x in a2.items()}

    assert val1 == val2
    assert books1 == books2
    assert all(x["round"] == 7 for x in a2.values())


def test_push_round_sends_to_server(mock_server, http_server):
    """_push_round POSTs the assignment plan to the gradient server."""
    import asyncio
    s = _service(http_server)
    assignments = {
        2: {"round": 1, "books": ["0"], "data": ["data/0/intervals/f.parquet"]},
        3: {"round": 1, "books": ["1"], "data": ["data/1/intervals/f.parquet"]},
    }
    ok = asyncio.run(s._push_round(1, assignments))
    assert ok is True
    assert len(mock_server.received_rounds) == 1
    payload = mock_server.received_rounds[0]
    assert payload["round"] == 1
    assert "2" in payload["assignments"]
    assert "3" in payload["assignments"]


def test_should_advance_round_timer_mode(mock_server, http_server):
    """In timer mode (no get_block_fn), round advances after poll_interval."""
    s = _service(http_server, poll_interval=0.01)
    # First call: should advance (time since last push > poll_interval)
    result = s._should_advance_round()
    assert result is not None
    assert result == 1  # current_round=0 → next=1


def test_should_advance_round_block_mode(mock_server, http_server):
    """In block-synced mode, round = block // blocks_per_round."""
    current_block = [100]

    def get_block():
        return current_block[0]

    s = _service(http_server, blocks_per_round=10, get_block_fn=get_block)
    s._current_round = 9  # block 100 // 10 = 10, which is > 9
    result = s._should_advance_round()
    assert result == 10

    s._current_round = 10  # already at round 10
    result = s._should_advance_round()
    assert result is None  # no advance


def test_poll_and_deliver_full_cycle(mock_server, http_server):
    """Full cycle: advance round → fetch data → create assignments → push → deliver."""
    delivered = {}

    async def deliver(assignments):
        delivered.update(assignments)

    s = _service(
        http_server,
        poll_interval=0,
        deliver_fn=deliver,
        miner_uids=[2, 3],
    )
    # Bypass the warmup gate (state-push normally bumps this in production).
    s._max_sim_ts_pushed = s._window_ns
    asyncio.run(s.poll_and_deliver())

    # Assignments should have been created, pushed to server, and delivered
    assert len(delivered) == 2
    assert 2 in delivered
    assert 3 in delivered
    assert len(mock_server.received_rounds) == 1


def test_poll_and_deliver_no_data(mock_server, http_server):
    """When no pages are flushed yet, no assignments are created or delivered."""
    mock_server.max_ts = 0  # no data
    for b in mock_server.books.values():
        b["parquets"] = []

    delivered = []

    async def deliver(assignments):
        delivered.append(assignments)

    s = _service(http_server, poll_interval=0, deliver_fn=deliver, miner_uids=[0])
    s._max_sim_ts_pushed = s._window_ns  # bypass warmup gate
    asyncio.run(s.poll_and_deliver())
    assert delivered == []
    assert len(mock_server.received_rounds) == 0


def test_insufficient_data_is_gated_in_poll_not_create_assignments(mock_server, http_server):
    """Before the sim has pushed one full window, poll_and_deliver assigns no
    round even though pages are listed. _create_assignments picks from the
    listed pages and does not read max_ts, so the window gate is here."""
    delivered = []

    async def deliver(assignments):
        delivered.append(assignments)

    s = _service(http_server, poll_interval=0, deliver_fn=deliver, miner_uids=[0])
    s._max_sim_ts_pushed = s._window_ns - 1
    asyncio.run(s.poll_and_deliver())
    assert delivered == []
    assert len(mock_server.received_rounds) == 0

    s._max_sim_ts_pushed = s._window_ns
    asyncio.run(s.poll_and_deliver())
    assert len(mock_server.received_rounds) == 1


def test_receive_scores_updates_store(http_server):
    """Scores received via receive_scores are stored and retrievable."""
    s = _service(http_server, miner_uids=[0, 1])
    payload = {
        "round": 5,
        "n_accepted": 2,
        "n_scored": 2,
        "scores": {
            "0": {"score": 0.15, "accepted": True, "books": ["1"]},
            "1": {"score": -0.05, "accepted": False, "books": ["2"]},
        },
    }
    s.receive_scores(payload)
    scores = s.get_scores()
    assert scores[0]["score"] == 0.15
    assert scores[1]["accepted"] is False


def test_push_state_calls_packager(http_server):
    """push_state extracts state via packager and POSTs."""
    packager = MagicMock()
    packager.extract_state.return_value = {"step": 0, "books": {}}
    s = GenTRXService(packager=packager, gradient_server_url=http_server)
    s.push_state(MagicMock())
    packager.extract_state.assert_called_once()


def test_push_state_swallows_exceptions(http_server):
    """push_state errors don't propagate (would crash handle_state)."""
    packager = MagicMock()
    packager.extract_state.side_effect = RuntimeError("extraction failed")
    s = GenTRXService(packager=packager, gradient_server_url=http_server)
    s.push_state(MagicMock())  # should not raise


# ---------------------------------------------------------------------------
# TX worker, retry, spool durability
# ---------------------------------------------------------------------------


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.02) -> bool:
    import time as _t
    deadline = _t.time() + timeout
    while _t.time() < deadline:
        if predicate():
            return True
        _t.sleep(interval)
    return predicate()


def test_push_state_enqueues_and_drains(http_server, mock_server):
    packager = MagicMock()
    packager.extract_state.return_value = {"step": 0, "books": {}}
    s = GenTRXService(packager=packager, gradient_server_url=http_server)
    s.push_state(MagicMock())
    assert _wait_until(lambda: len(mock_server.received_states) == 1)


def test_push_state_does_not_block_on_failure(http_server, mock_server):
    import time as _t
    mock_server.state_fail_remaining = 100
    packager = MagicMock()
    packager.extract_state.return_value = {"step": 0, "books": {}}
    s = GenTRXService(packager=packager, gradient_server_url=http_server)
    t0 = _t.time()
    s.push_state(MagicMock())
    assert (_t.time() - t0) < 0.1


def test_tx_worker_retries_on_500(http_server, mock_server):
    mock_server.state_fail_remaining = 1
    packager = MagicMock()
    packager.extract_state.return_value = {"step": 0, "books": {}}
    s = GenTRXService(packager=packager, gradient_server_url=http_server)
    s.push_state(MagicMock())
    assert _wait_until(lambda: len(mock_server.received_states) == 1, timeout=3.0)


def test_tx_queue_drops_oldest_without_spool(http_server, mock_server):
    mock_server.state_fail_remaining = 100_000
    packager = MagicMock()
    counter = {"n": 0}
    def _extract(_):
        counter["n"] += 1
        return {"step": counter["n"], "books": {}}
    packager.extract_state.side_effect = _extract
    s = GenTRXService(packager=packager, gradient_server_url=http_server)
    # Spool disabled, so queue is bounded by DEFAULT_TX_QUEUE_SIZE → drop-oldest.
    s._tx_queue.maxsize = 4
    while not s._tx_queue.empty():
        try:
            s._tx_queue.get_nowait()
        except Exception:
            break
    for _ in range(14):
        s.push_state(MagicMock())
    assert _wait_until(lambda: s._tx_drops >= 10, timeout=2.0)


def test_spool_persists_unsent_across_instance(http_server, mock_server, tmp_path):
    import time as _t
    spool = str(tmp_path / "spool.bin")
    mock_server.state_fail_remaining = 100
    packager = MagicMock()
    packager.extract_state.return_value = {"step": 1, "books": {}}
    s1 = GenTRXService(packager=packager, gradient_server_url=http_server, tx_spool_path=spool)
    s1.push_state(MagicMock())
    _t.sleep(0.2)
    s1._tx_stop.set()

    mock_server.state_fail_remaining = 0
    GenTRXService(packager=packager, gradient_server_url=http_server, tx_spool_path=spool)
    assert _wait_until(lambda: len(mock_server.received_states) >= 1, timeout=2.0)


def test_spool_unbounded_queue_replays_all(http_server, mock_server, tmp_path):
    """C3: when spool is enabled, queue is unbounded so replay > 256 still drains."""
    spool = str(tmp_path / "spool.bin")
    packager = MagicMock()
    packager.extract_state.return_value = {"step": 1, "books": {}}
    s1 = GenTRXService(packager=packager, gradient_server_url=http_server, tx_spool_path=spool)
    # Maxsize 0 = unbounded.
    assert s1._tx_queue.maxsize == 0
    s1._tx_stop.set()
