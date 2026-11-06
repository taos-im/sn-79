"""The clock the tail flush measures each book's interval against is the median last-event time over the books that
have ticked since the server started, so books restored from an earlier run's buffer and silent since cannot hold it
back. An aggregator whose sim id never changes restores its buffer across every restart, and with two of its three
restored books silent the median would sit at the earlier run's time and no interval would ever be written.

Run: pytest GenTRX/tests/test_a_book_restored_and_silent_since_does_not_hold_the_clock_back.py -v
"""
S = 10**9
OLD = 1_000_000 * S
NOW = OLD + 86_400 * S


def _make_aggregator(tmp_path):
    from GenTRX.src.gradient_server import GradientAggregator

    return GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path=str(tmp_path / "val"),
        output_path=str(tmp_path / "out.pt"),
        books_per_miner=1,
        interval=60,
        window_ns=50,
        warmup_rounds=0,
        rollback=False,
    )


def _restarted(tmp_path, books, ts):
    """An aggregator restarted on the buffer an earlier one saved after ticking `books` at `ts`, through the server's
    own staging file: _save_pending_rows, then _restore_pending_rows on a new server, which gives each restored book
    its rows, interval start, clock and an engine placeholder."""
    first = _make_aggregator(tmp_path)
    first._process_tick(_tick(ts, books))
    first._save_pending_rows()
    again = _make_aggregator(tmp_path)
    again._restore_pending_rows()
    assert {b: again._last_ts.get(b, 0) > 0 for b in books} == {b: True for b in books}, "the buffer was restored"
    assert set(books) <= set(again._engines), "the restore gives each book an engine placeholder"
    return again


def _tick(ts, books):
    events = [{"y": "o", "s": k % 2, "p": 300.0 + 0.01 * k, "t": ts + k, "i": 7_000_000_000 + k, "q": 1.0}
              for k in range(4)]
    book = {"bids": [[299.9, 5.0]], "asks": [[300.1, 5.0]], "events": events}
    cfg = {"simulation_id": "SIM_A", "priceDecimals": 2, "volumeDecimals": 4}
    return {"ts": ts, "books": {str(b): dict(book) for b in books}, "config": cfg, "sim_id": "SIM_A"}


def test_books_restored_and_silent_since_do_not_hold_the_clock_at_the_earlier_run(tmp_path):
    agg = _restarted(tmp_path, (2, 5, 10), OLD)
    agg._process_tick(_tick(NOW, (10,)))
    assert agg._book_clock() >= NOW, "the clock follows the book that ticks, not the two silent since the restore"


def test_before_any_book_ticks_the_restored_books_define_the_clock(tmp_path):
    agg = _restarted(tmp_path, (2, 5, 10), OLD)
    assert OLD <= agg._book_clock() < NOW


def test_one_ticking_book_far_ahead_still_cannot_define_the_clock(tmp_path):
    agg = _make_aggregator(tmp_path)
    agg._process_tick(_tick(NOW, (1, 2)))
    agg._process_tick(_tick(NOW + 1_000_000 * S, (3,)))
    assert agg._book_clock() < NOW + 1_000_000 * S, "the median, not the maximum"


def test_the_tail_flush_writes_the_restored_silent_books_once_a_ticking_book_is_an_interval_ahead(tmp_path):
    agg = _restarted(tmp_path, (2, 5, 10), OLD)
    agg._process_tick(_tick(NOW, (10,)))
    assert all(agg._pending_rows.get(b) for b in (2, 5)), "the restored books hold rows to flush"
    flushed = []
    agg._flush_book_parquet = flushed.append
    agg._tail_flush_stalled_books()
    assert sorted(flushed) == [2, 5], "the silent books flush and the ticking one keeps filling its page"


def test_the_tail_flush_holds_the_restored_books_until_a_book_ticks(tmp_path):
    agg = _restarted(tmp_path, (2, 5, 10), OLD)
    flushed = []
    agg._flush_book_parquet = flushed.append
    agg._tail_flush_stalled_books()
    assert flushed == []
