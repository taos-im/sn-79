"""One book with a bad timestamp must not stall every other book.

Book 70 held an exchange-scale timestamp (1.53e15 ns, 425 sim-hours) while the
other 128 books sat at ~2.45e12 (0.68h). The tail-flush test compared each book against the GLOBAL
MAX, so all 128 were 1,528,085s "behind" a 300s interval and flushed on every loop -- producing
1-3 second pages of ~217 rows against train_seq_len=512, which surfaces downstream as
"no trainable pages" with nothing actually wrong with training.
"""
import statistics


INTERVAL_NS = 300_000_000_000          # parquet_interval_ns: 5 sim-minutes
SIM_NOW = 2_455_000_000_000            # ~0.68 sim-hours, where the healthy books sit
POISON = 1_530_540_000_000_000         # ~425 sim-hours, exchange scale


def _healthy_books(n=128):
    # Books tick at slightly different times, all within a few seconds of each other.
    return {b: SIM_NOW - (b % 7) * 1_000_000_000 for b in range(n)}


def _stalled_by_max(last_ts):
    clock = max(last_ts.values())
    return [b for b, t in last_ts.items() if (clock - t) >= INTERVAL_NS]


def _stalled_by_median(last_ts):
    vals = sorted(v for v in last_ts.values() if v > 0)
    clock = vals[len(vals) // 2]
    return [b for b, t in last_ts.items() if (clock - t) >= INTERVAL_NS]


def test_max_clock_stalls_everything_when_one_book_is_poisoned():
    """The old behaviour, pinned so the regression is recognisable if it returns."""
    books = _healthy_books()
    books[70] = POISON
    stalled = _stalled_by_max(books)
    # 127 = every book except the poisoned one itself, which is zero from its own clock.
    assert len(stalled) == 127, len(stalled)
    assert 70 not in stalled


def test_median_clock_is_unmoved_by_one_poisoned_book():
    books = _healthy_books()
    books[70] = POISON
    stalled = _stalled_by_median(books)
    assert stalled == [], stalled                  # nothing stalled; the outlier cannot move a median


def test_median_clock_still_detects_a_genuinely_stalled_book():
    """The fix must not blind the check -- a book that really has stopped is still flushed."""
    books = _healthy_books()
    books[42] = SIM_NOW - (INTERVAL_NS * 2)        # genuinely 10 sim-minutes behind
    assert _stalled_by_median(books) == [42]


def test_median_survives_several_poisoned_books():
    books = _healthy_books()
    for b in (70, 71, 72):
        books[b] = POISON
    assert _stalled_by_median(books) == []
