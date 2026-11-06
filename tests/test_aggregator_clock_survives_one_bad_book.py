"""One book's outlier clock cannot hold the aggregator's clock, so it cannot stall every other book.

The tail flush flushes a book once it is a whole interval behind the aggregator's clock. That clock is the median of
the books' last timestamps, not their maximum: a single book whose timestamp sits far ahead of the rest leaves the
clock where the other books are, so none of them is flushed on every loop into pages too short to train on, while a
book that has stopped is still a whole interval behind and is flushed.
"""


INTERVAL_NS = 300_000_000_000          # parquet_interval_ns: 5 sim-minutes
SIM_NOW = 3_600_000_000_000            # one sim-hour, where the healthy books sit
OUTLIER = 1_000 * SIM_NOW              # a clock far ahead of every other book's
BOOKS = 16


def _healthy_books(n=BOOKS):
    # Books tick at slightly different times, all within a few seconds of each other.
    return {b: SIM_NOW - (b % 7) * 1_000_000_000 for b in range(n)}


def _stalled_by_max(last_ts):
    clock = max(last_ts.values())
    return [b for b, t in last_ts.items() if (clock - t) >= INTERVAL_NS]


def _stalled_by_median(last_ts):
    vals = sorted(v for v in last_ts.values() if v > 0)
    clock = vals[len(vals) // 2]
    return [b for b, t in last_ts.items() if (clock - t) >= INTERVAL_NS]


def test_max_clock_stalls_everything_when_one_book_is_an_outlier():
    """A clock taken as the maximum flags every other book: the case the median clock is chosen against."""
    books = _healthy_books()
    books[5] = OUTLIER
    stalled = _stalled_by_max(books)
    # every book except the outlier itself, which is zero from its own clock
    assert len(stalled) == BOOKS - 1, len(stalled)
    assert 5 not in stalled


def test_median_clock_is_unmoved_by_one_outlier_book():
    books = _healthy_books()
    books[5] = OUTLIER
    stalled = _stalled_by_median(books)
    assert stalled == [], stalled                  # nothing stalled; the outlier cannot move a median


def test_median_clock_still_detects_a_genuinely_stalled_book():
    """The median clock still flags a book that has stopped."""
    books = _healthy_books()
    books[42] = SIM_NOW - (INTERVAL_NS * 2)        # genuinely 10 sim-minutes behind
    assert _stalled_by_median(books) == [42]


def test_median_survives_several_outlier_books():
    books = _healthy_books()
    for b in (5, 6, 7):
        books[b] = OUTLIER
    assert _stalled_by_median(books) == []
