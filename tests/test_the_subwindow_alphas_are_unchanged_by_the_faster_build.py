"""book_alphas_by_subwindow is faster and its output is unchanged.

It was 40% of the scoring time on a filled scoring window (65 million calls of its
per-entry bucket arithmetic) and runs in the live validator's reward compute. The build now computes each timestamp's
sub-window once per call. The requirement is identical output: this test holds it against the function as it
was, kept below verbatim, on histories that reach every edge.
"""
import random

from taos.im.validator.debeta import book_alphas_by_subwindow


# The function as shipped before row 39, verbatim.
def _reference(mtm_hist, invsum_hist, invn_hist, drift_hist, t_start, t_end, k):
    """The windowed alpha per (uid, book) in each of k equal sub-windows of [t_start, t_end].

    Built from the timestamped per-batch histories the rolling window already keeps, so it costs no
    new accumulator. Each sub-window strips its own drift; a sub-window in which the book did not
    trade is None. With k = 1 the single entry equals book_alphas_by_book over the same span.

    Returns:
        dict: ``{uid: {book: [alpha or None] * k}}``, or None when k <= 1 or the span is empty.
    """
    k = int(k)
    if k <= 1 or t_end <= t_start:
        return None
    span = float(t_end - t_start)

    def bucket(ts):
        i = int((float(ts) - float(t_start)) * k / span)
        return min(max(i, 0), k - 1)

    invn_k, drift_k = {}, {}
    for b, tsd in (invn_hist or {}).items():
        row = invn_k.setdefault(b, [0.0] * k)
        for ts, v in tsd.items():
            if t_start <= ts <= t_end:
                row[bucket(ts)] += v
    for b, tsd in (drift_hist or {}).items():
        row = drift_k.setdefault(b, [0.0] * k)
        for ts, v in tsd.items():
            if t_start <= ts <= t_end:
                row[bucket(ts)] += v
    out = {}
    for uid in set(mtm_hist or {}) | set(invsum_hist or {}):
        by_book = {}
        for b in set((mtm_hist or {}).get(uid, {})) | set((invsum_hist or {}).get(uid, {})):
            m_k, s_k = [0.0] * k, [0.0] * k
            for ts, v in (mtm_hist or {}).get(uid, {}).get(b, {}).items():
                if t_start <= ts <= t_end:
                    m_k[bucket(ts)] += v
            for ts, v in (invsum_hist or {}).get(uid, {}).get(b, {}).items():
                if t_start <= ts <= t_end:
                    s_k[bucket(ts)] += v
            n_row = invn_k.get(b, [0.0] * k)
            d_row = drift_k.get(b, [0.0] * k)
            by_book[b] = [(m_k[i] - (s_k[i] / n_row[i]) * d_row[i]) if n_row[i] > 0 else None
                          for i in range(k)]
        out[uid] = by_book
    return out


def _histories(seed, n_uids=12, n_books=9, n_ts=40, t0=1_000_000_000_000, step=5_000_000_000):
    rng = random.Random(seed)
    ts = [t0 + i * step for i in range(n_ts)]
    mtm, invsum = {}, {}
    for u in range(n_uids):
        for b in rng.sample(range(n_books), rng.randint(0, n_books)):
            mtm.setdefault(u, {})[b] = {t: rng.uniform(-3, 3) for t in rng.sample(ts, rng.randint(0, n_ts))}
            if rng.random() < 0.8:
                invsum.setdefault(u, {})[b] = {t: rng.uniform(-50, 50) for t in rng.sample(ts, rng.randint(0, n_ts))}
    # A uid with inventory and no mark-to-market, and books the counts do not cover.
    invsum.setdefault(n_uids, {})[0] = {ts[3]: 1.5}
    invn = {b: {t: float(rng.choice([0, 0, 1, 2, 7])) for t in ts} for b in range(n_books - 2)}
    drift = {b: {t: rng.uniform(-1, 1) for t in ts} for b in range(n_books)}
    return (mtm, invsum, invn, drift), ts


def _same(a, b):
    assert (a is None) == (b is None)
    if a is None:
        return
    assert a.keys() == b.keys()
    for u in a:
        assert a[u].keys() == b[u].keys()
        for bk in a[u]:
            assert a[u][bk] == b[u][bk], (u, bk, a[u][bk], b[u][bk])


def test_the_output_is_the_shipped_functions_on_every_edge():
    for seed in range(30):
        h, ts = _histories(seed)
        spans = [
            (ts[0], ts[-1]),
            (ts[5], ts[25]),
            (ts[5] + 1, ts[25] - 1),
            (ts[0] - 10**12, ts[-1] + 10**12),
            (ts[10], ts[10] + 3),
        ]
        for t_start, t_end in spans:
            for k in (1, 2, 3, 4, 6, 7):
                _same(book_alphas_by_subwindow(*h, t_start, t_end, k), _reference(*h, t_start, t_end, k))


def test_the_degenerate_calls_answer_as_before():
    h, ts = _histories(1)
    for args in ((ts[5], ts[5], 3), (ts[9], ts[2], 3), (ts[0], ts[-1], 1), (ts[0], ts[-1], 0)):
        assert book_alphas_by_subwindow(*h, *args) is None and _reference(*h, *args) is None
    empty = ({}, {}, {}, {})
    _same(book_alphas_by_subwindow(*empty, ts[0], ts[-1], 3), _reference(*empty, ts[0], ts[-1], 3))
    _same(book_alphas_by_subwindow(None, None, None, None, ts[0], ts[-1], 3), _reference(None, None, None, None, ts[0], ts[-1], 3))
