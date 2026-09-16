"""De-beta scoring component: balanced two-sided spread capture (making) + directional-skill
(kappa-of-alpha), drift-riding stripped. Combined, rank-weighted by w_make (operator dial).

Offline-validated on mainnet-sim L3. Two properties are deliberate and load-bearing:

  Making credit is BALANCED and proportional: two-sided capture measured against a centred mid, with a
  counterparty-concentration discount (p11_discount) so credit earned repeatedly from the same small set
  of counterparties is worth less than the same capture spread across the field. Resistance here is
  economic rather than identity-based, and the making weight is kept modest.

  Directional skill requires magnitude as well as consistency: a per-book alpha must clear a magnitude
  floor before it counts, and skill is assessed across books rather than from a single one. Consistency
  at negligible size is not skill.

ALWAYS COMPUTED, WEIGHTED IN BY ONE DIAL. trade.py accumulates per-uid/book buy/sell capture and
the inventory mark-to-market path unconditionally; persistence.py and report.py carry the sums
across restarts and snapshots; reward.py computes the scores every cycle and blends them into the
trading score by --scoring.debeta.weight (default 0.0: legacy emissions with the decomposition
fully published; 1.0: full replacement). The old --scoring.debeta.enabled boolean is deprecated
and ignored.
"""
import statistics
from bisect import bisect_left, insort
from collections import defaultdict, deque


# --- timestamped-history helpers (mirror trade.py's volume-history machinery) ------------------------
# Every windowed de-beta accumulator is a running SUM plus a {..., ts: increment} history, so it can be
# live-pruned and shifted at a sim boundary exactly like trade_volumes. Invariant: running == sum(history
# within window). These helpers keep that invariant. See SCORING_PATH_ARCHITECTURE.md §7.

def _hadd2(hist, k1, k2, ts, val):
    d = hist.setdefault(k1, {}).setdefault(k2, {})
    d[ts] = d.get(ts, 0.0) + val


def _hadd1(hist, k, ts, val):
    d = hist.setdefault(k, {})
    d[ts] = d.get(ts, 0.0) + val


def prune_hist_2level(hist, running, threshold):
    """hist {k1:{k2:{ts:val}}}, running {k1:{k2:val}}. Drop ts<threshold, subtract pruned mass from
    running. Keeps running == sum(kept). A pair whose history empties is removed from both maps: it has
    no fills inside the window, and a key left behind would carry a running total of a floating-point
    residue into every pool built from the map (the skill floor's median, kappa's book count), where
    such residues accumulate until they outnumber the live pairs.

    Args:
        hist: ``{k1: {k2: {ts: val}}}`` history, pruned in place.
        running: ``{k1: {k2: val}}`` running totals, reduced by the pruned mass.
        threshold: Timestamps strictly below this are dropped.
    """
    for k1 in list(hist):
        d2 = hist[k1]
        for k2 in list(d2):
            tsd = d2[k2]
            pruned = 0.0
            keep = {}
            for ts, v in tsd.items():
                if ts >= threshold:
                    keep[ts] = v
                else:
                    pruned += v
            if len(keep) != len(tsd):
                d2[k2] = keep
                if pruned and k1 in running and k2 in running.get(k1, {}):
                    running[k1][k2] = running[k1][k2] - pruned
            if not keep:
                del d2[k2]
                if k1 in running:
                    running[k1].pop(k2, None)
        if not d2:
            del hist[k1]
            if k1 in running and not running[k1]:
                del running[k1]


def prune_hist_1level(hist, running, threshold):
    """hist {k:{ts:val}}, running {k:val}. A key whose history empties is removed from both maps, for
    the reason given on prune_hist_2level.

    Args:
        hist: ``{k: {ts: val}}`` history, pruned in place.
        running: ``{k: val}`` running totals, reduced by the pruned mass.
        threshold: Timestamps strictly below this are dropped.
    """
    for k in list(hist):
        tsd = hist[k]
        pruned = 0.0
        keep = {}
        for ts, v in tsd.items():
            if ts >= threshold:
                keep[ts] = v
            else:
                pruned += v
        if len(keep) != len(tsd):
            hist[k] = keep
            if pruned and k in running:
                running[k] = running[k] - pruned
        if not keep:
            del hist[k]
            running.pop(k, None)


def shift_hist_2level(hist, running, old_ts, new_ts, threshold):
    """Remap every ts (new = new_ts-(old_ts-ts)); prune those still outside threshold on the new clock;

    Args:
        hist: ``{k1: {k2: {ts: val}}}`` history, remapped and pruned in place.
        running: ``{k1: {k2: val}}`` running totals, reduced by any pruned mass.
        old_ts: The clock the stored timestamps were recorded on.
        new_ts: The clock they are being remapped onto.
        threshold: Remapped timestamps strictly below this are dropped.
    subtract pruned mass from running. Values unchanged, so running stays == sum(kept)."""
    for k1, d2 in list(hist.items()):
        for k2, tsd in list(d2.items()):
            newd = {}
            pruned = 0.0
            for ts, v in tsd.items():
                nts = new_ts - (old_ts - ts)
                if nts >= threshold:
                    newd[nts] = newd.get(nts, 0.0) + v
                else:
                    pruned += v
            d2[k2] = newd
            if pruned and k1 in running and k2 in running.get(k1, {}):
                running[k1][k2] = running[k1][k2] - pruned
            if not newd:
                del d2[k2]
                if k1 in running:
                    running[k1].pop(k2, None)
        if not d2:
            del hist[k1]
            if k1 in running and not running[k1]:
                del running[k1]


def shift_hist_1level(hist, running, old_ts, new_ts, threshold):
    """Age a one-level history window forward, dropping entries older than the threshold.

    Args:
        hist: The history mapping being aged.
        running: The running total to adjust as entries drop.
        old_ts: Previous window edge.
        new_ts: New window edge.
        threshold: Age limit for retained entries.
    """
    for k, tsd in list(hist.items()):
        newd = {}
        pruned = 0.0
        for ts, v in tsd.items():
            nts = new_ts - (old_ts - ts)
            if nts >= threshold:
                newd[nts] = newd.get(nts, 0.0) + v
            else:
                pruned += v
        hist[k] = newd
        if pruned and k in running:
            running[k] = running[k] - pruned


def sum_hist_2level(hist):
    """{k1:{k2:{ts:val}}} -> {k1:{k2: sum}}. Rebuilds a running sum from its history (used on load, so
    running == sum(history) holds by construction)."""
    return {k1: {k2: sum(tsd.values()) for k2, tsd in d2.items()} for k1, d2 in hist.items()}


def sum_hist_1level(hist):
    """{k:{ts:val}} -> {k: sum}."""
    return {k: sum(tsd.values()) for k, tsd in hist.items()}


def centered_mid(prices, W):
    """Symmetric (non-lagging) benchmark mid. In a pure trend it equals the current price, so capture
    measures only deviation from the trend line (drift removed). W = half-window in trades."""
    n = len(prices)
    if n == 0:
        return []
    csum = [0.0]
    for p in prices:
        csum.append(csum[-1] + p)
    mid = []
    for i in range(n):
        lo = max(0, i - W)
        hi = min(n, i + W + 1)
        mid.append((csum[hi] - csum[lo]) / (hi - lo))
    return mid


def caps_from_fills(fills):
    """fills: iterable of (buyer, seller, price, mid, volume). Returns {agent: [buy_cap, sell_cap]}.
    Per-trade buyer capture (mid-price)*v + seller capture (price-mid)*v == 0 (zero-sum: no making
    is created ex nihilo; a ring can only TRANSFER it, hence H1)."""
    caps = defaultdict(lambda: [0.0, 0.0])
    for (buyer, seller, price, mid, vol) in fills:
        if buyer is not None:
            caps[buyer][0] += (mid - price) * vol
        if seller is not None:
            caps[seller][1] += (price - mid) * vol
    return caps


def _uid(x):
    """Coerce a maker/taker id to int, mapping None (a pool/AMM side with no miner) to -1 so the >=0 miner
    guards skip it. Sim 't' events carry int ids (identity); exchange ET notices may carry None."""
    return -1 if x is None else int(x)


def _attribute_capture(buy_sums, sell_sums, book_id, t, mid, buy_hist, sell_hist, ts):
    """Book one fill's capture against `mid`: buyer gets (mid-price)*q, seller the negation
    (zero-sum per fill under ANY mid, so a degraded mid can shift capture between counterparties
    but never mint it)."""
    buy_cap = (mid - float(t["p"])) * float(t["q"])
    ma = t.get("Ma", -1)
    ta = t.get("Ta", -1)
    buyer, seller = (ta, ma) if int(t["s"]) == 0 else (ma, ta)
    if buyer is not None and buyer >= 0:
        buy_sums[buyer][book_id] = buy_sums[buyer].get(book_id, 0.0) + buy_cap
        if buy_hist is not None:
            _hadd2(buy_hist, buyer, book_id, ts, buy_cap)
    if seller is not None and seller >= 0:
        sell_sums[seller][book_id] = sell_sums[seller].get(book_id, 0.0) - buy_cap
        if sell_hist is not None:
            _hadd2(sell_hist, seller, book_id, ts, -buy_cap)


CAPTURE_FLUSH_NS = 60_000_000_000  # force-finalize a pending fill after 60 sim-s without W forward prints


def _finalize_ready(st, book_id, buy_sums, sell_sums, W, buy_hist, sell_hist, ts,
                    now_ns, flush_ns, force=False):
    """Finalize every pending fill whose forward window is complete (W prints arrived after it),
    stale (older than flush_ns of sim time, so a quiet book cannot hold capture hostage), or
    force-flushed. The window truncates only at genuine stream edges, never at batch edges."""
    prices, pend = st["prices"], st["pend"]
    base, n = st["base"], st["n"]
    while pend:
        idx, arrive_ns, t = pend[0]
        if not (force or n - 1 >= idx + W
                or (now_ns is not None and arrive_ns is not None and now_ns - arrive_ns >= flush_ns)):
            break
        lo = max(0, idx - W) - base
        hi = min(n, idx + W + 1) - base
        window = prices[lo:hi]
        _attribute_capture(buy_sums, sell_sums, book_id, t, sum(window) / len(window),
                           buy_hist, sell_hist, ts)
        pend.pop(0)
    new_base = max(base, (pend[0][0] if pend else n) - W)
    if new_base > base:
        del prices[:new_base - base]
        st["base"] = new_base


def accumulate_book_capture(buy_sums, sell_sums, book_id, trades, W, *,
                            buy_hist=None, sell_hist=None, ts=None,
                            mid_state=None, flush_ns=CAPTURE_FLUSH_NS):
    """Accumulate per-uid two-sided spread capture for ONE book's ordered trade batch into
    buy_sums / sell_sums ({uid: {book: cap}}), in place. Each `trade` is dict-like with keys
    p (price), q (quantity), s (side), Ma (maker uid), Ta (taker uid). side==0 => taker buys /
    maker sells; side==1 => maker buys / taker sells. buyer captures (mid-price)*q, seller
    (price-mid)*q, vs a non-lagging centered mid. Self-trades (Ma==Ta) earn nothing but their
    prints still shape the mid (first-line wash guard; cross-uid rings are netted by P11).

    mid_state=None (offline callers passing a whole run as one batch): the centered mid is computed
    over THIS batch, truncating at its edges. At live print density that truncation is the rule,
    not the exception (on a live board: median 4 prints per state update,
    90.9% of batches degrade to the plain batch mean, 12.8% of prints see a full window), so live
    callers pass mid_state ({book: st}) to carry the print window ACROSS batches: each fill is held
    (bounded: at most W pending fills and 2W+1 prices per book) until W forward prints arrive, then
    booked against its full centered window at the finalizing call's ts. A fill with no W forward
    prints inside flush_ns of sim time finalizes with a truncated forward side, so the lag is
    bounded and a quiet book still settles. Batch shape then cannot move capture: only the print
    stream itself can.

    When buy_hist/sell_hist ({uid:{book:{ts:incr}}}) + ts are given, increments are also recorded
    at ts so the sums can be windowed (pruned/shifted)."""
    if mid_state is None:
        prices = [float(t["p"]) for t in trades]
        mids = centered_mid(prices, W)
        for t, mid in zip(trades, mids):
            if t.get("Ma", -1) == t.get("Ta", -1):
                continue
            _attribute_capture(buy_sums, sell_sums, book_id, t, mid, buy_hist, sell_hist, ts)
        return
    st = mid_state.setdefault(book_id, {"prices": [], "pend": [], "base": 0, "n": 0})
    for t in trades:
        st["prices"].append(float(t["p"]))
        if t.get("Ma", -1) != t.get("Ta", -1):
            st["pend"].append((st["n"], ts, t))
        st["n"] += 1
    _finalize_ready(st, book_id, buy_sums, sell_sums, W, buy_hist, sell_hist, ts, ts, flush_ns)


def flush_capture_state(mid_state, buy_sums, sell_sums, W, *,
                        buy_hist=None, sell_hist=None, ts=None,
                        flush_ns=CAPTURE_FLUSH_NS, force=False):
    """Finalize stale pending fills on EVERY book (books with no new trades never reach
    accumulate_book_capture, so the live loop calls this each cycle; force=True drains everything,
    for offline end-of-run and the sim-boundary re-base)."""
    for book_id, st in mid_state.items():
        _finalize_ready(st, book_id, buy_sums, sell_sums, W, buy_hist, sell_hist, ts,
                        ts, flush_ns, force=force)


def balanced_reward(caps):
    """{agent: [buy_cap, sell_cap]} -> {agent: 2*min(buy_cap, sell_cap)} clamped at 0. The MAKING
    (liquidity) component: genuine two-sided spread capture; a one-sided accumulator/drift-rider -> ~0."""
    return {a: max(0.0, 2.0 * min(bc, sc)) for a, (bc, sc) in caps.items()}


def balanced_reward_per_book(capture_buy_sums, capture_sell_sums, uids):
    """Two-sided capture summed PER BOOK: sum_b 2*min(buy_b, sell_b), clamped at 0.

    Two-sidedness must hold on each book separately: capture is a fill-level quantity, so summing
    the sides before taking the minimum does not measure it. Strictly tighter than balanced_reward,
    since min is subadditive and the clamp is on the total, so this can only lower a score.
    """
    out = {}
    for u in uids:
        cb = capture_buy_sums.get(u) or {}
        cs = capture_sell_sums.get(u) or {}
        tot = 0.0
        for b in set(cb) | set(cs):
            # The clamp belongs on the total, not here. Capture is (mid - p) * q and is genuinely
            # negative on 43% of per-book values, so clamping inside the loop would drop a miner's
            # loss-making books instead of counting them, breaking the subadditivity the docstring
            # relies on.
            tot += 2.0 * min(cb.get(b, 0.0), cs.get(b, 0.0))
        out[u] = max(0.0, tot)
    return out


# Making resistance is ECONOMIC and identity-neutral: on the exchange, capital is really committed and
# really lost by a counterparty, and credit is proportional rather than winner-take-all. Identity-based
# grouping is deliberately NOT used. In simulation, where capital is free, a residual is inherent to any
# spread-capture reward; it is bounded by keeping the making weight modest.


def kappa_of_alpha(book_alphas):
    """DIRECTIONAL SKILL. book_alphas = per-book benchmark-relative (excess) PnL for one agent
    (each = book_total_pnl - mean_inventory*book_drift). Consistency across independent books
    (MAD-normalised downside-adjusted mean): a skilled forecaster is consistently positive; a
    drift-rider is positive only where drift helped (inconsistent) -> low. Needs >=4 books."""
    v = [float(x) for x in book_alphas]
    if len(v) < 4:
        return 0.0
    med = statistics.median(v)
    mad = max(statistics.median([abs(x - med) for x in v]), 1e-9)
    r = [x / mad for x in v]
    mean = sum(r) / len(r)
    sd = (sum((x - mean) ** 2 for x in r) / len(r)) ** 0.5
    lpm3 = sum(max(-x, 0.0) ** 3 for x in r) / len(r)
    reg = (abs(mean) + sd) ** 3 * 1e-3 + 1e-9
    return mean / (lpm3 + reg) ** (1.0 / 3.0)


def kappa_floored(book_alphas, floor):
    """Magnitude floor on directional skill.

    kappa measures consistency and is blind to magnitude, so a book only counts toward it once its
    |alpha| clears `floor`, and skill is assessed across several qualifying books rather than one.
    Consistency at negligible size is not skill. `floor` is scaled to real per-book traded notional and
    calibrated offline against the observed per-book alpha distribution."""
    q = [float(x) for x in book_alphas if abs(float(x)) >= floor]
    return kappa_of_alpha(q) if len(q) >= 4 else 0.0


def _rank01(vals):
    """Ties share their block's MINIMUM rank: equal raw values must map to equal ranks (uid order
    must never decide emissions; 88% of miners share making_raw==0.0 on a live board), and a
    zero-making mass earns zero relative making credit rather than a positional lottery."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    rank = [0.0] * len(vals)
    denom = max(len(vals) - 1, 1)
    pos = 0
    for k, i in enumerate(order):
        if k and vals[i] != vals[order[k - 1]]:
            pos = k
        rank[i] = pos / denom
    return rank


# Counterparty bucket for takers that are not miners (background agents, external exchange flow, pool
# fills). It counts in a maker's total flow but is never a "top counterparty": concentration is only
# ever measured over miner takers, against the whole flow.
MARKET_FLOW = -1


def accumulate_counterparties(cp, book_id, trades, *, cp_hist=None, ts=None):
    """P11 input: accumulate per-MAKER counterparty (taker) volume over ONE book's trade batch, IN
    PLACE (carried across batches). cp: {maker_uid: {taker_uid: vol}}. A maker fed by a dedicated
    counterparty concentrates here; a maker hit by the diverse market spreads out. Self-trades and
    background agents (<0) are excluded (same as the capture guard).

    When cp_hist ({maker:{taker:{ts:vol}}}) + ts are given, the per-batch volume is recorded at ts for

    Args:
        cp: ``{maker_uid: {taker_uid: vol}}`` counterparty volumes, accumulated in place.
        book_id: The book this batch belongs to.
        trades: Ordered trade batch, each dict-like with ``p``, ``q``, ``s``, ``Ma``, ``Ta``.
    windowing (making window). Offline callers pass none.

    Non-miner takers (simulation background agents, exchange external flow and pool fills, all with a
    negative or missing id) are kept under the single MARKET_FLOW bucket. They are the diverse market
    the discount is meant to leave untouched: dropping them judged a maker on its miner slice alone.
    A maker quoting into the broad market can take almost all of its flow from non-miner
    counterparties, and judging it on the miner slice alone read that as concentration."""
    for t in trades:
        ma = _uid(t.get("Ma", -1))
        ta = _uid(t.get("Ta", -1))
        if ma < 0 or ma == ta:
            continue
        if ta < 0:
            ta = MARKET_FLOW
        q = float(t["q"])
        d = cp.setdefault(ma, {})
        d[ta] = d.get(ta, 0.0) + q
        if cp_hist is not None:
            _hadd2(cp_hist, ma, ta, ts, q)


def et_book_batches(notices, seen_tids, ts):
    """Regroup exchange ET settled-fill notices into per-book ordered trade batches for de-beta.
    notices: {uid: [notice, ...]} from state.notices; seen_tids: {(book, trade_id): ts} carried across calls
    (mutated) so a fill counted once is never recounted -- neutralizes the maker+taker duplicate listing
    and the deliberate redelivery of settled fills across blocks; ts: current sample ts. Returns
    {book_id: [{p,q,s,Ma,Ta,i}, ...]} ordered by trade id (monotone engine mint order), the shape the
    accumulators expect. Pool fills carry Ma or Ta = None (no miner on that side); the accumulators skip

    Args:
        notices: ``{uid: [notice, ...]}`` from ``state.notices``.
        seen_tids: ``{(book, trade_id): ts}`` carried across calls and mutated, so no fill is recounted.
        ts: The current sample timestamp.

    Returns:
        dict: ``{book_id: [{p, q, s, Ma, Ta, i}, ...]}`` ordered by trade id.
    the None side via _uid. Sim mode has no ET notices, so this yields {} and the sim path is unchanged."""
    by_book = {}
    for uid_notices in notices.values():
        for n in uid_notices:
            if n.get("y") != "ET":
                continue
            tid = n.get("i")
            # A trade id names a trade within its book: the exchange mints them from one counter
            # today, but the ledger must not depend on that.
            seen_key = (int(n["b"]), tid)
            if seen_key in seen_tids:
                continue
            seen_tids[seen_key] = ts
            by_book.setdefault(int(n["b"]), []).append(
                # Mf/Tf carried through so making and skill CAN be measured net of fees. The
                # accumulators ignore unknown keys, so adding them changes no score by itself; the
                # dynamic fee policy is live and material (maker fees observed inverting from
                # -0.332% on ordinary fills to +0.626% on sweep fills), and legacy FIFO PnL counts
                # fees while de-beta does not, so replacing kappa+PnL silently drops them.
                {"p": n["p"], "q": n["q"], "s": n["s"], "Ma": n.get("Ma"), "Ta": n.get("Ta"),
                 "i": tid, "Mf": n.get("Mf"), "Tf": n.get("Tf")}
            )
    for b in by_book:
        by_book[b].sort(key=lambda e: (e["i"] is None, e["i"] if e["i"] is not None else 0))
    return by_book


def counterparty_ec(cp, maker, topk=2):
    """P11 excess concentration (leave-one-out): maker's top-k counterparty share MINUS those
    takers' share of the market's OTHER makers' flow. ~1.0 for a dedicated feeder (a big share of the
    maker, ~0 of everyone else); ~0 for a maker served by the diverse market. Identity-agnostic by

    Args:
        cp: ``{maker_uid: {taker_uid: vol}}`` counterparty volumes.
        maker: The maker uid being scored.
        topk: How many top counterparties the share is computed over.

    Returns:
        float: Excess concentration, ~1.0 for a feeder-fed maker and ~0 for a diverse one.
    design: it reads flow concentration, so it needs no account linkage to see a dedicated feeder."""
    my = cp.get(maker, {})
    tot = sum(my.values())
    if tot <= 0:
        return 0.0
    other = {}
    for mk, cps in cp.items():
        if mk == maker:
            continue
        for t, v in cps.items():
            other[t] = other.get(t, 0.0) + v
    other_tot = sum(other.values()) or 1.0
    # Shares are over the WHOLE flow, MARKET_FLOW included; the top-k is drawn from miner takers only.
    top = sorted((t for t in my if t != MARKET_FLOW), key=lambda t: my[t], reverse=True)[:topk]
    my_share = sum(my[t] for t in top) / tot
    mkt_share = sum(other.get(t, 0.0) for t in top) / other_tot
    return my_share - mkt_share


def p11_discount(own, cp, uids, strength, topk=2):
    """Apply the P11 counterparty-diversity discount to the making leg: making *= (1 - strength*EC+),
    EC+ = max(0, excess concentration). strength 0 disables. A feeder-fed maker (EC~1) loses ~strength
    of its making; a diverse maker (EC~0) is untouched. Applied to the MAGNITUDE before ranking, so a

    Args:
        own: The making magnitudes being discounted, mutated in place.
        cp: ``{maker_uid: {taker_uid: vol}}`` counterparty volumes.
        uids: The uids to score.
        strength: Discount strength; 0 disables.
        topk: Counterparty count for the concentration measure.
    feeder cannot rank-buy a top making slot it did not earn from the diverse market."""
    if strength <= 0:
        return own
    # O(total entries), not O(makers x total entries): build the GLOBAL taker aggregate once and
    # derive each maker's leave-one-out by subtraction. Numerically identical to the per-maker
    # rebuild (asserted in tests); the naive form cost 609ms of a 629ms scoring cycle at 251 uids.
    g = {}
    g_tot = 0.0
    for cps in cp.values():
        for t, v in cps.items():
            g[t] = g.get(t, 0.0) + v
            g_tot += v
    out = {}
    for u in uids:
        my = cp.get(u, {})
        tot = sum(my.values())
        if tot <= 0:
            out[u] = own.get(u, 0.0)
            continue
        other_tot = (g_tot - tot) or 1.0
        # Same rule as counterparty_ec: the market bucket is flow, never a counterparty.
        top = sorted((t for t in my if t != MARKET_FLOW), key=lambda t: my[t], reverse=True)[:topk]
        my_share = sum(my[t] for t in top) / tot
        mkt_share = sum((g.get(t, 0.0) - my.get(t, 0.0)) for t in top) / other_tot
        ec = max(0.0, my_share - mkt_share)
        out[u] = own.get(u, 0.0) * max(0.0, 1.0 - strength * ec)
    return out


SKILL_RANK_SCOPES = ("positives", "whole")


PRESENCE_WINDOW_DEFAULT = 50


def absent_uids(presence, window=PRESENCE_WINDOW_DEFAULT):
    """The uids whose last `window` query outcomes hold no successful response.

    `presence` is {uid: outcomes}, each outcome truthy for a valid (HTTP 200) response, kept by
    forward.update_stats as a bounded deque, or a list when shipped to the scoring child. A uid with
    fewer than `window` outcomes is present: a fresh validator has not seen enough to judge, so a
    restart never zeroes the board. One success anywhere in the window makes the uid present again.

    Without the gate a uid that had stopped answering kept earning on the timing of resting fills
    already inside the window: the skill leg is built from fills, so credit continues to accrue after
    the agent goes away.

    Args:
        presence: ``{uid: outcomes}``.
        window: Number of most recent outcomes that must all be failures.
    Returns:
        set: The absent uids.
    """
    n = max(int(window or PRESENCE_WINDOW_DEFAULT), 1)
    out = set()
    for u, outcomes in (presence or {}).items():
        seq = list(outcomes)
        if len(seq) >= n and not any(bool(x) for x in seq[-n:]):
            out.add(int(u))
    return out


def absent_now(validator):
    """The absent set from a live validator's presence records and dials, sorted for the wire; None when
    the gate is off (so the child applies none either)."""
    dcfg = getattr(getattr(getattr(validator, "config", None), "scoring", None), "debeta", None)
    gate = getattr(dcfg, "presence_gate", None)
    if gate is not None and int(gate) == 0:
        return None
    window = int(getattr(dcfg, "presence_window", None) or PRESENCE_WINDOW_DEFAULT)
    return sorted(absent_uids(getattr(validator, "miner_presence", {}) or {}, window))


def _rank_positive_leg(values, scope, dial):
    if scope not in SKILL_RANK_SCOPES:
        raise ValueError(f"{dial} must be one of {SKILL_RANK_SCOPES}, got {scope!r}")
    if scope == "whole":
        return _rank01([max(0.0, float(v)) for v in values])
    idx = [i for i, v in enumerate(values) if v > 0]
    out = [0.0] * len(values)
    if len(idx) == 1:
        # A lone positive is the best there is: _rank01 of one value is 0, which would pay the only
        # maker or the only skilled trader nothing on that leg (seen in the warm-up gate).
        out[idx[0]] = 1.0
    elif idx:
        for i, r in zip(idx, _rank01([float(values[i]) for i in idx])):
            out[i] = r
    return out


def rank_making(making, scope="positives"):
    """The making leg's rank, the twin of rank_skill. A uid with no two-sided capture ranks 0.

    scope="positives" (default): the positive makings are ranked among themselves, lowest positive 0,
    highest 1, a lone positive 1. scope="whole": ranked over the whole pool, the rule shipped previously, kept for
    rollback. Under "whole" the smallest positive making inherits the rank of the entire zero-maker
    block, and on both networks most of the pool makes nothing, so negligible two-sided capture is
    materially overpaid. Ranking among positives removes that without disturbing the top of the
    board.

    Args:
        making: Per-uid making values (post floor), any order.
        scope: "positives" or "whole".
    Returns:
        list: Ranks in [0, 1], aligned to `making`.
    """
    return _rank_positive_leg(making, scope, "making_rank_scope")


def rank_skill(skill, scope="positives"):
    """The skill leg's rank. Non-positive skill always ranks 0 (no directional skill, no credit).

    scope="positives" (default): the positive skills are ranked among themselves, lowest positive 0,
    highest 1, a lone positive 1. scope="whole": the clamped skills are ranked over the whole pool, the rule shipped previously,
    kept for rollback. Under "whole" the smallest positive skill inherits the rank of the entire
    non-positive block, so a negligible skill collects a mid score and near-zero skills square-wave as
    their sign flips between boards. Ranking among positives removes both effects without disturbing
    the top of the board.

    Args:
        skill: Per-uid skill values, any order.
        scope: "positives" or "whole".
    Returns:
        list: Ranks in [0, 1], aligned to `skill`.
    """
    return _rank_positive_leg(skill, scope, "skill_rank_scope")


def making_magnitude_floor(making, scale=0.5):
    """Magnitude floor for the MAKING rank, the making-side twin of median_abs_floor: scale times the
    median of the strictly positive making values across the field. Rank is magnitude-blind, so a
    two-sided quoter of negligible size ranked beside the field's real makers on testnet:
    two uids with 34k and 19k of daily maker volume (0.02 per cent of the scored field's) carried
    making raws of 5.4 and 3.8 against a median of about 200 and ranked 0.77 and 0.68 because only 8
    of 23 uids made at all. Below the floor a uid's making enters the rank as 0. 0 disables.

    Args:
        making: Per-uid making values (post-P11), any order.
        scale: Multiple of the positive-making median.
    Returns:
        float: The floor, 0.0 when scale is 0 or nobody made.
    """
    if scale <= 0:
        return 0.0
    pos = [float(m) for m in making if m > 0]
    if not pos:
        return 0.0
    return float(scale) * statistics.median(pos)


def debeta_scores(capture_buy_sums, capture_sell_sums, book_alphas_by_uid, floor=0.0, w_make=0.65,
                  cp=None, p11_strength=0.0, detail=None, making_floor_scale=0.0,
                  skill_rank_scope="positives", making_rank_scope="positives"):
    """Full per-uid de-beta score. making = per-uid two-sided spread capture, combined by RANK. Rank
    rather than magnitude-proportional, because proportional combining re-concentrates reward on the
    largest flow; and per-uid rather than netted across linked accounts, because grouping by identity is
    deliberately not used here. skill = floored kappa-of-alpha. Making resistance is economic, with rank
    bounding any one participant's share; the simulation free-capital residual is inherent to a
    spread-capture reward and is bounded by a modest w_make.
    P11: when cp (counterparty volumes) + p11_strength>0 are given, the making leg is discounted by
    excess counterparty concentration (feeder-ring defence) BEFORE ranking.

    capture_buy_sums/capture_sell_sums: {uid: {book: cap}}. book_alphas_by_uid: {uid: [per-book alpha]}.

    Args:
        capture_buy_sums: ``{uid: {book: capture}}`` buyer-side spread capture.
        capture_sell_sums: ``{uid: {book: capture}}`` seller-side spread capture.
        book_alphas_by_uid: ``{uid: [alpha per book]}`` drift-stripped alphas.
        floor: Kappa floor applied to the skill leg.
        w_make: Weight of the making leg in the combined score.
        cp: Counterparty volumes for the P11 discount, or None.
        p11_strength: P11 discount strength; 0 disables.
        making_floor_scale: Making magnitude floor as a multiple of the positive-making median
            (making_magnitude_floor); 0 disables.
        skill_rank_scope: How the skill leg is ranked, see rank_skill: "positives" (default) ranks the
            positive skills among themselves, "whole" ranks the clamped skills over the whole pool.
            Non-positive skill ranks 0 either way.
        making_rank_scope: How the making leg is ranked, see rank_making: "positives" (default) ranks
            the positive makings among themselves, "whole" ranks over the whole pool. Zero making
            ranks 0 either way.

    Returns:
        dict: ``{uid: combined score}``.
    cp: {maker_uid: {taker_uid: vol}}. Returns {uid: combined_score in [0,1]}."""
    uids = sorted(set(capture_buy_sums) | set(capture_sell_sums) | set(book_alphas_by_uid))
    if detail is not None:
        # Cleared BEFORE the empty-input return, not after: callers reuse one dict across cycles,
        # so an early return that skipped the clear would republish the previous cycle's legs.
        detail.clear()
    if not uids:
        return {}
    # Two-sidedness is required per book, not across global sums, and is deliberately not
    # configurable. Rollback lives one level up at scoring.debeta.enabled.
    own = balanced_reward_per_book(capture_buy_sums, capture_sell_sums, uids)
    pre_p11 = dict(own)
    if cp is not None and p11_strength > 0:
        own = p11_discount(own, cp, uids, p11_strength)
    making = [own.get(u, 0.0) for u in uids]
    skill = [kappa_floored(book_alphas_by_uid.get(u, []), floor) for u in uids]
    # What enters the two ranks. SKILL: only positive skill earns skill rank (rank_skill). Before the
    # Under the earlier clamp, a skill of exactly 0.0 (no book clears the magnitude floor) ranked ABOVE every
    # negative trader: testnet rung 2, 13 of 23 scored uids negative, 3 exactly 0.0, 4 positive, so the
    # zero block ranked 0.73 and at w_make 0.30 earned 0.51 of the score for no directional exposure.
    # The clamp alone then left the smallest positive skill on the step above the whole non-positive
    # block (see rank_skill), which "positives", the default scope, removes. MAKING below the
    # magnitude floor enters the rank as 0 for the same reason on the other leg (see
    # making_magnitude_floor). The raw legs in `detail` are the unclamped, unfloored values, so a
    # miner can still see what was measured.
    making_floor = making_magnitude_floor(making, making_floor_scale)
    making_ranked = [m if m >= making_floor else 0.0 for m in making] if making_floor > 0 else list(making)
    # Both legs rank their positives among themselves by default: the pool is mostly zero makers on
    # both networks (see rank_making), so a whole-pool making rank hands the smallest positive maker
    # the rank of the whole zero block, the making-side twin of the skill cliff.
    rm = rank_making(making_ranked, making_rank_scope)
    rs = rank_skill(skill, skill_rank_scope)
    comb = [w_make * rm[i] + (1.0 - w_make) * rs[i] for i in range(len(uids))]
    if detail is not None:
        # The SAME numbers the score was built from, not a recomputation: the dashboards publish
        # these and a recomputation could disagree with what was emitted, so
        # making_rank*w + skill_rank*(1-w) reproduces the score exactly.
        for i, u in enumerate(uids):
            base = pre_p11.get(u, 0.0)
            detail[u] = {
                "making_raw": making[i],          # POST-P11, i.e. what was ranked
                "making_rank": rm[i],
                "skill_raw": skill[i],
                "skill_rank": rs[i],
                "p11_factor": (making[i] / base) if base else 1.0,
                # Books whose |alpha| clears the floor: the de-beta coverage count. The kappa leg's
                # num_scored_books is a penalty-floor artifact (a constant for idle miners) and must
                # not stand in for this on any surface.
                "skill_books": sum(1 for a in (book_alphas_by_uid.get(u) or []) if abs(float(a)) >= floor),
            }
    return {uids[i]: comb[i] for i in range(len(uids))}


def combined_reward(making, skill, w_make=0.65):
    """Combined de-beta: w_make*rank01(making) + (1-w_make)*rank01(skill). Both legs are RANKS (0-1,
    comparable scales). w_make is the operator dial. Offline: drift-OPPOSING at every w_make; genuine
    top-skill decile preserved (reward-rank 0.83-0.95); de-concentrates (combined Gini 0.29).

    Making is RANK, NOT magnitude-proportional, by decision: (1) rank de-concentrates best
    (combined Gini 0.286 vs 0.62-0.68 for any magnitude leg, since making magnitude is Gini-0.89
    heavy-tailed); (2) rank BOUNDS a wash to top-rank reward (~one top-decile slot) which it cannot
    exceed, whereas magnitude-proportional lets a wash out-burn the field to approach the full making

    Args:
        making: ``{uid: making magnitude}``.
        skill: ``{uid: skill score}``.
        w_make: Weight of the making leg.

    Returns:
        dict: ``{uid: w_make*rank01(making) + (1-w_make)*rank01(skill)}``.
    weight (domination). So rank is both more de-concentrating AND more wash-safe against domination."""
    rm, rs = _rank01(list(making)), _rank01(list(skill))
    return [w_make * rm[i] + (1.0 - w_make) * rs[i] for i in range(len(rm))]


def accumulate_book_mtm(mtm, invsum, invn, inv, p_first, p_last, book_id, trades, *,
                        mtm_hist=None, invsum_hist=None, invn_hist=None,
                        drift=None, drift_hist=None, ts=None,
                        mark_state=None, mark_mode="last", mark_window=0):
    """Accumulate the drift-strip inputs for kappa-of-alpha over ONE book's ordered trade batch,
    IN PLACE and CARRIED ACROSS BATCHES (inv + p_last persist between publish intervals).
    Reconstructs, per miner, the MTM PnL path-integral (sum inv*dp), the inventory time-sum, and the
    price drift (sum dp), from the SAME fill stream as accumulate_book_capture. Each `trade` is dict-like
    with p (price), q (quantity), s (side), Ma (maker uid), Ta (taker uid): s==0 => taker buys / maker
    sells, s==1 => maker buys / taker sells. Only miner uids (>=0) are tracked. Faithful to the offline
    L3 reconstruction (build_debeta_report.process_l3): the alpha numerator is the MTM path integral,
    NOT realized PnL (realized double-subtracts drift for a holder).

    Drift: `drift` ({book: sum dp}) telescopes to p_last-p_first over the window, so it REPLACES the
    p_first/p_last extent for the windowed drift-strip (book_alphas_from_drift). p_first/p_last are still
    maintained for the offline full-run finalizer (book_alphas_from_mtm). When the *_hist ({...:{ts}}) +
    ts are given, per-batch increments are recorded at ts for windowing. A boundary re-base of p_last to

    Args:
        mtm: ``{uid: {book: pnl}}`` MTM path-integral, accumulated in place.
        invsum: ``{uid: {book: sum}}`` inventory time-sums, accumulated in place.
        invn: ``{book: n}`` trade counts, accumulated in place.
        inv: ``{uid: {book: inventory}}`` current inventories, carried across batches.
        p_first: ``{book: price}`` first seen price per book, carried across batches.
        p_last: ``{book: price}`` last seen price per book, carried across batches.
        book_id: The book this batch belongs to.
        trades: Ordered trade batch, each dict-like with ``p``, ``q``, ``s``, ``Ma``, ``Ta``.
        mark_state: ``{book: state}`` rolling settlement-mark state, carried across batches (M1).
        mark_mode: ``"last"`` (default, byte-identical to the pre-M1 path), ``"vwap"`` (rolling
            volume-weighted mean of the last mark_window prints) or ``"median"`` (rolling window
            median, robust to both price and volume outliers).
        mark_window: Window length in prints for vwap/median marking; <=0 disables.
    None means the first new-sim trade emits no dp, so the boundary jump never enters drift.

    M1 settlement-style marking (mark_mode != "last"): the marked series is a rolling reference
    over the last mark_window prints, so ONE print at an extreme price cannot revalue a holder's
    whole position - the same reason real venues settle on a window, not the last trade. "vwap"
    is the naive settlement analogue but is itself movable by a single large wash print (the
    unload leg drags the volume-weighted mean toward the manufactured price - measured in
    mark_impl_check); "median" needs a sustained majority of window prints to move, fusing the
    window mark (M1) with erroneous-print exclusion (M2). The drift accumulator and
    p_first/p_last then track the SAME marked series, so the drift-strip finalizers stay
    consistent (alpha = mtm - mean_inv * drift telescopes over the marked series either way).
    mark_state is NOT persisted: after a restart the window re-warms and the first trade emits
    no dp - it can miss a marking move, never invent one."""
    binv = inv[book_id]  # {uid: current signed inventory in this book}, carried across batches
    use_mark = mark_mode != "last" and mark_window > 0 and mark_state is not None
    if use_mark:
        st = mark_state.get(book_id)
        if st is None:
            st = mark_state[book_id] = {"pv": deque(), "vv": deque(), "pvs": 0.0, "vvs": 0.0,
                                        "pq": deque(), "sorted": [], "prev": None}
        prev = st["prev"]
    else:
        prev = p_last.get(book_id)
    for t in trades:
        p = float(t["p"])
        q = float(t["q"])
        if use_mark:
            if mark_mode == "vwap":
                st["pv"].append(p * q)
                st["vv"].append(q)
                st["pvs"] += p * q
                st["vvs"] += q
                if len(st["pv"]) > mark_window:
                    st["pvs"] -= st["pv"].popleft()
                    st["vvs"] -= st["vv"].popleft()
                p = st["pvs"] / st["vvs"] if st["vvs"] > 0 else p
            else:  # median
                sl = st["sorted"]
                insort(sl, p)
                st["pq"].append(p)
                if len(st["pq"]) > mark_window:
                    old = st["pq"].popleft()
                    del sl[bisect_left(sl, old)]
                m = len(sl) // 2
                p = sl[m] if len(sl) % 2 else 0.5 * (sl[m - 1] + sl[m])
        if prev is not None and p != prev:
            dp = p - prev
            for uid, iv in binv.items():
                if iv:
                    mtm[uid][book_id] = mtm[uid].get(book_id, 0.0) + iv * dp
                    if mtm_hist is not None:
                        _hadd2(mtm_hist, uid, book_id, ts, iv * dp)
            if drift is not None:
                drift[book_id] = drift.get(book_id, 0.0) + dp
                if drift_hist is not None:
                    _hadd1(drift_hist, book_id, ts, dp)
        ma = _uid(t.get("Ma", -1))
        ta = _uid(t.get("Ta", -1))
        buyer, seller = (ta, ma) if int(t["s"]) == 0 else (ma, ta)
        if buyer >= 0:
            binv[buyer] = binv.get(buyer, 0.0) + q
        if seller >= 0:
            binv[seller] = binv.get(seller, 0.0) - q
        for uid, iv in binv.items():
            invsum[uid][book_id] = invsum[uid].get(book_id, 0.0) + iv
            if invsum_hist is not None:
                _hadd2(invsum_hist, uid, book_id, ts, iv)
        invn[book_id] = invn.get(book_id, 0) + 1
        if invn_hist is not None:
            _hadd1(invn_hist, book_id, ts, 1)
        if p_first.get(book_id) is None:
            p_first[book_id] = p
        prev = p
    p_last[book_id] = prev
    if use_mark:
        st["prev"] = prev


def book_alphas_from_mtm(mtm, invsum, invn, p_first, p_last):
    """Finalize per-uid per-book alpha = MTM_pnl - mean_inventory * book_drift (drift-beta stripped),
    returning {uid: [alpha per book]} for kappa_of_alpha / kappa_floored. mean_inventory = inventory
    time-sum / book trade count; book_drift = p_last - p_first. A book with no trades is skipped.

    Args:
        mtm: ``{uid: {book: pnl}}`` MTM path-integrals.
        invsum: ``{uid: {book: sum}}`` inventory time-sums.
        invn: ``{book: n}`` trade counts.
        p_first: ``{book: price}`` first price per book.
        p_last: ``{book: price}`` last price per book.

    Returns:
        dict: ``{uid: [alpha per book]}`` with drift-beta stripped.
    FULL-RUN finalizer (offline reference); production uses book_alphas_from_drift (windowed)."""
    out = {}
    for uid in set(mtm) | set(invsum):
        alphas = []
        for b in set(mtm.get(uid, {})) | set(invsum.get(uid, {})):
            n = invn.get(b, 0)
            if n <= 0:
                continue
            mi = invsum.get(uid, {}).get(b, 0.0) / n
            drift = p_last.get(b, 0.0) - (p_first.get(b) or 0.0)
            tb = mtm.get(uid, {}).get(b, 0.0)
            alphas.append(tb - mi * drift)
        out[uid] = alphas
    return out


def book_alphas_from_drift(mtm, invsum, invn, drift):
    """WINDOWED finalizer: same as book_alphas_from_mtm but book_drift = drift[b] (the telescoped sum of
    dp over the window) instead of p_last-p_first. Over a full non-pruned run drift[b] == p_last-p_first,
    so this equals book_alphas_from_mtm exactly (asserted in tests); under pruning it is the windowed

    Args:
        mtm: ``{uid: {book: pnl}}`` MTM path-integrals.
        invsum: ``{uid: {book: sum}}`` inventory time-sums.
        invn: ``{book: n}`` trade counts.
        drift: ``{book: drift}`` telescoped windowed price drift.

    Returns:
        dict: ``{uid: [alpha per book]}`` with the windowed drift-beta stripped.
    drift-strip. mean_inventory = invsum[uid][b] / invn[b]."""
    return {uid: list(by_book.values())
            for uid, by_book in book_alphas_by_book(mtm, invsum, invn, drift).items()}


def book_alphas_by_book(mtm, invsum, invn, drift):
    """Same arithmetic as book_alphas_from_drift, but BOOK IDENTITY IS PRESERVED: {uid: {book: alpha}}.

    The list form loses it. Each miner's list is built from the set of books THAT MINER traded, so
    position i is a different book for different miners (measured: 66 distinct list lengths across
    256 miners on one window). Kappa does not care, being order-blind. Anything CROSS-SECTIONAL does:
    comparing miners at the same list index silently compares different books, which invalidates any
    per-book aggregate computed that way."""
    out = {}
    for uid in set(mtm) | set(invsum):
        by_book = {}
        for b in set(mtm.get(uid, {})) | set(invsum.get(uid, {})):
            n = invn.get(b, 0)
            if n <= 0:
                continue
            mi = invsum.get(uid, {}).get(b, 0.0) / n
            tb = mtm.get(uid, {}).get(b, 0.0)
            by_book[b] = tb - mi * drift.get(b, 0.0)
        out[uid] = by_book
    return out


def traded_book_alphas(alphas_by_book, capture_buy_sums, capture_sell_sums):
    """Keep, per uid, the books on which the uid FILLED inside the window: {uid: {book: alpha}}.

    The mark-to-market and inventory accumulators run for every miner holding a position on a book,
    on every trade in that book, so a miner that merely holds a static position through the window
    carries an alpha entry for it, and by the defining invariant that alpha is exactly zero. On a
    long-running validator those held-not-traded pairs come to be half of all pairs, so a floor taken
    over every pair collapses to a rounding residue and kappa counts books the miner never traded. The
    capture maps hold exactly the (uid, book) pairs with fills inside the window (they are pruned on
    the same clock), so they define the pool for both the floor and the skill leg.

    Args:
        alphas_by_book: ``{uid: {book: alpha}}`` from book_alphas_by_book.
        capture_buy_sums: ``{uid: {book: capture}}`` buyer-side capture, windowed.
        capture_sell_sums: ``{uid: {book: capture}}`` seller-side capture, windowed.

    Returns:
        dict: ``{uid: {book: alpha}}`` restricted to books with fills; a uid with none keeps an empty map.
    """
    out = {}
    for uid, by_book in alphas_by_book.items():
        traded = set((capture_buy_sums.get(uid) or {})) | set((capture_sell_sums.get(uid) or {}))
        out[uid] = {b: a for b, a in by_book.items() if b in traded}
    return out


def median_abs_floor(book_alphas_by_uid, scale=0.5):
    """E5 magnitude floor for kappa_floored: scale * median(|alpha|) over the per-book alphas passed
    in. Kappa is magnitude-blind, so a tiny-consistent spammer ranks high without a floor. The caller
    passes the pool of (uid, book) pairs with fills inside the window (traded_book_alphas): a held but
    untraded book has an alpha of exactly zero by the invariant and would drag the median to nothing."""
    mags = [abs(a) for al in book_alphas_by_uid.values() for a in al if a is not None]
    if not mags:
        return 0.0
    return scale * statistics.median(mags)
