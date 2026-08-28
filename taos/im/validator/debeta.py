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

WIRED AND DEFAULT-OFF. trade.py accumulates per-uid/book buy/sell capture and the inventory
mark-to-market path; persistence.py and report.py carry the sums across restarts and snapshots;
reward.py computes the scores and substitutes them for the trading score ONLY when
--scoring.debeta.enabled is set, which defaults to false. Nothing about live scoring changes until
an operator turns it on.
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

    Args:
        hist: ``{k1: {k2: {ts: val}}}`` history, pruned in place.
        running: ``{k1: {k2: val}}`` running totals, reduced by the pruned mass.
        threshold: Timestamps strictly below this are dropped.
    running. Keeps running == sum(kept)."""
    for k1, d2 in hist.items():
        for k2, tsd in d2.items():
            if not tsd:
                continue
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


def prune_hist_1level(hist, running, threshold):
    """hist {k:{ts:val}}, running {k:val}.

    Args:
        hist: ``{k: {ts: val}}`` history, pruned in place.
        running: ``{k: val}`` running totals, reduced by the pruned mass.
        threshold: Timestamps strictly below this are dropped.
    """
    for k, tsd in hist.items():
        if not tsd:
            continue
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


def accumulate_book_capture(buy_sums, sell_sums, book_id, trades, W, *,
                            buy_hist=None, sell_hist=None, ts=None):
    """Accumulate per-uid two-sided spread capture for ONE book's ordered trade batch into
    buy_sums / sell_sums ({uid: {book: cap}}), in place. Each `trade` is dict-like with keys
    p (price), q (quantity), s (side), Ma (maker uid), Ta (taker uid). side==0 => taker buys /
    maker sells; side==1 => maker buys / taker sells. buyer captures (mid-price)*q, seller
    (price-mid)*q, vs a non-lagging centered mid over this batch. Self-trades (Ma==Ta) are excluded
    (first-line wash guard; cross-uid rings are netted at score time by coldkey cluster).

    When buy_hist/sell_hist ({uid:{book:{ts:incr}}}) + ts are given, the per-batch increments are also

    Args:
        buy_sums: ``{uid: {book: capture}}`` buyer-side sums, accumulated in place.
        sell_sums: ``{uid: {book: capture}}`` seller-side sums, accumulated in place.
        book_id: The book this batch belongs to.
        trades: Ordered trade batch, each dict-like with ``p``, ``q``, ``s``, ``Ma``, ``Ta``.
        W: Optional windowing context ``(buy_hist, sell_hist, ts)``; None for offline callers.
    recorded at ts so the sums can be windowed (pruned/shifted). Offline callers pass none (full-run)."""
    prices = [float(t["p"]) for t in trades]
    mids = centered_mid(prices, W)
    for t, mid in zip(trades, mids):
        ma = t.get("Ma", -1)
        ta = t.get("Ta", -1)
        if ma == ta:
            continue
        buy_cap = (mid - float(t["p"])) * float(t["q"])
        buyer, seller = (ta, ma) if int(t["s"]) == 0 else (ma, ta)
        if buyer is not None and buyer >= 0:
            buy_sums[buyer][book_id] = buy_sums[buyer].get(book_id, 0.0) + buy_cap
            if buy_hist is not None:
                _hadd2(buy_hist, buyer, book_id, ts, buy_cap)
        if seller is not None and seller >= 0:
            sell_sums[seller][book_id] = sell_sums[seller].get(book_id, 0.0) - buy_cap
            if sell_hist is not None:
                _hadd2(sell_hist, seller, book_id, ts, -buy_cap)


def balanced_reward(caps):
    """{agent: [buy_cap, sell_cap]} -> {agent: 2*min(buy_cap, sell_cap)} clamped at 0. The MAKING
    (liquidity) component: genuine two-sided spread capture; a one-sided accumulator/drift-rider -> ~0."""
    return {a: max(0.0, 2.0 * min(bc, sc)) for a, (bc, sc) in caps.items()}


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
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    rank = [0.0] * len(vals)
    for pos, i in enumerate(order):
        rank[i] = pos / max(len(vals) - 1, 1)
    return rank


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
    windowing (making window). Offline callers pass none."""
    for t in trades:
        ma = _uid(t.get("Ma", -1))
        ta = _uid(t.get("Ta", -1))
        if ma == ta or ma < 0 or ta < 0:
            continue
        q = float(t["q"])
        d = cp.setdefault(ma, {})
        d[ta] = d.get(ta, 0.0) + q
        if cp_hist is not None:
            _hadd2(cp_hist, ma, ta, ts, q)


def et_book_batches(notices, seen_tids, ts):
    """Regroup exchange ET settled-fill notices into per-book ordered trade batches for de-beta.
    notices: {uid: [notice, ...]} from state.notices; seen_tids: {trade_id: ts} carried across calls
    (mutated) so a fill counted once is never recounted -- neutralizes the maker+taker duplicate listing
    and the deliberate redelivery of settled fills across blocks; ts: current sample ts. Returns
    {book_id: [{p,q,s,Ma,Ta,i}, ...]} ordered by trade id (monotone engine mint order), the shape the
    accumulators expect. Pool fills carry Ma or Ta = None (no miner on that side); the accumulators skip

    Args:
        notices: ``{uid: [notice, ...]}`` from ``state.notices``.
        seen_tids: ``{trade_id: ts}`` carried across calls and mutated, so no fill is recounted.
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
            if tid in seen_tids:
                continue
            seen_tids[tid] = ts
            by_book.setdefault(int(n["b"]), []).append(
                {"p": n["p"], "q": n["q"], "s": n["s"], "Ma": n.get("Ma"), "Ta": n.get("Ta"), "i": tid}
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
    top = sorted(my, key=lambda t: my[t], reverse=True)[:topk]
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
    out = {}
    for u in uids:
        ec = max(0.0, counterparty_ec(cp, u, topk))
        out[u] = own.get(u, 0.0) * max(0.0, 1.0 - strength * ec)
    return out


def debeta_scores(capture_buy_sums, capture_sell_sums, book_alphas_by_uid, floor=0.0, w_make=0.65,
                  cp=None, p11_strength=0.0):
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

    Returns:
        dict: ``{uid: combined score}``.
    cp: {maker_uid: {taker_uid: vol}}. Returns {uid: combined_score in [0,1]}."""
    uids = sorted(set(capture_buy_sums) | set(capture_sell_sums) | set(book_alphas_by_uid))
    if not uids:
        return {}
    caps = {u: [sum(capture_buy_sums.get(u, {}).values()), sum(capture_sell_sums.get(u, {}).values())]
            for u in uids}
    own = balanced_reward(caps)
    if cp is not None and p11_strength > 0:
        own = p11_discount(own, cp, uids, p11_strength)
    making = [own.get(u, 0.0) for u in uids]
    skill = [kappa_floored(book_alphas_by_uid.get(u, []), floor) for u in uids]
    comb = combined_reward(making, skill, w_make)
    return {uids[i]: comb[i] for i in range(len(uids))}


def combined_reward(making, skill, w_make=0.65):
    """Combined de-beta: w_make*rank01(making) + (1-w_make)*rank01(skill). Both legs are RANKS (0-1,
    comparable scales). w_make is the operator dial. Offline: drift-OPPOSING at every w_make; genuine
    top-skill decile preserved (reward-rank 0.83-0.95); de-concentrates (combined Gini 0.29).

    Making is RANK, NOT magnitude-proportional (decided 2026-08-06): (1) rank de-concentrates best
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


def demean_cross_section(alphas_by_book):
    """Remove each book's CROSS-SECTIONAL mean alpha, leaving alpha relative to the field on that book.

    WHY THIS EXISTS. alpha = MTM - mean_inventory x book_drift is drift-neutral in AGGREGATE, but the
    per-book residuals of a directionally-positioned miner stay CONSISTENTLY SIGNED, and kappa scores
    consistency rather than magnitude. So the strip cancels in the sum and survives in exactly the
    pattern kappa rewards: measured corr(net-inventory, skill) of +0.284 on real mainnet L3, when the
    design requires under 0.20. Subtracting the per-book mean removes the component every miner on
    that book shares, which is the market move itself, and leaves the part that is genuinely yours.

    This is the standard cross-sectional residualisation used to separate a common factor from
    idiosyncratic return. It also raises the number of scored miners, because a miner who merely kept
    pace with the field on every book is no longer credited for the field's move.

    Books with a single participant contribute nothing (their mean IS that participant) and are
    dropped rather than zeroed, since a book you alone traded carries no cross-sectional information.
    """
    per_book = {}
    for uid, by_book in alphas_by_book.items():
        for b, a in by_book.items():
            per_book.setdefault(b, []).append(a)
    means = {b: sum(v) / len(v) for b, v in per_book.items() if len(v) > 1}
    return {uid: {b: a - means[b] for b, a in by_book.items() if b in means}
            for uid, by_book in alphas_by_book.items()}


def median_abs_floor(book_alphas_by_uid, scale=0.5):
    """E5 magnitude floor for kappa_floored: scale * median(|alpha|) over ALL per-book alphas across
    all miners. Kappa is magnitude-blind, so a tiny-consistent spammer ranks high without a floor."""
    mags = [abs(a) for al in book_alphas_by_uid.values() for a in al if a is not None]
    if not mags:
        return 0.0
    return scale * statistics.median(mags)
