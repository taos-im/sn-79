# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Which book carries which records across a change of layout, and the move itself.

A layout, as scoring sees it, is its asset classes in order: each class's name, its books and the signature of its
market (layout_of). Comparing two layouts gives every new book its source (book_changes). A class whose name and
market are the same in both keeps its records in order: its k-th book takes the old layout's k-th book of that class,
whatever the two ids are, so a class that grows keeps every record on its first books and its added books start
empty, and a class whose ids shift (a class before it grew) takes its records along. A class whose market changed
starts fresh and restarts its pool; a class new to the layout starts empty; an old book that is no new book's source
is dropped.

rekey_books moves every book-keyed structure the validator and the scoring service hold by such a map, before the
boundary's shift puts them on the new run's clock. BOOK_KEYED says where each structure holds its book; a structure
added to the scoring state must be named there or in NOT_BOOK_KEYED, which a test checks.
"""
import hashlib
import json
from collections import defaultdict

# where each structure holds its book: first ("book"), under the uid ("uid"), under the uid and a timestamp
# ("uid_ts"), or in kappa's per-uid books maps ("kappa")
BOOK_KEYED = {
    "book": (
        "recent_trades", "debeta_inv", "debeta_inv_alt", "debeta_invn", "debeta_invn_alt", "debeta_pfirst",
        "debeta_pfirst_alt", "debeta_plast", "debeta_plast_alt", "debeta_drift", "debeta_drift_alt",
        "debeta_mark_state", "debeta_mark_state_alt", "debeta_capture_mid", "debeta_invn_hist", "debeta_drift_hist",
        "debeta_invn_alt_hist", "debeta_drift_alt_hist", "debeta_conj_farm", "debeta_conj_farm_hist",
        "debeta_conj_ref", "debeta_conj_samples",
    ),
    "uid": (
        "trade_volumes", "roundtrip_volumes", "volume_sums", "maker_volume_sums", "taker_volume_sums",
        "self_volume_sums", "roundtrip_volume_sums", "fee_sums", "agent_pnl_by_book", "open_positions",
        "initial_balances", "recent_miner_trades", "activity_factors", "pnl_factors", "inventory_quote",
        "capture_buy_sums", "capture_sell_sums", "realized_buy_sums", "realized_sell_sums", "capture_taker_buy_sums",
        "capture_taker_sell_sums", "debeta_mtm", "debeta_invsum", "debeta_heldn", "debeta_heldinv",
        "debeta_helddrift", "debeta_notional", "debeta_mtm_alt", "debeta_invsum_alt", "debeta_maker_buy_notional",
        "debeta_maker_sell_notional", "debeta_at_best", "debeta_best_size", "debeta_conj",
        "debeta_capbuy_hist", "debeta_capsell_hist", "debeta_realbuy_hist", "debeta_realsell_hist",
        "debeta_captkbuy_hist", "debeta_captksell_hist", "debeta_mtm_hist", "debeta_invsum_hist",
        "debeta_heldn_hist", "debeta_heldinv_hist", "debeta_helddrift_hist", "debeta_notional_hist",
        "debeta_mtm_alt_hist", "debeta_invsum_alt_hist", "debeta_maker_buy_notional_hist",
        "debeta_maker_sell_notional_hist", "debeta_at_best_hist", "debeta_best_size_hist", "debeta_conj_hist",
    ),
    "uid_ts": ("realized_pnl_history", "inventory_history"),
    "kappa": ("kappa_values",),
}
# keyed by class index: follow their class by name, and start empty for a class that restarts
CLASS_KEYED = ("debeta_cpc", "debeta_cpc_hist")
NOT_BOOK_KEYED = (
    "agent_pnl_total", "debeta_cp", "debeta_cp_hist", "debeta_conj_clock", "_debeta_class_vectors",
    "_debeta_handover", "_debeta_seen_tids", "_volume_seen_tids", "_debeta_young_books",
)


def layout_of(sim) -> dict | None:
    """``{"classes": [{"name", "books", "market"}]}`` for a simulation config, the same dict back for a layout already
    described (a saved state's), None where the config has no classes to describe."""
    if sim is None or isinstance(sim, dict):
        return sim
    try:
        classes = list(sim.asset_classes())
    except (AttributeError, KeyError, TypeError):
        return None
    from taos.im.validator.trade import _book_signature
    out = []
    for c in classes:
        books = [int(b) for b in c.books]
        signature = _book_signature(sim, books[0]) if books else None
        market = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()
        out.append({"name": str(c.name), "books": books, "market": market})
    return {"classes": out}


def book_changes(old: dict | None, new: dict | None) -> dict:
    """The change map from layout ``old`` to layout ``new`` (layout_of's form).

    ``books``: every new book's disposition, keep (its records stay), move (they come from another id), clear (its
    class's market changed: it starts fresh), new (no predecessor: it starts empty), and every old book that is no new
    book's source and not a new book, drop. ``dest``: every old book to the new id its records go to, or None.
    ``classes``: the classes whose pools restart, those whose market changed; a class that keeps, moves or gains books
    keeps its pool, and a class new to the layout has no pool to restart and ramps in. ``class_sources``: each
    new class's index in the old layout, where it had one. ``names_old``/``names_new``: the class names in order."""
    if not old or not new:
        return {"books": {}, "classes": []}
    old_classes, new_classes = old["classes"], new["classes"]
    by_name = {c["name"]: (j, c) for j, c in enumerate(old_classes)}
    books, dest, restart, class_sources = {}, {}, [], {}
    for i, c in enumerate(new_classes):
        found = by_name.get(c["name"])
        same_market = found is not None and found[1]["market"] == c["market"]
        if found is not None:
            class_sources[i] = found[0]
            if not same_market:
                restart.append(i)
        prev = found[1]["books"] if same_market else []
        for k, b in enumerate(c["books"]):
            if k < len(prev):
                books[b] = "keep" if prev[k] == b else "move"
                dest[prev[k]] = b
            else:
                books[b] = "clear" if found is not None and not same_market else "new"
    for c in old_classes:
        for b in c["books"]:
            dest.setdefault(b, None)
            if b not in books:
                books[b] = "drop"
    return {"books": books, "dest": dest, "classes": restart, "class_sources": class_sources,
            "names_old": [c["name"] for c in old_classes], "names_new": [c["name"] for c in new_classes]}


def _book_of(key):
    if isinstance(key, tuple) and len(key) == 2:
        return key[1]
    return key


def _rekey(d, dest, keep_unmapped):
    """``d`` with each book's value under ``dest[book]`` (dropped for None); a book ``dest`` does not name keeps its
    key if ``keep_unmapped``, else it is dropped. The container keeps its type and default factory."""
    if not isinstance(d, dict):
        return d
    out = defaultdict(d.default_factory) if isinstance(d, defaultdict) else {}
    for key, value in d.items():
        book = _book_of(key)
        try:
            book = int(book)
        except (TypeError, ValueError):
            out[key] = value
            continue
        if book in dest:
            to = dest[book]
        elif keep_unmapped:
            to = book
        else:
            continue
        if to is None:
            continue
        out[(key[0], to) if isinstance(key, tuple) else to] = value
    return out


def rekey_books(holder, dest: dict, class_sources: dict | None = None, restart_classes=(), *,
                keep_unmapped: bool) -> None:
    """Move every book-keyed structure on ``holder`` (the validator or the scoring service's state) by ``dest``, old
    book to new book or None. With ``keep_unmapped`` a book ``dest`` does not name stays where it is (a strip of some
    books); without it, it is dropped (a layout change, where ``dest`` names every old book). The class-keyed pools
    follow ``class_sources`` (new index to old) when given, and a class in ``restart_classes`` starts empty."""
    dest = {int(b): (None if t is None else int(t)) for b, t in (dest or {}).items()}
    for name in BOOK_KEYED["book"]:
        d = getattr(holder, name, None)
        if isinstance(d, dict):
            setattr(holder, name, _rekey(d, dest, keep_unmapped))
    for name in BOOK_KEYED["uid"]:
        d = getattr(holder, name, None)
        if isinstance(d, dict):
            for uid in list(d.keys()):
                d[uid] = _rekey(d[uid], dest, keep_unmapped)
    for name in BOOK_KEYED["uid_ts"]:
        d = getattr(holder, name, None)
        if isinstance(d, dict):
            for uid in list(d.keys()):
                by_ts = d[uid]
                if not isinstance(by_ts, dict):
                    continue
                for ts in list(by_ts.keys()):
                    had = bool(by_ts[ts])
                    by_ts[ts] = _rekey(by_ts[ts], dest, keep_unmapped)
                    if had and not by_ts[ts]:
                        del by_ts[ts]
    for name in BOOK_KEYED["kappa"]:
        d = getattr(holder, name, None)
        if isinstance(d, dict):
            for kv in d.values():
                if isinstance(kv, dict):
                    for key in ("books", "books_weighted"):
                        if isinstance(kv.get(key), dict):
                            kv[key] = _rekey(kv[key], dest, keep_unmapped)
    restart = {int(c) for c in restart_classes}
    for name in CLASS_KEYED:
        d = getattr(holder, name, None)
        if not isinstance(d, dict):
            continue
        if class_sources is not None:
            moved = {int(i): d[int(j)] for i, j in class_sources.items() if int(j) in d}
            d.clear()
            d.update(moved)
        for c in restart:
            if c in d:
                d[c] = {}
