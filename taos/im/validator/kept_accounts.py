# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Account entries an engine under dirtyAccountsOnly left out, filled in from the last full copy.

Such an engine sends an agent's entry for a book only when the account changed on that book since the last state the
validator answered, and leaves every other entry out of the state; the fees an untiered policy charges move with the
book, so the state carries each book's shared rates once ("bookFees"). An older engine sent a left-out entry as its
ids and fees with the eight fields below as nil; both are filled. Each process that reads the engine's states (main,
the scoring process, the proxy) keeps the last full entry of every (agent, book) and fills the rest in, so everything
downstream sees a complete state, as from an engine that sends every account. An engine that sends everything in full
only refreshes the copy."""

OMITTED = ("bb", "qb", "bl", "ql", "bc", "qc", "o", "l")


def fill_accounts(kept: dict, accounts: dict, books=None, book_fees: dict | None = None) -> int:
    """Fill the left-out entries of a flattened state's accounts ({agent: {book: entry}}) in place from `kept`, and
    keep every full entry in `kept`. `books` are the state's book ids: an agent's entry for one of them that the state
    does not carry is filled at `book_fees`' rates for its book; a book outside them is never filled, so a copy kept
    across a layout change puts no old book back. Returns how many entries could not be filled because no full copy
    was ever seen (a validator that has just started): the state is then incomplete, and the engine should be asked
    for every account in full.

    A filled entry shares its fields' values with the kept copy. Nothing downstream writes inside them (main only
    adds the traded volume 'v' at the entry's own level), so no copy is made of them. An entry left out entirely is a
    C-level copy of the kept one without main's 'v', its fees the book's shared rates, one dict a book: a Python-level
    rebuild of each of the 55,000 entries a 256-book state leaves out cost more than the smaller state saved."""
    missing = 0
    shared_fees = {b: {"v": None, "m": r["m"], "t": r["t"]} for b, r in (book_fees or {}).items()}
    book_ids = None if books is None else (books.keys() if isinstance(books, dict) else frozenset(books))
    for agent_id, agent_books in accounts.items():
        agent_kept = kept.setdefault(agent_id, {})
        for book_id, entry in agent_books.items():
            if entry.get("bb") is not None:
                agent_kept[book_id] = entry
                continue
            last = agent_kept.get(book_id)
            if last is None:
                missing += 1
                continue
            for key in OMITTED:
                entry[key] = last[key]
        if book_ids is None:
            continue
        for book_id in book_ids - agent_books.keys():
            last = agent_kept.get(book_id)
            if last is None:
                missing += 1
                continue
            entry = last.copy()
            entry.pop("v", None)
            fees = shared_fees.get(book_id)
            if fees is not None:
                entry["f"] = fees
            agent_books[book_id] = entry
    return missing
