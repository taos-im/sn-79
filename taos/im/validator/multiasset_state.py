# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The multi-asset state flatten, shared by the simulation engine and the shadow scoring replica.

It lives outside the engine module so the shadow child, which replays the raw state bytes main received, can
apply the same flatten without importing the engine (and torch with it). Both readers of a round must see the
same books, or the replica accounts for nothing while main accounts for everything."""


def flatten_multiasset_state(raw_dict: dict) -> tuple[dict, list[dict]]:
    """Flatten a multi-asset state request into the single-market shape.

    The simulator publishes a multi-asset run as {logDir, timestamp, model,
    backgrounds: [{bg, name, instances: [<single-market request>]}]}, where every
    instance is a full single-market request over one realization and all book ids
    are canonical (consecutive across realizations, each numbered from the sum of the
    book counts before it) — globally unique by construction. Flattening is therefore a pure merge:
      books   — disjoint key union across instances;
      accounts— per-agent union of the (disjoint) per-book maps;
      notices — per-agent concatenation.
    Returns (flat_request, instance_metas) where each meta records the instance's
    background index/name, instance index, flat index and own log directory —
    everything per-realization consumers (e.g. fundamental loading) need.

    A single-market request (no 'backgrounds' key) passes through unchanged.
    """
    if "backgrounds" not in raw_dict:
        return raw_dict, []

    books: dict = {}
    accounts: dict = {}
    notices: dict = {}
    metas: list[dict] = []
    flat_idx = 0
    for bg in raw_dict.get("backgrounds") or []:
        for instance_idx, instance in enumerate(bg.get("instances") or []):
            metas.append({
                "bg": bg.get("bg"),
                "name": bg.get("name"),
                "instance": instance_idx,
                "flat_idx": flat_idx,
                "logDir": instance.get("logDir"),
            })
            flat_idx += 1
            instance_books = instance.get("books") or {}
            # THE MERGE IS LOSSLESS ONLY WHILE THE ID SPACE IS INJECTIVE, and that is a property of
            # the SIMULATOR's numbering, not of anything this function can see. A bare update() means a
            # C++/Python numbering disagreement -- the F3 class in the multi-asset review, which still
            # has no cross-language test -- would silently drop one realization's book behind
            # another's and leave every downstream consumer reading a plausible, wrong market.
            # Refusing costs nothing on a correct run: canonical ids are disjoint by construction.
            collisions = books.keys() & instance_books.keys()
            if collisions:
                raise ValueError(
                    f"multi-asset state: realization {flat_idx - 1} republishes canonical book id(s) "
                    f"{sorted(collisions)} already seen in an earlier realization. The published ids "
                    "are not injective, which means the simulator's numbering and the validator's "
                    "disagree; merging would silently discard a market."
                )
            books.update(instance_books)
            for agent_id, holdings in (instance.get("accounts") or {}).items():
                account = accounts.setdefault(agent_id, {})
                overlap = account.keys() & (holdings or {}).keys()
                if overlap:
                    raise ValueError(
                        f"multi-asset state: agent {agent_id} already holds canonical book(s) "
                        f"{sorted(overlap)} from an earlier realization; per-book account maps must "
                        "be disjoint across realizations."
                    )
                account.update(holdings or {})
            for agent_id, agent_notices in (instance.get("notices") or {}).items():
                notices.setdefault(agent_id, []).extend(agent_notices or [])

    flat = {k: v for k, v in raw_dict.items() if k != "backgrounds"}
    flat["books"] = books
    flat["accounts"] = accounts
    flat["notices"] = notices
    return flat, metas
