# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""What the price band cut and refused, as totals for the validator's gauges.

An engine whose exchange carries a price band lists each order or close the band cut in its response's 'band' list
and reports each order it refused whole among its rejects with the reason PRICE_BAND. The engine calls count() on every
response, and the report publishes the totals since the process started. A response without a band list (the simulation engine's, or an engine's from before the band) adds
nothing, and an engine that never sent one has no totals, so a reader sees 'not measured' rather than a zero."""

REASON = "PRICE_BAND"


def count(engine, response) -> None:
    """Add a response's band cuts and band refusals to the engine's totals; nothing without a band list."""
    if not isinstance(response, dict) or not isinstance(response.get("band"), list):
        return
    totals = getattr(engine, "_price_band_totals", None)
    if totals is None:
        totals = {"cuts": 0, "cut_closes": 0, "refusals": 0}
        engine._price_band_totals = totals
    for cut in response["band"]:
        if not isinstance(cut, dict):
            continue
        totals["cuts"] += 1
        if cut.get("k") == "close":
            totals["cut_closes"] += 1
    for reject in response.get("rejects") or []:
        if isinstance(reject, dict) and reject.get("m") == REASON:
            totals["refusals"] += 1


def health(engine) -> dict | None:
    """The totals for the report, None before the first response with a band list."""
    totals = getattr(engine, "_price_band_totals", None)
    return None if totals is None else dict(totals)


def gauge_name(key: str) -> str:
    """price_band_cuts, price_band_cut_closes, price_band_refusals."""
    return f"price_band_{key}"
