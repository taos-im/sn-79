# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator's push to the mvtrx service carries, beside the live fills, a reconciliation list of every fill
seen in the simulation state. That list stated no fee, so the service fell back to the book's schedule for it,
which is what a trade WOULD cost, not what the engine charged (dev/c26a, 8 October 2026, after the tape fee fix in
the service). The trade event carries the engine's charge as Tf and Mf; the reconciliation fill states both, so
the service's _resolve_fill_fees keeps them (a stated fee wins, an explicit zero included) and the tape holds the
charge the engine took."""
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "taos" / "im" / "neurons" / "validator.py").read_text()


def _reconciliation_fill_literal():
    start = SRC.index("_sim_fills.append({")
    return SRC[start:SRC.index("})", start)]


def test_the_reconciliation_fill_carries_the_engines_taker_and_maker_fee():
    fill = _reconciliation_fill_literal()
    assert re.search(r'"taker_fee":\s*_ev\.get\("Tf"\)', fill), "the taker fee is the event's Tf, the engine's charge"
    assert re.search(r'"maker_fee":\s*_ev\.get\("Mf"\)', fill), "the maker fee is the event's Mf, the engine's charge"


def test_a_stated_zero_fee_reaches_the_service_as_zero_not_as_missing():
    """The service treats None as 'no fee stated' and schedules one; the engine's ZeroFeePolicy charges 0.0, which
    must arrive as 0.0. So the fill passes the event's value through unchanged, with no `or` default that would
    turn a zero into None."""
    fill = _reconciliation_fill_literal()
    for key in ("taker_fee", "maker_fee"):
        line = next(ln for ln in fill.splitlines() if f'"{key}"' in ln)
        assert " or " not in line, f"{line.strip()}: an `or` default would turn the engine's zero charge into a missing fee"
