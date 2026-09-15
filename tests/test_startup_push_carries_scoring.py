# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Every ingest push that carries accounts also carries the per-agent scoring maps.

The exchange validator's startup push (neurons/validator.py, _listen) sent books, accounts and open
orders with no agent_* maps. The service builds an agent_snapshots row per uid from any payload that
carries accounts, so each run of that push wrote a whole board of score 0, volume 0, PnL 0 rows
between real ones: seen on the localnet exchange DB, 256 or 257 rows at 16:06, 16:25,
16:58, 17:31 and 18:04 local, one per "MVTRX startup push" log line (the block re-enters about every
33 minutes, not only at start). The service now writes NULL for a payload without scoring, which
removes the false zeros, and the push now carries the restored maps so the rows are real.
"""
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "taos/im/neurons/validator.py").read_text()


def _push_payloads():
    """Each `_push_mvtrx({` payload literal, as the text up to its closing `}, url=`."""
    out = []
    for m in re.finditer(r"_push_mvtrx\(\{", SRC):
        end = SRC.index("}, url=", m.end())
        out.append(SRC[m.end():end])
    return out


def test_every_push_with_accounts_carries_the_scoring_maps():
    payloads = _push_payloads()
    assert len(payloads) >= 2, "expected the per-block push and the startup push"
    for body in payloads:
        if '"accounts":' in body:
            assert "_build_agent_scoring_maps()" in body or "_scoring_maps" in body, (
                "a push that carries accounts without the scoring maps writes a board of zero "
                "snapshot rows on the service side:\n" + body[:300]
            )
