# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The de-beta decomposition, the blend scores and the eligibility flag are per-miner gauges in their
own right: the report must publish them whenever a miner carries them, not only where kappa-3 was
computed.

On the first cycle at full de-beta weight (kappa 0, pnl 0, de-beta 1.0), every
debeta_* series, trading_score, combined_score and scorable vanished from /metrics for every miner,
while the scoring itself kept running (score EMAs moved, the per-book alpha and capture gauges stayed
up). The emission sat inside `if m['kappa'] is not None`, and at kappa weight 0 every uid's carrier is
the skipped stub with no median. The same gate had already hidden the decomposition for the 2 of 23
scored uids without a kappa at rung 2. These are structural assertions on the report source, in the
style of the other report wiring tests: they fail if the gauges are ever nested under the kappa test again.
"""
import ast
from pathlib import Path

DEV = Path(__file__).resolve().parents[1]
REPORT_PATH = DEV / "taos/im/validator/report.py"

# The gauge names the kappa test must not gate.
UNGATED = ("debeta_score", "debeta_making", "debeta_making_rank", "debeta_skill", "debeta_skill_rank",
           "debeta_p11_factor", "debeta_weight", "debeta_w_make", "debeta_skill_floor", "num_scored_books",
           "scorable", "trading_score", "combined_score")


def _kappa_gate(node):
    """True for `if m['kappa'] is not None` (or `m.get('kappa') is not None`)."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    src = ast.unparse(node.test)
    return "'kappa'" in src and "is not None" in src


def _string_constants(node):
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _statements_under_kappa_gates(tree):
    """Every statement that lives inside a kappa gate's body (not its else branch)."""
    found = []
    for node in ast.walk(tree):
        if _kappa_gate(node):
            for stmt in node.body:
                found.extend(ast.walk(stmt))
    return found


def test_the_report_publishes_the_debeta_gauges_outside_the_kappa_gate():
    tree = ast.parse(REPORT_PATH.read_text())
    gated = set()
    for n in _statements_under_kappa_gates(tree):
        gated |= _string_constants(n) & set(UNGATED)
    assert not gated, (
        f"{sorted(gated)} are published only when kappa is not None; at kappa weight 0 no miner has a "
        "kappa and these series disappear for the whole board"
    )


def test_the_miners_table_blanks_the_uncomputed_kappa_and_pnl_labels():
    """At kappa weight 0 the kappa-3 batch is skipped and at pnl weight 0 the pnl score is not
    computed. The miners table gauge must carry an EMPTY label value for them: Prometheus treats an
    empty label as absent, so the Agents table columns vanish and return with the computation and no
    dashboard edit is ever needed. str(None) rendered as 0.0000 on the final rung, and the kappa stub
    carries penalty and score as 0.0, so those two must key on the missing median, not on their own
    value."""
    src = REPORT_PATH.read_text()
    assert "kappa=(\"\" if m['kappa'] is None else m['kappa'])" in src
    assert "kappa_penalty=(\"\" if m['kappa'] is None or m['kappa_penalty'] is None else m['kappa_penalty'])" in src
    assert "kappa_score=(\"\" if m['kappa'] is None or m['kappa_score'] is None else m['kappa_score'])" in src
    assert "pnl_score=(\"\" if m['pnl_score'] is None else m['pnl_score'])" in src


def test_an_empty_label_value_is_legal_for_the_client_and_dropped_by_prometheus():
    """The mechanism the blanking relies on: the client exposes label="" and the server drops it.
    The first half is checked here against the installed prometheus_client; the second is Prometheus
    data-model semantics (an empty label value is equivalent to the label not being present)."""
    from prometheus_client import CollectorRegistry, Gauge, generate_latest
    reg = CollectorRegistry()
    g = Gauge("t_miners", "t", ["agent_id", "kappa"], registry=reg)
    g.labels(agent_id="1", kappa="").set(1.0)
    g.labels(agent_id="2", kappa="0.5").set(1.0)
    out = generate_latest(reg).decode()
    assert 'kappa=""' in out and 'kappa="0.5"' in out


def test_quiet_books_still_get_a_trade_price_and_a_fundamental_price():
    """The book gauges are persistent, so after a reporting restart a book publishes trade_price,
    trade_volume and fundamental_price only once it trades again: testnet, 31 of 128 books
    had none a few minutes after a restart. The last price is seeded from the validator's
    recent-trades buffer, the step volume is published as zero, and the fundamental price is emitted
    independently of the step's trades."""
    src = REPORT_PATH.read_text()
    loop = src[src.index("Collecting book metrics"):src.index("Book metrics collected")]
    assert loop.index('"fundamental_price"') < loop.index("if trades:"), (
        "fundamental_price must be published before and independently of the trades branch"
    )
    tail = loop[loop.index("else:", loop.index("has_new_trades = True")):]
    assert "_recent[-1].price" in tail and '"trade_price"' in tail, "seed trade_price from recent_trades on a quiet step"
    assert 'for _g in ("trade_volume", "trade_buy_volume", "trade_sell_volume")' in tail, "publish zero step volume"


def test_the_report_still_publishes_the_debeta_gauges_somewhere():
    """The fix must move the emission, not delete it."""
    src = REPORT_PATH.read_text()
    for name in UNGATED:
        assert f'"{name}"' in src or f"'{name}'" in src, f"the report no longer publishes {name}"
