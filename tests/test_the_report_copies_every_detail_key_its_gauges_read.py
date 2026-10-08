# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The report builds one dict per miner from uid_kappa (where reward.py stored the de-beta detail) with an explicit
key list, then its gauge loop and the miners table read that dict by detail key. A key reward.py stores and the
gauge loop names, but the dict does not copy, is silently dark: the gauge is never published and the table label
reads 0.0 for every miner, with no error anywhere. The two making bases (debeta_making_realized,
debeta_making_captured) and debeta_making_basis went dark that way on the 0.6.3 testnet (6 October 2026): the
making-basis line of the board monitor, the agents dashboard's making-credit panel and the acceptance's live-parity
check all read nothing. This test holds the dict and the loop together."""
import ast
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "taos" / "im" / "validator" / "report.py"


def _str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _per_miner_dict_keys(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = {_str(k) for k in node.keys} - {None}
            if {"making_rank", "skill_rank", "debeta_score", "making_share"} <= keys:
                return keys
    raise AssertionError("the per-miner dict (making_rank, skill_rank, debeta_score, making_share) was not found")


def _gauge_loop_pairs(tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Tuple) and node.elts and all(isinstance(e, ast.Tuple) and len(e.elts) == 2 for e in node.elts)):
            continue
        pairs = [(_str(e.elts[0]), _str(e.elts[1])) for e in node.elts]
        if all(k and g for k, g in pairs) and ("making_rank", "debeta_making_rank") in pairs:
            return pairs
    raise AssertionError("the de-beta gauge loop (detail key, gauge name) tuple was not found")


def test_every_detail_key_the_gauge_loop_reads_is_copied_into_the_per_miner_dict():
    tree = ast.parse(REPORT.read_text())
    keys = _per_miner_dict_keys(tree)
    pairs = _gauge_loop_pairs(tree)
    assert ("making_realized", "debeta_making_realized") in pairs and ("making_captured", "debeta_making_captured") in pairs
    missing = [k for k, _ in pairs if k not in keys]
    assert missing == [], f"the gauge loop reads detail keys the per-miner dict never copies, so they publish nothing: {missing}"
