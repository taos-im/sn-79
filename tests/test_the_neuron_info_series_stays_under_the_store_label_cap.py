# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The neuron_info series carries the validator's identity, the scoring dials and the fee policy. On 6 October 2026
it also carried a copy of every market parameter (94 labels) that simulation_config_info already publishes per asset
class, 150 labels in all, and the metrics store behind the dashboards refused the series (VictoriaMetrics ignores a
series over its per-series label cap) while the 100-label per-class series went through: the Scoring Config and Fee
Policy tables read No data. The series is held to what its two readers need and to a count the store takes."""
import ast
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from taos.im.protocol.models import MarketSimulationConfig
from taos.im.validator.report import neuron_info_labels

ROOT = Path(__file__).resolve().parents[1]
INFO_OWN_LABELS = 3      # wallet, netuid, sim_id on the Info metric itself
SCRAPE_LABELS = 2        # job, instance added by the scraper
BOUND = 100              # the per-class series at 100 labels is ingested by the store that refused 150


def _scoring_keys():
    """The keys of the report payload's validator_config['scoring'] dict, read off the validator source."""
    tree = ast.parse((ROOT / "taos" / "im" / "neurons" / "validator.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == "validator_config" and isinstance(v, ast.Dict):
                for k2, v2 in zip(v.keys, v.values):
                    if isinstance(k2, ast.Constant) and k2.value == "scoring" and isinstance(v2, ast.Dict):
                        return [k3.value for k3 in v2.keys if isinstance(k3, ast.Constant)]
    raise AssertionError("no validator_config scoring dict in the report payload")


def _duck():
    sim = MarketSimulationConfig.from_xml(ET.parse(ROOT / "simulate" / "trading" / "run" / "config" / "simulation_0.xml").getroot())
    hot = "5E7RbxVyLz681r9tF9a9bfB9ZuBQa4ZARbJ8HApjRruGaNtY"
    return SimpleNamespace(
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address=hot), coldkeypub=SimpleNamespace(ss58_address="5HDgAzc4E7")),
        metagraph=SimpleNamespace(hotkeys=["5A", hot]),
        config=SimpleNamespace(subtensor=SimpleNamespace(network="test"), wallet=SimpleNamespace(name="taos_testnet", hotkey="validator")),
        validator_config={"scoring": {k: 1 for k in _scoring_keys()}},
        simulation=sim,
    )


def test_the_series_carries_identity_dials_and_fee_policy_and_nothing_else():
    labels = neuron_info_labels(_duck())
    groups = {"identity": 0, "config_scoring_": 0, "simulation_fee_policy_": 0, "other": []}
    for k in labels:
        if k in ("uid", "network", "coldkey", "coldkey_name", "hotkey", "name"):
            groups["identity"] += 1
        elif k.startswith("config_scoring_"):
            groups["config_scoring_"] += 1
        elif k.startswith("simulation_fee_policy_"):
            groups["simulation_fee_policy_"] += 1
        else:
            groups["other"].append(k)
    assert groups["other"] == [], f"labels the two readers never use, the market parameters belong to simulation_config_info: {groups['other']}"
    assert groups["identity"] == 6 and groups["config_scoring_"] == len(_scoring_keys()) and groups["simulation_fee_policy_"] >= 1
    assert labels["uid"] == "1" and labels["hotkey"].startswith("5E7")


def test_the_series_stays_well_under_the_store_label_cap():
    n = len(neuron_info_labels(_duck())) + INFO_OWN_LABELS + SCRAPE_LABELS
    assert n < BOUND, f"{n} labels: the store that refused 150 and took 100 would refuse this series again"


def test_the_fee_policy_labels_keep_the_name_the_fee_policy_table_reads():
    labels = neuron_info_labels(_duck())
    fee = {k for k in labels if k.startswith("simulation_fee_policy_")}
    assert "simulation_fee_policy_type" in fee, "the Fee Policy table filters on simulation_fee_policy_.*"
