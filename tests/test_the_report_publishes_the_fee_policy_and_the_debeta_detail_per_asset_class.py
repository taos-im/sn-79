# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Operator, morning of 8 October 2026: the dashboards' asset-class selector must reach the fee policy and the
per-miner de-beta figures too. Each class's own configuration carries its own FeePolicy, so the report publishes
one fee_policy_info row per class; the class summaries are labelled by the class's name, as book_asset_class and
simulation_config_info already are; and the per-class per-uid detail the scorer computes for the class pools is
kept and published as agent_class_gauges, so an agent's making, ranks, factors and books can be read per class.
No scoring math changes: the scorer keeps a reference to what it already computed."""
import inspect
from types import SimpleNamespace

from taos.im.validator.report import (
    CLASS_DETAIL_GAUGES,
    class_names,
    publish_agent_class_gauges,
    publish_class_gauges,
    publish_simulation_config_info,
    simulation_config_rows,
)


class _Fee:
    def __init__(self, **p):
        self.p = p

    def to_prom_info(self):
        return {"simulation_fee_policy_type": "dynamic"} | {f"simulation_fee_policy_{k}": str(v) for k, v in self.p.items()}


class _Cfg:
    def __init__(self, fee_policy=None, **kw):
        self._d = dict(kw, fee_policy=fee_policy)

    def model_dump(self):
        return dict(self._d)

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name)


def _cls(name, books, fee=None, **cfg):
    return SimpleNamespace(name=name, books=books, config=_Cfg(fee_policy=fee, **cfg))


def _sim(*classes):
    return SimpleNamespace(asset_classes=lambda: list(classes), simulation_id="sim-1")


class _Labelled:
    def __init__(self, log):
        self.log = log

    def labels(self, **kw):
        return SimpleNamespace(info=lambda d: self.log.append(("info", kw, dict(d))), set=lambda v: self.log.append(("set", kw, v)))

    def clear(self):
        self.log.append(("clear",))


def test_each_class_row_carries_its_own_fee_policy_in_the_labels_the_fee_policy_table_reads():
    fast, slow = _Fee(takerFee="0.00023", targetMTR="0.4"), _Fee(takerFee="0.00050", targetMTR="0.5")
    rows = simulation_config_rows(_sim(_cls("simulation_0", [0, 1], fee=fast, init_price=300.0),
                                       _cls("simulation_1", [2, 3], fee=slow, init_price=29.57),
                                       _cls("bare", [4], init_price=1.0)))
    assert {k: v for k, v in rows["simulation_0"].items() if k.startswith("simulation_fee_policy_")} == fast.to_prom_info()
    assert rows["simulation_1"]["simulation_fee_policy_takerFee"] == "0.00050"
    assert rows["simulation_1"]["simulation_init_price"] == "29.57", "the market fields stay beside the fee fields"
    assert not any(k.startswith("simulation_fee_policy") for k in rows["bare"])
    assert "simulation_fee_policy" not in rows["simulation_0"], "the raw dump key never reaches the row, only the table's labels"


def test_the_fee_policy_is_published_per_class_on_the_config_row():
    log = []
    duck = SimpleNamespace(
        prometheus_simconfig_info=_Labelled(log), prometheus_book_class=_Labelled(log),
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hot")), config=SimpleNamespace(netuid=79),
        simulation=_sim(_cls("simulation_0", [0, 1], fee=_Fee(takerFee="0.00023"), init_price=300.0),
                        _cls("simulation_1", [2], fee=_Fee(takerFee="0.00050"), init_price=29.57)),
    )
    publish_simulation_config_info(duck)
    rows = [(kw["asset_class"], d) for kind, kw, d in [e for e in log if e[0] == "info"]]
    assert [c for c, _ in rows] == ["simulation_0", "simulation_1"]
    assert rows[0][1]["simulation_fee_policy_takerFee"] == "0.00023" and rows[1][1]["simulation_fee_policy_takerFee"] == "0.00050"
    assert rows[0][1]["simulation_init_price"] == "300.0" and "_book_ids" not in rows[0][1]


def test_the_per_class_series_stays_under_the_store_label_cap_with_the_fee_fields():
    """The store behind the dashboards took the per-class series at 100 labels and refused neuron_info at 150 (6 October
    2026); the real class configuration with its fee policy must stay well inside that."""
    from pathlib import Path
    from xml.etree import ElementTree as ET

    from taos.im.protocol.models import MarketSimulationConfig

    root = Path(__file__).resolve().parents[1]
    cfg = MarketSimulationConfig.from_xml(ET.parse(root / "simulate" / "trading" / "run" / "config" / "simulation_0.xml").getroot())
    row = simulation_config_rows(_sim(SimpleNamespace(name="simulation_0", books=list(range(96)), config=cfg)))["simulation_0"]
    assert any(k.startswith("simulation_fee_policy_") for k in row), "the shipped configuration carries a fee policy"
    n = len(row) - 1 + 4 + 2  # fields less _book_ids, plus wallet, netuid, sim_id, asset_class, plus job and instance
    assert n <= 120, f"{n} labels on the per-class series"


def test_class_names_follow_the_configuration_order_and_degrade_to_nothing():
    sim = _sim(_cls("simulation_0", [0]), _cls("simulation_1", [1]))
    assert class_names(sim) == {0: "simulation_0", 1: "simulation_1"}

    def boom():
        raise RuntimeError("no classes here")

    assert class_names(SimpleNamespace(asset_classes=boom)) == {}
    assert class_names(None) == {}


def test_the_class_gauges_are_labelled_by_the_class_name():
    duck = SimpleNamespace(
        prometheus_class_gauges="CLASS", simulation=_sim(_cls("simulation_0", [0]), _cls("simulation_1", [1])),
        debeta_class_summary={0: {"n_makers": 12, "making_credit": 40.0}, 1: {"n_makers": 0}, 7: {"n_makers": 1}},
    )
    updates = []
    publish_class_gauges(duck, updates, "hot", 79, "sim-1")
    assert ("CLASS", 12.0, "hot", 79, "sim-1", "simulation_0", "n_makers") in updates
    assert ("CLASS", 40.0, "hot", 79, "sim-1", "simulation_0", "making_credit") in updates
    assert ("CLASS", 0.0, "hot", 79, "sim-1", "simulation_1", "n_makers") in updates
    # a class id the configuration does not name keeps its id, so nothing is silently dropped
    assert ("CLASS", 1.0, "hot", 79, "sim-1", "7", "n_makers") in updates


def test_the_agent_class_gauges_carry_the_per_class_detail_by_class_name_and_uid():
    duck = SimpleNamespace(
        prometheus_agent_class_gauges="AGENTCLASS", simulation=_sim(_cls("simulation_0", [0]), _cls("simulation_1", [1])),
        debeta_class_details={
            0: {5: {"making_raw": 2.5, "making_rank": 0.5, "skill_raw": 0.1, "skill_rank": 0.25, "p11_factor": 0.9,
                    "making_s3_factor": 1.0, "bg_share": None, "skill_coverage_factor": 1.0, "skill_net_alpha": -0.2,
                    "skill_books": 3, "coverage_books": 7, "present": True, "notional": 1234.5}},
            1: {5: {"making_raw": 0.0, "making_rank": 0.0, "present": False, "coverage_books": 0}},
        },
    )
    updates = []
    publish_agent_class_gauges(duck, updates, "hot", 79, "sim-1")
    rows = {(u[5], u[6], u[7]): u[1] for u in updates}
    assert all(u[0] == "AGENTCLASS" and u[2:5] == ("hot", 79, "sim-1") for u in updates)
    assert rows[("simulation_0", 5, "making")] == 2.5 and rows[("simulation_0", 5, "making_rank")] == 0.5
    assert rows[("simulation_0", 5, "skill")] == 0.1 and rows[("simulation_0", 5, "skill_rank")] == 0.25
    assert rows[("simulation_0", 5, "s3_factor")] == 1.0 and rows[("simulation_0", 5, "coverage_factor")] == 1.0
    assert rows[("simulation_0", 5, "net_alpha")] == -0.2 and rows[("simulation_0", 5, "notional")] == 1234.5
    assert rows[("simulation_0", 5, "skill_books")] == 3.0 and rows[("simulation_0", 5, "coverage_books")] == 7.0
    assert rows[("simulation_0", 5, "present")] == 1.0 and rows[("simulation_1", 5, "present")] == 0.0
    assert ("simulation_0", 5, "bg_share") not in rows, "a None reads as absent, not as zero"
    assert all(isinstance(v, float) for v in rows.values())
    assert {g for _, g in CLASS_DETAIL_GAUGES} >= {"making", "making_rank", "skill", "skill_rank", "p11_factor", "s3_factor",
                                                   "bg_share", "coverage_factor", "net_alpha", "skill_books", "coverage_books",
                                                   "present", "notional"}


def test_the_scorer_keeps_the_class_details_and_the_validator_hands_them_to_the_report():
    from taos.im.neurons import validator
    from taos.im.validator import report, reward

    rsrc = inspect.getsource(reward)
    assert "self._debeta_class_details = _details" in rsrc, "the per-class detail the class pools are built from is kept for the report"
    assert "self._debeta_class_details = None" in rsrc, "and reset with the class summary every round"
    assert "'debeta_class_details': getattr(self, '_debeta_class_details', None) or {}" in inspect.getsource(validator)
    assert "self.debeta_class_details = " in inspect.getsource(report)
