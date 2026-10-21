"""The validator publishes the scoring dials that decide 0.6.3 and 0.6.4 pay in its reporting config (neuron_info), so
the dashboards' Scoring Config table can show them: the making basis and its horizon, the tether multiple, the class
weights, the skill bar share and the skill hurdle, and from 0.6.4 the making fee weight, the capture market votes
switch and the skill share EMA switch. The whitelist is hand-built in the validator's report payload, so a dial
missing from it never reaches a dashboard however it is set. The PnL Score's daily-return bounds are published from
the dials the PnL Score reads, scoring.pnl.normalization, not from a path the config does not carry."""
import argparse
import ast
import pathlib
from types import SimpleNamespace

SRC = pathlib.Path(__file__).resolve().parents[1] / "taos" / "im" / "neurons" / "validator.py"
EXPECTED = {
    "debeta_making_basis": "realized",
    "debeta_making_horizon_s": 20.0,
    "debeta_s3_k": 4.0,
    "debeta_class_weights": "books",
    "debeta_skill_min_books_share": 0.15625,
    "debeta_skill_hurdle_bps": 2.3,
    "debeta_making_fee_weight": 0.005,
    "debeta_capture_market_votes": 0,
    "debeta_skill_share_ema": 0,
}


def _reporting_scoring_dict():
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == "validator_config" and isinstance(v, ast.Dict):
                for k2, v2 in zip(v.keys, v.values):
                    if isinstance(k2, ast.Constant) and k2.value == "scoring" and isinstance(v2, ast.Dict):
                        return {k3.value: v3 for k3, v3 in zip(v2.keys, v2.values) if isinstance(k3, ast.Constant)}
    raise AssertionError("no validator_config scoring dict in the report payload")


def _shipped_config():
    """The validator's parsed defaults, nested the way bittensor's config exposes them."""
    from taos.im.config import add_im_validator_args
    p = argparse.ArgumentParser()
    add_im_validator_args(None, p)
    root = SimpleNamespace()
    for a in p._actions:
        if not a.option_strings or a.dest in ("help",):
            continue
        node = root
        parts = a.dest.split(".")
        for part in parts[:-1]:
            if not hasattr(node, part):
                setattr(node, part, SimpleNamespace())
            node = getattr(node, part)
        setattr(node, parts[-1], a.default)
    return root


def _scope(self):
    """What the payload's expressions see: the validator, its de-beta block read once and the dial reader."""
    from taos.im.config import debeta_dial
    return {"self": self, "_dcfg": getattr(self.config.scoring, "debeta", None), "debeta_dial": debeta_dial}


def test_the_063_scoring_dials_are_in_the_reporting_config():
    missing = set(EXPECTED) - set(_reporting_scoring_dict())
    assert not missing, f"not published to neuron_info: {sorted(missing)}"


def test_each_published_value_is_the_shipped_default_and_a_label_can_carry_it():
    entries = _reporting_scoring_dict()
    self = SimpleNamespace(config=_shipped_config())
    for key, want in EXPECTED.items():
        value = eval(compile(ast.Expression(entries[key]), str(SRC), "eval"), {}, _scope(self))
        assert isinstance(value, (str, int, float)), (key, type(value))
        assert value == want, (key, value, want)


def test_an_older_config_without_the_dials_still_publishes():
    entries = _reporting_scoring_dict()
    self = SimpleNamespace(config=SimpleNamespace(scoring=SimpleNamespace(debeta=SimpleNamespace())))
    for key in EXPECTED:
        value = eval(compile(ast.Expression(entries[key]), str(SRC), "eval"), {}, _scope(self))
        assert value == EXPECTED[key], (key, value, "a dial the config lacks publishes the parser default")
        assert isinstance(value, (str, int, float)), (key, value)


def test_the_pnl_return_bounds_are_published_from_the_dials_the_pnl_score_reads():
    entries = _reporting_scoring_dict()
    self = SimpleNamespace(config=_shipped_config())
    for key, want in (("pnl_normalization_min_daily_return", -1.0), ("pnl_normalization_max_daily_return", 1.0)):
        value = eval(compile(ast.Expression(entries[key]), str(SRC), "eval"), {}, {"self": self})
        assert value == want, (key, value, want)
