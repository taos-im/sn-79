# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A de-beta dial a config lacks falls back to the validator's own parser default.

The scorer, the trade path, the scoring child's config, the config dict shipped to it and the validator's published
labels each carried their own literal fallbacks, and dozens disagreed with the parser: p11_topk 2 against 10, s3_k 0
against 4, w_make 0 or 0.30 against 0.5, floor_scale 0.5 against 0, making_pool 'rank' against 'proportional_both',
making_basis 'captured' against 'realized', the maker fee weight 0 against 0.005, and others. A full validator
config carries every dial, so production never reached them; a config without one scored with a different rule.
The parser is now the one table: debeta_defaults() reads it, and every fallback goes through debeta_dial(). A dial
set to zero keeps its zero.
"""
import inspect
import pathlib
import re
import types

from taos.im.config import debeta_defaults, debeta_dial

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "taos/im/validator").glob("*.py")) + [ROOT / "taos/im/neurons/validator.py"]
RECV = (r"(?:dcfg|_dcfg|debeta_cfg|_debeta_cfg|_d|debeta|\(dcfg or \{\}\)"
        r"|getattr\(\s*[\w.]+\s*,\s*'debeta'\s*,\s*None\s*\))")
GETATTR = re.compile(r"getattr\(\s*" + RECV + r"\s*,\s*['\"](\w+)['\"]\s*,(?!\s*None\s*\))\s*([^()]*?)\)")
GET = re.compile(RECV + r"\.get\(\s*['\"](\w+)['\"]\s*,(?!\s*None\s*\))\s*([^()]*?)\)")
NONE_OR = re.compile(r"getattr\(\s*" + RECV + r"\s*,\s*['\"](\w+)['\"]\s*,\s*None\s*\)\s*or\s*([\w.'\"]+)")


def test_the_table_is_the_parser():
    d = debeta_defaults()
    assert d["w_make"] == 0.5 and d["p11_topk"] == 10 and d["s3_k"] == 4.0 and d["making_fee_weight"] == 0.005
    assert d is debeta_defaults(), "read once"


def test_a_missing_dial_reads_the_parser_default_and_a_set_one_its_value():
    assert debeta_dial(types.SimpleNamespace(), "p11_topk") == 10
    assert debeta_dial(None, "w_make") == 0.5
    assert debeta_dial({}, "w_make") == 0.5
    assert debeta_dial({"p11_topk": 3}, "p11_topk") == 3
    assert debeta_dial(types.SimpleNamespace(s3_k=0.0), "s3_k") == 0.0, "an explicit 0 is kept"
    assert debeta_dial({"w_make": None}, "w_make") == 0.5


def test_no_dial_read_falls_back_to_a_literal():
    dials = set(debeta_defaults())
    bad = []
    for f in FILES:
        src = f.read_text()
        for rx in (GETATTR, GET):
            for m in rx.finditer(src):
                if m.group(1) in dials:
                    bad.append(f"{f.relative_to(ROOT)}:{src[:m.start()].count(chr(10)) + 1} {m.group(1)} -> {m.group(2)}")
    assert not bad, "\n".join(bad)


def test_a_none_fallback_lands_on_the_parser_default():
    d = debeta_defaults()
    bad = []
    for f in FILES:
        src = f.read_text()
        for m in NONE_OR.finditer(src):
            name, lit = m.groups()
            if name not in d:
                continue
            value = {"PRESENCE_WINDOW_DEFAULT": 50}.get(lit, lit.strip("'\""))
            if str(value) != str(d[name]):
                bad.append(f"{f.name}:{src[:m.start()].count(chr(10)) + 1} {name} or {lit}, parser {d[name]!r}")
    assert not bad, "\n".join(bad)


def test_the_child_carries_every_dial_the_parser_declares():
    from taos.im.validator.scoring_shadow import _debeta_duck
    duck = _debeta_duck(types.SimpleNamespace(w_make=0.0))
    assert set(vars(duck)) == set(debeta_defaults())
    assert duck.w_make == 0.0 and duck.p11_topk == 10 and duck.making_pool == "proportional_both"


def test_the_payout_reads_the_resolved_skill_bar():
    from taos.im.validator import reward, scoring_shadow
    assert "skill_bar(self)," in inspect.getsource(reward.get_rewards)
    assert "skill_bar(shadow)," in inspect.getsource(scoring_shadow.shadow_score)
