"""The export surface of taos.im.protocol.models is pinned.

The module is star-imported by the protocol package, the miner agent base and the example agents that
third-party miners copy, and it has no `__all__`, so its whole public namespace is API. These fixtures
were generated from the module before its classes were split into domain modules behind a facade; the
test holds the facade to the same names, kinds, bases, fields and enum members, and to byte-identical
serialisation of a filled instance of every pydantic class.
"""
import enum
import inspect
import json
import re
from pathlib import Path

from pydantic import BaseModel as _PB

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SURFACE = json.loads((FIXTURES / "protocol_models_surface.json").read_text(encoding="utf-8"))
ROUNDTRIP = json.loads((FIXTURES / "protocol_models_roundtrip.json").read_text(encoding="utf-8"))


def _module():
    import taos.im.protocol.models as m
    return m


def _norm(annotation: str) -> str:
    # A nested class's module path is part of str(annotation); the classes moved to domain modules, so
    # the path is normalised to the package and the class name is what is compared.
    return re.sub(r"taos\.im\.protocol\.\w+\.", "taos.im.protocol.", annotation)


def test_every_recorded_name_still_resolves_with_the_same_kind():
    m = _module()
    missing = [n for n in SURFACE if not hasattr(m, n)]
    assert not missing, f"names gone from the facade: {missing}"
    for name, rec in SURFACE.items():
        obj = getattr(m, name)
        if rec["kind"] == "class":
            assert inspect.isclass(obj), name
            assert obj.__qualname__ == rec["qualname"], name
            assert [b.__name__ for b in obj.__bases__] == rec["bases"], name
        elif rec["kind"] == "module":
            assert inspect.ismodule(obj) and obj.__name__ == rec["name"], name
        elif rec["kind"] == "function":
            assert callable(obj), name


def test_pydantic_fields_and_enum_members_are_unchanged():
    m = _module()
    for name, rec in SURFACE.items():
        if rec["kind"] != "class":
            continue
        obj = getattr(m, name)
        if "fields" in rec:
            got = {k: {"annotation": _norm(str(v.annotation)), "alias": v.alias, "required": v.is_required()}
                   for k, v in obj.model_fields.items()}
            exp = {k: {**f, "annotation": _norm(f["annotation"])} for k, f in rec["fields"].items()}
            assert got == exp, f"{name}: fields differ"
        if "members" in rec:
            assert {e.name: e.value for e in obj} == rec["members"], f"{name}: enum members differ"


def test_star_import_yields_at_least_the_recorded_public_names():
    ns = {}
    exec("from taos.im.protocol.models import *", ns)
    exported = {k for k in ns if not k.startswith("_")}
    missing = sorted(set(SURFACE) - exported)
    assert not missing, f"star import lost: {missing}"


def test_every_pydantic_class_serialises_a_filled_instance_identically():
    m = _module()
    for name, rec in ROUNDTRIP.items():
        cls = getattr(m, name)
        assert inspect.isclass(cls) and issubclass(cls, _PB), name
        inst = cls.model_validate(rec["input"]) if rec["how"] == "validate" else cls.model_construct()
        assert inst.model_dump_json() == rec["json"], f"{name}: serialisation changed"


def test_the_facade_keeps_the_subclass_relations_agents_rely_on():
    m = _module()
    assert issubclass(m.LazyBook, m.Book)
    assert issubclass(m.EventHistory, m.History) and issubclass(m.L2History, m.History)
    for e in (m.OrderDirection, m.STP, m.TimeInForce, m.OrderCurrency, m.LoanSettlementOption):
        assert issubclass(e, enum.IntEnum)
