# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A notice has one shape, whichever mode produced it.

Both state updates carry notices through msgpack, so both used to hand consumers raw JSON-shaped dicts
with stringified keys. Two shims existed to absorb that:

    events.py       `_etype(e)` -- e.get("type") if isinstance(e, dict) else getattr(e, "type", None)
    agents/__init__ notices looked up by self.uid, then again by str(self.uid)

Both decompress paths now run their tree's `parse_notices`, which builds event models and keys them by
int uid. A consumer reads `.type` and `notices[uid]` without asking what shape it received, and the
shims are gone.

The earlier version of this file described the asymmetry as "simulation parses into objects, exchange
leaves dicts". That was wrong in a way worth recording, because it scoped the fix to one path: NEITHER
path parsed. What actually differed was that two same-named `AgentEventHistory` classes existed, and
only one tolerated dicts -- so which one an agent got, and therefore whether it worked, was decided by
import order. `test_protocol_trees_do_not_shadow.py` covers that half.
"""

import re
from pathlib import Path

DEV = Path(__file__).resolve().parents[1]
EVENTS = DEV / "taos/im/protocol/events.py"
AGENTS = DEV / "taos/im/agents/__init__.py"


def test_events_no_longer_needs_the_dict_or_object_shim():
    src = EVENTS.read_text()
    assert "isinstance(e, dict)" not in src, (
        "_etype existed only because notices could be dicts; with the parse in place it reads e.type"
    )


def test_the_agent_does_not_look_notices_up_twice():
    src = AGENTS.read_text()
    doubled = re.search(r"_nt\.get\(self\.uid\).*?_nt\.get\(str\(self\.uid\)", src, re.S)
    assert not doubled, "one parse produces one key type, so one lookup is enough"


def test_notices_parse_into_event_models_on_both_paths():
    """Behaviour, not a source-text probe.

    The previous version of this checked for the string "model_construct" in the exchange models module,
    after splitting it on "notices" -- a word that does not appear in that file, so the split returned
    the whole module and the assertion passed on unrelated text. It would not have failed if the parse
    had never been written.
    """
    from taos.im.protocol.events import parse_notices as sim_parse
    from taos.im.protocol.exchange.events import parse_notices as xch_parse

    wire = {"171": [{"type": "ET", "timestamp": 5, "agentId": 171, "id": 9}]}
    for label, parse in (("simulation", sim_parse), ("exchange", xch_parse)):
        out = parse({k: [dict(e) for e in v] for k, v in wire.items()})
        assert list(out) == [171], f"{label}: uid keys must be ints, got {list(out)}"
        (event,) = out[171]
        assert not isinstance(event, dict), f"{label}: notice still a dict"
        assert event.type == "ET", f"{label}: .type unreadable ({event!r})"
        assert event.timestamp == 5, f"{label}: .timestamp unreadable"


def test_an_unrecognised_notice_is_passed_through_rather_than_dropped():
    """LOSING A MINER'S NOTICE IS WORSE THAN A MIXED SHAPE.

    This test asserted the opposite until 2026-08-18, on the reasoning that consumers select by type so
    an unparsed notice is unreadable anyway. It is readable -- as a dict, which is how every consumer
    read notices before the parse existed. Dropping cost the exchange path its RDCP close notices,
    because that tree has no ClosePositionsEvent, and s_sltp reported it as a trigger that never fired.

    A code that neither tree's dispatcher knows would be a genuinely new wire type. It survives here and
    is logged, so it shows up as an unhandled type rather than as missing data.
    """
    from taos.im.protocol.events import parse_notices

    out = parse_notices({"1": [{"type": "NOT_A_REAL_TYPE", "timestamp": 3}]})
    assert list(out) == [1], f"uid key lost: {out}"
    (survivor,) = out[1]
    assert isinstance(survivor, dict), f"expected the raw notice preserved, got {type(survivor).__name__}"
    assert survivor.get("type") == "NOT_A_REAL_TYPE"


def test_empty_and_missing_notices_are_passed_through_untouched():
    from taos.im.protocol.events import parse_notices

    assert parse_notices(None) is None
    assert parse_notices({}) == {}


def test_every_shim_this_removed_is_named_here():
    """If a new notice-shape shim appears, this fails rather than letting one drift back in."""
    hits = []
    for path in (EVENTS, AGENTS):
        src = path.read_text()
        for pat in (r"isinstance\((?:e|ev|entry), dict\)", r"get\(str\(self\.uid\)"):
            if re.search(pat, src):
                hits.append(f"{path.name}:{pat}")
    assert not hits, f"notice-shape shims are back: {hits}"


def _dispatcher_codes():
    """Every short wire code either tree's from_json dispatcher names, read from the source.

    Derived rather than listed, so a code added to one dispatcher is covered here without anyone
    remembering to update a constant.
    """
    codes = set()
    for rel in ("taos/im/protocol/events.py", "taos/im/protocol/exchange/events.py"):
        src = (DEV / rel).read_text()
        body = src.split("def from_json", 1)[-1][:4000]
        for m in re.finditer(r'case ([^:]+):', body):
            for lit in re.findall(r'"([A-Z_]+)"', m.group(1)):
                # Short codes only; the long RESPONSE_DISTRIBUTED_* forms come off the simulator, not
                # the wire, and carry a different payload shape.
                if len(lit) <= 8:
                    codes.add(lit)
    return sorted(codes)


def test_every_wire_code_parses_on_both_paths():
    """NO NOTICE IS EVER DROPPED, on either path.

    The first version of parse_notices skipped whatever its own tree's dispatcher did not recognise,
    reasoning that consumers select by type so an unparsed notice is unreadable anyway. Wrong twice: a
    dict is perfectly readable, and the exchange tree has no ClosePositionsEvent -- so RDCP was silently
    discarded and an SL/TP trigger's close notice never reached the miner. s_sltp caught it as "0 close
    notice(s) after the move", which reads like a product defect in the trigger.
    """
    from taos.im.protocol.events import parse_notices as sim_parse
    from taos.im.protocol.exchange.events import parse_notices as xch_parse

    codes = _dispatcher_codes()
    assert "RDCP" in codes, f"expected RDCP among the derived codes, got {codes}"

    problems = []
    for code in codes:
        for label, parse in (("simulation", sim_parse), ("exchange", xch_parse)):
            wire = {"type": code, "timestamp": 1, "agentId": 4, "id": 1, "r": 0, "l": "d"}
            got = (parse({4: [wire]}) or {}).get(4, [])
            if not got:
                problems.append(f"{label}/{code}: DROPPED")
            elif isinstance(got[0], dict):
                problems.append(f"{label}/{code}: passed through as a dict")
            elif getattr(got[0], "type", None) != code:
                problems.append(f"{label}/{code}: .type reads {getattr(got[0], 'type', None)!r}")
    assert not problems, "notice codes not handled uniformly:\n  " + "\n  ".join(problems)


def test_parsing_a_notice_loses_no_field_the_validator_stamped():
    """A NOTICE MUST SURVIVE THE ROUND-TRIP WHOLE.

    Event classes declare the fields the simulator emits. The validator stamps others on top -- `cr` (the
    SL/TP close reason), `xo` (the external-order UUID the data service keys on), `seq` -- and no class
    declares them. Under pydantic's default `extra='ignore'` they were dropped as soon as the wire dict
    became a model, and `model_dump()` could not emit what was never kept.

    That is the worst shape of this bug: the notice still arrives, still looks complete, and is missing
    only the field its consumer selects on. Measured 2026-08-18: an SL/TP trigger closed the position
    correctly, the close notice reached the miner without `cr`, and the scenario reported a trigger that
    never fired. `Ma` masked it in testing by happening to be a declared field.

    Asserted as "no key is lost" rather than "cr is kept", because the next stamped field will not be cr.
    """
    from taos.im.protocol.events import parse_notices as sim_parse
    from taos.im.protocol.exchange.events import parse_notices as xch_parse

    wire = {"y": "ET", "t": 111, "a": 4, "b": 5, "q": 0.5, "p": 0.07, "i": 9,
            "cr": "TP", "xo": "uuid-abc", "seq": 42}

    for label, parse in (("simulation", sim_parse), ("exchange", xch_parse)):
        event = (parse({4: [dict(wire)]}) or {}).get(4, [None])[0]
        assert event is not None and not isinstance(event, dict), f"{label}: notice not parsed"
        dumped = event.model_dump()
        lost = sorted(k for k in wire if k not in dumped)
        assert not lost, (
            f"{label}: {lost} dropped by the parse/serialise round-trip. Every consumer downstream of "
            f"the miner reads these off the published notice, so a dropped key is invisible data loss."
        )
        assert str(dumped.get("cr")) == "TP", f"{label}: cr did not survive as a value"
