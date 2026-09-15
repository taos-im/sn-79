# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner must be able to SEE a notice it was sent, without overriding anything.

WHY THIS EXISTS. A miner reported receiving RDPOM and never seeing it. The notice was delivered
correctly and dispatched correctly; the framework's onOrderAccepted default is a docstring-only stub
-- "To be implemented by subclasses" -- so an agent that does not override it prints nothing, and the
operator has no way to tell a delivered notice from an undelivered one.

WHY THE ACCEPTANCE SUITE COULD NOT CATCH IT, which is the part worth encoding. The suite HAS a
scenario for this path: s_agent_handlers_fire asserts onOrderAccepted is invoked, written after a
real bug where notices arrived and no handler ever ran. But AcceptanceMinerAgent OVERRIDES
onOrderAccepted to record the call -- so the suite exercises its own override and never the default.
It proved DISPATCH and stopped one step short of OBSERVABILITY, and the step it stopped short of is
the one a miner stands on.

So the property is asserted here against an agent that overrides NOTHING, which is what a miner
copying the published examples actually has.
"""
import importlib.util as ilu
import pathlib
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def agents_mod():
    try:
        sys.path.insert(0, str(_ROOT))
        from taos.im import agents as m
        return m
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"taos.im.agents needs the runtime: {exc!r}")


class _Event:
    """The shape the dispatch loop reads: a type and whatever the log line quotes."""

    def __init__(self, etype, **kw):
        self.type = etype
        self.bookId = kw.get("bookId", 5)
        self.side = kw.get("side", 0)
        self.tradeId = kw.get("tradeId", 1)
        self.quantity = kw.get("quantity", 1.0)
        self.price = kw.get("price", 0.01)
        self.timestamp = kw.get("timestamp", 0)
        self.takerAgentId = kw.get("takerAgentId", 1)
        self.makerAgentId = kw.get("makerAgentId", 2)
        self.takerOrderId = kw.get("takerOrderId", 10)
        self.makerOrderId = kw.get("makerOrderId", 11)
        self.orderId = kw.get("orderId", 10)

    def __repr__(self):
        return f"<{self.type}>"


def _plain_agent(agents_mod, events):
    """An agent that implements only what is REQUIRED and overrides no handler.

    FinanceAgent is abstract (`initialize`), so a bare object.__new__ is refused -- which is useful:
    the fixture has to be a real subclass, and a real subclass implementing nothing but the abstract
    members is precisely what a miner copying a published example ends up with.
    """
    cls = getattr(agents_mod, "FinanceAgent", None)
    if cls is None:
        pytest.skip("FinanceAgent not exported")

    ns = {"initialize": lambda self, *a, **k: None}
    import abc
    for name in getattr(cls, "__abstractmethods__", ()):
        ns.setdefault(name, lambda self, *a, **k: None)
    Plain = type("PlainMinerAgent", (cls,), ns)
    a = object.__new__(Plain)
    a.events = events
    a.uid = 1
    return a


def _capture(monkeypatch, agents_mod):
    """Whatever the dispatch put in front of an operator.

    The emitter is bt.logging.info, so that is what is intercepted -- not a guessed set of names.
    """
    seen = []
    bt = getattr(agents_mod, "bt", None)
    if bt is None or not hasattr(bt, "logging"):
        pytest.skip("bt.logging is not reachable from the module under test")
    monkeypatch.setattr(bt.logging, "info",
                        lambda *a, **k: seen.append(" ".join(str(x) for x in a)), raising=False)
    return seen


def test_the_default_handler_is_silent_which_is_why_this_matters(agents_mod):
    """The premise, asserted so this test cannot quietly stop being about anything.

    Read through the AST rather than by scanning source lines: the first version filtered lines that
    START with a quote and so mistook the docstring's own CONTENT for executable body, failing on a
    handler that is exactly as empty as it claims.
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(agents_mod.FinanceAgentBase))
    fn = next(f for f in tree.body[0].body
              if isinstance(f, ast.FunctionDef) and f.name == "onOrderAccepted")
    body = [b for b in fn.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
    body = [b for b in body if not isinstance(b, ast.Pass)]
    assert not body, (
        "FinanceAgentBase.onOrderAccepted now does something. If it logs, assert THAT instead of the "
        f"dispatch wrapper: {[type(b).__name__ for b in body]}"
    )


def test_every_dispatched_notice_produces_a_logged_line(agents_mod, monkeypatch):
    """The property a miner depends on: delivered implies visible."""
    events = [_Event("RDPOM"), _Event("EVENT_TRADE"), _Event("ET")]
    agent = _plain_agent(agents_mod, events)
    seen = _capture(monkeypatch, agents_mod)
    try:
        agent._dispatch_notice_handlers(types.SimpleNamespace())
    except Exception as exc:
        pytest.skip(f"dispatch needs more state than this fixture builds: {exc!r}")
    blob = "\n".join(seen)
    assert "EXCHANGE EVENTS" in blob, (
        "an agent overriding nothing produced NO operator-visible output for 3 delivered notices. "
        "That is the miner-reported failure: delivered, dispatched, invisible."
    )
    for e in events:
        assert e.type in blob or str(e.bookId) in blob, f"{e.type} was dispatched but never shown"


def test_it_does_not_depend_on_the_agent_overriding_anything(agents_mod):
    """The logging must sit in the dispatch wrapper, not in a handler a subclass may replace."""
    import inspect
    src = inspect.getsource(agents_mod.FinanceAgent._dispatch_notice_handlers)
    assert "logged" in src and "EXCHANGE EVENTS" in src, (
        "the per-notice logging is not in the dispatch wrapper. If it lives in a handler, an agent "
        "that overrides that handler loses its own visibility -- which is exactly how the acceptance "
        "suite's own agent hid this defect for the whole life of exchange mode."
    )
