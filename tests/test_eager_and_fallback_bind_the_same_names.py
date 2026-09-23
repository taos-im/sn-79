"""Both arms of the scoring-proc dispatch must bind every name the request reads.

The dispatch has two arms: an eager one that unpacks inputs the shadow pinned at tee time, and a
fallback that recomputes them from the validator. They feed one `request_scores` call through a
lambda, so a name bound in only one arm is not an error until that arm runs — and the eager arm runs
at a boundary, on the live validator, where the NameError kills the score for that round. That is
what happened to `presence` on 22 September: it was added to the fallback and to the call, and the
eager arm left it unbound.

Read from source rather than executed, because the block sits inside the validator's run loop.
"""
import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "taos" / "im" / "neurons" / "validator.py"

# bound before the branch, not inside either arm
_OUTER = {"_eager", "_shadow"}


@pytest.fixture(scope="module")
def dispatch():
    """The `if _eager is not None:` statement that splits the two arms, with its sibling list."""
    tree = ast.parse(SRC.read_text())
    for parent in ast.walk(tree):
        for _, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for child in value:
                    if isinstance(child, ast.AST):
                        child.sibling_body = value
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not node.orelse:
            continue
        t = node.test
        if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "_eager"
                and any(isinstance(o, ast.IsNot) for o in t.ops)):
            return node
    raise AssertionError("the eager dispatch branch is gone — this guard needs rewriting")


def _bound(body):
    names = set()
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
    return names


def _request_reads(node):
    """Underscore-prefixed locals the request lambda closes over, among the branch's siblings."""
    reads = set()
    for sib in node.sibling_body:
        for sub in ast.walk(sib):
            if isinstance(sub, ast.Lambda):
                for n in ast.walk(sub):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id.startswith("_"):
                        reads.add(n.id)
    return reads - _OUTER


def test_the_guard_still_finds_something_to_guard(dispatch):
    """A refactor that moves the request out of the lambda would empty the set below in silence."""
    assert len(_request_reads(dispatch)) >= 8


def test_every_input_the_request_reads_is_bound_on_both_arms(dispatch):
    eager, fallback = _bound(dispatch.body), _bound(dispatch.orelse)
    for name in sorted(_request_reads(dispatch)):
        assert name in eager, f"{name} reaches request_scores but the eager arm never binds it"
        assert name in fallback, f"{name} reaches request_scores but the fallback arm never binds it"


def test_presence_specifically_survives_on_both_arms(dispatch):
    """The regression itself, named, so a rewrite of the guard above cannot quietly drop it."""
    assert "_presence" in _bound(dispatch.body)
    assert "_presence" in _bound(dispatch.orelse)
