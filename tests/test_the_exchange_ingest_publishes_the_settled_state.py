# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The exchange ingest push publishes the last settled state, not the engine's prediction of this block's match.

After execute() the validator re-reads books and accounts from the engine's post-match state, and the push carried
that: an order the chain later declined showed as filled, and the balances as moved, on every published surface
until a later push corrected it. The push now carries the books and accounts receive() reconciled with the chain,
with this block's settled fills appended to the books' events. Orders placed and cancelled this block appear one
block later, as the reconciliation reports them.

handle_state is a single very large coroutine driving a live engine and chain, so the composition is a helper
tested directly, and the wiring is pinned by its order in the source.
"""
import inspect
import re

from taos.im.neurons import validator as validator_mod

_compose = validator_mod._settled_ingest_books


def _handle_state_source() -> str:
    src = inspect.getsource(validator_mod)
    i = src.index("def handle_state")
    return src[i:src.index("\n        def ", i + 1)]


def test_settled_fills_are_appended_to_the_settled_books():
    settled = {3: {"b": [[10.0, 2.0]], "a": [[11.0, 1.0]], "e": [{"y": "o", "i": 7}]}}
    fill = {"y": "t", "nid": 3, "i": "t1", "p": 10.5, "q": 1.0}
    out = _compose(settled, {3: [fill]})
    assert out[3]["b"] == [[10.0, 2.0]] and out[3]["a"] == [[11.0, 1.0]], "the settled levels are what is published"
    assert out[3]["e"] == [{"y": "o", "i": 7}, fill]


def test_a_fill_on_a_book_the_settled_state_lacks_gets_a_book_of_its_own():
    fill = {"y": "t", "nid": 9, "i": "t2"}
    assert _compose({}, {9: [fill]}) == {9: {"b": [], "a": [], "e": [fill]}}


def test_the_settled_snapshot_is_left_unchanged():
    settled = {3: {"b": [], "a": [], "e": []}}
    _compose(settled, {3: [{"y": "t"}]})
    assert settled == {3: {"b": [], "a": [], "e": []}}


def test_books_without_fills_pass_through():
    settled = {1: {"b": [[1.0, 1.0]], "a": [], "e": []}, 2: {"b": [], "a": [], "e": []}}
    assert _compose(settled, {}) == settled


def test_the_snapshot_is_taken_before_the_engine_executes():
    src = _handle_state_source()
    snap = src.index("_settled_books = dict(")
    assert snap < src.index("await self.engine.execute(state, miner_responses)")
    assert "_settled_accounts = " in src[snap - 400:snap + 400]


def test_every_settled_fill_is_recorded_for_the_push():
    src = _handle_state_source()
    loop = src[src.index("for _ef in _recon_now.get("):]
    assert "_settled_fill_events.setdefault(int(_nid), []).append(_ev)" in loop[:loop.index("_settled_fill_notice(")]


def test_the_push_publishes_the_settled_books_accounts_and_open_orders():
    src = _handle_state_source()
    assert re.search(r'"books":\s+_settled_ingest_books\(_settled_books, _settled_fill_events\)', src)
    assert re.search(r'"accounts":\s+_settled_accounts', src)
    assert "for _uid_str, _uid_data in (_settled_accounts or {}).items():" in src
    assert not re.search(r'"books":\s+getattr\(state, .books.', src)
    assert not re.search(r'"accounts":\s+getattr\(state, .accounts.', src)
