"""In exchange mode GenTRX must receive the state AFTER matching, not before.

handle_state pushed the state to GenTRX near the top of the block. In SIMULATION that is correct:
the simulator has already matched, so book['e'] is populated when the state arrives. In EXCHANGE
nothing has matched yet at that point -- orders are sent to the LOB and settled by
`self.engine.execute(state, miner_responses)` further down, and the authoritative 't' events are only
then injected into state.books[nid]['e'] from the reconciliation's executed_fills.

So the exchange aggregator received books with empty event lists on every single tick.
With both agents trading:

    [GTX] ingest: 8 tick(s), 80 book-slot(s), 0 event(s)
    [GTX] ingest: 5 tick(s), 50 book-slot(s), 0 event(s)

Ten book-slots per tick, valid advancing timestamps, and never one event. No rows, so no parquet
page, so /data-status reported one book, so the val split consumed it and every round was skipped.
The exchange mechanism could not produce a gradient for a reason that had nothing to do with GenTRX.

This is a source-ordering guard rather than a behavioural test: handle_state is a single very large
coroutine driving a live engine, chain and axon set, and the defect IS the ordering, so the ordering
is what must be pinned.
"""
import inspect
import re

from taos.im.neurons import validator as validator_mod


# Anchor on the injection LOOP, not on any mention of executed_fills: the explanatory comments
# name it too, and a comment is not the code whose position matters.
def _handle_state_source() -> str:
    src = inspect.getsource(validator_mod)
    i = src.index("def handle_state")
    return src[i:]


def test_exchange_push_happens_after_the_fill_injection():
    src = _handle_state_source()
    inject = src.index("for _ef in _recon_now.get(")
    pushes = [m.start() for m in re.finditer(r"push_state\(state\)", src)]
    assert pushes, "no push_state call found in handle_state"
    assert any(p > inject for p in pushes), (
        "every push_state runs BEFORE the executed_fills injection, so exchange books are pushed "
        "with empty events"
    )


def test_the_early_push_is_not_used_for_exchange():
    src = _handle_state_source()
    inject = src.index("for _ef in _recon_now.get(")
    early = [p for p in [m.start() for m in re.finditer(r"push_state\(state\)", src)] if p < inject]
    for p in early:
        window = src[max(0, p - 600):p]
        assert "exchange" in window, (
            "an early push_state is not gated away from exchange mode; exchange would still be "
            "pushed pre-match"
        )


def test_simulation_still_pushes_early():
    """Simulation must keep its pre-existing behaviour: the sim has already matched."""
    src = _handle_state_source()
    inject = src.index("for _ef in _recon_now.get(")
    pushes = [m.start() for m in re.finditer(r"push_state\(state\)", src)]
    assert any(p < inject for p in pushes), "simulation lost its early push"
