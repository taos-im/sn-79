# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A uid slot changing hands must be detected even when no axon changed.

`resync_metagraph` returned early on

    if previous_metagraph.axons == self.metagraph.axons and len(self.hotkeys) == len(self.metagraph.hotkeys):
        bt.logging.debug("No axon changes!")
        return

which skips the hotkey comparison below it entirely. The early return is a real optimisation: a resync that
changed nothing has nothing to do. But the condition asks about AXONS and the work it skips is about
HOTKEYS, and those are different questions.

A deregistration usually changes the axon list too, because the new occupant has not published one yet, so
the bug stays latent. It bites when the new occupant publishes the same axon info the departed miner had:
the same operator re-registering on the same box and port, which is the common case on a test chain and not
rare in production. Then the slot changes hands, `handle_deregistration` is never called, the departed
miner's score and history are never cleared, and the new occupant inherits the standing.

Found 2026-08-05 while building the acceptance suite's deregistration stage. The fix keeps the early return
and widens its condition to cover the hotkeys as well, so it still skips a genuinely unchanged resync.
"""

import re


_SRC = "taos/common/neurons/validator.py"


def _resync_source() -> str:
    src = open(_SRC).read()
    start = src.index("def resync_metagraph")
    end = src.index("\n    def ", start + 1)
    return src[start:end]


def test_the_guard_is_the_tested_function_and_not_an_inline_condition():
    """Keep the shipped guard and the tested guard the same object.

    The condition below is verified directly in test_a_changed_hotkey_alone_is_enough_to_proceed. That is
    only worth anything while resync_metagraph actually calls it, so this pins the call. If someone
    reinstates an inline `if previous_metagraph.axons == ...` the extracted function becomes dead code that
    still passes its own tests, which is exactly the failure this file exists to prevent.
    """
    body = _resync_source()
    assert "resync_inputs_unchanged(" in body, (
        "resync_metagraph no longer calls the guard function; if the condition moved back inline it is no "
        "longer covered by the behavioural test in this file"
    )
    inline = re.search(r"if \(?previous_metagraph\.axons == self\.metagraph\.axons", body)
    assert inline is None, (
        "the early-return condition is inline again; move it back behind "
        f"resync_inputs_unchanged so it stays tested. Found: {inline.group(0) if inline else ''}"
    )


def test_the_hotkey_comparison_is_still_reached_after_the_guard():
    """The guard must sit before the comparison, not replace it."""
    body = _resync_source()
    assert "self.handle_deregistration(uid)" in body, (
        "the per-uid deregistration call is gone from resync_metagraph"
    )
    guard = body.index("bt.logging.debug(\"No axon changes!\")")
    call = body.index("self.handle_deregistration(uid)")
    assert guard < call, "the early return must precede the hotkey comparison it guards"


def test_a_changed_hotkey_alone_is_enough_to_proceed():
    """Exercise the real condition, imported from the module resync_metagraph calls.

    An earlier version of this test defined a local copy of the condition and asserted against that. It
    would have passed with the bug still in place, because a copy is only ever evidence about the copy.
    The condition now lives in a named function that resync_metagraph calls, so this reaches the shipped
    code: nothing changed, only a hotkey changed, only an axon changed. The middle one is the regression.
    """
    from taos.common.neurons.validator import resync_inputs_unchanged as should_skip

    axons = ["1.2.3.4:8091", "1.2.3.5:8091"]
    old_hk = ["hkA", "hkB"]

    assert should_skip(axons, axons, old_hk, ["hkA", "hkB"]) is True, (
        "a resync where nothing changed should still skip, or every resync does the full work"
    )
    assert should_skip(axons, axons, old_hk, ["hkA", "hkNEW"]) is False, (
        "a hotkey change with an unchanged axon list must NOT skip: that is the deregistration case"
    )
    assert should_skip(axons, ["1.2.3.4:8091", "9.9.9.9:8091"], old_hk, old_hk) is False, (
        "an axon change must not skip either"
    )
