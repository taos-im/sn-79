# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The scoring process's INIT carries main's kept accounts as one msgpack frame, not through the per-structure
plainify and pickle the history structures take under the reward lock, and no checkpoint carries them.

On a 256-book state the kept copy is 67,584 entries; plainifying and pickling it took 0.95 to 1.87 s on the test
machine against 0.08 to 0.09 s for one msgpack pack, and the lock is held throughout. A checkpoint needs no copy:
after a restore the first state cannot be completed, so the engine is asked for every account in full."""
import test_scoring_shadow as tss
from test_account_entries_left_out_are_filled_in import full_accounts

from taos.im.validator.scoring_shadow import _STRUCT_NAMES, kept_frame, kept_from_frame


def test_the_kept_accounts_are_not_a_pickled_or_checkpointed_structure():
    assert "kept_accounts" not in _STRUCT_NAMES


def test_the_frame_carries_the_kept_accounts_exactly():
    main = tss._fresh_main()
    main.kept_accounts = full_accounts(5.0)
    main.kept_accounts[1][1]["o"] = [{"i": 7, "p": 301.25, "q": 0.25, "s": 1}]
    blob = kept_frame(main)
    assert isinstance(blob, bytes)
    assert kept_from_frame(blob) == main.kept_accounts, "int keys, order lists and floats come back as kept"


def test_no_kept_accounts_ship_no_frame():
    main = tss._fresh_main()
    assert kept_frame(main) is None
    main.kept_accounts = {}
    assert kept_frame(main) is None
