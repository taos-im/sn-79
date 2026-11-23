# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The presence records (miner_presence, {uid: outcomes}) are read on the listener thread for the scoring
service's inputs while the main thread's metagraph resync pops a departed uid from them. A read that iterates
the live mapping fails with 'dictionary changed size during iteration' when the two coincide, and that state's
handling is abandoned with it. Both reads, the absent set and the presence shares, iterate a snapshot of the
mapping's items instead, so a uid leaving mid-read costs nothing: the snapshot still carries the departed uid's
outcomes, which the resync's reset drops from the next read on.
"""
from collections import deque

from taos.im.validator import debeta


class _LeavesAnotherUid:
    """Outcomes whose first read pops another uid from the shared mapping, as the resync does mid-read."""

    def __init__(self, presence, other, outcomes):
        self._presence, self._other, self._outcomes = presence, other, list(outcomes)

    def __iter__(self):
        self._presence.pop(self._other, None)
        return iter(self._outcomes)

    def __len__(self):
        return len(self._outcomes)


def _presence():
    p = {}
    p[1] = _LeavesAnotherUid(p, 2, [True, False, True])
    p[2] = deque([False, False, False], maxlen=3)
    p[3] = deque([True, True, True], maxlen=3)
    return p


def test_the_presence_shares_survive_a_uid_leaving_mid_read():
    p = _presence()
    shares = debeta.presence_shares(p, window=3)
    assert shares[1] == 2 / 3 and shares[3] == 1.0
    assert shares[2] == 0.0, "the snapshot still carries the departed uid"
    assert 2 not in p, "the resync's pop took effect on the live mapping"


def test_the_absent_set_survives_a_uid_leaving_mid_read():
    p = _presence()
    assert debeta.absent_uids(p, window=3) == {2}
    assert 2 not in p
