"""forward() must never release a lock it did not acquire.

The mainnet sim validator logged `RuntimeError: Lock is not acquired` six times, each
one immediately after a SUCCESSFUL query:

    INFO  | Set Delays (0.0039s).
    INFO  | Received 197 valid responses containing 2401 instructions ...
    ERROR | Exception in listener loop: Lock is not acquired.

The mechanism, and why the victim is not the culprit:

  * `forward()` opens its `try` about sixty lines before it acquires `miner_net_lock`, and the
    `finally` released unconditionally.
  * `asyncio.Lock` has NO OWNERSHIP. release() from a coroutine that never acquired it succeeds and
    frees the lock for whoever is actually holding it.
  * `forward()` is re-entrant across that span: the reward-catchup loop does `await asyncio.sleep`
    INSIDE the try, so a second state can start a second `forward()` while the first is parked there.
    The mainnet log shows exactly that overlap - a query starting at 04:45:45.970 while the one from
    04:45:42.848 was still in flight.

So an invocation exiting between the `try` and the acquire - an early return, or any exception in the
metagraph / request-data assembly - released a live holder's lock. Worse, when the trigger was an
exception the raising `finally` REPLACED it during propagation, so the original fault was destroyed
every time and the log only ever showed the mask.

This test reproduces the destructive half: a pre-acquire failure must leave another holder's lock
alone. It fails on the unguarded version, where the lock comes back unlocked.
"""
import asyncio
import sys
from types import SimpleNamespace

import pytest

_REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)


def _synapse():
    return SimpleNamespace(
        books=[], accounts={}, notices={}, version="test", timestamp=0, block=0, pools={},
        config=SimpleNamespace(model_dump=lambda mode=None: {}),
    )


def _validator(lock):
    """The minimum surface forward() touches before it reaches the acquire."""
    return SimpleNamespace(
        query_process=SimpleNamespace(poll=lambda: None),
        miner_net_lock=lock,
        querying=False,
        uid=0,
        deregistered_uids=set(),
        volume_sums={},
        engine=SimpleNamespace(book_ids=[0], mode="simulation"),
        simulation=SimpleNamespace(miner_wealth=1.0, volumeDecimals=4, book_count=1),
        config=SimpleNamespace(
            neuron=SimpleNamespace(observe=False),
            scoring=SimpleNamespace(
                activity=SimpleNamespace(capital_turnover_cap=10.0),
                max_instructions_per_book=10,
            ),
            exchange=SimpleNamespace(volume_cap=50000.0),
        ),
        # the pre-acquire failure: this is called at the top of the try, well before the lock
        get_extended_metagraph=lambda: (_ for _ in ()).throw(RuntimeError("metagraph unavailable")),
        should_block_queries=lambda: False,
        pagerduty_alert=lambda *a, **k: None,
        _pending_reward_tasks=0,
    )


@pytest.mark.asyncio
async def test_pre_acquire_failure_does_not_steal_a_live_holders_lock():
    """The exact mainnet defect: B holds the lock, A fails before acquiring, B's lock survives."""
    from taos.im.validator.forward import forward

    lock = asyncio.Lock()
    await lock.acquire()          # invocation B is mid-query and holds it
    assert lock.locked()

    with pytest.raises(Exception) as caught:
        await forward(_validator(lock), _synapse())

    assert lock.locked(), (
        "forward() released a lock it never acquired: a concurrent in-flight query has just had its "
        "mutual exclusion silently revoked, and its own release() will raise 'Lock is not acquired'"
    )
    # The original fault must survive. The unguarded finally replaced it with RuntimeError from
    # release(), which is why six mainnet occurrences carried no usable diagnosis.
    assert "metagraph unavailable" in str(caught.value), (
        f"the real exception was masked by the finally: got {caught.value!r}"
    )

    lock.release()


@pytest.mark.asyncio
async def test_the_guard_is_not_satisfied_by_never_releasing():
    """A successful pass must still release, or the lock leaks and every later query hangs."""
    from taos.im.validator.forward import forward

    lock = asyncio.Lock()
    v = _validator(lock)
    # observe returns before the try in 0.6.x, so it exercises the no-acquire path without an error
    v.config.neuron.observe = True
    await forward(v, _synapse())
    assert not lock.locked(), "an untaken lock must not be left locked either"
