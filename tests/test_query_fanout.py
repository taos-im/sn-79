# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Orchestration tests for QueryService._query_fanout.

_query_fanout delegates the wire protocol (signing, process_server_response) to
bittensor's own Dendrite.call — unchanged — so these tests cover only the NEW
logic: concurrent fan-out, per-task timeout -> 408, global-timeout cancel ->
dropped (query_miners then 408-stubs it), and the collected count. A fake
dendrite stands in for Dendrite.call so the test is deterministic and offline.
"""
import asyncio
from types import SimpleNamespace


from taos.im.validator.query import QueryService


def test_fanout_is_default_with_env_optout_and_sentinel_killswitch(monkeypatch):
    import taos.im.validator.query as q

    # No env, no sentinel -> fan-out is the default.
    monkeypatch.setattr(q.os.path, "exists", lambda p: False)
    monkeypatch.delenv("QUERY_FANOUT", raising=False)
    assert q._query_fanout_enabled() is True

    # QUERY_FANOUT=0 -> legacy threaded path.
    monkeypatch.setenv("QUERY_FANOUT", "0")
    assert q._query_fanout_enabled() is False

    # QUERY_FANOUT=1 -> fan-out (explicit).
    monkeypatch.setenv("QUERY_FANOUT", "1")
    assert q._query_fanout_enabled() is True

    # .query_threaded sentinel wins over env=1 (emergency kill-switch).
    monkeypatch.setattr(q.os.path, "exists", lambda p: str(p).endswith(".query_threaded"))
    assert q._query_fanout_enabled() is False


def _synapse(uid):
    # Minimal stand-in: only .dendrite.status_code is touched by the fan-out.
    return SimpleNamespace(uid=uid, dendrite=SimpleNamespace(status_code=None))


def _fake_self(axons, neuron_timeout=0.15, global_timeout=0.30):
    async def _call(axon, synapse, timeout, deserialize):
        await asyncio.sleep(axon["delay"])
        synapse.dendrite.status_code = 200
        return synapse

    return SimpleNamespace(
        dendrite=SimpleNamespace(call=_call),
        metagraph=SimpleNamespace(axons=axons),
        config=SimpleNamespace(
            neuron=SimpleNamespace(timeout=neuron_timeout, global_query_timeout=global_timeout)
        ),
    )


async def _run(axons, uids, deregistered, per_task):
    self_ = _fake_self(axons)
    return await QueryService._query_fanout(
        self_,
        axon_synapses={u: _synapse(u) for u in uids},
        uid_list=uids,
        deregistered_uids=deregistered,
        query_start=asyncio.get_running_loop().time(),
        per_task_timeout=per_task,
    )


def test_healthy_miners_collected_with_200():
    axons = [{"delay": 0.02}, {"delay": 0.03}, {"delay": 0.02}]
    responses, count = asyncio.run(_run(axons, [0, 1, 2], set(), per_task=0.25))
    assert count == 3
    assert set(responses.keys()) == {0, 1, 2}
    assert all(r.dendrite.status_code == 200 for r in responses.values())


def test_deregistered_uid_is_not_queried():
    axons = [{"delay": 0.02}, {"delay": 0.02}, {"delay": 0.02}]
    responses, count = asyncio.run(_run(axons, [0, 1, 2], {1}, per_task=0.25))
    assert set(responses.keys()) == {0, 2}
    assert 1 not in responses


def test_slow_miner_past_per_task_becomes_408():
    # per_task (0.10) < global (0.30): the slow uid resolves via wait_for -> 408.
    axons = [{"delay": 0.02}, {"delay": 0.50}]
    responses, count = asyncio.run(_run(axons, [0, 1], set(), per_task=0.10))
    assert responses[0].dendrite.status_code == 200
    assert responses[1].dendrite.status_code == 408
    assert count == 2


def test_slow_miner_past_global_is_dropped_for_stubbing():
    # per_task == global (0.20): the slow uid is still pending at the global
    # ceiling, gets cancelled, and is NOT collected -> query_miners 408-stubs it.
    axons = [{"delay": 0.02}, {"delay": 1.0}]
    self_ = _fake_self(axons, neuron_timeout=0.15, global_timeout=0.20)

    async def go():
        return await QueryService._query_fanout(
            self_,
            axon_synapses={0: _synapse(0), 1: _synapse(1)},
            uid_list=[0, 1],
            deregistered_uids=set(),
            query_start=asyncio.get_running_loop().time(),
            per_task_timeout=0.20,
        )

    responses, count = asyncio.run(go())
    assert responses[0].dendrite.status_code == 200
    assert 1 not in responses  # dropped -> missing-UID 408 stub happens in query_miners
    assert count == 1
