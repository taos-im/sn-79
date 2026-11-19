# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The venue orders a round's instructions by their delays. Two rules on the validator: the measured policy,
under which a response's delay is its own process time, every instruction at that delay with the response's cancels
before its placements and no random spread (a response at or over the timeout still follows the curve); and the
queued order's delay, derived from the service's receipt stamp (receivedNs) against the previous round's open, 0
without a stamp or before the open, the engine's bound at most, stamped by the engines' shared drain and the stamp
taken off the instruction. Both modes share the drain; the policy is a dial, curve by default."""
import argparse
import random
from types import SimpleNamespace

from taos.im.config import add_im_validator_args
from taos.im.protocol import instructions as I
from taos.im.protocol.models import OrderDirection
from taos.im.protocol.response import FinanceAgentResponse
from taos.im.validator import delays as D
from taos.im.validator.engines import MarketEngine

TIMEOUT, MIN, MAX = 3.0, 10_000_000, 1_000_000_000
S = 1_000_000_000


def _validator(**dials):
    base = dict(min_delay=MIN, max_delay=MAX, min_instruction_delay=5_000_000, max_instruction_delay=25_000_000)
    base.update(dials)
    return SimpleNamespace(config=SimpleNamespace(neuron=SimpleNamespace(timeout=TIMEOUT, max_instruction_delay_s=600),
                                                  scoring=SimpleNamespace(**base)))


def _limit(uid, book, delay=0):
    return I.PlaceLimitOrderInstruction(agentId=uid, bookId=book, delay=delay, direction=OrderDirection.BUY, quantity=1.0, price=100.0,
                                        clientOrderId=None)


def _cancel(uid, book, delay=0):
    return I.CancelOrdersInstruction(agentId=uid, bookId=book, delay=delay, cancellations=[I.CancelOrderInstruction(orderId=1, volume=None)])


def test_the_measured_policy_is_the_process_time_within_the_timeout_and_the_curve_at_it():
    assert D.base_delay(0.25, TIMEOUT, MIN, MAX, "measured") == 250_000_000
    assert D.base_delay(0.0, TIMEOUT, MIN, MAX, "measured") == 0
    assert D.base_delay(TIMEOUT, TIMEOUT, MIN, MAX, "measured") == D.base_delay(TIMEOUT, TIMEOUT, MIN, MAX, "curve")
    assert "measured" in D.DELAY_POLICIES
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    assert "measured" in next(a for a in parser._actions if a.dest == "scoring.delay_policy").choices


def test_under_measured_every_instruction_waits_the_measured_time_cancels_first_and_nothing_is_drawn(monkeypatch):
    def no_draw(*a, **k):
        raise AssertionError("no random spread under the measured policy")
    monkeypatch.setattr(random, "randint", no_draw)
    resp = FinanceAgentResponse(agent_id=4, instructions=[_limit(4, 0), _limit(4, 0), _cancel(4, 1), _limit(4, 1), _cancel(4, 0)])
    out = D.set_delays(_validator(delay_policy="measured"), {4: SimpleNamespace(response=resp, dendrite=SimpleNamespace(process_time=0.4))})
    kinds = [i.type for i in out[0].instructions]
    assert kinds == ["CANCEL_ORDERS", "CANCEL_ORDERS", "PLACE_ORDER_LIMIT", "PLACE_ORDER_LIMIT", "PLACE_ORDER_LIMIT"]
    assert [i.bookId for i in out[0].instructions] == [1, 0, 0, 0, 1], "a stable sort: the miner's order within each kind"
    assert {i.delay for i in out[0].instructions} == {400_000_000}


def test_the_curve_policy_is_untouched_by_the_measured_one():
    random.seed(1)
    resp = FinanceAgentResponse(agent_id=4, instructions=[_limit(4, 0), _limit(4, 0)])
    out = D.set_delays(_validator(), {4: SimpleNamespace(response=resp, dendrite=SimpleNamespace(process_time=0.4))})
    a, b = (i.delay for i in out[0].instructions)
    assert b > a, "the second instruction on a book still draws its spread under the curve"


def test_a_queued_orders_delay_is_its_receipt_after_the_previous_open_bounded():
    assert D.queued_order_delay(None, 100, 600 * S) == 0
    assert D.queued_order_delay(100, None, 600 * S) == 0
    assert D.queued_order_delay(50, 100, 600 * S) == 0, "received before the open: at the open"
    assert D.queued_order_delay(100 + 3 * S, 100, 600 * S) == 3 * S
    assert D.queued_order_delay(100 + 900 * S, 100, 600 * S) == 600 * S, "the engine's bound"
    assert D.queued_order_delay("x", 100, 600 * S) == 0


def _engine(validator=None):
    """A stand-in with the shared drain and the attributes it reads; the engines are abstract."""
    e = SimpleNamespace(_external_buffer=[], validator=validator, _ROUND_OPENS_KEPT=MarketEngine._ROUND_OPENS_KEPT)
    e._drain_external_orders = lambda: MarketEngine._drain_external_orders(e)
    e._note_round_open = lambda open_ns: MarketEngine._note_round_open(e, open_ns)
    e._open_of_receipt = lambda received_ns: MarketEngine._open_of_receipt(e, received_ns)
    return e


def test_the_drain_stamps_each_queued_order_from_the_previous_open_and_strips_the_stamp(monkeypatch):
    clock = iter([1000 * S, 1010 * S])
    monkeypatch.setattr(D, "max_instruction_delay_ns", lambda config: 600 * S)
    import time as _t
    monkeypatch.setattr(_t, "time_ns", lambda: next(clock))
    e = _engine(validator=_validator())
    assert e._drain_external_orders() == [], "the first drain opens the first round"
    e._external_buffer = [{"agentId": 7, "type": "PLACE_ORDER_LIMIT", "payload": {"bookId": 0}, "receivedNs": 1004 * S},
                          {"agentId": 8, "type": "CANCEL_ORDERS", "payload": {"bookId": 0}},
                          {"agentId": 9, "type": "PLACE_ORDER_LIMIT", "payload": {"bookId": 0}, "receivedNs": 999 * S}]
    out = e._drain_external_orders()
    assert [o["delay"] for o in out] == [4 * S, 0, 0]
    assert all("receivedNs" not in o for o in out), "the stamp is not sent to the engine"
    assert e._external_buffer == []


def test_the_pollers_start_opens_the_first_round_so_the_first_drain_has_an_open(monkeypatch):
    import asyncio
    import time as _t
    clock = iter([2000 * S, 2003 * S])
    monkeypatch.setattr(_t, "time_ns", lambda: next(clock))
    monkeypatch.setattr(D, "max_instruction_delay_ns", lambda config: 600 * S)
    e = SimpleNamespace(_external_buffer=[], validator=_validator(), _external_poll_task=None)
    e._external_orders_loop = lambda: asyncio.sleep(0)
    e._ensure_external_poller = lambda: MarketEngine._ensure_external_poller(e)
    e._drain_external_orders = lambda: MarketEngine._drain_external_orders(e)
    e._note_round_open = lambda open_ns: MarketEngine._note_round_open(e, open_ns)
    e._open_of_receipt = lambda received_ns: MarketEngine._open_of_receipt(e, received_ns)
    e._ROUND_OPENS_KEPT = MarketEngine._ROUND_OPENS_KEPT

    async def run():
        e._ensure_external_poller()
        assert e._round_open_ns == 2000 * S
        e._external_buffer = [{"agentId": 7, "type": "PLACE_ORDER_LIMIT", "payload": {}, "receivedNs": 2002 * S}]
        out = e._drain_external_orders()
        assert out[0]["delay"] == 2 * S, "measured from the poller's start, not 0"
        e._external_poll_task.cancel()
    asyncio.run(run())


def test_an_exchange_validator_runs_the_measured_policy_by_default_and_a_simulation_the_curve():
    resp = FinanceAgentResponse(agent_id=4, instructions=[_limit(4, 0), _limit(4, 0)])
    v = _validator(delay_policy=None)
    v.engine = SimpleNamespace(mode="exchange")
    out = D.set_delays(v, {4: SimpleNamespace(response=resp, dendrite=SimpleNamespace(process_time=0.4))})
    assert {i.delay for i in out[0].instructions} == {400_000_000}
    resp = FinanceAgentResponse(agent_id=4, instructions=[_limit(4, 0), _limit(4, 0)])
    v = _validator(delay_policy=None)
    v.engine = SimpleNamespace(mode="simulation")
    out = D.set_delays(v, {4: SimpleNamespace(response=resp, dendrite=SimpleNamespace(process_time=0.4))})
    a, b = (i.delay for i in out[0].instructions)
    assert a != 400_000_000 and b > a, "the curve, with its spread"
    # the query process scores from the settings main hands it, the engine mode among them
    resp = FinanceAgentResponse(agent_id=4, instructions=[_limit(4, 0)])
    v = _validator(delay_policy=None, engine_mode="exchange")
    out = D.set_delays(v, {4: SimpleNamespace(response=resp, dendrite=SimpleNamespace(process_time=0.25))})
    assert out[0].instructions[0].delay == 250_000_000
    import inspect
    from taos.im.validator import forward
    assert "engine_mode=getattr(getattr(self, 'engine', None), 'mode', None)" in inspect.getsource(forward)


def test_the_drain_counts_what_it_handed_over_for_the_gauges(monkeypatch):
    import time as _t
    clock = iter([3000 * S, 3010 * S])
    monkeypatch.setattr(_t, "time_ns", lambda: next(clock))
    monkeypatch.setattr(D, "max_instruction_delay_ns", lambda config: 600 * S)
    e = _engine(validator=_validator())
    e._drain_external_orders()
    assert e.queued_order_counts == {"drained": 0, "unstamped": 0, "late": 0, "delay_max_ns": 0}
    e._external_buffer = [{"agentId": 7, "type": "PLACE_ORDER_LIMIT", "payload": {}, "receivedNs": 3007 * S},
                          {"agentId": 8, "type": "CANCEL_ORDERS", "payload": {}},
                          {"agentId": 9, "type": "PLACE_ORDER_LIMIT", "payload": {}, "receivedNs": 3002 * S}]
    e._drain_external_orders()
    assert e.queued_order_counts == {"drained": 3, "unstamped": 1, "late": 0, "delay_max_ns": 7 * S}


def test_a_receipt_the_poller_buffered_after_its_drain_keeps_its_phase_in_the_round_it_arrived_in(monkeypatch):
    import time as _t
    clock = iter([100 * S, 106 * S, 112 * S])
    monkeypatch.setattr(_t, "time_ns", lambda: next(clock))
    monkeypatch.setattr(D, "max_instruction_delay_ns", lambda config: 600 * S)
    e = _engine(validator=_validator())
    e._drain_external_orders()   # opens at 100 s
    e._drain_external_orders()   # opens at 106 s
    e._external_buffer = [{"agentId": 7, "type": "PLACE_ORDER_LIMIT", "payload": {}, "receivedNs": 105 * S + 500_000_000},
                          {"agentId": 8, "type": "PLACE_ORDER_LIMIT", "payload": {}, "receivedNs": 109 * S},
                          {"agentId": 9, "type": "PLACE_ORDER_LIMIT", "payload": {}, "receivedNs": 50 * S}]
    out = e._drain_external_orders()   # opens at 112 s
    assert out[0]["delay"] == 5 * S + 500_000_000, "received in the round that opened at 100 s: its phase there, not 0"
    assert out[1]["delay"] == 3 * S, "received in the round that opened at 106 s: its phase there"
    assert out[2]["delay"] == 0, "older than every kept open: at the open"
    assert e.queued_order_counts == {"drained": 3, "unstamped": 0, "late": 2, "delay_max_ns": 5 * S + 500_000_000}
    assert list(e._round_opens) == [100 * S, 106 * S, 112 * S]
