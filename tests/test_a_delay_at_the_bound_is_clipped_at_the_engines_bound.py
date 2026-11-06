# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A miner's delay at the bound stays inside the engine's bound once the validator adds its own.

The query path refuses a requested delay above --neuron.max_instruction_delay_s (600 s by default) and the engine holds
the same bound (maxInstructionDelay, refused on arrival). set_delays adds the base and per-instruction delays and clips
the sum at the validator's own bound, not the protocol's 86,400 s, so an instruction asking 600 s reaches the engine at
600 s. Both paths do: main's (forward.py) and the query process's under neuron.orders_packed, whose settings
(delay_settings) carry the bound.
"""
import random
from types import SimpleNamespace

from taos.im.protocol import instructions as I
from taos.im.protocol.models import OrderDirection
from taos.im.protocol.response import FinanceAgentResponse
from taos.im.validator.delays import max_instruction_delay_ns, set_delays
from taos.im.validator.forward import delay_settings

SECOND = 1_000_000_000
SCORING = dict(min_delay=10_000_000, max_delay=1_000_000_000, min_instruction_delay=5_000_000,
               max_instruction_delay=25_000_000)


def _validator(bound_s=None):
    neuron = SimpleNamespace(timeout=3.0)
    if bound_s is not None:
        neuron.max_instruction_delay_s = bound_s
    return SimpleNamespace(config=SimpleNamespace(neuron=neuron, scoring=SimpleNamespace(**SCORING)),
                           simulation=SimpleNamespace(time_unit="ns"))


def _synapses(delay, books=(0, 0, 1)):
    def limit(book):
        return I.PlaceLimitOrderInstruction(agentId=3, bookId=book, direction=OrderDirection.BUY, quantity=1.0,
                                            price=100.0, clientOrderId=None, delay=delay)
    return {3: SimpleNamespace(response=FinanceAgentResponse(agent_id=3, instructions=[limit(b) for b in books]),
                               dendrite=SimpleNamespace(process_time=2.9))}


def _delays(out):
    return [i.delay for r in out for i in r.instructions]


def test_a_delay_at_the_bound_reaches_the_engine_at_the_bound():
    random.seed(5)
    out = _delays(set_delays(_validator(), _synapses(600 * SECOND)))
    assert out == [600 * SECOND] * 3, "600 s plus the validator's delay is clipped at the engine's 600 s"


def test_a_delay_under_the_bound_keeps_the_validators_whole_addition():
    random.seed(5)
    out = _delays(set_delays(_validator(), _synapses(10 * SECOND)))
    most = SCORING["max_delay"] + SCORING["max_instruction_delay"]  # the base delay's top and the per-instruction one's
    assert all(10 * SECOND < d <= 10 * SECOND + most for d in out), out


def test_the_configured_bound_is_the_clip_on_both_paths():
    v = _validator(bound_s=300.0)
    assert max_instruction_delay_ns(v.config) == 300 * SECOND
    random.seed(5)
    main = _delays(set_delays(v, _synapses(300 * SECOND)))
    settings = delay_settings(v)
    assert settings.config.neuron.max_instruction_delay_s == 300.0, "the query process is handed the bound"
    random.seed(5)
    packed = _delays(set_delays(settings, _synapses(300 * SECOND)))
    assert main == packed == [300 * SECOND] * 3


def test_an_unset_bound_is_the_engines_default():
    assert max_instruction_delay_ns(SimpleNamespace(neuron=SimpleNamespace())) == 600 * SECOND
    random.seed(5)
    out = _delays(set_delays(_validator(bound_s=0.0), _synapses(I.MAX_INSTRUCTION_DELAY)))
    assert out == [600 * SECOND] * 3, "a bound of 0 reads as the default, as the query path reads it"
