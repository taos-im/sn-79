# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Miner response delays: when each miner's instructions act in the simulation. Apart from reward.py so the query process
can apply them (neuron.orders_packed) without importing the scoring stack.
"""
import json
import random
import time
from typing import TYPE_CHECKING

import bittensor as bt
import numpy as np

from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.protocol.instructions import MAX_INSTRUCTION_DELAY

if TYPE_CHECKING:
    from taos.im.neurons.validator import Validator


DELAY_POLICIES = ("curve", "flat", "jitter")


def _curve_delay(p_time: float, timeout: float, min_delay: int, max_delay: int) -> int:
    """Exponential scaling of process time into delay."""
    t = p_time / timeout
    exp_scale = 5
    delay_frac = (np.exp(exp_scale * t) - 1) / (np.exp(exp_scale) - 1)
    delay = min_delay + delay_frac * (max_delay - min_delay)
    return int(delay)


def base_delay(p_time: float, timeout: float, min_delay: int, max_delay: int, policy: str = "curve",
               jitter: int = 0, prompt_cutoff: float = 0.0, rng=random) -> int:
    """The base delay a response's process time earns under a delay policy.

    curve: the exponential mapping from min_delay to max_delay over the timeout. flat: min_delay. jitter: min_delay
    plus a uniform draw in [0, jitter] at or under prompt_cutoff seconds; above it the curve, but never under the
    draw's top, so a slower response never waits less than a prompt one. A process time at or over the timeout follows
    the curve under every policy.
    """
    if policy == "curve" or p_time >= timeout:
        return _curve_delay(p_time, timeout, min_delay, max_delay)
    if policy == "flat":
        return int(min_delay)
    if policy == "jitter":
        if p_time <= prompt_cutoff:
            return int(min_delay) + rng.randint(0, int(jitter))
        return max(_curve_delay(p_time, timeout, min_delay, max_delay), int(min_delay) + int(jitter))
    raise ValueError(f"unknown delay policy {policy!r}")


def latency_profile(self: 'Validator') -> dict[int, float]:
    """The per-uid process times the latency_profile dial names, read once and kept on the validator; empty when the
    dial is unset or the file cannot be read, which warns and never stops a round."""
    cached = getattr(self, '_latency_profile', None)
    if cached is not None:
        return cached
    path = getattr(self.config.scoring, 'latency_profile', None) or ''
    profile: dict[int, float] = {}
    if path:
        try:
            with open(path) as fh:
                raw = json.load(fh)
            profile = {int(uid): float(seconds) for uid, seconds in raw.items()}
            bt.logging.info(f"Latency profile loaded from {path}: {len(profile)} uids")
        except (OSError, ValueError, TypeError, AttributeError) as e:
            bt.logging.warning(f"Latency profile at {path} could not be read ({e}); measured response times are used")
            profile = {}
    self._latency_profile = profile
    return profile


DEFAULT_MAX_INSTRUCTION_DELAY_S = 600.0


def max_instruction_delay_ns(config) -> int:
    """The longest delay a miner's instruction may ask for, --neuron.max_instruction_delay_s in nanoseconds (600 s, the
    engine's maxInstructionDelay default, when it is unset or 0); the query service refuses a longer one at acceptance
    and the engine holds the same bound."""
    seconds = float(getattr(getattr(config, 'neuron', None), 'max_instruction_delay_s', None) or 0.0)
    return int((seconds or DEFAULT_MAX_INSTRUCTION_DELAY_S) * 1_000_000_000)


def set_delays(self: 'Validator', synapse_responses: dict[int, MarketSimulationStateUpdate]) -> list[FinanceAgentResponse]:
    """
    Applies base delay based on process time under the configured delay policy (base_delay),
    and adds a per-book random latency instruction_delay to instructions,
    with zero instruction_delay applied to the first instruction per book.

    Args:
        self (taos.im.neurons.validator.Validator): Validator instance.
        synapse_responses (dict[int, MarketSimulationStateUpdate]): Latest state updates.

    Returns:
        list[FinanceAgentResponse]: Delayed finance responses.
    """
    start_time = time.time()
    responses = []
    timeout = self.config.neuron.timeout
    scoring = self.config.scoring
    min_delay = scoring.min_delay
    max_delay = scoring.max_delay
    min_instruction_delay = scoring.min_instruction_delay
    max_instruction_delay = scoring.max_instruction_delay
    policy = getattr(scoring, 'delay_policy', None) or "curve"
    if policy not in DELAY_POLICIES:
        bt.logging.warning(f"Unknown delay policy {policy!r}; the curve applies")
        policy = "curve"
    jitter = int(getattr(scoring, 'delay_jitter', None) or 0)
    prompt_cutoff = float(getattr(scoring, 'delay_prompt_cutoff', None) or 0.0)
    profile = latency_profile(self)
    # the engine refuses an instruction delayed past its bound, so the validator's addition stops there
    bound = min(MAX_INSTRUCTION_DELAY, max_instruction_delay_ns(self.config))

    log_messages = []
    for _uid, synapse_response in synapse_responses.items():
        response = synapse_response.response
        if response:
            measured = synapse_response.dendrite.process_time
            p_time = measured
            if profile:
                try:
                    p_time = profile.get(int(_uid), measured)
                except (TypeError, ValueError):
                    p_time = measured
            delay = base_delay(p_time, timeout, min_delay, max_delay, policy, jitter, prompt_cutoff)
            seen_books = set()
            for instruction in response.instructions:
                book_id = instruction.bookId

                if book_id not in seen_books:
                    instruction_delay = 0
                    seen_books.add(book_id)
                else:
                    instruction_delay = random.randint(min_instruction_delay, max_instruction_delay)
                instruction.delay = min(instruction.delay + delay + instruction_delay, bound)
            responses.append(response)
            log_messages.append(
                f"UID {response.agent_id} responded with {len(response.instructions)} instructions "
                f"after {measured:.4f}s – base delay {delay}{self.simulation.time_unit if hasattr(self, 'simulation') else 'ns'}"
                + (f" (profile {p_time:.4f}s)" if p_time is not measured else "")
            )
    if policy == "flat":
        random.shuffle(responses)
    if log_messages:
        bt.logging.info("\n".join(log_messages))    
    elapsed = time.time() - start_time
    if elapsed > 0.1:
        bt.logging.warning(f"set_delays took {elapsed:.4f}s for {len(synapse_responses)} responses")
    return responses
