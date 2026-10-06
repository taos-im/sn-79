# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A NOTIFICATION THE MINER'S AXON CANNOT ROUTE IS NOT A NOTIFICATION.

bittensor's axon dispatches a request by the synapse's class NAME, not by its class hierarchy:
`Axon.attach` reads the forward function's parameter annotation, stores the handler under
`annotation.__name__`, and the incoming request's `name` field has to match that string exactly. A
subclass of an attached type does not match, and the axon answers UnknownSynapseError.

WHAT THIS COSTS WHEN IT IS WRONG. `taos.im.validator.forward.notify` dispatches
`FinanceEventNotification`. `BaseMinerNeuron` attaches `update` annotated with the base
`EventNotification`. Those two names differ, so from the first commit until 2 October 2026 every
notification the validator sent came back refused and `TaosAgent.process` was never once called on a
live miner -- the dispatch existed, was maintained, and delivered nothing. `forward` was specialised
to `MarketSimulationStateUpdate` for exactly this reason; `update` was left behind, and nothing
compared the two sides.

So the check is the comparison nobody was making: the class the validator sends and the class the
miner attaches must be the same class. It is a static read of both annotations rather than a live
axon, because the failure is static -- it is decided at attach time, identically on every miner.
"""

import inspect
import typing

import pytest

from taos.common.neurons.miner import BaseMinerNeuron
from taos.im.protocol import FinanceEventNotification


def _annotation_of(fn, module):
    """The resolved first-parameter annotation of `fn`, as Axon.attach reads it."""
    sig = inspect.signature(fn)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    assert params, f"{fn.__qualname__} takes no synapse parameter"
    hints = typing.get_type_hints(fn, vars(module))
    return hints[params[0].name]


@pytest.fixture(scope="module")
def miner_module():
    """The subnet miner module. It guards its imports behind a try, so import it the plain way."""
    import taos.im.neurons.miner as m

    return m


def test_the_subnet_miner_attaches_the_class_the_validator_actually_sends(miner_module):
    attached = _annotation_of(miner_module.Miner.update, miner_module)
    assert attached is FinanceEventNotification, (
        f"the miner attaches `update` under {attached.__name__!r}, and the validator sends "
        f"{FinanceEventNotification.__name__!r}. bittensor routes by name, so every notification "
        "will be refused with UnknownSynapseError and no agent's process() will run."
    )


def test_the_override_is_load_bearing_because_the_base_attaches_a_different_name(miner_module):
    """If the base ever matched, the override could go. It does not, so deleting it reopens the bug."""
    base = _annotation_of(BaseMinerNeuron.update, inspect.getmodule(BaseMinerNeuron))
    assert base is not FinanceEventNotification, (
        "the base class now attaches the same class the validator sends; if that is deliberate, "
        "fold the two together rather than leaving an override that no longer overrides anything"
    )
    assert miner_module.Miner.update is not BaseMinerNeuron.update, (
        "the subnet miner no longer overrides `update`, so its axon is back on the base annotation "
        f"({base.__name__}) while the validator sends {FinanceEventNotification.__name__}"
    )


@pytest.mark.parametrize("name", ["blacklist_update", "priority_update"])
def test_the_blacklist_and_priority_hooks_take_the_same_class(miner_module, name):
    """Axon.attach type-checks all three against each other and raises if they disagree."""
    got = _annotation_of(getattr(miner_module.Miner, name), miner_module)
    assert got is FinanceEventNotification, (
        f"{name} takes {got.__name__}, but `update` takes {FinanceEventNotification.__name__}; "
        "Axon.attach requires the forward, blacklist and priority functions to agree"
    )
