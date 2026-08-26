"""A miner that explicitly asks for NO_STP must get NO_STP.

`validate_responses` contained:

    if stp_value == 'NO_STP' or stp_value == 0:
        instruction.stp = STP.CANCEL_OLDEST

which silently rewrote every explicit NO_STP into CANCEL_OLDEST on every market and limit order.
Two consequences, both measured on 2026-08-07:

* `s_stp` "STP=NONE: flag accepted end-to-end" failed, correctly: the flag was not accepted.
* Every marketable limit order got CO, so the sweep the engine creates at submission to fill it
  looked like the same agent as the resting order and STP cancelled the order the sweep existed to
  fill. Nothing filled, nothing rested, no notice, in BOTH directions -- and a miner could not opt
  out, because opting out was exactly what this line discarded.

It cannot have been guarding "the miner did not set a flag". `STP.NO_STP == 0` and the instruction
model already defaults `stp` to `CANCEL_OLDEST`, so an unset flag never reaches this code as 0. The
only value that does is one the miner chose. A sentinel written for "zero means unset" caught the
one case where zero means something.
"""

import warnings

warnings.filterwarnings("ignore")

from taos.im.protocol.exchange.instructions import (  # noqa: E402
    PlaceLimitOrderInstruction,
    PlaceMarketOrderInstruction,
)
from taos.im.protocol.models import STP  # noqa: E402


def test_no_stp_is_zero_and_the_model_default_is_cancel_oldest():
    """The two facts that make the override provably redundant, pinned so they cannot drift."""
    assert STP.NO_STP.value == 0
    unset = PlaceLimitOrderInstruction(
        agentId=1, bookId=5, direction=0, quantity=1.0, price=0.01,
        clientOrderId=None, delegate="",
    )
    assert unset.stp == STP.CANCEL_OLDEST, (
        "if the model default changes to NO_STP, an unset flag becomes indistinguishable from an "
        "explicit NONE and the override below would become defensible again"
    )


def test_validator_does_not_rewrite_an_explicit_no_stp():
    """THE REGRESSION. Removing the rewrite is the whole fix; assert it stays removed."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "taos" / "im" / "validator" / "query.py"
    # CODE ONLY. The fix leaves a comment quoting the removed line so the next reader knows why it
    # went; matching the raw file text would match that explanation and fail on the documentation.
    text = "\n".join(
        l for l in src.read_text().splitlines() if not l.lstrip().startswith("#")
    )
    assert "instruction.stp = STP.CANCEL_OLDEST" not in text, (
        "validate_responses is rewriting an explicit NO_STP to CANCEL_OLDEST again. A miner cannot "
        "then disable self-trade prevention, and marketable limit orders are cancelled by the sweep "
        "created to fill them"
    )


def test_an_explicit_no_stp_survives_construction_on_both_order_types():
    for cls, extra in (
        (PlaceLimitOrderInstruction, {"price": 0.01}),
        (PlaceMarketOrderInstruction, {}),
    ):
        i = cls(
            agentId=1, bookId=5, direction=0, quantity=1.0,
            clientOrderId=None, delegate="", stp=STP.NO_STP, **extra,
        )
        assert i.stp == STP.NO_STP, f"{cls.__name__} lost the explicit NO_STP"
        assert i.payload()["stp"] == STP.NO_STP, f"{cls.__name__} did not put NO_STP on the wire"
