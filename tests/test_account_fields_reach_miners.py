"""Every field the Account model declares must survive to the miner.

FOUR SEPARATE DEFECTS OF THIS SHAPE IN ONE NIGHT (2026-08-07/08). Each was a hand-maintained field
list that drifted from its schema, and every one failed silently because the missing field had a
benign default:

  1. PlaceMarketOrderInstruction.payload() omitted leverage and settleFlag; msgpack ignores unknown
     keys and defaults absent ones, so the engine used its struct defaults.
  2. The exchange engine's account dicts (engines/exchange.py _normalize) had no 'v' key at all.
  3. MarketSimulationStateUpdate's from_json path constructed Account with 7 of 12 kwargs, so
     base_loan, quote_loan, base_collateral, quote_collateral and traded_volume all defaulted --
     a miner carrying a margin loan read zero for it.
  4. UnifiedAccount.traded_volume read self._raw.get('tv'); the key is 'v'. Simulation was unaffected
     because there _raw is an Account object and the sibling branch is correct, which is exactly why
     it survived: the mode that worked masked the mode that did not.

Defect 4 took five attempts to find, because the value was correct at every hop that COMPUTES or
CARRIES it -- volume_sums held 1383.308, the state dict held it, it survived compression and the
wire -- and wrong only at the accessor. None versus 0.0 was the tell throughout: None means never
carried or never exposed, 0.0 means computed as empty.

These tests are cheap and structural. They cannot catch a wrong VALUE, only a field that cannot
arrive at all, which is the failure mode that keeps recurring.
"""

import ast
import inspect
import types

import pytest

from taos.im.agents import UnifiedAccount
from taos.im.protocol.models import Account


def _model_aliases() -> set[str]:
    """The public (alias) names of every field Account declares."""
    out = set()
    for name, f in Account.model_fields.items():
        out.add(f.alias or name)
    return out


def test_unified_account_exposes_every_account_field():
    """The wrapper miners actually read must expose the whole model.

    UnifiedAccount has NO __getattr__, so a field without an explicit property is unreachable --
    silently, since attribute access raises rather than returning a default and callers routinely
    guard with getattr(..., None). traded_volume was missing entirely on the exchange path.
    """
    missing = sorted(a for a in _model_aliases() if not hasattr(UnifiedAccount, a))
    assert not missing, (
        f"UnifiedAccount exposes no accessor for {missing}. Miners on the exchange path cannot read "
        f"these at all. Add a property for each, handling both the dict and Account branches."
    )


def test_unified_account_reads_the_exchange_dict_keys():
    """Each property must read the key the producers actually write.

    This is the one that bit: the accessor looked for 'tv' while every producer writes 'v'. A wrong
    key is indistinguishable from missing data, because both yield the default.
    """
    raw = {
        "i": 171, "b": 5,
        "bb": {"f": 5.0}, "qb": {"f": 100.0},
        "bl": 1.0, "ql": 2.0, "bc": 3.0, "qc": 4.0,
        "o": [], "l": {}, "f": None, "v": 1383.308,
    }
    acct = UnifiedAccount(raw)
    assert acct.traded_volume == 1383.308, "traded_volume must read the 'v' key"
    assert acct.base_loan == 1.0 and acct.quote_loan == 2.0
    assert acct.base_collateral == 3.0 and acct.quote_collateral == 4.0
    assert acct.agent_id == 171 and acct.book_id == 5
    assert acct.base_balance.free == 5.0 and acct.quote_balance.free == 100.0


def test_unified_account_simulation_path_is_unchanged():
    """The wrapper is shared by both modes; fixing exchange must not disturb simulation.

    Simulation passes an Account-like object rather than a dict, and that branch was always correct.
    Asserted explicitly because the exchange fix edits the same properties.
    """
    class FakeAccount:
        agent_id, book_id = 171, 5
        base_balance = types.SimpleNamespace(free=5.0)
        quote_balance = types.SimpleNamespace(free=100.0)
        base_loan, quote_loan = 1.0, 2.0
        base_collateral, quote_collateral = 3.0, 4.0
        orders, loans, fees = [], {}, None
        traded_volume = 42.0

    acct = UnifiedAccount(FakeAccount())
    assert acct.traded_volume == 42.0
    assert acct.base_loan == 1.0 and acct.quote_collateral == 4.0
    assert acct.agent_id == 171 and acct.book_id == 5


def test_state_update_reconstruction_passes_every_field():
    """Any Account(...) built in the protocol module must pass the full field set.

    MarketSimulationStateUpdate rebuilt accounts with an explicit kwarg list and quietly dropped five
    fields. Parsed with ast rather than executed, so this holds for every construction site in the
    file regardless of which code path a given mode takes.
    """
    import taos.im.protocol as proto

    tree = ast.parse(inspect.getsource(proto))
    aliases = _model_aliases()
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Account":
            passed = {kw.arg for kw in node.keywords if kw.arg}
            if not passed:
                continue  # positional or **kwargs construction; not this test's business
            missing = sorted(aliases - passed)
            if missing:
                offenders.append((getattr(node, "lineno", "?"), missing))
    assert not offenders, (
        "Account(...) constructed without every declared field:\n"
        + "\n".join(f"  line {ln}: missing {m}" for ln, m in offenders)
        + "\nA field omitted here takes its model default and reaches the miner as that default, "
          "which is indistinguishable from real data."
    )


@pytest.mark.parametrize("absent,expected", [({}, None), ({"v": 0.0}, 0.0)])
def test_absent_is_distinguishable_from_zero(absent, expected):
    """None must not be coerced to 0.0 anywhere on this path.

    The distinction is the only reason defect 4 was findable: a working write of an empty volume
    sends 0.0, so None meant the value was never carried or never exposed. Collapsing them would
    have hidden it completely.
    """
    assert UnifiedAccount(absent).traded_volume == expected
