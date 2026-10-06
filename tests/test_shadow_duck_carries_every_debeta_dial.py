"""Every scoring.debeta dial the scorer reads must reach the scoring child's config duck.

The child re-scores from a hand-built namespace rather than the real config object. A dial added to
`compute_debeta_scores` but not to `_config_duck` therefore takes its `getattr` default inside the
child while main uses the operator's value, and the two sides score differently with nothing failing
loudly: it surfaces later as a parity mismatch, which reads as a shadow fault rather than a missing
field. Two dials were added on 22 September and both had to be wired by hand, which is what this
guards.

Scope is deliberately what the SCORER reads off `config.scoring.debeta`, not every declared
argument: `publish_book_gauges` is read only by the reporter. `weight` is included: the fallback that
carries the previous de-beta map through a warming or failed cycle reads it off the de-beta namespace,
so a duck without it made that fallback a no-op in the scoring child. If this test starts failing on a name
that is genuinely carried elsewhere, add it to the exclusion with the reason, do not widen the
regex.
"""
import re

from taos.im.validator.scoring_shadow import _config_duck


def _dials_the_scorer_reads():
    """Names read off the de-beta config object in reward.py, taken from the source so the test
    tracks the code rather than a list someone remembers to update."""
    src = open("taos/im/validator/reward.py", encoding="utf-8").read()
    names = set(re.findall(r"getattr\(dcfg,\s*'([a-z0-9_]+)'", src))
    # `dcfg` names two objects in that file; 'debeta' is read off the scoring config to reach the de-beta
    # namespace itself, so it is not a duck field.
    return names - {"debeta"}


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_every_dial_the_scorer_reads_reaches_the_child():
    needed = _dials_the_scorer_reads()
    assert needed, "no dcfg reads found in reward.py; the scorer moved"
    duck = _config_duck(_NS(scoring=_NS(
        debeta=_NS(**{n: 0.5 for n in needed}),
        kappa=_NS(lookback=10800_000_000_000),
        activity=_NS(trade_volume_assessment_period=1, trade_volume_sampling_interval=1),
    )))
    missing = sorted(needed - set(vars(duck.scoring.debeta)))
    assert not missing, (
        "these scoring.debeta dials never reach the scoring child, so it will silently use its own "
        f"defaults while main uses the operator's: {missing}"
    )
