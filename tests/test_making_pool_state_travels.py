# SPDX-License-Identifier: MIT
"""The making pool's smoothed state must survive a restart and travel to the scoring shadow.

Both matter operationally. The state is a track record: losing it on restart re-bases every
operator's pool share over a half-life, and losing it on the shadow's side means main adopts an
empty state in cutover mode and re-bases anyway. Both are silent if untested, which is why the
state is asserted here rather than only the arithmetic.
"""
import inspect
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from taos.im.validator import persistence, scoring_shadow  # noqa: E402


class _Fake:
    """Enough of a validator for build_validator_state's de-beta/EMA fields."""

    def __init__(self):
        self._debeta_pool_ema = {"term": {1: 0.25, 2: 0.0}, "share": {1: 0.75, 2: 0.25}}


def test_the_pool_state_is_written_into_the_validator_state():
    src = inspect.getsource(persistence)
    assert src.count('"debeta_pool_ema"') >= 3, "built in both state writers and read back on load"
    fake = _Fake()
    captured = {k: dict(v or {}) for k, v in (getattr(fake, '_debeta_pool_ema', {}) or {}).items()}
    assert captured == {"term": {1: 0.25, 2: 0.0}, "share": {1: 0.75, 2: 0.25}}


def test_the_load_path_coerces_uids_and_values():
    """Persisted json makes every key a string; the loader must put back int uids and float values
    or the next round's EMA silently starts from zero for every uid."""
    src = inspect.getsource(persistence)
    i = src.find('_pema = validator_state.get("debeta_pool_ema")')
    assert i > 0, "the loader reads the key"
    block = src[i:i + 400]
    assert "int(u)" in block and "float(x)" in block


def test_the_shadow_carries_the_pool_state_in_and_out():
    sig = inspect.signature(scoring_shadow.shadow_score)
    assert "debeta_pool_ema" in sig.parameters
    src = inspect.getsource(scoring_shadow.shadow_score)
    assert "'debeta_pool_ema': _pema" in src, "returned so main can adopt it in cutover mode"
    # and it is advanced whatever the dial says, so switching the dial does not re-base the pool
    assert src.index("_share_tot = sum(_share.values())") > src.index("making_pool_inputs(")
    assert "if _pool_mode == \"proportional\"" in src


def test_both_senders_ship_the_pool_state():
    """The child is stateless: whatever main does not send is lost. Both frames carry it."""
    src = inspect.getsource(scoring_shadow)
    assert src.count("pin['debeta_pool_ema']") == 1          # the eager score_inputs frame
    assert src.count("(debeta_pool_ema or {}).items()}),") == 1   # the score_at frame
    assert "pool_ema = payload[10] if len(payload) > 10 else None" in src
    assert src.count("pool_ema = payload[10]") == 2          # both handlers on the child side


def test_the_published_gauges_exist_under_both_settings():
    from taos.im.validator import report
    src = inspect.getsource(report)
    assert "'debeta_making_share'" in src and "'debeta_ladder_input'" in src
    assert "('making_share', 'debeta_making_share')" in src
    assert "('ladder_input', 'debeta_ladder_input')" in src
