# SPDX-License-Identifier: MIT
"""The skill bar is a share of the layout's books, not a count. Twenty of mainnet's 128 books is 16 per cent of
the field; the same twenty on a 28-book multi-asset layout is 71 per cent, and a four-book asset class could never
qualify on its own. From 0.6.3 the bar resolves from scoring.debeta.skill_min_books_share (0.15625, so mainnet
keeps 20) against the books the validator scores, on the mechanism's four-book floor; an explicit skill_min_books
still overrides it, and an unknown layout keeps the 0.6.2 launch value rather than dropping to the floor."""
import argparse
import inspect
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taos.im.config import add_im_validator_args  # noqa: E402
import taos.im.validator.reward as reward  # noqa: E402


def test_the_share_gives_the_launch_bar_on_mainnet_and_scales_with_the_layout():
    assert reward.resolve_skill_min_books(0, 0.15625, 128) == 20
    assert reward.resolve_skill_min_books(0, 0.15625, 28) == 5
    assert reward.resolve_skill_min_books(0, 0.15625, 16) == 4
    assert reward.resolve_skill_min_books(0, 0.15625, 4) == 4


def test_an_explicit_count_overrides_the_share_and_never_falls_below_the_floor():
    assert reward.resolve_skill_min_books(30, 0.15625, 128) == 30
    assert reward.resolve_skill_min_books(2, 0.15625, 128) == 4


def test_an_unknown_layout_or_an_unset_share_keeps_the_launch_value():
    assert reward.resolve_skill_min_books(0, 0.15625, 0) == 20
    assert reward.resolve_skill_min_books(0, 0.0, 128) == 20
    assert reward.resolve_skill_min_books(None, None, None) == 20


def _duck(share=0.15625, explicit=0, book_ids=None, book_count=None, sim_config=None):
    v = types.SimpleNamespace()
    v.config = types.SimpleNamespace(
        scoring=types.SimpleNamespace(
            debeta=types.SimpleNamespace(skill_min_books=explicit, skill_min_books_share=share)))
    if book_ids is not None:
        v.engine = types.SimpleNamespace(book_ids=book_ids)
    if book_count is not None:
        v.simulation = types.SimpleNamespace(book_count=book_count)
    if sim_config is not None:
        v.simulation_config = sim_config
    return v


def test_the_validator_reads_the_layout_from_its_engine_first_then_the_simulation():
    sparse = list(range(16)) + list(range(16, 20)) + list(range(32, 36)) + list(range(48, 52))
    assert reward.skill_bar(_duck(book_ids=sparse, book_count=128)) == 5
    assert reward.skill_bar(_duck(book_count=128)) == 20
    assert reward.skill_bar(_duck(sim_config={"book_count": 28})) == 5
    assert reward.skill_bar(_duck(sim_config={"book_count": 64, "book_ids": list(range(28))})) == 5
    assert reward.skill_bar(_duck()) == 20
    assert reward.skill_bar(_duck(explicit=12, book_count=128)) == 12


def test_every_consumer_resolves_through_the_bar():
    for fn in (reward.compute_debeta_scores, reward.build_scoring_config):
        assert "skill_bar(self" in inspect.getsource(fn)
    assert "getattr(dcfg, 'skill_min_books', 4)" not in inspect.getsource(reward.compute_debeta_scores)


def test_the_defaults_derive_the_bar_from_the_share():
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    known, _ = parser.parse_known_args([])
    assert getattr(known, "scoring.debeta.skill_min_books") == 0
    assert getattr(known, "scoring.debeta.skill_min_books_share") == 0.15625
