# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The per-book debeta_alpha gauge reads the alphas of the scoring its boundary paid.

The gauge reads the validator's debeta_alphas_by_book, which compute_debeta_scores sets. A boundary whose scoring main
adopts from the scoring service runs no compute_debeta_scores in main, so the service returns its per-book alphas
beside its class summary and main takes them with the map it keeps for the report."""
import inspect

import pytest

from taos.im.validator import scoring_shadow


def test_the_scoring_service_returns_its_per_book_alphas():
    src = inspect.getsource(scoring_shadow.shadow_score)
    tail = src[src.rindex("return {"):]
    assert "'debeta_alphas_by_book': getattr(shadow, 'debeta_alphas_by_book', None) or {}" in tail, tail


def test_main_takes_the_alphas_with_the_map_it_keeps():
    validator = pytest.importorskip("taos.im.neurons.validator")
    src = inspect.getsource(validator)
    at = src.index("self._scored_debeta = {")
    block = src[at:src.index("bt.logging.info(", at)]
    assert "adopted.get('debeta_alphas_by_book')" in block, block
    assert "self.debeta_alphas_by_book = " in block, block
