# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The report publishes the de-beta map its boundary's scoring paid.

On a boundary whose scoring main adopts from the scoring service, the service's map is kept for the report
(_scored_debeta). On a VERIFY boundary main computes the boundary itself and pays its own result, the service's
answer only checked against it, so the service's map is not kept there: the report reads main's own computation, as
on a cycle main scored in process."""
import inspect

import pytest


def test_the_service_map_is_kept_only_when_main_paid_the_services_result():
    validator = pytest.importorskip("taos.im.neurons.validator")
    src = inspect.getsource(validator)
    at = src.index("self._scored_debeta = {")
    guard = src[src.rindex("\n", 0, src.rindex("if ", 0, at)):at]
    assert "adopted is not None" in guard and "not _verify" in guard, guard
