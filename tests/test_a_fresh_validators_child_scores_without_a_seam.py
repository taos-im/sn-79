# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A validator started with no saved state has no run change behind it, so its seam (debeta_seam_ts, the run
change on the new clock) is None until a run rolls. Four places have to agree on that: the validator starts with the
attribute None; the scoring service's INIT ships a scalar it does not find as None, not as an empty map; the seam
check treats anything that is not an integer timestamp as no seam; and a saved state that carries a non-integer
seam loads as None. Otherwise a fresh validator's service fails every de-beta cycle on the first comparison of a
timestamp with a map, and every trading score is zero for its whole first run.
"""
import inspect

from taos.im.validator import debeta, persistence, scoring_shadow


def test_a_seam_that_is_not_an_integer_is_no_seam():
    hist = {7: {5: 1.0, 9: 2.0}}
    assert debeta.seam_inside(hist, None) is False
    assert debeta.seam_inside(hist, {}) is False
    assert debeta.seam_inside(hist, "7") is False
    assert debeta.seam_inside(hist, True) is False
    assert debeta.seam_inside(hist, 7) is True, "an entry before the seam"
    assert debeta.seam_inside(hist, 5) is False, "none before it"


def test_the_init_ships_a_missing_scalar_as_none_and_a_missing_map_as_empty():
    assert scoring_shadow.init_default("debeta_seam_ts") is None
    assert scoring_shadow.init_default("debeta_capbuy_hist") == {}
    src = inspect.getsource(scoring_shadow)
    assert "getattr(validator, name, init_default(name))" in src, "send_init reads the default through it"


def test_a_saved_seam_loads_as_an_integer_or_none():
    assert persistence.seam_ts_or_none(5_000_000_000) == 5_000_000_000
    assert persistence.seam_ts_or_none(None) is None
    assert persistence.seam_ts_or_none({}) is None
    assert persistence.seam_ts_or_none("5") is None
    assert persistence.seam_ts_or_none(True) is None
    src = inspect.getsource(persistence)
    assert "self.debeta_seam_ts = seam_ts_or_none(validator_state.get('debeta_seam_ts'))" in src


def test_a_fresh_validator_starts_with_no_seam():
    from taos.im.neurons import validator
    src = inspect.getsource(validator.Validator.__init__)
    assert "self.debeta_seam_ts = None" in src
