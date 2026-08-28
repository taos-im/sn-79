# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Pools must carry whether the subnet can settle a trade at all.

A subnet with SubtokenEnabled=False rejects every stake extrinsic, so a fill there can never
settle however healthy its pool looks: netuid 90 has 81k alpha in its pool and staking switched
off. Without the flag on the pool, the UI cannot mark the subnet or stop a miner trading it, and
the only feedback is a failed settlement after the fact.

The merge must never DROP a pool, and must never claim a subnet is tradable when the chain has
not been asked -- absent is left absent so a stale or failed lookup cannot read as "enabled".
"""
from taos.im.validator.subtoken import merge_subtoken_flags


def test_flag_is_attached_per_pool():
    pools = {"5": {"tao_in": 1.0}, "90": {"tao_in": 2.0}}
    out = merge_subtoken_flags(pools, {5: True, 90: False})
    assert out["5"]["subtoken_enabled"] is True
    assert out["90"]["subtoken_enabled"] is False


def test_integer_and_string_keys_both_work():
    out = merge_subtoken_flags({5: {"tao_in": 1.0}}, {5: False})
    assert out[5]["subtoken_enabled"] is False


def test_unknown_subnet_is_left_absent_not_assumed_enabled():
    """A subnet the chain was not asked about must not be presented as tradable."""
    out = merge_subtoken_flags({"7": {"tao_in": 1.0}}, {5: True})
    assert "subtoken_enabled" not in out["7"]


def test_no_pool_is_ever_dropped():
    pools = {str(n): {"tao_in": float(n)} for n in range(20)}
    out = merge_subtoken_flags(pools, {3: False})
    assert len(out) == len(pools)
    assert out["3"]["subtoken_enabled"] is False


def test_empty_inputs_are_safe():
    assert merge_subtoken_flags({}, {}) == {}
    assert merge_subtoken_flags(None, {5: True}) == {}
    out = merge_subtoken_flags({"5": {"tao_in": 1.0}}, None)
    assert "subtoken_enabled" not in out["5"]


def test_non_dict_pool_entry_is_left_alone():
    """Malformed input must not raise inside the payload builder."""
    out = merge_subtoken_flags({"5": "not-a-dict"}, {5: False})
    assert out["5"] == "not-a-dict"


def test_original_pool_dict_is_not_mutated():
    pools = {"5": {"tao_in": 1.0}}
    merge_subtoken_flags(pools, {5: False})
    assert "subtoken_enabled" not in pools["5"]


def test_failed_query_omits_the_subnet_rather_than_guessing():
    """A subnet whose read failed must not be presented as tradable."""
    from taos.im.validator.subtoken import reset_cache, subtoken_flags

    class Flaky:
        def query(self, _m, _s, args):
            if args[0] == 90:
                raise RuntimeError("rpc blew up")

            class V:
                value = True
            return V()

    reset_cache()
    flags = subtoken_flags(Flaky(), [5, 90])
    assert flags.get(5) is True
    assert 90 not in flags, "a failed read must not become an implicit True"
    merged = merge_subtoken_flags({"5": {}, "90": {}}, flags)
    assert merged["5"]["subtoken_enabled"] is True
    assert "subtoken_enabled" not in merged["90"]
    reset_cache()


def test_no_substrate_returns_cache_without_raising():
    from taos.im.validator.subtoken import reset_cache, subtoken_flags
    reset_cache()
    assert subtoken_flags(None, [5]) == {}
