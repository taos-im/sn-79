# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""0.6.1 testnet: the scoring child (the authoritative scorer in CUTOVER mode) failed parity
after every INIT and re-initialised every two minutes, and each INIT restarted its de-beta making and
skill windows from zero, because the INIT snapshot (_STRUCT_NAMES) carried the kappa structures only.
The child-side save offload had the same gap: main's persistence writes the de-beta histories, the
child's file did not, so a restart with the shadow on zeroed the 24-hour making window as well.

Three properties pin the fix:
- every validator attribute trade.update_trade_volumes touches is either snapshotted or a knob, derived
  from the source so the list cannot drift again when an accumulator is added;
- the de-beta accumulators round-trip through the INIT path with their shells intact, and a shadow built
  from the snapshot keeps accumulating identically to main from the next round on;
- the child-side save carries the same de-beta block as main's, produced by one shared function.
"""
import ast
import pickle
from collections import defaultdict, deque
from pathlib import Path

import msgpack

import test_scoring_shadow as tss
from taos.im.validator.scoring_shadow import (
    _STRUCT_NAMES,
    child_save_validator_state,
    compute_parity_components,
)
from taos.im.validator.trade import update_trade_volumes

DEV = Path(__file__).resolve().parents[1]
TRADE = DEV / "taos/im/validator/trade.py"
PERSISTENCE = DEV / "taos/im/validator/persistence.py"
_S = tss._S

# Not history state: configuration, clocks and services the ShadowState provides as knobs.
KNOBS = {
    "config", "simulation", "step", "effective_max_uids", "_last_prune_timestamp", "pagerduty_alert",
    "engine", "kappa_cache", "deregistered_uids", "scoring_config", "simulation_config",
    "scoring_interval", "validator_uid",
}

DEBETA_ATTRS = [
    "capture_buy_sums", "capture_sell_sums", "debeta_mtm", "debeta_invsum", "debeta_inv",
    "debeta_invn", "debeta_pfirst", "debeta_plast", "debeta_drift", "debeta_mark_state",
    "debeta_capture_mid", "debeta_cp",
    "debeta_capbuy_hist", "debeta_capsell_hist", "debeta_mtm_hist", "debeta_invsum_hist",
    "debeta_invn_hist", "debeta_drift_hist", "debeta_cp_hist",
]


def _self_attributes_reachable_from(func_name: str) -> set:
    """Every `self.<attr>`, `hasattr/getattr/setattr(self, '<attr>')` and `for _n in ('a', 'b'): setattr(self,
    _n, ...)` name in trade.py reachable from `func_name` through calls to trade.py's own functions."""
    tree = ast.parse(TRADE.read_text())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    seen, todo, attrs = set(), [func_name], set()
    while todo:
        name = todo.pop()
        if name in seen or name not in funcs:
            continue
        seen.add(name)
        fn = funcs[name]
        loop_names = defaultdict(set)
        for node in ast.walk(fn):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and isinstance(node.iter, (ast.Tuple, ast.List)):
                for elt in node.iter.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        loop_names[node.target.id].add(elt.value)
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
                attrs.add(node.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in funcs:
                    todo.append(node.func.id)
                if node.func.id in ("hasattr", "getattr", "setattr") and len(node.args) >= 2 \
                        and isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
                    target = node.args[1]
                    if isinstance(target, ast.Constant) and isinstance(target.value, str):
                        attrs.add(target.value)
                    elif isinstance(target, ast.Name):
                        attrs |= loop_names.get(target.id, set())
    return attrs


def _plain(obj):
    """Structural view: defaultdicts and dicts to dict, deques to list, so main and shadow compare by content."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (deque, list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _fill_state(ts, fills):
    """A round whose books carry the 't' events the de-beta accumulators read (Ma/Ta/p/q/s) and whose
    notices carry the same fills for the kappa accounting. fills: (taker, maker, book, qty, price, side)."""
    st = tss._state(ts, fills)
    books = defaultdict(lambda: {"e": []})
    for n, (taker, maker, book, qty, price, side) in enumerate(fills):
        books[book]["e"].append({
            "y": "t", "i": ts // _S * 1000 + n, "s": side, "t": ts, "q": qty, "p": price,
            "Ti": 10 * n, "Ta": taker, "Tf": qty * price * 0.002,
            "Mi": 10 * n + 1, "Ma": maker, "Mf": qty * price * 0.001,
        })
    st.books = dict(books)
    return st


def _debeta_rounds(start=1, count=6):
    """Enough two-sided prints on two books for a non-trivial capture mid and MTM path."""
    rounds = []
    for k in range(start, start + count):
        fills = []
        for j in range(8):
            side = j % 2
            taker, maker = (1 + j % 3, 2 + (j + 1) % 3)
            price = 100.0 + ((k * 7 + j * 3) % 11) * 0.25
            fills.append((taker, maker, j % 2, 1.0 + j * 0.5, price, side))
            fills.append((maker, taker, 1 - j % 2, 0.5 + j * 0.25, price * 0.55, 1 - side))
        rounds.append(_fill_state(k * _S, fills))
    return rounds


def _debeta_main(rounds=None):
    main = tss._fresh_main()
    for st in rounds or _debeta_rounds():
        update_trade_volumes(main, st)
    assert main.capture_buy_sums and main.debeta_capbuy_hist and main.debeta_mtm_hist, (
        "the fixture must exercise the de-beta accumulators or the tests below prove nothing"
    )
    return main


def test_every_state_attribute_trade_py_touches_is_snapshotted():
    touched = _self_attributes_reachable_from("update_trade_volumes")
    assert "capture_buy_sums" in touched and "_volume_seen_tids" in touched, "the scan must see the accumulators"
    missing = sorted((touched - KNOBS) - set(_STRUCT_NAMES))
    assert not missing, (
        f"update_trade_volumes maintains {missing} on the validator but the INIT snapshot does not carry "
        f"them: every INIT (and every parity re-INIT) restarts these from zero in the child while main's "
        f"keep going. Add them to _STRUCT_NAMES with a matching shell in _rebuild_structs."
    )


def test_init_snapshot_carries_the_debeta_accumulators_with_their_shells():
    main = _debeta_main()
    shadow = tss._snapshot_to_shadow(main)
    for name in DEBETA_ATTRS:
        assert hasattr(shadow, name), f"{name} not in the snapshot"
        assert _plain(getattr(shadow, name)) == _plain(getattr(main, name)), f"{name} differs after INIT"
    # Load-bearing shells: the running sums auto-create like main's defaultdicts.
    shadow.capture_buy_sums[99][7] += 1.0
    assert shadow.capture_buy_sums[99][7] == 1.0
    shadow.debeta_inv[98][3] += 2.0
    assert shadow.debeta_inv[98][3] == 2.0
    assert isinstance(shadow.debeta_mtm[97], defaultdict)


def test_shadow_keeps_accumulating_identically_after_init():
    main = _debeta_main()
    shadow = tss._snapshot_to_shadow(main)
    for st in _debeta_rounds(start=7, count=4):
        update_trade_volumes(main, st)
        update_trade_volumes(shadow, st)
    for name in DEBETA_ATTRS:
        assert _plain(getattr(shadow, name)) == _plain(getattr(main, name)), f"{name} drifted after INIT"
    assert compute_parity_components(shadow) == compute_parity_components(main)


def test_parity_digest_covers_the_debeta_legs():
    main = _debeta_main()
    before = compute_parity_components(main)
    assert "debeta_cap" in before and "debeta_mtm" in before
    main.capture_buy_sums[1][0] += 1.0
    after_cap = compute_parity_components(main)
    assert after_cap["debeta_cap"] != before["debeta_cap"]
    assert {k for k in before if before[k] != after_cap[k]} == {"debeta_cap"}
    main.debeta_mtm[1][0] += 1.0
    after_mtm = compute_parity_components(main)
    assert after_mtm["debeta_mtm"] != after_cap["debeta_mtm"]


def test_snapshot_pickles_and_rebuilds_every_struct_name():
    main = _debeta_main()
    parts = {n: pickle.loads(pickle.dumps(tss._plainify(getattr(main, n, {})), protocol=5)) for n in _STRUCT_NAMES}
    rebuilt = tss._rebuild_structs(parts)
    assert set(rebuilt) == set(_STRUCT_NAMES), "every snapshotted structure must be rebuilt on the child"


def test_child_save_persists_the_debeta_histories(tmp_path):
    from taos.im.validator.persistence import build_debeta_state

    main = _debeta_main()
    shadow = tss._snapshot_to_shadow(main)
    path = str(tmp_path / "validator_state.mp")
    child_save_validator_state(shadow, tss._light_fields(), path)
    saved = msgpack.unpackb(open(path, "rb").read(), raw=False, strict_map_key=False)

    expected = build_debeta_state(main)
    assert expected["debeta_capbuy_hist"] and expected["debeta_mtm_hist"], "fixture must populate the histories"
    for key, val in expected.items():
        assert key in saved, f"child save lacks {key}: a restart with the shadow on zeroes the de-beta window"
        assert saved[key] == tss._mp_norm(val), f"{key} differs from main's persistence block"
    keys = list(saved)
    assert keys.index("roundtrip_volume_sums") < keys.index("debeta_capbuy_hist") < keys.index("miner_stats"), (
        "the de-beta block sits where main's build_validator_state puts it"
    )


def test_main_and_child_saves_share_one_debeta_block():
    src = PERSISTENCE.read_text()
    block = src[src.index("def build_validator_state("):]
    block = block[:block.index("\ndef ", 1)]
    assert "build_debeta_state(self)" in block, "main's save must use the shared de-beta block"
    for literal in ("debeta_capbuy_hist", "debeta_plast"):
        assert f'"{literal}":' not in block, f"{literal} is still spelled out in build_validator_state (drift risk)"
