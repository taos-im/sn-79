# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The per-book de-beta gauges cross two process boundaries, and both were silently unwired on
first live use: the reporting child parses its own argv so the publish flag never
arrived, and the capture/alpha accumulators live in the validator so the child's gauge block read
None forever. These are source-level wiring assertions: cheap, and they fail loudly if either hop
is dropped in a refactor. Behavioural coverage of the gauge block itself lives in the report tests.
"""
from pathlib import Path

DEV = Path(__file__).resolve().parents[1]
VALIDATOR = (DEV / "taos/im/neurons/validator.py").read_text()
REPORT = (DEV / "taos/im/validator/report.py").read_text()


def test_publish_flag_is_forwarded_to_the_reporting_child():
    spawn = VALIDATOR[VALIDATOR.index("'../validator/report.py'"):]
    spawn = spawn[:spawn.index("subprocess.Popen")]
    assert "cmd.append('--scoring.debeta.publish_book_gauges')" in spawn, (
        "report.py parses its own argv: without forwarding, the flag is dead in the child"
    )


def test_accumulators_ship_in_the_reporting_payload():
    for key in ("'debeta_capture_buy_sums'", "'debeta_capture_sell_sums'", "'debeta_alphas_by_book'"):
        assert key in VALIDATOR, f"reporting payload must carry {key}"


def test_report_unpacks_accumulators_with_int_keys():
    for attr in ("self.capture_buy_sums", "self.capture_sell_sums", "self.debeta_alphas_by_book"):
        assert f"{attr} = _int_keyed_books(" in REPORT, (
            f"{attr} must be set from the payload (int-keyed: IPC stringifies keys), "
            "or the book-gauge block evicts every series"
        )


def test_miner_gauge_evictions_pair_with_their_publish():
    """Each publish-or-evict miner gauge must evict THE SAME series it publishes. A mismatched name
    (found live: the trading_score else-branch evicted pnl_score) deletes a healthy
    neighbour gauge on every fallback cycle and leaves the intended one stale forever."""
    import re
    # Indent-aware: publish and evict statements must sit at the SAME indent, or the regex would
    # pair a publish with an outer guard's else (whose different-name evict is correct).
    pairs = re.findall(
        r'\n( +)updates\.append\(\(miner_gauges,[^)]*, "(\w+)"\)\)\n *else:\n'
        r'\1_remove_and_evict\([^)]*miner_gauges[^)]*, "(\w+)"\)',
        REPORT,
    )
    pairs = [(pub, ev) for _indent, pub, ev in pairs]
    assert pairs, "expected publish/evict miner gauge pairs in report.py"
    mismatched = [(pub, ev) for pub, ev in pairs if pub != ev]
    assert not mismatched, f"evict name must match published gauge: {mismatched}"
