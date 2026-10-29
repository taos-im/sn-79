# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator builds the miners' payloads on one thread unless told otherwise.

The batch path compresses each miner's state on a thread pool, and the threads contend for the GIL: at 256 miners the
payloads took 66 ms on half the cores against 51 ms on one thread on the developers' bench, and 49.7 against 43.0 ms on
a 32-core test host (median of 15 rounds, 128 books). So --compression.parallel_workers defaults to 0, the value the
query process, which builds the payloads, already declares for itself; a positive count or -1 still selects the pool.
"""
import argparse

from taos.im.config import add_im_validator_args
from taos.im.validator import query


def _validator(args):
    parser = argparse.ArgumentParser()
    add_im_validator_args(None, parser)
    known, _ = parser.parse_known_args(args)
    return getattr(known, "compression.parallel_workers")


def test_the_validator_builds_payloads_on_one_thread_by_default():
    assert _validator([]) == 0


def test_the_query_process_reads_the_same_default():
    known, _ = query.service_parser().parse_known_args([])
    assert getattr(known, "compression.parallel_workers") == _validator([])


def test_a_pool_can_still_be_chosen():
    assert _validator(["--compression.parallel_workers", "-1"]) == -1
    assert _validator(["--compression.parallel_workers", "4"]) == 4
