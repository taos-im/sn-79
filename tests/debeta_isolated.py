# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The de-beta dials a mechanism test fixes so it exercises one mechanism alone.

A dial a config does not set reads the parser default (taos.im.config.debeta_dial), and the shipped defaults switch
on the realized basis, the notional hurdle, the inactive-book and counterparty discounts, the class split and the
maker fee. A test written against one mechanism on a handful of books names these so the rest stay off; it then
sets the dials it is about on top.
"""

ISOLATED = {
    "making_basis": "captured",
    "making_pool": "rank",
    "weight": 0.0,
    "skill_hurdle_bps": 0.0,
    "skill_max_inactive_books": 0.0,
    "s3_k": 0.0,
    "p11_strength": 0.0,
    "skill_p11_strength": 0.0,
    "p11_topk": 2,
    "class_weights": "",
    "making_fee_weight": 0.0,
    "mark_mode": "last",
}


def isolated(**dials):
    """ISOLATED with the test's own dials on top."""
    return {**ISOLATED, **dials}
