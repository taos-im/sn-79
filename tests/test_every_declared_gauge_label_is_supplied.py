# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A label declared on a gauge must be supplied wherever that gauge is set.

prometheus_client raises ValueError("Incorrect label names") when a labels() call omits one of the
declared names, and it rejects the WHOLE call. So a label added to a gauge's declaration without a
matching argument at the call site does not degrade one series -- it stops the entire gauge
publishing, and the validator reports only "Unable to publish metrics" with a traceback.

That is easy to introduce, because the declaration and the call are hundreds of lines apart and a
new scoring field naturally gets added to one of them first. This pins the two together so the pair
is checked rather than remembered.
"""

import re
from pathlib import Path

_REPORT = Path(__file__).resolve().parents[1] / "taos" / "im" / "validator" / "report.py"
_SRC = _REPORT.read_text(encoding="utf-8")


def _declared_miner_labels() -> set[str]:
    i = _SRC.index("self.prometheus_miners = Gauge(")
    blob = _SRC[i:_SRC.index("registry=self.registry_miner", i)]
    return set(re.findall(r"'([a-z0-9_]+)'", blob)) - {"miners"}


def _supplied_miner_labels() -> set[str]:
    j = _SRC.index("_set_if_changed_metric(\n                self.prometheus_miners,")
    blob = _SRC[j:]
    depth, end = 0, 0
    for k, ch in enumerate(blob):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = k
                break
    return set(re.findall(r"^\s{16}([a-z0-9_]+)=", blob[:end], re.M))


def test_no_declared_label_is_left_unsupplied():
    missing = _declared_miner_labels() - _supplied_miner_labels()
    assert not missing, (
        f"the miners gauge declares {sorted(missing)} but the call site never supplies them; "
        "prometheus_client will reject every publish of this gauge, not just those series"
    )


def test_no_supplied_label_is_undeclared():
    """The mirror case: an argument with no declared label is rejected just as hard."""
    extra = _supplied_miner_labels() - _declared_miner_labels()
    assert not extra, f"the call site supplies {sorted(extra)}, which the gauge does not declare"
