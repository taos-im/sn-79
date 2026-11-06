# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The launchers' settings header says whether the PagerDuty key is set, never the key itself.

The header goes to the terminal and to whatever keeps the launcher's output. Every launcher in the tree is checked:
its header line runs under the launcher's own shell, and no echo or printf in it expands a secret.
"""
import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHERS = sorted(p.name for p in ROOT.glob("run_*.sh"))
DUMMY = "DUMMYPAGERDUTYKEY0123456789abcdef"
_SECRET = re.compile(
    r"(?<!\\)\$\{?(PD_KEY|[A-Z0-9_]*(?:SECRET|SECRET_KEY|TOKEN|PASSWORD|API_KEY|PROXY_KEY|INTEGRATION_KEY))"
    r"(?![A-Za-z0-9_])(?!:\+)"
)
# The gradient server's launcher prints its API key once it is up, as its header says, to be set on the validator host.
HANDED_OVER = {("run_vast.sh", "API_KEY")}


def _header_lines():
    found = []
    for name in LAUNCHERS:
        text = (ROOT / name).read_text()
        shell = text.splitlines()[0].removeprefix("#!").split()[0]
        found += [(name, shell, ln) for ln in text.splitlines() if "PAGERDUTY KEY" in ln]
    return found


def test_the_launchers_print_the_pagerduty_header():
    names = {name for name, _, _ in _header_lines()}
    assert "run_validator.sh" in names
    readers = {n for n in LAUNCHERS if "PD_KEY" in (ROOT / n).read_text()}
    assert names == readers, "every launcher that reads the key prints the header"


@pytest.mark.parametrize(("value", "shown"), [(DUMMY, "set"), ("", "not set"), ('""', "not set")])
def test_the_header_says_whether_the_key_is_set(value, shown):
    for name, shell, line in _header_lines():
        out = subprocess.run([shell, "-c", line], env={"PD_KEY": value}, capture_output=True, text=True).stdout
        assert out.strip() == f"PAGERDUTY KEY: {shown}", f"{name}: {line.strip()}"


def test_no_launcher_echoes_a_secret():
    leaks = []
    for name in LAUNCHERS:
        for n, line in enumerate((ROOT / name).read_text().splitlines(), 1):
            said = re.split(r"\b(?:echo|printf)\b", line, maxsplit=1)
            if len(said) == 2:
                leaks += [f"{name}:{n}" for var in _SECRET.findall(said[1]) if (name, var) not in HANDED_OVER]
    assert leaks == []
