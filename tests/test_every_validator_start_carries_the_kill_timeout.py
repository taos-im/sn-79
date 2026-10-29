# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Every pm2 start of the validator in the launchers registers a 60 s kill timeout, as the engine's starts do.

pm2 SIGKILLs 1600 ms after the stop signal by default. On a stop the validator first waits up to 30 s for a state
save, reward, report or archive compression in flight, then stops its services and its event loop
(taos/im/validator/cleanup.py), so a kill at 1600 ms loses the save it was writing. Every launcher in the tree is
checked.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHERS = sorted(p.name for p in ROOT.glob("run_*.sh"))
_VALIDATOR_START = re.compile(r"^\s*pm2 start\b.*--name=\"?\$VALIDATOR_NAME\b")


def _validator_starts(name):
    lines = (ROOT / name).read_text().splitlines()
    return [(i + 1, ln) for i, ln in enumerate(lines) if _VALIDATOR_START.search(ln)]


def test_the_published_launcher_starts_the_validator():
    assert "run_validator.sh" in LAUNCHERS
    assert _validator_starts("run_validator.sh"), "run_validator.sh starts no validator under pm2"


def test_every_validator_start_carries_the_kill_timeout():
    missing = []
    for name in LAUNCHERS:
        for lineno, ln in _validator_starts(name):
            if not re.search(r"--kill-timeout 60000\b", ln):
                missing.append(f"{name}:{lineno}")
    assert not missing, f"pm2 start of the validator without --kill-timeout 60000 at {missing}"
