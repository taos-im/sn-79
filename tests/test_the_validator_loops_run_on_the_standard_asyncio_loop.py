# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator's own event loops, the listener's and the main loop, run on the standard asyncio loop.

forward.py installed uvloop as the process-wide event loop policy, so every loop the validator created was a uvloop
loop, and an abort inside uvloop's native loop code ends the process without a message: libuv aborts when it cannot
re-arm a descriptor it watches. The standard loop raises an error with a traceback in the same case, which the
validator logs and survives, and which names the descriptor. Nothing the tree ships imports uvloop, so an install does
not pull it either.
"""
import asyncio
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_importing_forward_leaves_the_standard_policy():
    import taos.im.validator.forward  # noqa: F401
    policy = asyncio.get_event_loop_policy()
    assert type(policy).__module__.startswith("asyncio"), type(policy)
    loop = policy.new_event_loop()
    try:
        assert type(loop).__module__.startswith("asyncio"), type(loop)
    finally:
        loop.close()


def test_no_validator_module_installs_uvloop():
    for p in (ROOT / "taos").rglob("*.py"):
        src = p.read_text(errors="replace")
        assert "uvloop.EventLoopPolicy" not in src and "uvloop.install" not in src, p


def test_no_install_pulls_uvloop():
    listed = re.compile(r"^\s*uvloop\b", re.I | re.M)
    imported = re.compile(r"^\s*(import|from)\s+uvloop\b", re.M)
    for name in ("requirements.txt", "constraints.txt"):
        p = ROOT / name
        if p.exists():
            assert not listed.search(p.read_text()), name
    for p in _shipped_py():
        assert not imported.search(p.read_text(errors="replace")), p


def _shipped_py():
    """The tree's own Python files: the tracked ones, so the engine's build and vendor trees are not read."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--", "*.py"],
                             capture_output=True, text=True, check=True).stdout.split()
        return [ROOT / p for p in out if (ROOT / p).is_file()]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in ROOT.rglob("*.py") if not {"build", "vcpkg", ".git"} & set(p.parts)]
