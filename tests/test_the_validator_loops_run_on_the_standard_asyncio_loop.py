# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The validator's own event loops, the listener's and the main loop, run on the standard asyncio loop.

forward.py installed uvloop as the process-wide event loop policy, so every loop the validator created was a uvloop
loop, and an abort inside uvloop's native loop code ends the process without a message: libuv aborts when it cannot
re-arm a descriptor it watches. The standard loop raises an error with a traceback in the same case, which the
validator logs and survives, and which names the descriptor.
"""
import asyncio
import pathlib

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
