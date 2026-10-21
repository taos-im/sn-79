# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""seed.py can be imported from any thread, and keeps its early SIGUSR1 handler on the main thread.

The seed service process installs its handler at import, before the slow imports, so a SIGUSR1 sent
right after spawn does not kill it. The proxy imports the same module from its seed thread, where
signal.signal raises ValueError. Each case runs in a fresh interpreter so the module is really
imported, not served from a cache.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, cwd=ROOT)


def test_importing_from_a_thread_does_not_raise():
    code = (
        "import signal, threading\n"
        "errors = []\n"
        "def imp():\n"
        "    try:\n"
        "        import taos.im.validator.seed\n"
        "    except BaseException as e:\n"
        "        errors.append(repr(e))\n"
        "t = threading.Thread(target=imp); t.start(); t.join()\n"
        "print('ERR', errors)\n"
        "print('DFL', signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL)\n"
    )
    r = _run(code)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ERR []" in r.stdout, f"import off the main thread raised: {r.stdout}"
    assert "DFL True" in r.stdout, "a thread import changed the process's SIGUSR1 disposition"


def test_importing_on_the_main_thread_installs_the_handler():
    code = (
        "import signal\n"
        "import taos.im.validator.seed as s\n"
        "print('INSTALLED', signal.getsignal(signal.SIGUSR1) is s._flag_sigusr1)\n"
    )
    r = _run(code)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "INSTALLED True" in r.stdout, "the main-thread import no longer installs the SIGUSR1 handler"
