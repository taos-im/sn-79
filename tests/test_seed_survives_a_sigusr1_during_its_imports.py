"""The seed service must survive the validator's SIGUSR1 while it is still importing.

sn79_20261007_174656: the validator spawned the seed at 18:08:52, waited its two seconds, wrote the
handoff file and sent SIGUSR1 at 18:08:56; at 18:09:23 it found the seed dead with exit code -10,
which is SIGUSR1's default disposition. seed.py installed its handler inside run_seed_service, after
`import bittensor`, which takes longer than two seconds on a loaded box. With the seed dead nothing
recorded seeds for the new run. The handler has to be in place before the slow imports.

The prelude (every top-level statement before `import bittensor`) is run for real in a subprocess
that then signals itself.
"""
import ast
import signal
import subprocess
import sys
from pathlib import Path

SEED_PY = Path(__file__).resolve().parents[1] / "taos/im/validator/seed.py"


def _prelude_source() -> str:
    tree = ast.parse(SEED_PY.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == "bittensor" for a in node.names):
            break
        body.append(node)
    else:
        raise AssertionError("seed.py no longer imports bittensor; re-point this test")
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_a_sigusr1_before_the_heavy_imports_does_not_kill_the_seed():
    code = (
        _prelude_source()
        + "\nimport os, signal, time\nos.kill(os.getpid(), signal.SIGUSR1)\ntime.sleep(0.2)\nprint('alive')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert r.returncode != -signal.SIGUSR1, (
        "seed.py dies on a SIGUSR1 that arrives before `import bittensor` finishes: install the handler "
        "before the heavy imports"
    )
    assert r.returncode == 0 and "alive" in r.stdout, r.stderr


def test_the_flag_an_early_signal_sets_is_not_reset_afterwards():
    """A module-level `_log_dir_changed = False` after the imports would discard an early signal."""
    tree = ast.parse(SEED_PY.read_text(encoding="utf-8"))
    seen_import = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == "bittensor" for a in node.names):
            seen_import = True
        if seen_import and isinstance(node, ast.Assign) and "_log_dir_changed" in ast.unparse(node.targets):
            raise AssertionError("_log_dir_changed is re-initialised after the imports")
