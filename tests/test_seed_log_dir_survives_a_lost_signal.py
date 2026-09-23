"""The seed service must not write to a directory from a previous run.

Seen on mainnet. The handoff file `/tmp/validator_log_dir.txt` held the CORRECT run directory
the whole time, the simulator was writing there correctly, and the seed service was still appending
to `/tmp/taos-sim-0001` from a run that had long ended - emitting

    ERROR | Exception in seed handling: Seed=79654.05 | Error=[Errno 2] No such file or directory:
            '/tmp/taos-sim-0001/fundamental_seed.csv'

several times a second, losing every seed in between. `kill -USR1 <pid>` clears it instantly, which
is what distinguishes a missing file from a stale PROCESS.

Three independent faults had to line up, and each is guarded here:

  1. the reader refreshed only on SIGUSR1, and `_log_dir_changed` was cleared only on the branch
     where the directory had NOT changed - so after the first successful read it latched forever;
  2. the signal is sent to `v.seed_process`, a handle that goes stale as soon as this service
     outlives the validator that spawned it. The validator then writes the file, finds its handle
     dead, logs "cannot notify" and returns, and nothing ever tells this process;
  3. `open(path, 'a')` raises for a directory that has been cleared from /tmp or not yet created,
     which is an ordinary condition here, not an exceptional one.

`check_log_dir_change` is a closure inside `seed()`, which needs live websockets, so its two
properties are asserted structurally. `_ensure_log_dir` is module-level and tested for real.
"""
import ast
import os
import sys

_REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)

SEED_PY = _REPO_ROOT + "/taos/im/validator/seed.py"


def _fn(name):
    tree = ast.parse(open(SEED_PY, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in seed.py")


def test_refresh_does_not_depend_on_the_signal_arriving():
    """The early return must consider the file's mtime, or a lost signal latches it forever."""
    fn = _fn("check_log_dir_change")
    src = ast.unparse(fn)
    assert "st_mtime" in src, (
        "check_log_dir_change must stat the handoff file. Refreshing only on SIGUSR1 means a signal "
        "sent to a stale process handle leaves this service writing to a dead directory forever."
    )
    # the guard that returns early must be qualified by the mtime, not just by the cached value
    early = [n for n in ast.walk(fn)
             if isinstance(n, ast.If) and any(isinstance(b, ast.Return) for b in n.body)]
    assert any("mtime" in ast.unparse(n.test) for n in early), (
        "the early-return guard ignores the file's mtime, so a changed handoff file is never noticed"
    )


def test_the_changed_flag_is_cleared_on_both_paths():
    """It was cleared only in the `else`, so a run that DID change directory left it set."""
    fn = _fn("check_log_dir_change")
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            for stmt in node.orelse:
                if isinstance(stmt, ast.Assign) and "_log_dir_changed" in ast.unparse(stmt.targets):
                    raise AssertionError(
                        "_log_dir_changed is cleared inside an `else`; it must be cleared on both "
                        "branches or the directory-changed path re-reads the file on every seed"
                    )


def test_the_handoff_path_is_overridable():
    """A fixed /tmp path is both wiped by systemd and shared across runs; allow a deployment out."""
    src = ast.unparse(_fn("check_log_dir_change"))
    assert "TAOS_VALIDATOR_LOG_DIR_FILE" in src, (
        "the handoff location must be overridable so a deployment can move it off /tmp"
    )


def test_ensure_log_dir_creates_a_missing_directory(tmp_path):
    """The real repair: a cleared run directory must not cost every subsequent seed."""
    from taos.im.validator.seed import _ensure_log_dir

    target = tmp_path / "run" / "20260827_020642" / "fundamental_seed.csv"
    assert not target.parent.exists()

    _ensure_log_dir(str(target))

    assert target.parent.is_dir(), "the directory the CSV appends into was not created"
    with open(target, "a") as fh:          # this is the call that raised [Errno 2] on mainnet
        fh.write("1,100.0\n")
    assert target.read_text() == "1,100.0\n"


def test_ensure_log_dir_does_not_raise_when_it_cannot_create(tmp_path):
    """It runs per trade inside the seed path: it must degrade, never add a second failure mode."""
    from taos.im.validator.seed import _ensure_log_dir

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file")
    # a path *through* a regular file cannot be created; the helper must swallow and log
    _ensure_log_dir(str(blocker / "deeper" / "seed.csv"))


def test_every_append_site_is_guarded():
    """All three CSVs share the failure; guarding only the seed file leaves two open."""
    src = open(SEED_PY, encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.splitlines()
    unguarded = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "open" and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == "a"):
            window = "\n".join(lines[max(0, node.lineno - 3):node.lineno])
            if "_ensure_log_dir" not in window:
                unguarded.append(node.lineno)
    assert not unguarded, (
        f"append-open without a directory guard at line(s) {unguarded}: a cleared run directory "
        "will raise per trade there"
    )
