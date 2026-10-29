"""The launcher's monitor window opens the validator's and the engine's log panes and keeps them open.

run_validator.sh used to start each log pane with `pm2 logs <name>` as the pane's own command. tmux runs that in a
non-interactive shell, which does not read the operator's shell setup, so where pm2 is installed under nvm the command
is not found and the pane closes at once: the window showed htop alone. The panes also used
`split-window -p`, which current tmux replaces with `-l <n>%`. Each log pane is now a shell with the command typed into
it, as the SL/TP and GenTRX windows already were.
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_validator.sh"
# The validator's own log panes belong to the full launcher. Where run_mvtrx.sh is absent, run_validator.sh is the
# testnet launcher, which opens none, so those two checks have nothing to read.
FULL_LAUNCHER = (SCRIPT.parent / "run_mvtrx.sh").exists()


def _function(name):
    text = SCRIPT.read_text()
    m = re.search(rf"^{name}\(\) \{{.*?^\}}\n", text, re.S | re.M)
    assert m, f"{name} is not defined in run_validator.sh"
    return m.group(0)


LAUNCHERS = ["run_validator.sh", "run_validator_testnet.sh", "run_gradients.sh", "run_miner.sh", "run_mvtrx.sh"]


@pytest.mark.parametrize("name", LAUNCHERS)
def test_no_split_starts_a_pane_with_its_command_or_sizes_it_with_p(name):
    path = SCRIPT.parent / name
    if not path.exists():
        pytest.skip(f"{name} is not in this tree")
    splits = [line for line in path.read_text().splitlines() if "split-window" in line and not line.lstrip().startswith("#")]
    for line in splits:
        assert not re.search(r"split-window\b.*\s-p\s", line), f"{name}: -p is gone from current tmux: {line.strip()}"
        # Whatever follows the -F format is a pane command, which runs in a non-interactive shell.
        tail = line.split("#{pane_id}'")[-1] if "#{pane_id}'" in line else re.sub(r"^.*?-t\s+\S+", "", line)
        assert not re.search(r"['\"]\s*$|['\"][^'\"]*['\"]\)?\s*$", tail.strip()) or tail.strip() in ("", ")"), (
            f"{name}: a pane started with its own command closes when the command is not found: {line.strip()}")


@pytest.mark.skipif(not FULL_LAUNCHER, reason="this run_validator.sh opens no log panes")
def test_every_log_pane_goes_through_the_shell_pane_helper():
    text = SCRIPT.read_text()
    assert text.count('_log_pane "$P0" "-v -l 75%" "$VALIDATOR_NAME"') == 3, (
        "the validator's pane: the full launch, the preserve-simulator launch and its observe variant")
    assert text.count('_log_pane "$P1" "-h -l 50%" simulator') == 2, "the engine's pane, in both launch branches"
    assert text.count('_log_pane "$P1" "-h -l 50%" exchange') == 2


@pytest.mark.skipif(not FULL_LAUNCHER, reason="this run_validator.sh opens no log panes")
@pytest.mark.skipif(shutil.which("tmux") is None, reason="no tmux on this host")
def test_the_log_pane_stays_open_and_runs_pm2_logs(tmp_path):
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "pm2").write_text('#!/bin/sh\necho "STUB-PM2 $*"\nexec sleep 30\n')
    (stub / "pm2").chmod(0o755)
    env = dict(os.environ, TMUX_TMPDIR=str(tmp_path), PATH=f"{stub}:{os.environ['PATH']}")
    env.pop("TMUX", None)

    def run(*a):
        return subprocess.run(["tmux", *a], env=env, capture_output=True, text=True, timeout=20)

    assert run("new-session", "-d", "-s", "t", "-x", "200", "-y", "50", "sleep 60").returncode == 0
    try:
        p0 = run("list-panes", "-t", "t", "-F", "#{pane_id}").stdout.strip()
        helper = _function("_log_pane")
        out = subprocess.run(["bash", "-c", helper + f'_log_pane "{p0}" "-v -l 75%" validator-sim-test'],
                             env=env, capture_output=True, text=True, timeout=20)
        assert out.returncode == 0, out.stderr
        pane = out.stdout.strip()
        assert re.fullmatch(r"%\d+", pane), f"the helper returns the new pane's id, got {pane!r}"
        # The command is typed into a fresh shell, which runs it once its profile has loaded: on a heavily
        # loaded machine that can take longer than 10 s.
        # The wait ends as soon as the stub prints, so only a pane that never runs it pays the minute.
        seen = ""
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            time.sleep(0.25)
            seen = run("capture-pane", "-p", "-t", pane).stdout
            if "STUB-PM2 logs validator-sim-test" in seen:
                break
        assert "STUB-PM2 logs validator-sim-test" in seen, f"the pane never ran pm2 logs: {seen!r}"
        assert pane in run("list-panes", "-t", "t", "-F", "#{pane_id}").stdout, "the log pane closed"
    finally:
        run("kill-server")
