# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""The launchers reinstall cyscale at the version constraints.txt pins, the one bittensor itself pins.

A launch removes the old sync substrate-interface, the scalecodec it pulls in and cyscale, then reinstalls cyscale.
Under the lockfile it takes the exact version bittensor and bittensor-cli require. Every launcher in the tree is
checked, and the lockfile's path reaches pip as one argument when the repository path holds a space.
"""
import pathlib
import re
import shlex
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHERS = sorted(p.name for p in ROOT.glob("run_*.sh"))
_LOCKFILE = re.compile(r"""-c\s+"?(\$REPO_ROOT/)?constraints\.txt"?|\$\{?_CONSTRAINTS\}?""")


def _cyscale_installs(name):
    lines = (ROOT / name).read_text().splitlines()
    return [(n, ln) for n, ln in enumerate(lines, 1)
            if re.search(r"\bpip3? install\b.*\bcyscale\b", ln) and not ln.lstrip().startswith("#")]


def test_the_lockfile_pins_cyscale():
    assert re.search(r"^cyscale==\d", (ROOT / "constraints.txt").read_text(), re.M)


def test_some_launcher_reinstalls_cyscale():
    assert any(_cyscale_installs(name) for name in LAUNCHERS)


def test_every_cyscale_reinstall_takes_the_lockfile():
    unpinned = [f"{name}:{n}" for name in LAUNCHERS for n, ln in _cyscale_installs(name) if not _LOCKFILE.search(ln)]
    assert unpinned == []


def test_the_constraints_a_reinstall_uses_are_set_before_it():
    unset = []
    for name in LAUNCHERS:
        lines = (ROOT / name).read_text().splitlines()
        for n, ln in _cyscale_installs(name):
            if "_CONSTRAINTS" in ln and not any(re.match(r"\s*_CONSTRAINTS=", earlier) for earlier in lines[: n - 1]):
                unset.append(f"{name}:{n}")
    assert unset == []


def test_the_lockfile_reaches_pip_as_one_argument_under_a_path_with_a_space(tmp_path):
    repo = tmp_path / "a repo"
    repo.mkdir()
    (repo / "constraints.txt").write_text("cyscale==0\n")
    checked = []
    for name in LAUNCHERS:
        lines = (ROOT / name).read_text().splitlines()
        for n, ln in _cyscale_installs(name):
            if "_CONSTRAINTS" not in ln:
                continue
            setter = [e for e in lines[: n - 1] if re.match(r"\s*_CONSTRAINTS=", e)][-1].strip()
            script = f'pip() {{ printf "%s\\n" "$@"; }}\nREPO_ROOT={shlex.quote(str(repo))}\n{setter}\n{ln.strip()}\n'
            run = subprocess.run(["bash", "-c", script], cwd=repo, capture_output=True, text=True, check=True)
            args = run.stdout.splitlines()
            assert "-c" in args, f"{name}:{n} passes no lockfile"
            assert args[args.index("-c") + 1] in ("constraints.txt", str(repo / "constraints.txt")), f"{name}:{n}"
            checked.append(f"{name}:{n}")
    assert checked
