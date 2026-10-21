# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""In the signed update mode, a validator updates only to a release tag signed by an allowed signer of the tree it
already runs.

With `--repo.auto_update signed` (AUTO_UPDATE=signed in the launchers; opt-in, the default follows the tracked
branch's tip) the update check (hourly, at startup, at the run end) targets the newest release tag
(vMAJOR.MINOR.PATCH) on the tracked branch, applied as a fast-forward, and only when `git verify-tag` accepts its
SSH signature against .github/allowed_signers as it stands in the CURRENT checkout: a tag whose commit adds its own
key to that file cannot vouch for itself."""
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DEV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEV / "taos" / "im" / "validator"))
import signed_release  # noqa: E402  (the module is dependency-free so the launcher can run it by path)

pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs ssh-keygen")


def _run(*args, cwd=None):
    r = subprocess.run(list(args), cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, f"{args}: {r.stderr}"
    return r.stdout.strip()


def _key(tmp, name):
    path = tmp / f"key_{name}"
    _run("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name, "-f", str(path))
    return path, (tmp / f"key_{name}.pub").read_text().strip()


def _git(repo, *args):
    return _run("git", "-C", str(repo), *args)


def _commit(repo, name, text):
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", name)


def _tag(repo, tag, key=None):
    if key is None:
        _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "tag", "-a", tag, "-m", tag)
    else:
        _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "gpg.format=ssh",
             "-c", f"user.signingkey={key}", "tag", "-s", tag, "-m", tag)


@pytest.fixture
def world(tmp_path):
    """An upstream with one release, a validator clone at it, and two keys: A allowed, B not."""
    key_a, pub_a = _key(tmp_path, "a")
    key_b, pub_b = _key(tmp_path, "b")
    up = tmp_path / "upstream"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _commit(up, ".github/allowed_signers", f"owner@sn79 namespaces=\"git\" {pub_a}\n")
    _commit(up, "taos/v.txt", "0.6.3\n")
    _tag(up, "v0.6.3", key_a)
    val = tmp_path / "validator"
    _run("git", "clone", "-q", str(up), str(val))
    _git(val, "checkout", "-q", "main")
    return dict(up=up, val=val, key_a=key_a, key_b=key_b, pub_b=pub_b)


def _head(repo):
    return _git(repo, "rev-parse", "HEAD")


def test_a_release_signed_by_an_allowed_signer_is_applied(world):
    _commit(world["up"], "taos/v.txt", "0.6.4\n")
    _tag(world["up"], "v0.6.4", world["key_a"])
    plan = signed_release.plan_update(world["val"], "origin", "main")
    assert plan["action"] == "update" and plan["tag"] == "v0.6.4", plan
    signed_release.apply_update(world["val"], plan["tag"])
    assert (world["val"] / "taos" / "v.txt").read_text() == "0.6.4\n"


def test_a_release_signed_by_an_unknown_key_is_refused(world):
    _commit(world["up"], "taos/v.txt", "0.6.4-evil\n")
    _tag(world["up"], "v0.6.4", world["key_b"])
    before = _head(world["val"])
    plan = signed_release.plan_update(world["val"], "origin", "main")
    assert plan["action"] == "refuse", plan
    assert _head(world["val"]) == before


def test_an_unsigned_release_is_refused(world):
    _commit(world["up"], "taos/v.txt", "0.6.4\n")
    _tag(world["up"], "v0.6.4")
    assert signed_release.plan_update(world["val"], "origin", "main")["action"] == "refuse"


def test_a_push_cannot_vouch_for_itself_by_adding_its_key(world):
    """The release adds key B to the allowed signers AND is signed by B: the running tree's file decides."""
    _commit(world["up"], ".github/allowed_signers",
            (world["up"] / ".github" / "allowed_signers").read_text() + f"evil@x namespaces=\"git\" {world['pub_b']}\n")
    _tag(world["up"], "v0.6.4", world["key_b"])
    assert signed_release.plan_update(world["val"], "origin", "main")["action"] == "refuse"


def test_commits_after_the_newest_release_are_not_followed(world):
    """The branch tip moves on without a tag: nothing to apply, whoever pushed it."""
    _commit(world["up"], "taos/v.txt", "tip, untagged\n")
    plan = signed_release.plan_update(world["val"], "origin", "main")
    assert plan["action"] == "none", plan
    assert (world["val"] / "taos" / "v.txt").read_text() == "0.6.3\n"


def test_a_release_that_is_not_a_fast_forward_is_not_applied(world):
    _commit(world["val"], "local.txt", "a local commit\n")
    _git(world["up"], "checkout", "-q", "-b", "side", "v0.6.3^")
    _commit(world["up"], "taos/v.txt", "side\n")
    _tag(world["up"], "v0.6.9", world["key_a"])
    _git(world["up"], "checkout", "-q", "main")
    _git(world["up"], "merge", "-q", "--no-edit", "-s", "ours", "side")
    plan = signed_release.plan_update(world["val"], "origin", "main")
    assert plan["action"] != "update", plan


def test_a_non_release_tag_is_ignored(world):
    _commit(world["up"], "taos/v.txt", "nightly\n")
    _tag(world["up"], "nightly-20261009", world["key_a"])
    assert signed_release.plan_update(world["val"], "origin", "main")["action"] == "none"


def test_the_update_check_and_the_run_end_pull_follow_the_mode():
    """check_repo plans through signed_release in the default mode, and update_repo applies the verified tag
    instead of pulling the branch; the launchers do the same before starting."""
    import inspect

    from taos.im.validator import update
    src = inspect.getsource(update)
    assert "signed_release.plan_update(" in src and "def pull_release(" in src
    neuron = (DEV / "taos" / "im" / "neurons" / "validator.py").read_text()
    assert "remote.pull()" not in neuron, "update_repo pulls through update.pull_release, never the branch tip"
    for launcher in ("run_validator.sh", "run_validator_testnet.sh"):
        path = DEV / launcher
        if path.exists():
            text = path.read_text()
            assert "signed_release.py" in text and "AUTO_UPDATE" in text, launcher


def test_the_default_mode_is_branch_until_a_signing_key_ships():
    """0.6.4 ships the signed mode as an opt-in and keeps 0.6.3's branch-tip update as the default: no
    signing key exists yet, and with none listed the signed mode would never update a validator."""
    import inspect

    from taos.im import config as im_config
    from taos.im.validator.update import auto_update_mode
    src = inspect.getsource(im_config.add_im_validator_args)
    i = src.index('"--repo.auto_update"')
    assert 'default="branch"' in src[i:i + 900]
    def mode(**repo):
        return auto_update_mode(SimpleNamespace(config=SimpleNamespace(repo=SimpleNamespace(**repo))))

    assert auto_update_mode(SimpleNamespace(config=SimpleNamespace())) == "branch"
    assert mode() == "branch" and mode(auto_update="bogus") == "branch" and mode(auto_update="signed") == "signed"
    for launcher in ("run_validator.sh", "run_validator_testnet.sh"):
        path = DEV / launcher
        if path.exists():
            text = path.read_text()
            assert "${AUTO_UPDATE:-signed}" not in text and "${AUTO_UPDATE:-branch}" in text, launcher
    assert (DEV / "run_validator.sh").exists(), "the public tree ships run_validator.sh"


def test_a_signers_file_with_no_key_refuses_every_release(world):
    """The file ships with no key until the owner team adds theirs: the signed mode then updates nothing."""
    (world["val"] / ".github" / "allowed_signers").write_text("# no key listed yet\n")
    _commit(world["up"], "taos/v.txt", "0.6.4\n")
    _tag(world["up"], "v0.6.4", world["key_a"])
    assert signed_release.plan_update(world["val"], "origin", "main")["action"] == "refuse"


def test_the_shipped_signers_file_is_in_the_tree_and_the_carve():
    """The public tree is the carve, so the file being there is the proof; the private tree also names it in the carve
    script that builds the public one."""
    assert (DEV / ".github" / "allowed_signers").is_file()
    carve = DEV / "apply_testnet.sh"
    if carve.is_file():
        assert ".github/allowed_signers" in carve.read_text()
