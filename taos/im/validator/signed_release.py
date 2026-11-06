# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Validator updates go only to a release tag signed by an allowed signer.

The update check (hourly, at startup and at the run end) and the launcher follow one of three modes:

    signed  (opt-in until a release signing key ships) the newest release tag (vMAJOR.MINOR.PATCH) on the tracked
            branch, applied as a fast-forward, and only when ``git verify-tag`` accepts its SSH signature against
            ``.github/allowed_signers`` as it stands in the tree the validator is already running. The trust anchor
            is the code in place: a release that adds a key to that file is verified against the file before it.
    branch  (default) the tracked branch's tip.
    off     no self-update.

This module imports nothing from the package, so the launcher can run it by path before installing anything:

    python3 taos/im/validator/signed_release.py --remote origin --branch main [--apply]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

MODES = ("signed", "branch", "off")
RELEASE_TAG = re.compile(r"^v\d+\.\d+\.\d+$")
SIGNERS = Path(".github") / "allowed_signers"


def _git(repo, *args, timeout=120):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout)


def verify_tag(repo, tag, signers_file):
    """(accepted, git's message) for ``tag`` against ``signers_file``."""
    signers = Path(signers_file)
    if not signers.is_file():
        return False, f"no allowed signers file at {signers}"
    r = _git(repo, "-c", "gpg.format=ssh", "-c", f"gpg.ssh.allowedSignersFile={signers.resolve()}", "verify-tag", tag)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def newest_release_tag(repo, ref):
    """The highest release tag whose commit is on ``ref``, or None."""
    listed = _git(repo, "for-each-ref", "--sort=-v:refname", "--format=%(refname:short)", "refs/tags")
    for tag in listed.stdout.split():
        if RELEASE_TAG.match(tag) and _git(repo, "merge-base", "--is-ancestor", f"{tag}^{{commit}}", ref).returncode == 0:
            return tag
    return None


def plan_update(repo, remote, branch, signers_file=None):
    """What the signed mode would do now: {"action": "update"|"none"|"refuse", "reason", "tag", "commit"}.

    Reads the allowed signers BEFORE anything moves, from the current checkout unless a file is given."""
    repo = Path(repo)
    signers_file = Path(signers_file) if signers_file else repo / SIGNERS
    fetched = _git(repo, "fetch", "--tags", "--force", remote, branch)
    if fetched.returncode != 0:
        return {"action": "none", "reason": f"fetch from {remote} failed: {fetched.stderr.strip()[:200]}"}
    tag = newest_release_tag(repo, f"{remote}/{branch}")
    if tag is None:
        return {"action": "none", "reason": f"no release tag on {remote}/{branch}"}
    commit = _git(repo, "rev-parse", f"{tag}^{{commit}}").stdout.strip()
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    if commit == head:
        return {"action": "none", "reason": f"at {tag}", "tag": tag, "commit": commit}
    if _git(repo, "merge-base", "--is-ancestor", head, commit).returncode != 0:
        return {"action": "none", "reason": f"{tag} is not ahead of HEAD (behind or diverged); not updating",
                "tag": tag, "commit": commit}
    ok, why = verify_tag(repo, tag, signers_file)
    if not ok:
        return {"action": "refuse", "reason": f"{tag} is not signed by an allowed signer: {why[:300]}",
                "tag": tag, "commit": commit}
    return {"action": "update", "reason": f"{tag} verified", "tag": tag, "commit": commit}


def apply_update(repo, tag):
    """Fast-forward the current branch to the verified tag's commit."""
    r = _git(repo, "merge", "--ff-only", f"{tag}^{{commit}}")
    if r.returncode != 0:
        raise RuntimeError(f"could not fast-forward to {tag}: {(r.stderr or r.stdout).strip()[:300]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Update this checkout to the newest signed release tag, if any.")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default=None, help="the tracked branch (default: the current one)")
    ap.add_argument("--apply", action="store_true", help="fast-forward to the verified tag; otherwise only report")
    a = ap.parse_args(argv)
    branch = a.branch or _git(a.repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    plan = plan_update(a.repo, a.remote, branch)
    print(f"signed release check: {plan['action']}: {plan['reason']}")
    if plan["action"] == "refuse":
        return 2
    if plan["action"] == "update" and a.apply:
        apply_update(a.repo, plan["tag"])
        print(f"updated to {plan['tag']} ({plan['commit'][:10]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
