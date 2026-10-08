"""The launchers install the package against constraints.txt, as the install scripts already do.

run_validator.sh and run_miner.sh reinstall the package on every start with `pip install -e .`, and
requirements.txt leaves most packages unpinned, so a host that already had an old version kept it as long
as it satisfied the requirement. That is how the 0.6.3 testnet validators served HTTP 500 on every metrics
family: their prometheus_client predated the 0.22 escaping API that constraints.txt pins past, and nothing
on the launch path asked for the pinned version. The install scripts used the lockfile; the launchers,
which every update runs, did not.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
# In the development tree run_validator.sh is the development launcher, which is not published;
# run_validator_testnet.sh is published AS run_validator.sh. So check whichever is the published one.
_VALIDATOR = "run_validator_testnet.sh" if (ROOT / "run_validator_testnet.sh").exists() else "run_validator.sh"
LAUNCHERS = [p for p in (_VALIDATOR, "run_miner.sh") if (ROOT / p).exists()]


def _package_installs(text):
    return [ln for ln in text.splitlines()
            if re.search(r"\bpip3? install\b.*-e \.(\s|$|\[)", ln) and not ln.lstrip().startswith("#")]


def test_there_is_a_launcher_to_check():
    assert LAUNCHERS, "no launcher found at the repository root"


def test_every_launcher_installs_the_package_against_the_lockfile():
    for name in LAUNCHERS:
        installs = _package_installs((ROOT / name).read_text())
        assert installs, f"{name} does not install the package at all"
        for ln in installs:
            assert "constraints.txt" in ln or "$_CONSTRAINTS" in ln, (
                f"{name} installs the package without the lockfile: {ln.strip()}")


def test_the_lockfile_is_applied_only_when_present():
    for name in LAUNCHERS:
        text = (ROOT / name).read_text()
        assert re.search(r'\[ -f [^\]]*constraints\.txt[^\]]*\]', text), (
            f"{name} must fall back to an unconstrained install where no constraints.txt ships")
