"""The public repository's test workflow exists, is well-formed and runs only there.

This test pins what the workflow runs: pytest on Python 3.10 from requirements.txt under constraints.txt, on pull
requests and on pushes to test and main. Where the release copy script is present, it also checks that the script
copies the workflow.
"""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
CARVE = ROOT / "apply_testnet.sh"


def test_the_carve_names_the_workflow():
    if not CARVE.exists():
        pytest.skip("the carve script is not part of the public tree")
    text = CARVE.read_text()
    assert '"$SRC/.github/workflows/tests.yml"' in text
    assert '"$SRC/.github/allowed_signers"' in text, "the signers copy is the model this one follows"


def test_the_workflow_runs_the_public_suite_only_on_the_public_repository():
    doc = yaml.safe_load(WORKFLOW.read_text())
    on = doc.get(True, doc.get("on"))
    assert "pull_request" in on
    assert on["push"]["branches"] == ["test", "main"]
    job = doc["jobs"]["pytest"]
    assert job["if"] == "github.repository == 'taos-im/sn-79'"
    steps = job["steps"]
    python = next(s for s in steps if str(s.get("uses", "")).startswith("actions/setup-python"))
    assert python["with"]["python-version"] == "3.10"
    runs = "\n".join(s.get("run", "") for s in steps)
    assert "pip install -r requirements.txt -c constraints.txt" in runs
    assert "pytest-asyncio" in runs, "the suite has async tests; without the plugin they fail, not skip"
    assert "python -m pytest -q -p no:cacheprovider tests" in runs
