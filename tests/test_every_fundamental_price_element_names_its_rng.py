"""Every FundamentalPrice element in a tracked XML file carries ownRng explicitly.

The engine's default decides whether the fundamental draws from a private per-book stream (1) or reseeds
the shared simulation RNG every seed step (0). A class file without the attribute would take the default
silently, whichever it is. Every file in the tree therefore says which it means, and this test keeps it so.
"""
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


SKIPPED_DIRS = {".git", "vcpkg"}


def _tracked_xml(root=ROOT):
    # Outside a git checkout of this tree, such as an unpacked release archive, or without git, every XML file under
    # the tree is checked instead.
    if (Path(root) / ".git").exists():
        try:
            out = subprocess.run(["git", "ls-files", "*.xml", "**/*.xml"], cwd=root, capture_output=True, text=True,
                                 check=True)
            return sorted(set(out.stdout.split()))
        except (OSError, subprocess.CalledProcessError):
            pass
    found = []
    for path, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIPPED_DIRS and not d.startswith("build")]
        found.extend(Path(path, f).relative_to(root).as_posix() for f in files if f.endswith(".xml"))
    return sorted(found)


def _is_record(rel, root=ROOT):
    # A release's calibration records are kept exactly as their owners delivered them, each named in the SOURCES.tsv
    # of its calibration directory: a record, not a configuration any launcher loads, so it is not held to the rule.
    path = root / rel
    sources = path.parent / "SOURCES.tsv"
    if path.parent.name != "calibration" or not sources.exists():
        return False
    return path.name in {line.split("\t", 1)[0] for line in sources.read_text().splitlines()}


def _fundamental_price_files():
    found = []
    for rel in _tracked_xml():
        path = ROOT / rel
        if _is_record(rel) or not path.exists() or "<FundamentalPrice" not in path.read_text(errors="replace"):
            continue
        found.append(rel)
    return found


FILES = _fundamental_price_files()


def test_only_a_file_its_calibration_record_names_is_left_out(tmp_path):
    cal = tmp_path / "release" / "calibration"
    cal.mkdir(parents=True)
    (cal / "SOURCES.tsv").write_text("file\twhat it records\tsource\ndelivered.xml\tthe trial\tits owner\n")
    (cal / "unnamed.xml").write_text("<FundamentalPrice/>")
    (tmp_path / "release" / "delivered.xml").write_text("<FundamentalPrice/>")
    assert _is_record("release/calibration/delivered.xml", tmp_path)
    assert not _is_record("release/calibration/unnamed.xml", tmp_path), "a file its SOURCES.tsv does not name"
    assert not _is_record("release/delivered.xml", tmp_path), "a file outside a calibration directory"


def test_outside_a_git_checkout_every_xml_file_in_the_tree_is_checked(tmp_path):
    (tmp_path / "run" / "config").mkdir(parents=True)
    (tmp_path / "run" / "config" / "a.xml").write_text("<FundamentalPrice ownRng=\"1\"/>")
    (tmp_path / "b.xml").write_text("<Simulation/>")
    (tmp_path / "notes.txt").write_text("not xml")
    for skipped in (".git", "build", "simulate/trading/build", "vcpkg"):
        (tmp_path / skipped).mkdir(parents=True)
        (tmp_path / skipped / "c.xml").write_text("<FundamentalPrice/>")
    assert _tracked_xml(tmp_path) == ["b.xml", "run/config/a.xml"]


TAG = re.compile(r"<FundamentalPrice\b[^>]*>")
OWN_RNG = re.compile(r"""\sownRng\s*=\s*["'][01]["']""")


@pytest.mark.skipif(not FILES, reason="no tracked XML with a FundamentalPrice element")
@pytest.mark.parametrize("rel", FILES)
def test_the_element_says_which_rng_it_draws_from(rel):
    # The opening tags are read as text: the engine's parser (pugixml) accepts a few files, such as the
    # integration bench, that the standard library's parser rejects over a "--" inside a comment.
    tags = TAG.findall((ROOT / rel).read_text(errors="replace"))
    assert tags, f"{rel}: no FundamentalPrice opening tag found"
    missing = [tag for tag in tags if not OWN_RNG.search(tag)]
    assert not missing, f"{rel}: {len(missing)} FundamentalPrice element(s) without ownRng=\"0\" or \"1\": {missing[0][:120]}"


def test_the_shipped_classes_draw_from_their_own_stream():
    cfg = ROOT / "simulate/trading/run/config"
    for name in ("simulation_0.xml", "simulation_1.xml", "simulation_2.xml"):
        root = ET.parse(cfg / name).getroot()
        assert [e.get("ownRng") for e in root.iter("FundamentalPrice")] == ["1"], name
