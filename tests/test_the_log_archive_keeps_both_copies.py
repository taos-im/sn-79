# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Two episodes can own the same log file name, and neither may be lost.

Block ranges restart with every new simulation, so after a restart a window like
`L3-19.00010000-00020000.log` exists in two episode directories at once. The archiver collected
files across directories, wrote each under its BASENAME, reopened the archive in append mode, and
removed the source afterwards. Both copies were therefore stored under one name and both sources
deleted -- and a reader fetching that name gets one of them, with the other unrecoverable.

Disambiguation happens only on collision, so an archive that never had one keeps the exact layout
its readers already expect. Nothing about the common case changes.
"""

import zipfile
from pathlib import Path


def _write_archive(tmp_path, sources):
    """Mirror the archiver's loop: append mode, basename, disambiguate on collision."""
    archive = tmp_path / "L3_00010000-00020000.zip"
    with zipfile.ZipFile(archive, "w" if not archive.exists() else "a", compression=zipfile.ZIP_DEFLATED) as zipf:
        taken = set(zipf.namelist())
        for src in sources:
            arc = Path(src).name
            if arc in taken:
                arc = f"{Path(src).parent.name}__{arc}"
            taken.add(arc)
            zipf.write(src, arcname=arc)
    return archive


def _episode(tmp_path, name, content):
    d = tmp_path / name
    d.mkdir()
    f = d / "L3-19.00010000-00020000.log"
    f.write_text(content)
    return f


def test_both_episodes_survive_the_same_name(tmp_path):
    a = _episode(tmp_path, "20260921_085739", "first episode")
    b = _episode(tmp_path, "20260921_172544", "second episode")
    archive = _write_archive(tmp_path, [a, b])
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        bodies = {z.read(n).decode() for n in names}
    assert len(names) == len(set(names)), f"the archive holds duplicate entry names: {names}"
    assert bodies == {"first episode", "second episode"}, "an episode's content was lost"


def test_the_uncollided_case_is_unchanged(tmp_path):
    """Readers fetch by basename; an archive without a collision must look exactly as before."""
    a = _episode(tmp_path, "20260921_085739", "only episode")
    archive = _write_archive(tmp_path, [a])
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ["L3-19.00010000-00020000.log"], "a non-colliding entry was renamed"


def test_the_disambiguated_name_says_which_episode(tmp_path):
    a = _episode(tmp_path, "20260921_085739", "first")
    b = _episode(tmp_path, "20260921_172544", "second")
    archive = _write_archive(tmp_path, [a, b])
    with zipfile.ZipFile(archive) as z:
        extra = [n for n in z.namelist() if n != "L3-19.00010000-00020000.log"]
    assert extra and extra[0].startswith("20260921_172544__"), (
        "the second copy must name the episode it came from, or it cannot be attributed"
    )
