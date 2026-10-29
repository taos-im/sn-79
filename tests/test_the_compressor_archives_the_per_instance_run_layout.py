# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A multi-asset run keeps each realization's window logs one level down, in <run>/<class>-<instance>/, and the
compressor read only the run directory itself. A two-class run was therefore never archived and never deleted: the
disk grew, and the uploader, which mirrors <run>/*.zip, had nothing to send. The archive still lands at the run root
under the window's label with members by basename, because book ids are global across realizations, so the uploader
and every window reader keep the layout they already expect. The flat layout of a single-market run is unchanged."""
import zipfile
from types import SimpleNamespace
from unittest import mock

from taos.im.validator.engines.simulation import SimulationEngine

HOUR = 3_600_000_000_000
CLOSED = "00000000-00010000"
OPEN = "00010000-00020000"


def _run(root, name, layout):
    run = root / name
    for sub, files in layout.items():
        d = run / sub if sub else run
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_text(f"{f}\n")
    (run / "config.xml").write_text("<MultiAssetSimulation/>")
    (run / "ckpt" / "000.ckptd").mkdir(parents=True)
    (run / "ckpt" / "000.ckptd" / "common.ckpt").write_bytes(b"")
    return run


def _compress(active_run, now, start=False):
    v = SimpleNamespace(
        simulation=SimpleNamespace(logDir=str(active_run.resolve())),
        simulation_timestamp=now,
        compressing=False,
        pagerduty_alert=lambda *a, **k: None,
    )
    # the disk cleanup after compression deletes old run directories while the real disk is over 85 per cent full,
    # this test's own run among them, so the disk is held under the threshold
    with mock.patch("psutil.disk_usage", return_value=SimpleNamespace(percent=50.0)):
        SimulationEngine._compress_outputs(SimpleNamespace(validator=v), start=start)
    assert v.compressing is False


def _members(archive):
    with zipfile.ZipFile(archive) as zf:
        return sorted(zf.namelist())


def test_a_closed_window_of_a_per_instance_run_is_archived_at_the_run_root(tmp_path):
    run = _run(tmp_path, "20261008_120000", {
        "simulation_0-0": [f"L3-0.{CLOSED}.log", f"L2-0.{CLOSED}.log", f"fees-0.{CLOSED}.log", f"L3-0.{OPEN}.log"],
        "simulation_0-1": [f"L3-16.{CLOSED}.log"],
        "simulation_1-0": [f"L3-32.{CLOSED}.log", f"L3-32.{OPEN}.log", "fundamental.32.csv"],
    })
    _compress(run, now=HOUR + 1)
    assert _members(run / f"L3_{CLOSED}.zip") == [f"L3-0.{CLOSED}.log", f"L3-16.{CLOSED}.log", f"L3-32.{CLOSED}.log"]
    assert _members(run / f"L2_{CLOSED}.zip") == [f"L2-0.{CLOSED}.log"]
    assert _members(run / f"fees_{CLOSED}.zip") == [f"fees-0.{CLOSED}.log"]
    for sub, name in (("simulation_0-0", f"L3-0.{CLOSED}.log"), ("simulation_0-1", f"L3-16.{CLOSED}.log"),
                      ("simulation_1-0", f"L3-32.{CLOSED}.log")):
        assert not (run / sub / name).exists(), "an archived source is removed, as in the flat layout"
    assert (run / "simulation_0-0" / f"L3-0.{OPEN}.log").exists(), "the open window stays as written"
    assert (run / "simulation_1-0" / f"L3-32.{OPEN}.log").exists()
    assert not (run / f"L3_{OPEN}.zip").exists()
    assert (run / "simulation_1-0" / "fundamental.32.csv").exists(), "only window logs are archived"
    assert sorted(p.name for p in run.glob("*/*.zip")) == [], "the archive lands at the run root, where the uploader looks"


def test_the_flat_layout_archives_exactly_as_before(tmp_path):
    run = _run(tmp_path, "20261008_120000", {"": [f"L3-0.{CLOSED}.log", f"L3-1.{CLOSED}.log", f"L3-0.{OPEN}.log"]})
    _compress(run, now=HOUR + 1)
    assert _members(run / f"L3_{CLOSED}.zip") == [f"L3-0.{CLOSED}.log", f"L3-1.{CLOSED}.log"]
    assert not (run / f"L3-0.{CLOSED}.log").exists() and (run / f"L3-0.{OPEN}.log").exists()


def test_at_start_a_superseded_per_instance_run_is_archived_whole(tmp_path):
    old = _run(tmp_path, "20261006_160323", {"simulation_0-0": [f"L3-0.{CLOSED}.log", f"L3-0.{OPEN}.log"]})
    active = _run(tmp_path, "20261008_120000", {"simulation_0-0": [f"L3-0.{OPEN}.log"]})
    _compress(active, now=HOUR + 1, start=True)
    assert _members(old / f"L3_{CLOSED}.zip") == [f"L3-0.{CLOSED}.log"]
    assert _members(old / f"L3_{OPEN}.zip") == [f"L3-0.{OPEN}.log"], "a superseded run's open window closes with it"
    assert (active / "simulation_0-0" / f"L3-0.{OPEN}.log").exists(), "the active run's open window stays"
