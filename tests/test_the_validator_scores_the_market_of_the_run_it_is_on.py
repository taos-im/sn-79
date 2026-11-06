# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A class's market is its whole class file, and the validator scores the market of the run it is on.

The class files a run reads are the ones its own config.xml names. A run resumed on class files whose market differs
from the start config's, with the same composition, is scored on the run's files: its classes, and so the metrics and
the next comparison, are the run's. A resume of that run keeps every book. The next run opens on the start config's
files, so the classes whose market changed start their books fresh and the class handover carries their standing.

Any parameter of a class file counts toward its market, including the ones the validator does not parse. Whitespace,
comments and the run-level attributes of the file's root, which the multi-asset wrapper governs, do not.
"""

import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

from taos.im.protocol.config import MultiAssetSimulationConfig
from taos.im.validator.engines.simulation import SimulationEngine
from taos.im.validator.trade import layout_book_changes

CONFIGS = Path(__file__).resolve().parents[1] / "simulate" / "trading" / "run" / "config"
CLASSES = ["simulation_0", "simulation_1", "simulation_2"]
# the 64/32/32 start config, kept beside the published default
START = "multiasset_simulation_0_64_32_32.xml"
NEEDED = [START] + [f"{c}.xml" for c in CLASSES]
pytestmark = pytest.mark.skipif(not all((CONFIGS / n).exists() for n in NEEDED), reason="the class files are not here")

RUN = "20000101_000000"
NEW_RUN = "20000102_000000"  # a later run: run names sort by their start stamp
COMPOSITION = [("simulation_0", 4), ("simulation_1", 2), ("simulation_2", 2)]


def _wrapper(paths):
    lines = "\n".join(f'    <Background name="{name}" path="{paths[name]}" instanceCount="{n}"/>'
                      for name, n in COMPOSITION)
    return (
        '<?xml version="1.0"?>\n<MultiAssetSimulation traceTime="1" useMessagePack="1" host="localhost" port="8000" '
        'generalMsgEndpoint="/account" threads="8" start="0" step="1000000000" duration="86400000000000" '
        f'gracePeriod="600000000000" ckptIntervalInSteps="600">\n{lines}\n</MultiAssetSimulation>\n')


def _edit(path, pattern, repl):
    text = path.read_text()
    new, n = re.subn(pattern, repl, text, count=1)
    assert n == 1, f"{pattern} not in {path.name}"
    path.write_text(new)


def _class_copies(dst, change=None):
    """Copies of the three class files in dst; change maps a class to its (pattern, replacement) edit."""
    dst.mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        shutil.copy(CONFIGS / f"{c}.xml", dst / f"{c}.xml")
        if change and c in change:
            _edit(dst / f"{c}.xml", *change[c])
    return {c: dst / f"{c}.xml" for c in CLASSES}


def _layout(root, paths):
    wrapper = root / "config.xml"
    wrapper.write_text(_wrapper(paths))
    return MultiAssetSimulationConfig.from_multiasset_xml(ET.parse(wrapper).getroot(), root)


def _changed_classes(old, new):
    books = layout_book_changes(old, new)["books"]
    return {cls.name for cls in new.asset_classes() if any(books[b] != "keep" for b in cls.books)}


@pytest.fixture
def tree(tmp_path):
    config_dir = tmp_path / "simulate" / "trading" / "run" / "config"
    config_dir.mkdir(parents=True)
    for n in NEEDED:
        shutil.copy(CONFIGS / n, config_dir / n)
    state = tmp_path / "state"
    state.mkdir()
    (state / f"validator.simulation.localnet.{RUN[:13]}.mp").write_bytes(b"")
    v = SimpleNamespace(
        config=SimpleNamespace(
            simulation=SimpleNamespace(xml_config=str(config_dir / START)),
            neuron=SimpleNamespace(full_path=str(state)),
            subtensor=SimpleNamespace(network="local"),
        ),
        repo_path=tmp_path,
    )
    engine = SimulationEngine(None, v)
    engine._load_config()
    return SimpleNamespace(root=tmp_path, config_dir=config_dir, v=v, engine=engine)


def _run_dir(tree, name, paths):
    d = tree.root / "simulate" / "trading" / "run" / "logs" / name
    d.mkdir(parents=True)
    (d / "config.xml").write_text(_wrapper(paths))
    (d / "ckpt" / "1000000000000.ckptd").mkdir(parents=True)
    (d / "ckpt" / "1000000000000.ckptd" / "common.ckpt").write_text("")
    return d


@pytest.mark.parametrize("cls, pattern, repl", [
    ("simulation_2", r'(<MagneticField [^>]*?alpha=")[^"]*', r"\g<1>99.9"),
    ("simulation_1", r'(Hurst=")[^"]*', r"\g<1>0.6"),
])
def test_a_parameter_the_validator_does_not_parse_changes_the_market(tmp_path, cls, pattern, repl):
    old = _layout(tmp_path / "a", _class_copies(tmp_path / "a"))
    new = _layout(tmp_path / "b", _class_copies(tmp_path / "b", {cls: (pattern, repl)}))
    assert _changed_classes(old, new) == {cls}


def test_whitespace_comments_and_the_roots_run_attributes_are_not_a_market_change(tmp_path):
    old = _layout(tmp_path / "a", _class_copies(tmp_path / "a"))
    change = {
        "simulation_0": (r"<Simulation ", "<!-- a note -->\n<Simulation   "),
        "simulation_1": (r'(<Simulation [^>]*?duration=")[^"]*', r"\g<1>3600000000000"),
        "simulation_2": (r"(<MagneticField )", "\n\n    \\g<1>"),
    }
    new = _layout(tmp_path / "b", _class_copies(tmp_path / "b", change))
    assert _changed_classes(old, new) == set()


def test_a_resumed_run_on_other_class_files_is_scored_on_them(tree):
    frozen = _class_copies(tree.root / "frozen", {"simulation_2": (r'( sigma=")0\.0125', r"\g<1>0.0200")})
    run = _run_dir(tree, RUN, frozen)
    default_label = tree.v.simulation.label()
    assert tree.engine._follow_resumed_run_layout() is True
    assert tree.v.simulation.label() == default_label, "the same composition"
    assert [bg.config.fp_sigma for bg in tree.v.simulation.backgrounds][2] == 0.0200, "the run's class file"
    assert {bg.path for bg in tree.v.simulation.backgrounds} == {str(p) for p in frozen.values()}
    assert tree.v.simulator_config_file.endswith(START), "the next run starts from the start config"
    assert run.exists()


def test_a_resume_of_that_run_keeps_every_book(tree):
    frozen = _class_copies(tree.root / "frozen", {"simulation_2": (r'( sigma=")0\.0125', r"\g<1>0.0200")})
    run = _run_dir(tree, RUN, frozen)
    tree.engine._follow_resumed_run_layout()
    before = tree.v.simulation
    tree.engine._load_config()
    tree.engine._adopt_run_layout(run / "simulation_0-0", "the first state of the run")
    assert _changed_classes(before, tree.v.simulation) == set()


def test_the_next_run_on_the_start_configs_files_starts_the_changed_class_fresh(tree):
    frozen = _class_copies(tree.root / "frozen", {"simulation_2": (r'( sigma=")0\.0125', r"\g<1>0.0200")})
    _run_dir(tree, RUN, frozen)
    tree.engine._follow_resumed_run_layout()
    before = tree.v.simulation
    new = _run_dir(tree, NEW_RUN, {c: tree.config_dir / f"{c}.xml" for c in CLASSES})
    tree.engine._load_config()
    assert tree.engine._adopt_run_layout(new / "simulation_0-0", "the run opened") is False, (
        "the new run is on the start config's files")
    assert _changed_classes(before, tree.v.simulation) == {"simulation_2"}
