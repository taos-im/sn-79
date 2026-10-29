"""The validator scores the class layout of the run it is on, not the layout of the config it was started with.

The 0.6.5 validator resumed mainnet's 0.6.4 run (six simulation_0 and two simulation_1, books 0-95 and
96-127) under 0.6.5's three-class default. It took its layout from the start config (engines/simulation.py
`_load_config`) and never from the run, so it scored and published 0-63 / 64-95 / 96-127, advertised the third class's
grid and minimum order for books 96-127, and started its simulation state fresh, because that file is named after the
layout. It was relaunched by hand on a copy of the run's layout and crossed over at the run end through a one-shot
watcher.

Now: at start the validator reads the layout of the run it resumes from that run's own config.xml; when a run opens it
reads the new run's; and the engine's next run still starts from the start config, the release default. A layout
whose books differ from the loaded ones is adopted as well: the book ids follow it, and at the run
open the books whose market changed start their histories fresh and the vanished ones are dropped.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from taos.im.validator.engines.simulation import SimulationEngine
from taos.im.validator.reward import class_map

CONFIGS = Path(__file__).resolve().parents[1] / "simulate" / "trading" / "run" / "config"
NEEDED = ["multiasset_simulation_0.xml", "simulation_0.xml", "simulation_1.xml", "simulation_2.xml"]
pytestmark = pytest.mark.skipif(not all((CONFIGS / n).exists() for n in NEEDED), reason="the class files are not here")

OLD_RUN = "20261019_102513"


def _wrapper(backgrounds, config_dir):
    """A run's config.xml as the engine writes it: absolute background paths."""
    lines = "\n".join(
        f'    <Background name="{name}" path="{config_dir / (name + ".xml")}" instanceCount="{n}"/>'
        for name, n in backgrounds)
    return (
        '<?xml version="1.0"?>\n<MultiAssetSimulation traceTime="1" useMessagePack="1" host="localhost" port="8000" '
        'generalMsgEndpoint="/account" threads="8" start="0" step="1000000000" duration="86400000000000" '
        f'gracePeriod="600000000000" ckptIntervalInSteps="600">\n{lines}\n</MultiAssetSimulation>\n')


def _run_dir(root, name, backgrounds, config_dir, checkpoint=True):
    d = root / "simulate" / "trading" / "run" / "logs" / name
    d.mkdir(parents=True)
    (d / "config.xml").write_text(_wrapper(backgrounds, config_dir))
    if checkpoint:
        (d / "ckpt" / "1000000000000.ckptd").mkdir(parents=True)
        (d / "ckpt" / "1000000000000.ckptd" / "common.ckpt").write_text("")
    return d


@pytest.fixture
def tree(tmp_path):
    config_dir = tmp_path / "simulate" / "trading" / "run" / "config"
    config_dir.mkdir(parents=True)
    for n in NEEDED:
        shutil.copy(CONFIGS / n, config_dir / n)
    state = tmp_path / "state"
    state.mkdir()
    (state / f"validator.simulation.localnet.{OLD_RUN[:13]}.mp").write_bytes(b"")
    v = SimpleNamespace(
        config=SimpleNamespace(
            simulation=SimpleNamespace(xml_config=str(config_dir / "multiasset_simulation_0.xml")),
            neuron=SimpleNamespace(full_path=str(state)),
            subtensor=SimpleNamespace(network="local"),
        ),
        repo_path=tmp_path,
    )
    engine = SimulationEngine(None, v)
    engine._load_config()
    return SimpleNamespace(root=tmp_path, config_dir=config_dir, v=v, engine=engine)


def _composition(sim):
    return [(bg.name, bg.instance_count) for bg in sim.backgrounds]


def test_the_start_config_is_the_three_class_default(tree):
    assert _composition(tree.v.simulation) == [("simulation_0", 4), ("simulation_1", 2), ("simulation_2", 2)]


def test_a_validator_started_on_a_resumed_run_scores_that_runs_layout(tree):
    default_file = tree.v.simulator_config_file
    default_label = tree.v.simulation.label()
    _run_dir(tree.root, OLD_RUN, [("simulation_0", 6), ("simulation_1", 2)], tree.config_dir)
    assert tree.engine._follow_resumed_run_layout() is True
    assert _composition(tree.v.simulation) == [("simulation_0", 6), ("simulation_1", 2)]
    cmap = class_map(tree.v)
    assert {b for b, c in cmap.items() if c == 0} == set(range(96))
    assert {b for b, c in cmap.items() if c == 1} == set(range(96, 128))
    assert tree.v.simulator_config_file == default_file, "the next run starts from the release default"
    assert tree.v.simulation_state_file.endswith(f"/{tree.v.simulation.label()}.mp")
    assert tree.v.simulation.label() != default_label, "the simulation state is the run's, named by its layout"
    assert tree.engine._book_ids == list(range(128))


def test_a_run_that_opens_on_the_default_layout_crosses_over(tree):
    _run_dir(tree.root, OLD_RUN, [("simulation_0", 6), ("simulation_1", 2)], tree.config_dir)
    tree.engine._follow_resumed_run_layout()
    new = _run_dir(tree.root, "20261024_031500", [("simulation_0", 4), ("simulation_1", 2), ("simulation_2", 2)],
                   tree.config_dir)
    reinits = []
    tree.v._scoring_shadow = SimpleNamespace(request_reinit=lambda: reinits.append(1))
    before = tree.v.simulation.label()
    tree.engine._load_config()
    assert tree.engine._adopt_run_layout(new / "simulation_0-0", "the run opened", previous_label=before) is False, (
        "the new run's layout is the start config's: nothing to adopt")
    assert _composition(tree.v.simulation) == [("simulation_0", 4), ("simulation_1", 2), ("simulation_2", 2)]
    assert reinits == [1], "the layout changed across the open, so the scoring service is re-initialised once"


def test_a_class_removed_across_a_resume_is_followed(tree):
    _run_dir(tree.root, OLD_RUN, [("simulation_0", 8)], tree.config_dir)
    assert tree.engine._follow_resumed_run_layout() is True
    assert _composition(tree.v.simulation) == [("simulation_0", 8)]
    assert set(class_map(tree.v).values()) == {0}


def test_a_layout_on_other_books_is_adopted_and_the_book_ids_follow(tree):
    _run_dir(tree.root, OLD_RUN, [("simulation_0", 6)], tree.config_dir)
    assert tree.engine._follow_resumed_run_layout() is True
    assert _composition(tree.v.simulation) == [("simulation_0", 6)]
    assert tree.engine._book_ids == list(range(96)), "the book-keyed structures follow the adopted layout's ids"


def test_a_newer_run_than_the_saved_state_leaves_the_layout_to_the_run_change(tree):
    default = _composition(tree.v.simulation)
    _run_dir(tree.root, OLD_RUN, [("simulation_0", 6), ("simulation_1", 2)], tree.config_dir)
    _run_dir(tree.root, "20261024_031500", [("simulation_0", 6), ("simulation_1", 2)], tree.config_dir)
    assert tree.engine._follow_resumed_run_layout() is False, (
        "the engine resumes the newest run, which this validator has not scored: its run change sets the layout")
    assert _composition(tree.v.simulation) == default


def test_no_run_directory_keeps_the_start_config(tree):
    default = _composition(tree.v.simulation)
    assert tree.engine._follow_resumed_run_layout() is False
    assert _composition(tree.v.simulation) == default


def test_start_and_every_run_open_and_the_first_state_follow_the_run():
    src = (Path(__file__).resolve().parents[1] / "taos" / "im" / "validator" / "engines" / "simulation.py").read_text()
    start = src.index("    def start(self)")
    assert src.index("self._follow_resumed_run_layout()", start) < src.index("self.initialize_structures()", start), (
        "the run's layout is adopted before any structure is built on it")
    op = src.index("    def _open_run(")
    assert src.index("self._adopt_run_layout(", op) < src.index("shift_simulation_histories(", op)
    tick = src.index("    def on_tick(")
    assert "self._adopt_run_layout(state.logDir" in src[tick:src.index("    def ", tick + 10)]
