# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Every simulation config in the tree must parse, or a validator restart is a crash loop.

`MarketSimulationConfig.from_xml` is called from `SimulationEngine.start()`, which runs inside the
validator's `__init__`. An attribute it reads with a bare `attrib['name']` and no default therefore
does not degrade one field: it raises KeyError before the validator exists, pm2 restarts it, and it
raises again. There is no partial-service mode and no log line that says "config".

It has happened: a background-agent recalibration renamed the HFT
attribute `psiHFT_constant="100"` to `psiHFT_quotes="5.8"` in `simulation_0.xml`, and config.py:485
still read the old name:

    validator.py:1083 __init__ -> engine.start() -> _load_config()
      -> MarketSimulationConfig.from_xml(v.xml_config)
        -> config.py:485 float(HFT_config.attrib['psiHFT_constant'])
    KeyError: 'psiHFT_constant'

The validator kept running on its already-parsed copy for ninety minutes and only died when the
chain recycle restarted it at 14:31 -- 54 restarts, 21 logged KeyErrors. A config change that breaks
the parser is invisible until the next restart, which is the worst time to discover it.

Both spellings must load. The rename is only in simulation_0.xml; ten other configs in the same
directory still carry `psiHFT_constant`, including the acceptance config the engine runs, and a
model that accepts only the new name would simply move the crash to those.

`scaleR` is the counter-example and the pattern to copy: config.py:506 and :531 read it as
`attrib['scaleR'] if 'scaleR' in attrib else 0.0`, so it has been absent from simulation_0.xml the
whole time and cost nothing.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pydantic
import pytest

from taos.im.protocol.config import MarketSimulationConfig

_CONFIGS = Path(__file__).resolve().parents[1] / "simulate" / "trading" / "run" / "config"
_PRODUCTION = _CONFIGS / "simulation_0.xml"
_ACCEPTANCE = _CONFIGS / "simulation_0_acceptance.xml"

# The published tree carries `simulation_0.xml` alone. Every case below that needs a SECOND config
# therefore has no subject there, and hardcoding a filename makes it fail on the file being absent
# rather than on anything it asserts.
#
# Absence must not become a silent way for these to stop running in a FULL tree, so the skip is
# conditional on the tree being the reduced one, discriminated by a config only a full tree carries.
# Where the second config is a ROLE rather than one file -- "something still on the old spelling" --
# it is discovered by reading the configs, not named: the exemplar has already moved once, when the
# copy that used to serve as it was aligned to production and picked up the new spelling.
_FULL_TREE = (_CONFIGS / "exchange_0.xml").exists()
_reduced = pytest.mark.skipif(not _FULL_TREE, reason="run/config/ is reduced here to the one config")


def _a_config_on_the_old_spelling() -> Path | None:
    for path in sorted(_CONFIGS.glob("simulation*.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        tags = {el.tag for el in root.iter()}
        # Must be a config a validator can actually be launched with: `_is_validator_config` below
        # says that is the presence of <FundamentalPrice>, and the engine-side fragment in this
        # directory carries the old spelling but no fundamental, so it cannot serve as the exemplar.
        if "FundamentalPrice" not in tags:
            continue
        hft = next((el for el in root.iter() if el.tag == "HighFrequencyTraderAgent"), None)
        if hft is not None and "psiHFT_constant" in hft.attrib:
            return path
    return None


_OLD_SPELLING = _a_config_on_the_old_spelling()


def _load(path: Path):
    return MarketSimulationConfig.from_xml(ET.parse(path).getroot())


def test_the_production_config_parses():
    """The one the validator is launched against: --simulation.xml_config .../simulation_0.xml."""
    assert _PRODUCTION.exists()
    cfg = _load(_PRODUCTION)
    assert cfg is not None


@_reduced
def test_the_acceptance_config_parses():
    """Aligned to production, and the engine runs it."""
    assert _ACCEPTANCE.exists()
    assert _load(_ACCEPTANCE) is not None


def _is_validator_config(path) -> bool:
    """Can a VALIDATOR be launched against this file?

    Not every `simulation*.xml` can. `simulation_0_RS.xml` carries no <FundamentalPrice> at all and
    never has -- `git log -S` shows it arrived complete in one commit, so its absence is the shape of
    the file rather than a regression, and nothing in the tree launches a validator against it. It is
    an engine-side background-agent fragment.

    The discriminator is that element rather than a filename list: a list is the next thing to drift,
    and a config that genuinely LOSES its <FundamentalPrice> should start failing here, which a
    hardcoded exclusion would hide. `_req` raising on it stays correct -- a validator config without
    one is malformed -- and this only says which files are validator configs in the first place.
    """
    try:
        return any(el.tag == "FundamentalPrice" for el in ET.parse(path).getroot().iter())
    except ET.ParseError:
        return True  # unparseable XML is a failure worth reporting, not a file to skip


_VALIDATOR_CONFIGS = [p for p in sorted(_CONFIGS.glob("simulation*.xml")) if _is_validator_config(p)]


def test_the_discriminator_still_selects_the_configs_that_matter():
    """Guards the helper above: if it ever selects nothing, every case below passes vacuously."""
    names = {p.name for p in _VALIDATOR_CONFIGS}
    _launched = {"simulation_0.xml"} | ({"simulation_0_acceptance.xml"} if _FULL_TREE else set())
    assert _launched <= names, (
        f"a config a validator is actually launched with is not being tested: {sorted(names)}"
    )
    # The directory holds 7; exactly one (simulation_0_RS.xml) is excluded. Asserted as "all but at
    # most one" rather than a fixed count, so adding a config does not fail this, but a discriminator
    # that suddenly starts rejecting several does.
    _all = sorted(_CONFIGS.glob("simulation*.xml"))
    assert len(_VALIDATOR_CONFIGS) >= len(_all) - 1, (
        f"the discriminator rejected {len(_all) - len(_VALIDATOR_CONFIGS)} of {len(_all)} configs: "
        f"{sorted(p.name for p in _all if p not in _VALIDATOR_CONFIGS)}"
    )


@pytest.mark.parametrize("path", _VALIDATOR_CONFIGS, ids=lambda p: p.name)
def test_every_simulation_config_in_the_tree_parses(path):
    """Any of these can be handed to a validator; one that cannot parse is a crash loop waiting."""
    try:
        _load(path)
    except KeyError as exc:
        pytest.fail(
            f"{path.name} does not parse: KeyError {exc}. A validator started against this config "
            f"dies in __init__ and pm2 restarts it into the same failure. Read the attribute with a "
            f"default, or accept both spellings, as config.py does for scaleR."
        )


@_reduced
def test_both_psi_spellings_are_accepted():
    """The new name must win where present; the old must still load."""
    assert _OLD_SPELLING is not None, "no config in this tree is still on the old spelling"
    root = ET.parse(_PRODUCTION).getroot()
    hft = next(el for el in root.iter() if el.tag == "HighFrequencyTraderAgent")
    assert "psiHFT_quotes" in hft.attrib, "the production config no longer carries the new spelling"

    old = ET.parse(_OLD_SPELLING).getroot()
    hft_old = next(el for el in old.iter() if el.tag == "HighFrequencyTraderAgent")
    assert "psiHFT_constant" in hft_old.attrib, f"{_OLD_SPELLING.name} no longer carries the old one"

    assert _load(_PRODUCTION).hft_agent_psi == pytest.approx(float(hft.attrib["psiHFT_quotes"]))
    assert _load(_OLD_SPELLING).hft_agent_psi == pytest.approx(
        float(hft_old.attrib["psiHFT_constant"])
    )


def test_no_hft_attribute_is_read_without_a_fallback():
    """The class of bug, not just this instance.

    Any bare attrib[...] on the HFT block is one config edit away from the same crash loop.
    """
    src = (Path(__file__).resolve().parents[1] / "taos" / "im" / "protocol" / "config.py").read_text(
        encoding="utf-8"
    )
    i = src.index("hft_agent_psi")
    block = src[i - 2000:i + 2000]
    bare = [
        ln.strip()
        for ln in block.splitlines()
        if "HFT_config.attrib[" in ln and " in HFT_config.attrib" not in ln and ".get(" not in ln
    ]
    assert not bare, (
        "these HFT attributes are read with no fallback, so removing or renaming any one of them in "
        f"a config makes every validator restart a crash loop: {bare}"
    )


# ── backward compatibility with miners running an older model ────────────────────────────────────
#
# MarketSimulationConfig is a WIRE CONTRACT. The validator serialises it into
# MarketSimulationStateUpdate and third-party miners validate it with whatever version of this class
# they happen to be running. Retyping an existing field is therefore a breaking change to everyone
# who has not upgraded, and the failure is silent and misdirected: the miner rejects the WHOLE
# config, `simulation_config` stays None, and their agent dies on `'NoneType' has no attribute
# 'book_count'` -- a symptom that points nowhere near the config.
#
# Seen exactly that way on both test miners, after the
# validator picked up tauF="1.4" and the miners were still on `sta_agent_tauF: int`.
#
# ADDING a field is safe -- the model ignores unknown keys -- so the exact value goes in a new
# optional field and the original keeps carrying an integer.

class _OldMinerConfig(pydantic.BaseModel):
    """The shape an un-upgraded miner validates against: tauF is an int, and it knows nothing else."""

    model_config = pydantic.ConfigDict(extra="ignore")
    sta_agent_tauF: int


def test_an_old_miner_still_accepts_what_the_validator_sends():
    """The whole point: no forced miner upgrade."""
    sent = _load(_PRODUCTION).model_dump()
    _OldMinerConfig(**sent)  # raises if the wire value is fractional


def test_the_wire_field_stays_an_integer_even_for_a_fractional_config():
    cfg = _load(_PRODUCTION)
    assert isinstance(cfg.sta_agent_tauF, int) and not isinstance(cfg.sta_agent_tauF, bool), (
        "sta_agent_tauF went back to a float on the wire, which every miner on the old model rejects"
    )


def test_the_exact_value_is_still_available():
    """Truncating on the wire must not lose the real figure: a config mirror that lies is its own bug."""
    root = ET.parse(_PRODUCTION).getroot()
    sta = next(el for el in root.iter() if el.tag == "StylizedTraderAgent")
    assert _load(_PRODUCTION).sta_agent_tauF_exact == pytest.approx(float(sta.attrib["tauF"]))


@_reduced
def test_a_config_with_an_integral_tauF_is_unchanged():
    assert _OLD_SPELLING is not None, "no config in this tree is still on the old spelling"
    cfg = _load(_OLD_SPELLING)
    assert cfg.sta_agent_tauF == 1 and cfg.sta_agent_tauF_exact == pytest.approx(1.0)


def test_an_updated_miner_still_accepts_an_OLD_validator_payload():
    """The other direction, and it is a requirement too.

    Miners upgrade before validators do, and a validator that has not been updated sends a payload
    with no `sta_agent_tauF_exact` at all. The new field must therefore be OPTIONAL with a default;
    declaring it required would break every updated miner against every un-updated validator, which
    is the same outage as the forward case and harder to attribute.
    """
    sent_by_old_validator = _load(_PRODUCTION).model_dump()
    sent_by_old_validator.pop("sta_agent_tauF_exact")
    cfg = MarketSimulationConfig(**sent_by_old_validator)
    assert cfg.sta_agent_tauF_exact is None, "the field must default rather than be demanded"
    assert cfg.sta_agent_tauF == int(sent_by_old_validator["sta_agent_tauF"])


def test_the_new_field_is_not_required():
    """Stated against the model itself, so it cannot be made required by a later edit."""
    assert MarketSimulationConfig.model_fields["sta_agent_tauF_exact"].is_required() is False


def test_the_wire_field_is_still_required():
    """The compatibility argument depends on the original field remaining present and integral."""
    f = MarketSimulationConfig.model_fields["sta_agent_tauF"]
    assert f.is_required() is True
    assert f.annotation is int
