"""Tests for the mission profile registry and :class:`MissionProfile`."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from metadata_asf.profiles import MISSIONS, UnknownMissionError, get_profile
from metadata_asf.profiles.base import MissionProfile


@pytest.fixture(name="clear_registry")
def fixture_clear_registry() -> Generator[None, None, None]:
    """Snapshot and restore the global registry so tests stay independent."""
    saved = dict(MISSIONS)
    yield
    MISSIONS.clear()
    MISSIONS.update(saved)


def test_profile_defaults() -> None:
    """A profile with minimal arguments keeps documented defaults."""
    prof = MissionProfile(name="NISAR", asf_dataset="NISAR", supported_products=[("L1", "RSLC")])
    assert prof.name == "NISAR"
    assert prof.asf_dataset == "NISAR"
    assert prof.supported_products == [("L1", "RSLC")]
    assert prof.default_ocean_wkt is None
    assert prof.field_mapping == {}


def test_profile_copies_supported_products() -> None:
    """Callers can mutate the list after construction without side effects."""
    pairs = [("L1", "RSLC"), ("L2", "GSLC")]
    prof = MissionProfile(name="NISAR", asf_dataset="NISAR", supported_products=pairs)
    pairs.append(("bogus", "X"))
    assert len(prof.supported_products) == 2


def test_profile_field_mapping_provided() -> None:
    """An explicit field mapping is stored, one source property list per column."""
    prof = MissionProfile(
        name="NISAR",
        asf_dataset="NISAR",
        supported_products=[("L1", "RSLC")],
        field_mapping={"platform": ["producer"], "polarization": ["p1", "p2"]},
    )
    assert prof.field_mapping == {"platform": ["producer"], "polarization": ["p1", "p2"]}


def test_profile_field_mapping_copies_lists() -> None:
    """Callers can mutate the source lists after construction without side effects."""
    props: list[str] = ["p1"]
    prof = MissionProfile(
        name="NISAR",
        asf_dataset="NISAR",
        supported_products=[("L1", "RSLC")],
        field_mapping={"polarization": props},
    )
    props.append("p2")
    assert prof.field_mapping["polarization"] == ["p1"]


def test_registry_contains_nisar() -> None:
    """NISAR is the first registered mission (AGENTS.md sections 2-3)."""
    assert "NISAR" in MISSIONS


def test_get_profile_known(clear_registry: None) -> None:
    """A registered name resolves back to its profile object."""
    prof = MissionProfile(name="NISAR", asf_dataset="NISAR", supported_products=[])
    MISSIONS["NISAR"] = prof
    assert get_profile("NISAR") is prof


def test_get_profile_unknown_lists_available(clear_registry: None) -> None:
    """Unknown mission raises with the list of available missions in the message."""
    MISSIONS["SENTINEL-1"] = MissionProfile(
        name="SENTINEL-1", asf_dataset="SENTINEL-1", supported_products=[]
    )
    with pytest.raises(UnknownMissionError) as excinfo:
        get_profile("ALOS-2")
    assert "Unknown mission 'ALOS-2'" in str(excinfo.value)
    assert "NISAR" in str(excinfo.value)
    assert "SENTINEL-1" in str(excinfo.value)
    assert set(excinfo.value.available) == {"NISAR", "SENTINEL-1"}


def test_get_profile_unknown_empty_registry(clear_registry: None) -> None:
    """Error message stays informative even when no profile exists yet."""
    MISSIONS.clear()  # fixture restores the snapshot afterwards.
    with pytest.raises(UnknownMissionError) as excinfo:
        get_profile("NISAR")
    # KeyError's str() re-quotes its payload, so assert on substrings rather than a regex.
    assert "Unknown mission 'NISAR'" in str(excinfo.value)
    assert "Available missions:" in str(excinfo.value)
    assert excinfo.value.available == []
