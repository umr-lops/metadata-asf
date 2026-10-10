"""Tests for the mission profile registry and :class:`MissionProfile`."""

from __future__ import annotations

from collections.abc import Callable, Generator

import pytest

from metadata_asf.profiles import MISSIONS, UnknownMissionError, get_profile
from metadata_asf.profiles.base import MissionProfile
from metadata_asf.profiles.nisar import nisar_profile

# A full, correctly-shaped NISAR RSLC product name (18 underscore-separated tokens).
_FULL_NISAR_NAME = (
    "NISAR_L1_PR_RSLC_010_098_D_052_4005_SHNA_A_" "20260115T235949_20260116T000025_X05010_N_P_J_001"
)


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


class TestNisarBeamModeDecoder:
    """The NISAR profile decodes its ``beam_mode`` from the product file name."""

    def _decode(self) -> Callable[[str], str | None]:
        decode = nisar_profile.decode_beam_mode
        assert decode is not None
        return decode

    def test_profile_exposes_decoder(self) -> None:
        assert nisar_profile.decode_beam_mode is not None
        assert MISSIONS["NISAR"].decode_beam_mode is not None

    def test_decodes_bandwidth_and_polarization(self) -> None:
        assert self._decode()(_FULL_NISAR_NAME) == "40 MHz, single-pol HH"

    def test_pole_variants(self) -> None:
        decode = self._decode()
        base = (
            "NISAR_L1_PR_RSLC_010_098_D_052_2005_{pole}_A_"
            "20260115T235949_20260116T000025_X05010_N_P_J_001"
        )
        assert decode(base.format(pole="DHDH")) == "20 MHz, dual-pol HH/HV"
        assert decode(base.format(pole="NASV")) == "20 MHz, single-pol VV"
        assert decode(base.format(pole="QPDH")) == "20 MHz, quad-pol HH/HV/VV/VH"
        # S-band primary: the first half is NA, so the S-band half (token 9) is used.
        assert decode(base.format(pole="NAQP")) == "20 MHz, quad-pol HH/HV/VV/VH"

    def test_bandwidth_variants(self) -> None:
        decode = self._decode()
        base = (
            "NISAR_L1_PR_RSLC_010_098_D_052_{mode}_SHNA_A_"
            "20260115T235949_20260116T000025_X05010_N_P_J_001"
        )
        for mode, prefix in (("7700", "77 MHz"), ("2005", "20 MHz"), ("0005", "5 MHz")):
            result = decode(base.format(mode=mode))
            assert result is not None and result.startswith(prefix)

    def test_extension_is_stripped(self) -> None:
        assert self._decode()(_FULL_NISAR_NAME + ".h5") == "40 MHz, single-pol HH"

    @pytest.mark.parametrize("bad", ["not a nisar", "", "NISAR_L1_RSLC", "SENTINEL-1_1_PR_RSLC"])
    def test_non_nisar_names_yield_none(self, bad: str) -> None:
        assert self._decode()(bad) is None
