"""Tests for :func:`metadata_asf.extract.to_dataframe` with fake ASF products."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from metadata_asf import extract
from metadata_asf.profiles import UnknownMissionError


class FakeProduct:
    """Minimal stand-in for ``asf.ASFProduct``: ``properties`` dict + ``geometry``."""

    def __init__(self, properties: dict[str, object] | None, geometry: object = None) -> None:
        self.properties = properties
        self.geometry = geometry
        self.sceneName: str | None = None


def _footprint() -> dict[str, object]:
    return {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
    }


def _full_rslc() -> FakeProduct:
    return FakeProduct(
        {
            "sceneName": "NISAR_L1_PR_RSLC_0005",
            "platform": "NISAR",
            "startTime": "2026-10-07T21:16:36Z",
            "stopTime": "2026-10-07T21:16:41Z",
            "mainBandPolarization": ["HH", "HV"],
            "sideBandPolarization": ["HH", "HV"],
            "processingLevel": "RSLC",
            "beamModeType": None,
        },
        geometry=_footprint(),
    )


def test_columns_and_order() -> None:
    df = extract.to_dataframe([_full_rslc()], mission="NISAR")
    assert list(df.columns) == list(extract.COLUMNS)
    assert len(df) == 1


def test_full_rslc_row() -> None:
    df = extract.to_dataframe([_full_rslc()], mission="NISAR")
    row = df.iloc[0]
    assert row["granule_id"] == "NISAR_L1_PR_RSLC_0005"
    assert row["platform"] == "NISAR"
    assert row["geometry"].startswith("POLYGON")
    assert row["start_time"] == pd.Timestamp("2026-10-07T21:16:36", tz="UTC")
    assert row["stop_time"] == pd.Timestamp("2026-10-07T21:16:41", tz="UTC")
    assert row["polarization"] == ["HH", "HV"]
    assert row["product_type"] == "RSLC"
    assert row["processing_level"] == "L1"
    assert pd.isna(row["beam_mode"])


def test_datetime_columns_are_utc() -> None:
    df = extract.to_dataframe([_full_rslc()], mission="NISAR")
    for column in ("start_time", "stop_time"):
        assert isinstance(df[column].dtype, pd.DatetimeTZDtype)
        assert df[column].dt.tz is not None
        assert str(df[column].dtype).endswith("UTC]")


def test_polarization_merged_and_deduplicated() -> None:
    product = FakeProduct(
        {
            "sceneName": "S",
            "platform": "NISAR",
            "startTime": "2026-10-07T21:16:36Z",
            "stopTime": "2026-10-07T21:16:41Z",
            "mainBandPolarization": "HH",
            "sideBandPolarization": ["hv", "HH"],
            "processingLevel": "RSLC",
        },
        geometry=_footprint(),
    )
    df = extract.to_dataframe([product], mission="NISAR")
    assert df.iloc[0]["polarization"] == ["HH", "HV"]


def test_gslc_derives_l2() -> None:
    product = _full_rslc()
    assert product.properties is not None
    product.properties = {**product.properties, "processingLevel": "GSLC", "sceneName": "G"}
    df = extract.to_dataframe([product], mission="NISAR")
    row = df.iloc[0]
    assert row["product_type"] == "GSLC"
    assert row["processing_level"] == "L2"


def test_beam_mode_decoded_from_nisar_filename() -> None:
    product = _full_rslc()
    assert product.properties is not None
    product.properties = {
        **product.properties,
        "sceneName": (
            "NISAR_L1_PR_RSLC_010_098_D_052_4005_DHDH_A_"
            "20260115T235949_20260116T000025_X05010_N_P_J_001"
        ),
    }
    df = extract.to_dataframe([product], mission="NISAR")
    assert df.iloc[0]["beam_mode"] == "40 MHz, dual-pol HH/HV"


def test_multipolygon_footprint() -> None:
    product = _full_rslc()
    product.geometry = {
        "type": "MultiPolygon",
        "coordinates": [[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]],
    }
    df = extract.to_dataframe([product], mission="NISAR")
    assert df.iloc[0]["geometry"].startswith("MULTIPOLYGON")


def test_missing_fields_leave_empty_cells_and_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    product = FakeProduct(
        {"sceneName": "S", "platform": "NISAR", "processingLevel": "RSLC"},
        geometry=None,
    )
    with caplog.at_level("WARNING"):
        df = extract.to_dataframe([product], mission="NISAR")
    row = df.iloc[0]
    assert row["granule_id"] == "S"
    assert row["processing_level"] == "L1"
    assert pd.isna(row["start_time"])
    assert pd.isna(row["stop_time"])
    assert pd.isna(row["geometry"])
    assert row["polarization"] == []
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    joined = "\n".join(warnings)
    assert "start_time" in joined
    assert "geometry" in joined
    assert "polarization" in joined


def test_one_warning_per_field_across_rows(caplog: pytest.LogCaptureFixture) -> None:
    def bad() -> FakeProduct:
        return FakeProduct({"sceneName": "S"}, geometry=None)

    with caplog.at_level("WARNING"):
        extract.to_dataframe([bad(), bad(), bad()], mission="NISAR")
    count = sum(1 for r in caplog.records if r.levelname == "WARNING" and "start_time" in r.message)
    assert count == 1


def test_unparseable_timestamp_degrades_to_empty(caplog: pytest.LogCaptureFixture) -> None:
    product = _full_rslc()
    assert product.properties is not None
    product.properties = {**product.properties, "startTime": "not-a-timestamp"}
    with caplog.at_level("WARNING"):
        df = extract.to_dataframe([product], mission="NISAR")
    assert pd.isna(df.iloc[0]["start_time"])
    assert any("start_time" in r.message for r in caplog.records if r.levelname == "WARNING")


def test_empty_product_list() -> None:
    df = extract.to_dataframe([], mission="NISAR")
    assert len(df) == 0
    assert list(df.columns) == list(extract.COLUMNS)


def test_unknown_mission_raises() -> None:
    with pytest.raises(UnknownMissionError):
        extract.to_dataframe([_full_rslc()], mission="SENTINEL-1")


def test_profile_mapping_of_structural_column_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from metadata_asf.profiles import MISSIONS
    from metadata_asf.profiles.base import MissionProfile

    MISSIONS["BROKEN"] = MissionProfile(
        name="BROKEN",
        asf_dataset="BROKEN",
        supported_products=[("L1", "RSLC")],
        field_mapping={"geometry": ["sceneName"]},
    )
    try:
        with pytest.raises(ValueError, match="not a mappable schema column"):
            extract.to_dataframe([_full_rslc()], mission="BROKEN")
    finally:
        MISSIONS.pop("BROKEN")


def test_simple_namespace_product_also_works() -> None:
    product = SimpleNamespace(
        properties={
            "sceneName": "S",
            "platform": "NISAR",
            "startTime": "2026-10-07T21:16:36Z",
            "stopTime": "2026-10-07T21:16:41Z",
            "processingLevel": "RSLC",
        },
        geometry=_footprint(),
        sceneName="S",
    )
    df = extract.to_dataframe([product], mission="NISAR")
    assert df.iloc[0]["granule_id"] == "S"
