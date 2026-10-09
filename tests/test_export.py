"""Tests for :func:`metadata_asf.export.write_daily_parquet` (AGENTS.md section 7)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from metadata_asf.export import write_daily_parquet


def _daily_frame(
    dates: list[str],
    platform: object = "NISAR",
) -> pd.DataFrame:
    n = len(dates)
    frame = pd.DataFrame(
        {
            "granule_id": [f"SCENE_{i:03d}" for i in range(n)],
            "platform": [platform] * n,
            "geometry": ["POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"] * n,
            "start_time": pd.to_datetime([f"{d}T05:00:00" for d in dates]).tz_localize("UTC"),
            "stop_time": pd.to_datetime([f"{d}T05:00:10" for d in dates]).tz_localize("UTC"),
            "polarization": [["HH", "HV"]] * n,
            "beam_mode": [None] * n,
            "product_type": ["RSLC"] * n,
            "processing_level": ["L1"] * n,
        }
    )
    return frame


def test_one_file_per_day_sorted(tmp_path: Path) -> None:
    frame = _daily_frame(["2025-01-03", "2025-01-01", "2025-01-02", "2025-01-01"])
    written = write_daily_parquet(frame, tmp_path, mission="NISAR")
    assert [p.name for p in written] == [
        "NISAR_ocean_20250101.parquet",
        "NISAR_ocean_20250102.parquet",
        "NISAR_ocean_20250103.parquet",
    ]


def test_content_roundtrip_and_snappy(tmp_path: Path) -> None:
    frame = _daily_frame(["2025-01-01", "2025-01-01", "2025-01-02"])
    written = write_daily_parquet(frame, tmp_path, mission="NISAR")
    day_one = pq.read_table(written[0])
    assert day_one.num_rows == 2
    assert day_one.column_names == list(frame.columns)
    assert list(day_one.column("granule_id").to_pylist()) == ["SCENE_000", "SCENE_001"]
    # Snappy compression, guaranteed by AGENTS.md section 7.
    metadata = pq.read_metadata(written[0])
    assert metadata.row_group(0).column(0).compression == "SNAPPY"
    # No index column is written.
    assert "__index_level_0__" not in day_one.column_names
    # Re-read with pandas to confirm the values survive the round trip.
    reread = pd.read_parquet(written[0])
    assert reread["start_time"].iloc[0] == pd.Timestamp("2025-01-01T05:00:00Z")
    assert list(reread["polarization"].iloc[0]) == ["HH", "HV"]


def test_empty_frame_writes_nothing(tmp_path: Path) -> None:
    frame = _daily_frame([])
    assert write_daily_parquet(frame, tmp_path, mission="NISAR") == []
    assert list(tmp_path.iterdir()) == []


def test_creates_missing_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out"
    frame = _daily_frame(["2025-01-01"])
    written = write_daily_parquet(frame, target, mission="NISAR")
    assert written[0].parent == target


def test_missing_start_time_rejected(tmp_path: Path) -> None:
    frame = _daily_frame(["2025-01-01"])
    del frame["start_time"]
    with pytest.raises(ValueError, match="start_time"):
        write_daily_parquet(frame, tmp_path, mission="NISAR")


def test_naive_start_time_rejected(tmp_path: Path) -> None:
    frame = _daily_frame(["2025-01-01"])
    frame["start_time"] = frame["start_time"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        write_daily_parquet(frame, tmp_path, mission="NISAR")


def test_non_utc_start_time_rejected(tmp_path: Path) -> None:
    frame = _daily_frame(["2025-01-01"])
    frame["start_time"] = frame["start_time"].dt.tz_convert("Europe/Paris")
    with pytest.raises(ValueError, match="UTC"):
        write_daily_parquet(frame, tmp_path, mission="NISAR")


def test_mission_falls_back_to_platform(tmp_path: Path) -> None:
    frame = _daily_frame(["2025-01-01"], platform="SENTINEL-1")
    written = write_daily_parquet(frame, tmp_path)
    assert written[0].name == "SENTINEL-1_ocean_20250101.parquet"


def test_explicit_mission_wins_over_platform(tmp_path: Path) -> None:
    frame = _daily_frame(["2025-01-01"], platform="SENTINEL-1")
    written = write_daily_parquet(frame, tmp_path, mission="NISAR")
    assert written[0].name == "NISAR_ocean_20250101.parquet"


def test_existing_dir_is_not_cleared(tmp_path: Path) -> None:
    (tmp_path / "keep_me.txt").write_text("keep me", encoding="utf-8")
    frame = _daily_frame(["2025-01-01"])
    write_daily_parquet(frame, tmp_path, mission="NISAR")
    assert (tmp_path / "keep_me.txt").exists()
    assert len(list(tmp_path.glob("*.parquet"))) == 1
