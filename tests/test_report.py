"""Tests for :mod:`metadata_asf.report`: catalog analysis and HTML rendering."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from metadata_asf import report


def _write_daily(catalog: Path, days: dict[str, int]) -> list[Path]:
    """Write one daily Parquet per key in ``days`` (value = number of records)."""
    catalog.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for iso, n in days.items():
        date = dt.date.fromisoformat(iso)
        df = pd.DataFrame(
            {
                "granule_id": [f"g{i}-{iso}" for i in range(n)],
                "platform": ["NISAR"] * n,
                "geometry": [
                    (
                        "POLYGON((0 0,1 0,1 1,0 1,0 0))"
                        if i % 3
                        else "POLYGON((179 10,-179 10,-179 20,179 20,179 10))"
                    )
                    for i in range(n)
                ],
                "start_time": pd.to_datetime([iso + "T05:00:00"] * n),
                "stop_time": pd.to_datetime([iso + "T05:00:10"] * n),
                "polarization": [["HH", "HV"] for _ in range(n)],
                "beam_mode": [None] * n,
                "product_type": ["RSLC"] * n,
                "processing_level": ["L1"] * n,
            }
        )
        # Force UTC datetime dtypes even for a zero-row file so the schema is stable.
        df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_localize("UTC")
        df["stop_time"] = pd.to_datetime(df["stop_time"]).dt.tz_localize("UTC")
        path = catalog / f"NISAR_ocean_{date:%Y%m%d}.parquet"
        df.to_parquet(path, index=False)
        paths.append(path)
    return paths


def test_analyze_catalog_counts_and_span(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 2, "2025-01-03": 3})
    stats = report.analyze_catalog(tmp_path)

    assert stats.mission == "NISAR"
    assert stats.total_records == 5
    assert stats.total_files == 2
    assert stats.span_start == dt.date(2025, 1, 1)
    assert stats.span_end == dt.date(2025, 1, 3)
    # 2025-01-02 sits between the two files but has none of its own.
    assert stats.missing_days == [dt.date(2025, 1, 2)]
    assert stats.busiest_day == 3
    assert (dt.date(2025, 1, 3), 3) in stats.per_day


def test_analyze_catalog_empty_day_and_mixes(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 0, "2025-01-02": 1})
    stats = report.analyze_catalog(tmp_path)

    assert stats.empty_days == [dt.date(2025, 1, 1)]
    assert stats.first_day_with_data == dt.date(2025, 1, 2)
    assert ("RSLC", 1) in stats.product_mix
    assert ("L1", 1) in stats.level_mix
    assert ("(empty)", 1) in stats.beam_mix
    assert ("HH", 1) in stats.polarization_mix
    assert ("HV", 1) in stats.polarization_mix


def test_analyze_catalog_geometry_quality(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 3})  # 1 antimeridian + 2 simple footprints
    stats = report.analyze_catalog(tmp_path)

    assert stats.geom_total == 3
    assert stats.geom_missing == 0
    assert stats.geom_invalid == 0
    assert stats.geom_antimeridian == 1
    # Every valid footprint's centroid lands over land or ocean (the whole point of the split).
    assert stats.geom_land + stats.geom_ocean == stats.geom_total


def test_beam_color_map_is_deterministic_and_excludes_empty() -> None:
    mix = [("40 MHz, dual-pol HH/HV", 10), ("5 MHz, single-pol VV", 7), ("(empty)", 2)]
    cmap = report._beam_color_map(mix)
    # Deterministic: same input -> same assignment; first mode gets the first palette colour.
    assert cmap["40 MHz, dual-pol HH/HV"] == report._BEAM_PALETTE[0]
    assert cmap["5 MHz, single-pol VV"] == report._BEAM_PALETTE[1]
    # "(empty)" / unknown are never assigned a palette colour (they use the neutral one).
    assert "(empty)" not in cmap
    assert len(cmap) == 2


def test_beam_color_for_unknown_uses_neutral() -> None:
    cmap = {"40 MHz, dual-pol HH/HV": "#2f6fed"}
    assert report._beam_color_for("40 MHz, dual-pol HH/HV", cmap) == "#2f6fed"
    assert report._beam_color_for("other", cmap) == report._BEAM_UNKNOWN
    assert report._beam_color_for(None, cmap) == report._BEAM_UNKNOWN


def test_analyze_catalog_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        report.analyze_catalog(tmp_path / "does-not-exist")


def test_analyze_catalog_no_parquet_raises(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("no parquets here", encoding="utf-8")
    with pytest.raises(ValueError):
        report.analyze_catalog(tmp_path)


def test_render_report_html_is_self_contained(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 2})
    stats = report.analyze_catalog(tmp_path)
    combined, *_ = report._read_catalog(report._find_files(tmp_path))
    html = report.render_report_html(stats, catalog=combined)

    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    # Inline CSS, no JavaScript, no external assets.
    assert "<style>" in html
    assert "<script" not in html
    assert 'src="http' not in html
    assert 'href="http' not in html
    # Figures are inlined as base64 PNG data URIs (volume, mix, map, geometry).
    assert html.count("data:image/png;base64,") >= 4
    # The three documented sections are present.
    assert "Volume and completeness" in html
    assert "Product and instrument mix" in html
    assert "Footprint geometry quality" in html


def test_geometry_labels_not_double_escaped_and_land_ocean_present(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 2})
    stats = report.analyze_catalog(tmp_path)
    combined, *_ = report._read_catalog(report._find_files(tmp_path))
    html = report.render_report_html(stats, catalog=combined)

    # No double-escaping of the ">" operator in the geometry table labels.
    assert "&amp;gt;" not in html
    assert "Centroid over land" in html
    assert "Centroid over ocean" in html
    # Beam/mode values carry a palette colour in the table (inline background).
    assert "background:#2f6fed" in html


def test_map_figure_color_codes_by_beam_mode(tmp_path: Path) -> None:
    """The footprint map is drawn from a beam-backfilled frame without error."""
    _write_daily(tmp_path, {"2025-01-01": 3})
    combined, *_ = report._read_catalog(report._find_files(tmp_path))
    combined = report._backfill_beam_mode(combined, mission="NISAR")
    # Rendering the map (colour-coded centroids + legend) must not raise.
    report._safe_figure("map", lambda: report._fig_map(combined))


def test_render_report_without_catalog_still_renders(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 1})
    html = report.render_report_html(report.analyze_catalog(tmp_path))

    # No map without a catalog frame, but the stats-only figures are present.
    assert "data:image/png;base64," in html
    assert "Volume and completeness" in html


def test_write_report_returns_path(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 1})
    out = tmp_path / "out" / "report.html"
    written = report.write_report(tmp_path, out, mission="NISAR")

    assert written == out
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_render_report_empty_catalog(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 0})
    html = report.render_report_html(report.analyze_catalog(tmp_path))

    assert "Volume and completeness" in html
    # No crash on an all-zero catalog; the empty-day is surfaced.
    assert "2025-01-01" in html


def test_backfill_beam_mode_from_nisar_granule_id() -> None:
    frame = pd.DataFrame(
        {
            "granule_id": [
                "NISAR_L1_PR_RSLC_010_098_D_052_4005_DHDH_A_"
                "20260115T235949_20260116T000025_X05010_N_P_J_001"
            ],
            "platform": ["NISAR"],
            "beam_mode": [None],
        }
    )
    filled = report._backfill_beam_mode(frame, mission="NISAR")
    assert filled["beam_mode"].iloc[0] == "40 MHz, dual-pol HH/HV"
    # The on-disk frame is never mutated.
    assert frame["beam_mode"].isna().all()


def test_backfill_beam_mode_keeps_existing_values() -> None:
    frame = pd.DataFrame(
        {
            "granule_id": [
                "NISAR_L1_PR_RSLC_010_098_D_052_4005_DHDH_A_"
                "20260115T235949_20260116T000025_X05010_N_P_J_001"
            ],
            "platform": ["NISAR"],
            "beam_mode": ["already set"],
        }
    )
    filled = report._backfill_beam_mode(frame, mission="NISAR")
    assert filled is frame  # no copy when nothing to do
    assert filled["beam_mode"].iloc[0] == "already set"


__all__ = []
