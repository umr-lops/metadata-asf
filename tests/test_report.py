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


def test_analyze_catalog_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        report.analyze_catalog(tmp_path / "does-not-exist")


def test_analyze_catalog_no_parquet_raises(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("no parquets here", encoding="utf-8")
    with pytest.raises(ValueError):
        report.analyze_catalog(tmp_path)


def test_render_report_html_is_self_contained(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 2})
    html = report.render_report_html(report.analyze_catalog(tmp_path))

    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    # Inline CSS, no JavaScript, no external assets.
    assert "<style>" in html
    assert "<script" not in html
    assert "src=" not in html
    assert 'href="http' not in html
    # The three documented sections are present.
    assert "Volume and completeness" in html
    assert "Product and instrument mix" in html
    assert "Footprint geometry quality" in html


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


__all__ = []
