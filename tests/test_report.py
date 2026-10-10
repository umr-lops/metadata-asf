"""Tests for :mod:`metadata_asf.report`: catalog analysis and HTML rendering."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
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
    # The busiest day is 2025-01-03 (3 records); the data spans 01-01 to 01-03.
    assert stats.busiest_day_date == dt.date(2025, 1, 3)
    assert stats.first_day_with_data == dt.date(2025, 1, 1)
    assert stats.last_day_with_data == dt.date(2025, 1, 3)
    # The first/last products are the granule names of the earliest/latest acquisitions.
    assert stats.first_product == "g0-2025-01-01"
    assert stats.last_product == "g2-2025-01-03"
    assert (dt.date(2025, 1, 3), 3) in stats.per_day


def test_analyze_catalog_per_month(tmp_path: Path) -> None:
    # Two days in January (3 + 2) and one day in February (4): months sum their days.
    _write_daily(tmp_path, {"2025-01-01": 3, "2025-01-31": 2, "2025-02-15": 4})
    stats = report.analyze_catalog(tmp_path)

    assert stats.per_month == [
        (dt.date(2025, 1, 1), 5),
        (dt.date(2025, 2, 1), 4),
    ]


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
    # Inline CSS, no JavaScript, no external assets (the "About" section may carry
    # outbound navigation links, but those load nothing into the page).
    assert "<style>" in html
    assert "<script" not in html
    assert 'src="http' not in html
    assert 'src="//' not in html
    assert "<link" not in html
    # Figures are inlined as base64 PNG data URIs (volume + 4 mix panels + map +
    # land/ocean + geometry = 8).
    assert html.count("data:image/png;base64,") >= 8
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
    # Rendering the map (footprint outlines coloured by beam/mode + legend) must not raise.
    report._safe_figure("map", lambda: report._fig_map(combined))


def test_footprint_maps_draw_polygon_outlines(tmp_path: Path) -> None:
    """Both footprint maps outline the actual polygon (PolyCollection), not centroid points."""
    _write_daily(tmp_path, {"2025-01-01": 5})
    combined, *_ = report._read_catalog(report._find_files(tmp_path))
    combined = report._backfill_beam_mode(combined, mission="NISAR")

    def _polygons_drawn(draw: Callable[[], None]) -> int:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection

        draw()
        fig = plt.gcf()
        n = 0
        for ax in fig.axes:
            for col in ax.collections:
                if isinstance(col, PolyCollection):
                    n += len(col.get_paths())
        plt.close("all")
        return n

    # Beam/mode map outlines one ring per footprint.
    assert _polygons_drawn(lambda: report._fig_map(combined)) >= 5
    # Land/ocean map likewise outlines footprints.
    assert _polygons_drawn(lambda: report._fig_land_ocean(combined)) >= 5


def test_per_day_beam_sums_to_per_day(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 2, "2025-01-02": 3})
    combined, *_ = report._read_catalog(report._find_files(tmp_path))
    stats = report.analyze_catalog(tmp_path)

    days = [d for d, _ in stats.per_day]
    breakdown = report._per_day_beam(combined, days)
    assert breakdown is not None
    # Each day's beam breakdown sums to that day's total record count.
    for day, count in stats.per_day:
        assert sum(breakdown[day].values()) == count
    # Empty beam_mode decodes to the "(empty)" bucket in the breakdown.
    assert breakdown[dt.date(2025, 1, 1)].get("(empty)", 0) == 2


def test_volume_figure_stacked_by_beam_mode_renders(tmp_path: Path) -> None:
    """The daily-volume figure is drawn as a beam/mode stack without error."""
    _write_daily(tmp_path, {"2025-01-01": 3, "2025-01-02": 2})
    combined, *_ = report._read_catalog(report._find_files(tmp_path))
    combined = report._backfill_beam_mode(combined, mission="NISAR")
    stats = report.analyze_catalog(tmp_path)
    fig = report._safe_figure("volume", lambda: report._fig_volume(stats, combined))
    assert "data:image/png;base64," in fig


def test_volume_figure_falls_back_without_frame(tmp_path: Path) -> None:
    """Without a catalog frame the volume figure still renders (single colour)."""
    _write_daily(tmp_path, {"2025-01-01": 2})
    stats = report.analyze_catalog(tmp_path)
    fig = report._safe_figure("volume", lambda: report._fig_volume(stats))
    assert "data:image/png;base64," in fig


def test_land_ocean_figure_renders(tmp_path: Path) -> None:
    """The land/ocean split map is drawn without error and produces a PNG."""
    _write_daily(tmp_path, {"2025-01-01": 4})
    combined, *_ = report._read_catalog(report._find_files(tmp_path))
    fig = report._safe_figure("land-ocean", lambda: report._fig_land_ocean(combined))
    assert "data:image/png;base64," in fig


def test_render_report_includes_land_ocean_map(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 4})
    stats = report.analyze_catalog(tmp_path)
    combined, *_ = report._read_catalog(report._find_files(tmp_path))
    html = report.render_report_html(stats, catalog=combined)

    # Eight inline figures now (volume + 4 mix panels + map + land/ocean + geometry).
    assert html.count("data:image/png;base64,") >= 8
    # The land/ocean note explains the centroid-based split.
    assert "centroid falls inside" in html


def test_mix_panels_render_as_four_distinct_serif_figures(tmp_path: Path) -> None:
    """Each mix (product/level/beam/pol) is its own figure, in a serif font."""
    _write_daily(tmp_path, {"2025-01-01": 4})
    stats = report.analyze_catalog(tmp_path)
    combined = report._read_catalog(report._find_files(tmp_path))[0]
    html = report.render_report_html(stats, catalog=combined)

    # The four mix figures are inlined and the section heading is present.
    assert "Product and instrument mix" in html
    assert html.count("data:image/png;base64,") >= 8
    # A single mix panel renders on its own without error.
    fig = report._safe_figure(
        "mix:Product type", lambda: report._fig_mix_panel("Product type", stats.product_mix, None)
    )
    assert "data:image/png;base64," in fig
    # The font resolver returns a real, installed font name.
    assert report._mix_font() in {"Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"}


def test_report_lists_provider_and_repo_links(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 1})
    html = report.render_report_html(report.analyze_catalog(tmp_path))

    # Provider (ASF) and tool (this repo) sources are both listed with real links.
    assert "Alaska Satellite Facility" in html
    assert 'href="https://asf.alaska.edu"' in html
    # The search endpoint asf_search actually calls: NASA CMR, queried with provider=ASF.
    assert 'href="https://cmr.earthdata.nasa.gov/search/granules.umm_json"' in html
    assert 'href="https://github.com/umr-lops/metadata-asf"' in html
    assert 'href="https://metadata-asf.readthedocs.io"' in html


def test_report_links_nisar_when_mission_is_nisar(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 1})  # platform is NISAR by construction
    html = report.render_report_html(report.analyze_catalog(tmp_path))

    assert "NISAR mission" in html
    assert 'href="https://nisar-docs.asf.alaska.edu/"' in html
    assert 'href="https://nisar-docs.asf.alaska.edu/products-overview/"' in html
    assert 'href="https://www.nisarmission.com/instruments/"' in html


def test_report_omits_nisar_links_for_other_mission(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 1})
    stats = report.analyze_catalog(tmp_path, mission="SENTINEL-1")
    html = report.render_report_html(stats)
    assert "NISAR mission" not in html


def test_kpi_cards_use_descriptive_labels(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 2, "2025-01-03": 3})
    html = report.render_report_html(report.analyze_catalog(tmp_path))

    # The headline cards are now phrased around the Ifremer-collected subset.
    assert "Total NISAR records at Ifremer" in html
    assert "Days without NISAR data in 2025-01-01 → 2025-01-03" in html
    assert "Full size of the NISAR metadata" in html
    # Per-day cards: descriptive titles + busiest day carries its date + last day present.
    assert "Mean records / day" in html
    assert "Median records / day" in html
    assert "Busiest day (2025-01-03)" in html
    assert "First day with data" in html
    assert "Last day with data" in html
    # The first/last cards name the product collected (truncated, full name in title).
    assert "g0-2025-01-01" in html
    assert "g2-2025-01-03" in html


def test_report_shows_nisar_logo_when_mission_is_nisar(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 1})  # platform is NISAR by construction
    html = report.render_report_html(report.analyze_catalog(tmp_path))

    # The NISAR logo is embedded as a self-contained base64 data URI (no external asset).
    assert 'class="logo"' in html
    assert "data:image/png;base64," in html
    assert 'src="http' not in html


def test_subfamily_of_maps_beam_to_frequency_family() -> None:
    assert report._subfamily_of("40 MHz, dual-pol HH/HV") == "40 MHz"
    assert report._subfamily_of("5 MHz") == "5 MHz"
    assert report._subfamily_of("77 MHz, single-pol HH") == "77 MHz"
    assert report._subfamily_of("(empty)") is None
    assert report._subfamily_of("unknown mode") is None


def test_beams_for_family_selects_the_family_beams(tmp_path: Path) -> None:
    stats, _combined = _frame_with_beams(tmp_path)

    # The 5 MHz family has two distinct beams (single-pol VV + dual-pol VV/VH); 40 MHz one.
    family_5 = {label for label, _ in report._beams_for_family(stats.beam_mix, "5 MHz")}
    assert family_5 == {"5 MHz, single-pol VV", "5 MHz, dual-pol VV/VH"}
    family_40 = {label for label, _ in report._beams_for_family(stats.beam_mix, "40 MHz")}
    assert family_40 == {"40 MHz, dual-pol HH/HV"}
    # Empty families return no beams.
    assert report._beams_for_family(stats.beam_mix, "20 MHz") == []
    assert report._beams_for_family(stats.beam_mix, "77 MHz") == []


def test_subfamily_figure_stacks_the_family_beams(tmp_path: Path) -> None:
    """The sub-family figure is drawn (stacked by beam) without error and yields a PNG."""
    stats, combined = _frame_with_beams(tmp_path)
    fig = report._safe_figure(
        "subfamily:5 MHz", lambda: report._fig_subfamily(stats, combined, "5 MHz")
    )
    assert "data:image/png;base64," in fig


def test_report_lists_per_subfamily_daily_figures(tmp_path: Path) -> None:
    stats, combined = _frame_with_beams(tmp_path)
    html = report.render_report_html(stats, catalog=combined)

    # Only the populated sub-families (5 and 40 MHz here) become collapsible panels.
    panels = [m.strip() for m in re.findall(r'<details class="beammap"><summary>([^<]*)', html)]
    assert "5 MHz" in panels
    assert "40 MHz" in panels
    assert "20 MHz" not in panels and "77 MHz" not in panels
    # The sub-family daily-record figures are inlined alongside the other figures.
    assert html.count("data:image/png;base64,") >= 4


def _frame_with_beams(tmp_path: Path) -> tuple[report.CatalogStats, pd.DataFrame]:
    """A catalog frame plus stats with three beam/modes.

    Two of them share the 5 MHz family (single-pol VV and dual-pol VV/VH) so the 5 MHz sub-family
    figure has more than one beam to stack; the third is a lone 40 MHz beam.
    """
    _write_daily(tmp_path, {"2025-01-01": 6})
    combined = report._read_catalog(report._find_files(tmp_path))[0]
    combined["beam_mode"] = [
        "5 MHz, single-pol VV",
        "5 MHz, single-pol VV",
        "5 MHz, dual-pol VV/VH",
        "5 MHz, dual-pol VV/VH",
        "40 MHz, dual-pol HH/HV",
        "40 MHz, dual-pol HH/HV",
    ]
    stats = report._summarize(
        combined,
        [dt.date(2025, 1, 1)],
        [6],
        [0],
        "NISAR",
    )
    return stats, combined


def test_report_lists_per_beam_maps_as_collapsible_sections(tmp_path: Path) -> None:
    stats, combined = _frame_with_beams(tmp_path)
    html = report.render_report_html(stats, catalog=combined)

    # Each populated beam/mode becomes a <details> panel; the empty bucket is skipped.
    beam_panels = [
        m.strip() for m in re.findall(r'<details class="beammap"><summary>([^<]*)', html)
    ]
    # The two beam/modes plus the two sub-families they belong to (5 and 40 MHz).
    assert "40 MHz, dual-pol HH/HV" in beam_panels
    assert "5 MHz, single-pol VV" in beam_panels
    assert "40 MHz" in beam_panels  # the sub-family panel
    assert "5 MHz" in beam_panels
    assert "(empty)" not in html


def test_beam_maps_skips_missing_beam_modes(tmp_path: Path) -> None:
    """_beam_maps returns only the beams that actually have records."""
    _stats, combined = _frame_with_beams(tmp_path)

    maps = report._beam_maps(combined)
    present = {
        label for label, _ in report._value_counts(combined, "beam_mode") if label != "(empty)"
    }
    assert set(maps) == present
    assert "(empty)" not in maps


def test_report_omits_nisar_logo_for_other_mission(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 1})
    stats = report.analyze_catalog(tmp_path, mission="SENTINEL-1")
    html = report.render_report_html(stats)
    assert 'class="logo"' not in html


def test_report_carries_scope_disclaimer(tmp_path: Path) -> None:
    _write_daily(tmp_path, {"2025-01-01": 1})
    html = report.render_report_html(report.analyze_catalog(tmp_path))

    # The report must warn that it is an Ifremer-collected subset, not the whole ASF collection.
    assert "Scope disclaimer" in html
    assert "Ifremer" in html
    assert "not" in html and "complete ASF collection" in html
    assert "NISAR" in html  # mission name is woven into the disclaimer


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
