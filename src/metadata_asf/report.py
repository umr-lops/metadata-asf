"""Read-only HTML report on a directory of daily Parquet catalogs.

The report describes what a harvest has produced: volume and completeness
(records, daily files, span, missing/empty days), the product/instrument mix,
and the geometry quality of the footprints. It is *read-only* and never
downloads anything (AGENTS.md section 13), and the HTML it emits is fully
self-contained (inline CSS, no JavaScript, no external assets) so a single
file stands alone.

Figures (daily volume, cumulative records, mix panels, footprint map,
geometry-issue chart) are rendered server-side with matplotlib and inlined
as base64 PNG data URIs, mirroring the IFR-EMER/LOPS RCM daily metadata
report that this module is inspired by. The "local archive cross-check"
section of that reference does not apply here, because ``metadata-asf``
holds no local product archive (that is ``fetch-asf``'s job).
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import html
import io
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import NDArray

from metadata_asf.profiles import UnknownMissionError, get_profile

logger = logging.getLogger(__name__)

#: Trailing ``YYYYMMDD`` in a daily file name, e.g. ``NISAR_ocean_20250101.parquet``.
_DAY_IN_NAME: re.Pattern[str] = re.compile(r"(\d{8})\.parquet$")

#: |latitude| above which a lon/lat four-corner ring stops describing a swath
#: (longitudes converge); such scenes are flagged, not repaired.
_NEAR_POLAR_LAT: float = 85.0
#: |latitude| above which a footprint is "within 1 degree of a pole".
_POLE_1DEG_LAT: float = 89.0

# --- Source links shown in the report's "About" section ----------------------

#: Data provider: the Alaska Satellite Facility, which curates the SAR metadata this tool harvests.
_PROVIDER_NAME = "Alaska Satellite Facility (ASF)"
_PROVIDER_SITE = "https://asf.alaska.edu"
#: The search endpoint ``asf_search`` actually calls: NASA CMR, queried with ``provider=ASF``.
_PROVIDER_API = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
#: ASF's documentation site.
_PROVIDER_DOCS = "https://docs.asf.alaska.edu/"

#: This tool.
_REPO_NAME = "metadata-asf"
_REPO_GITHUB = "https://github.com/umr-lops/metadata-asf"
_REPO_DOCS = "https://metadata-asf.readthedocs.io"

#: The NISAR mission (NASA/ISRO) — surfaced when the catalog's mission is NISAR.
_NISAR_NAME = "NISAR mission"
_NISAR_DOCS = "https://nisar-docs.asf.alaska.edu/"
_NISAR_PRODUCTS = "https://nisar-docs.asf.alaska.edu/products-overview/"
_NISAR_SENSOR = "https://www.nisarmission.com/instruments/"

#: Bundled NISAR mission logo (embedded as a base64 data URI so the report stays self-contained).
_NISAR_LOGO_PATH: Path = Path(__file__).parent / "assets" / "nisar_logo.png"
_NISAR_LOGO_DATA_URI: str | None = None


def _nisar_logo_data_uri() -> str | None:
    """The bundled NISAR logo as a ``data:image/png;base64,...`` URI, or ``None`` if unavailable.

    Read and cached once; the report never fetches the logo at render time, so it works offline.
    """
    global _NISAR_LOGO_DATA_URI
    if _NISAR_LOGO_DATA_URI is not None:
        return _NISAR_LOGO_DATA_URI
    try:
        data = base64.b64encode(_NISAR_LOGO_PATH.read_bytes()).decode("ascii")
        _NISAR_LOGO_DATA_URI = f"data:image/png;base64,{data}"
    except OSError as exc:
        logger.debug("Could not read the NISAR logo asset: %s", exc)
        _NISAR_LOGO_DATA_URI = ""
    return _NISAR_LOGO_DATA_URI or None


@dataclasses.dataclass(frozen=True)
class CatalogStats:
    """Aggregated, render-ready statistics about one daily-Parquet catalog."""

    mission: str
    generated_at: dt.datetime
    total_records: int
    total_files: int
    total_size_bytes: int
    span_start: dt.date | None
    span_end: dt.date | None
    missing_days: list[dt.date]
    empty_days: list[dt.date]
    first_day_with_data: dt.date | None
    last_day_with_data: dt.date | None
    mean_per_day: float
    median_per_day: float
    busiest_day: int
    busiest_day_date: dt.date | None
    per_day: list[tuple[dt.date, int]]
    per_month: list[tuple[dt.date, int]]
    first_product: str | None
    last_product: str | None
    platform_mix: list[tuple[str, int]]
    product_mix: list[tuple[str, int]]
    level_mix: list[tuple[str, int]]
    beam_mix: list[tuple[str, int]]
    polarization_mix: list[tuple[str, int]]
    geom_total: int
    geom_missing: int
    geom_invalid: int
    geom_antimeridian: int
    geom_near_polar: int
    geom_pole_1deg: int
    geom_land: int
    geom_ocean: int


def analyze_catalog(catalog_dir: Path, *, mission: str | None = None) -> CatalogStats:
    """Scan ``catalog_dir`` for daily Parquet files and aggregate their statistics.

    Args:
        catalog_dir: directory holding the ``*_ocean_YYYYMMDD.parquet`` files; every
            ``*.parquet`` entry is read.
        mission: label shown in the report header; falls back to the single distinct
            ``platform`` value, else ``"ASF"``.

    Returns:
        A :class:`CatalogStats` ready to hand to :func:`render_report_html`.

    Raises:
        FileNotFoundError: if ``catalog_dir`` does not exist.
        ValueError: if the directory exists but holds no ``*.parquet`` file.
    """
    combined, file_days, file_rows, file_sizes = _read_catalog(_find_files(catalog_dir))
    return _summarize(combined, file_days, file_rows, file_sizes, mission)


def render_report_html(stats: CatalogStats, *, catalog: pd.DataFrame | None = None) -> str:
    """Render :class:`CatalogStats` into a self-contained HTML document.

    Args:
        stats: the aggregated catalog statistics to render.
        catalog: optional combined record frame; when given, the footprint
            world map is drawn from its ``geometry`` column.

    Returns:
        The full HTML page as a string (UTF-8, inline CSS, figures inlined as
        base64 PNG data URIs, no JavaScript, no external assets).
    """
    parts: list[str] = [
        _doctype(),
        _head(),
        _header(stats),
        _about_section(stats),
        _kpi_cards(stats),
    ]
    # A sub-family gets a figure when at least one of its beam/modes has records in the catalog.
    subfamily_figs = {
        family: _safe_figure(
            f"subfamily:{family}", _subfamily_figure(stats, catalog, family), dpi=140
        )
        for family in _SUBFAMILY_ORDER
        if _beams_for_family(stats.beam_mix, family)
    }
    parts.append(
        _volume_section(
            stats,
            _safe_figure("volume", lambda: _fig_volume(stats, catalog)),
            subfamily_figs,
        )
    )
    beam_colors = _beam_color_map(stats.beam_mix)
    mix_panels: list[tuple[str, list[tuple[str, int]], dict[str, str] | None]] = [
        ("Product type", stats.product_mix, None),
        ("Processing level", stats.level_mix, None),
        ("Beam / mode", stats.beam_mix, beam_colors),
        ("Polarization", stats.polarization_mix, None),
    ]
    mix_figs: dict[str, str] = {
        title: _safe_figure(f"mix:{title}", _mix_panel_draw(title, items, colors))
        for title, items, colors in mix_panels
    }
    parts.append(_mix_section(stats, mix_figs))
    has_geom = catalog is not None and not catalog.empty and "geometry" in catalog.columns
    # The two footprint maps are the most detail-rich, so they render a bit sharper (higher DPI).
    map_html = _safe_figure("map", lambda: _fig_map(catalog), dpi=200) if has_geom else ""
    land_ocean_html = (
        _safe_figure("land-ocean", lambda: _fig_land_ocean(catalog), dpi=200) if has_geom else ""
    )
    geo_fig = _safe_figure("geometry", lambda: _fig_geometry_issues(stats))
    beam_maps = _beam_maps(catalog) if has_geom else {}
    parts.append(_geometry_section(stats, map_html, land_ocean_html, geo_fig, beam_maps))
    parts.append(_footer(stats))
    parts.append("</body>\n</html>\n")
    return "".join(parts)


def write_report(
    catalog_dir: Path,
    output_file: Path,
    *,
    mission: str | None = None,
) -> Path:
    """Analyze ``catalog_dir`` and write the self-contained HTML report to ``output_file``.

    Args:
        catalog_dir: directory of daily Parquet files to describe (see
            :func:`analyze_catalog`).
        output_file: where the HTML report is written; parent directories are created.
        mission: optional label for the report header (see :func:`analyze_catalog`).

    Returns:
        The path of the written report.
    """
    combined, file_days, file_rows, file_sizes = _read_catalog(_find_files(catalog_dir))
    combined = _backfill_beam_mode(combined, mission)
    stats = _summarize(combined, file_days, file_rows, file_sizes, mission)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_report_html(stats, catalog=combined), encoding="utf-8")
    logger.info(
        "Wrote report %s — %d records over %d daily file(s).",
        output_file,
        stats.total_records,
        stats.total_files,
    )
    return output_file


def _find_files(catalog_dir: Path) -> list[Path]:
    """The sorted ``*.parquet`` inventory of ``catalog_dir`` (validated)."""
    if not catalog_dir.is_dir():
        raise FileNotFoundError(f"Catalog directory not found: {catalog_dir}")
    files = sorted(catalog_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No *.parquet files in catalog directory: {catalog_dir}")
    return files


def _read_catalog(
    files: list[Path],
) -> tuple[pd.DataFrame, list[dt.date | None], list[int], list[int]]:
    """Read every catalog file; return the combined frame plus per-file metadata."""
    frames: list[pd.DataFrame] = []
    file_days: list[dt.date | None] = []
    file_rows: list[int] = []
    file_sizes: list[int] = []
    for path in files:
        frame = pd.read_parquet(path)
        frames.append(frame)
        file_days.append(_day_from_name(path.name))
        file_rows.append(len(frame))
        file_sizes.append(path.stat().st_size)
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["start_time", "geometry"])
    )
    return combined, file_days, file_rows, file_sizes


def _backfill_beam_mode(frame: pd.DataFrame, mission: str | None) -> pd.DataFrame:
    """Fill empty ``beam_mode`` cells from ``granule_id`` via the mission's decoder.

    Catalogs harvested before ``beam_mode`` was derived (e.g. NISAR, where it is only readable
    from the product file name) carry an empty column. Rather than force a re-harvest, the
    report decodes it on the fly — but only the *in-memory* frame is affected; the Parquet
    files on disk are never rewritten.

    Args:
        frame: the combined catalog frame.
        mission: optional explicit mission label; otherwise inferred from ``platform``.

    Returns:
        The frame with any decodable ``beam_mode`` cells filled (a copy when changed).
    """
    if frame.empty or "beam_mode" not in frame.columns or "granule_id" not in frame.columns:
        return frame

    label = mission or _mission_label(frame)
    try:
        profile = get_profile(label)
    except UnknownMissionError:
        return frame
    decoder = profile.decode_beam_mode
    if decoder is None:
        return frame

    filled: list[str | None] = []
    changed = False
    for beam_value, granule_id in zip(frame["beam_mode"], frame["granule_id"], strict=True):
        if beam_value is not None and not (isinstance(beam_value, float) and np.isnan(beam_value)):
            filled.append(str(beam_value))
            continue
        decoded = decoder(str(granule_id)) if granule_id is not None else None
        filled.append(decoded)
        if decoded is not None:
            changed = True
    if not changed:
        return frame
    result = frame.copy()
    result["beam_mode"] = filled
    return result


def _summarize(
    combined: pd.DataFrame,
    file_days: list[dt.date | None],
    file_rows: list[int],
    file_sizes: list[int],
    mission: str | None,
) -> CatalogStats:
    """Aggregate the combined frame and file inventory into :class:`CatalogStats`."""
    per_day, platform_mix, product_mix, level_mix, beam_mix, polarization_mix = _mixes(combined)
    per_month = _per_month(per_day)
    geom = _geometry_quality(combined)
    completeness = _completeness(file_days, file_rows)
    busiest_day_date = _busiest_day_date(per_day)
    first_product, last_product = _first_and_last_product(combined)
    return CatalogStats(
        mission=mission or _mission_label(combined),
        generated_at=dt.datetime.now(dt.timezone.utc),
        total_records=int(len(combined)),
        total_files=len(file_days),
        total_size_bytes=int(sum(file_sizes)),
        span_start=completeness.span_start,
        span_end=completeness.span_end,
        missing_days=completeness.missing_days,
        empty_days=completeness.empty_days,
        first_day_with_data=completeness.first_day_with_data,
        last_day_with_data=completeness.last_day_with_data,
        mean_per_day=completeness.mean_per_day,
        median_per_day=completeness.median_per_day,
        busiest_day=completeness.busiest_day,
        busiest_day_date=busiest_day_date,
        per_day=per_day,
        per_month=per_month,
        first_product=first_product,
        last_product=last_product,
        platform_mix=platform_mix,
        product_mix=product_mix,
        level_mix=level_mix,
        beam_mix=beam_mix,
        polarization_mix=polarization_mix,
        **geom,
    )


def _day_from_name(name: str) -> dt.date | None:
    """Extract the trailing ``YYYYMMDD`` acquisition day from a daily file name."""
    match = _DAY_IN_NAME.search(name)
    if not match:
        logger.debug("Cannot parse a day from catalog file name %r", name)
        return None
    try:
        return dt.datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        logger.debug("Unparseable day %r in catalog file name %r", match.group(1), name)
        return None


def _mission_label(frame: pd.DataFrame) -> str:
    """Best-effort mission label from the ``platform`` column (else ``"ASF"``)."""
    if "platform" in frame.columns and not frame.empty:
        platforms = frame["platform"].dropna().unique()
        if len(platforms) == 1:
            return str(platforms[0])
    return "ASF"


def _completeness(
    file_days: list[dt.date | None],
    file_rows: list[int],
) -> _Completeness:
    """Derive span, missing/empty days and per-day stats from the file inventory."""
    dated = [d for d in file_days if d is not None]
    empty_days = sorted(
        d for d, n in zip(file_days, file_rows, strict=False) if d is not None and n == 0
    )

    missing_days: list[dt.date] = []
    span_start: dt.date | None = None
    span_end: dt.date | None = None
    if dated:
        present = set(dated)
        span_start, span_end = min(dated), max(dated)
        cursor = span_start
        while cursor <= span_end:
            if cursor not in present:
                missing_days.append(cursor)
            cursor = _next_day(cursor)

    record_counts = [n for d, n in zip(file_days, file_rows, strict=False) if d is not None]
    days_with_data = sorted(
        d for d, n in zip(file_days, file_rows, strict=False) if d is not None and n > 0
    )
    return _Completeness(
        span_start=span_start,
        span_end=span_end,
        missing_days=missing_days,
        empty_days=empty_days,
        first_day_with_data=days_with_data[0] if days_with_data else None,
        last_day_with_data=days_with_data[-1] if days_with_data else None,
        mean_per_day=(sum(record_counts) / len(record_counts)) if record_counts else 0.0,
        median_per_day=float(pd.Series(record_counts).median()) if record_counts else 0.0,
        busiest_day=max(record_counts) if record_counts else 0,
    )


def _next_day(day: dt.date) -> dt.date:
    """The calendar day after ``day``."""
    return day + dt.timedelta(days=1)


def _mixes(
    frame: pd.DataFrame,
) -> tuple[
    list[tuple[dt.date, int]],
    list[tuple[str, int]],
    list[tuple[str, int]],
    list[tuple[str, int]],
    list[tuple[str, int]],
    list[tuple[str, int]],
]:
    """Compute the per-day series and the categorical mixes for the report."""
    per_day = _per_day(frame)
    platform = _value_counts(frame, "platform")
    product = _value_counts(frame, "product_type")
    level = _value_counts(frame, "processing_level")
    beam = _value_counts(frame, "beam_mode")
    polarization = _polarization_counts(frame)
    return per_day, platform, product, level, beam, polarization


def _per_day(frame: pd.DataFrame) -> list[tuple[dt.date, int]]:
    """``(day, record_count)`` pairs from ``start_time``, sorted by day."""
    if frame.empty or "start_time" not in frame.columns:
        return []
    counts = frame["start_time"].dt.date.value_counts().sort_index()
    return [(dt.date.fromisoformat(str(day)), int(n)) for day, n in counts.items() if pd.notna(day)]


def _busiest_day_date(per_day: list[tuple[dt.date, int]]) -> dt.date | None:
    """The calendar day on which the most records were collected (first on a tie)."""
    if not per_day:
        return None
    day, _count = max(per_day, key=lambda pair: pair[1])
    return day


def _first_and_last_product(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    """The product file names of the earliest and latest acquisitions by ``start_time``.

    The product file name is stored in ``granule_id`` (the ASF granule name), so the report can
    show *which* product was collected first and last.
    """
    if frame.empty or "start_time" not in frame.columns or "granule_id" not in frame.columns:
        return (None, None)
    ordered = frame.sort_values("start_time", kind="stable")
    values = ordered["granule_id"].dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return (None, None)
    return (str(values.iloc[0]), str(values.iloc[-1]))


def _per_month(per_day: list[tuple[dt.date, int]]) -> list[tuple[dt.date, int]]:
    """``(month, record_count)`` pairs derived from ``per_day``.

    Each entry's date is the first day of its month; months are ordered chronologically and a
    month's count is the sum of all its days.
    """
    totals: dict[dt.date, int] = {}
    for day, count in per_day:
        month = day.replace(day=1)
        totals[month] = totals.get(month, 0) + count
    return [(month, total) for month, total in sorted(totals.items())]


def _value_counts(frame: pd.DataFrame, column: str) -> list[tuple[str, int]]:
    """``(label, count)`` pairs for a column, most common first, blanks as ``"(empty)"``."""
    if frame.empty or column not in frame.columns:
        return []
    series = frame[column].fillna("(empty)").astype(str)
    series = series[series.str.strip() != ""]
    counts = series.value_counts()
    return [(str(label), int(n)) for label, n in counts.items()]


def _polarization_counts(frame: pd.DataFrame) -> list[tuple[str, int]]:
    """Flatten the per-row ``polarization`` lists and count individual codes."""
    if frame.empty or "polarization" not in frame.columns:
        return []
    counts: dict[str, int] = {}
    for values in frame["polarization"]:
        if values is None or (isinstance(values, float) and np.isnan(values)):
            continue
        # A value survives the Parquet round-trip as a list or a numpy array; a single
        # code arrives as a plain string, which must not be iterated character by character.
        items = values if not isinstance(values, str) else [values]
        for item in items:
            label = str(item).strip().upper()
            if label:
                counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _empty_geom_stats(n: int) -> dict[str, int]:
    """Zeroed geometry stats for a frame with no parseable footprints."""
    return {
        "geom_total": 0,
        "geom_missing": n,
        "geom_invalid": 0,
        "geom_antimeridian": 0,
        "geom_near_polar": 0,
        "geom_pole_1deg": 0,
        "geom_land": 0,
        "geom_ocean": 0,
    }


def _geometry_quality(frame: pd.DataFrame) -> dict[str, int]:
    """Vectorized footprint quality counts over the ``geometry`` WKT column."""
    if frame.empty or "geometry" not in frame.columns:
        return _empty_geom_stats(len(frame))

    present = np.asarray(frame["geometry"].notna(), dtype=bool)
    try:
        geoms = shapely.from_wkt(np.asarray(frame["geometry"], dtype=object), on_invalid="ignore")
    except (ValueError, TypeError) as exc:
        logger.debug("Could not parse footprints for the report: %s", exc)
        return _empty_geom_stats(len(frame))

    geom_total = int(present.sum())
    geom_missing = int(len(frame)) - geom_total
    # Unparseable WKT yields a null geometry here, so it reads as "invalid as delivered".
    validity = np.asarray(shapely.is_valid(geoms), dtype=bool)
    geom_invalid = int(((~validity) & present).sum())

    minx, miny, maxx, maxy = shapely.bounds(geoms).T
    width = np.asarray(maxx - minx)
    max_polar = np.maximum(np.abs(np.asarray(maxy)), np.abs(np.asarray(miny)))
    finite = np.isfinite(width)
    geom_antimeridian = int((finite & (width > 180.0)).sum())
    finite_lat = np.isfinite(max_polar)
    geom_near_polar = int((finite_lat & (max_polar > _NEAR_POLAR_LAT)).sum())
    geom_pole_1deg = int((finite_lat & (max_polar > _POLE_1DEG_LAT)).sum())

    # Land vs ocean, judged by each valid footprint's centroid against the bundled land mask.
    usable = present & validity
    geom_land = geom_ocean = 0
    land_union = _land_union()
    if land_union is not None:
        centroids = shapely.centroid(geoms)
        cx = np.asarray(shapely.get_x(centroids))
        cy = np.asarray(shapely.get_y(centroids))
        usable = usable & np.isfinite(cx) & np.isfinite(cy)
        on_land = np.asarray(shapely.contains(land_union, centroids), dtype=bool) & usable
        geom_land = int(on_land.sum())
        geom_ocean = int((usable & ~on_land).sum())

    return {
        "geom_total": geom_total,
        "geom_missing": geom_missing,
        "geom_invalid": geom_invalid,
        "geom_antimeridian": geom_antimeridian,
        "geom_near_polar": geom_near_polar,
        "geom_pole_1deg": geom_pole_1deg,
        "geom_land": geom_land,
        "geom_ocean": geom_ocean,
    }


@dataclasses.dataclass(frozen=True)
class _Completeness:
    span_start: dt.date | None
    span_end: dt.date | None
    missing_days: list[dt.date]
    empty_days: list[dt.date]
    first_day_with_data: dt.date | None
    last_day_with_data: dt.date | None
    mean_per_day: float
    median_per_day: float
    busiest_day: int


# --- Beam/mode color coding (shared by the mix figure, the beam table, the map) -

#: Distinct, colour-blind-friendly palette used to give every beam/mode a stable colour.
_BEAM_PALETTE: tuple[str, ...] = (
    "#2f6fed",  # blue
    "#e67e22",  # orange
    "#27ae60",  # green
    "#c0392b",  # red
    "#8e44ad",  # purple
    "#16a085",  # teal
    "#d35400",  # dark orange
    "#2c3e50",  # slate
    "#f1c40f",  # yellow
    "#1abc9c",  # light teal
)

#: Colour used for a footprint whose beam/mode is empty or unknown.
_BEAM_UNKNOWN: str = "#9aa5b1"


def _beam_legend_label(label: str) -> str:
    """Human-readable legend text for a beam/mode (blank → the neutral “unknown” wording)."""
    return "Unknown / no beam/mode" if label == "(empty)" else label


def _beam_color_map(beam_mix: list[tuple[str, int]]) -> dict[str, str]:
    """Assign each beam/mode label a stable colour (shared across figure, table and map).

    Labels are ordered by descending count (the mix's own order) so the most common modes get
    the first palette entries; the mapping is deterministic for a given catalog.
    """
    return {
        label: _BEAM_PALETTE[index % len(_BEAM_PALETTE)]
        for index, (label, _count) in enumerate(beam_mix)
        if label and label != "(empty)"
    }


# --- Figure rendering (matplotlib, inlined as base64 PNG) --------------------


#: Font family for the mix panels; Times New Roman is requested first (with a metric-compatible
#: fallback) so the figures read in the house serif style, degrading gracefully where it is absent.
_MIX_FONT_CANDIDATES: tuple[str, ...] = ("Times New Roman", "Liberation Serif", "DejaVu Serif")


#: One colour per NISAR frequency sub-family (derived from the beam/mode label).
_SUBFAMILY_COLORS: dict[str, str] = {
    "5 MHz": "#1f6fb2",
    "20 MHz": "#27ae60",
    "40 MHz": "#e67e22",
    "77 MHz": "#8e44ad",
}
#: Order in which the sub-family panels are listed.
_SUBFAMILY_ORDER: tuple[str, ...] = ("5 MHz", "20 MHz", "40 MHz", "77 MHz")


def _subfamily_of(beam: str) -> str | None:
    """The NISAR frequency sub-family (5/20/40/77 MHz) of a beam/mode label, else ``None``."""
    for name in ("5 MHz", "20 MHz", "40 MHz", "77 MHz"):
        if beam.startswith(name):
            return name
    return None


def _beams_for_family(beam_mix: list[tuple[str, int]], family: str) -> list[tuple[str, int]]:
    """The (label, count) beam/mode entries that belong to a frequency sub-family.

    The order follows ``beam_mix`` (most common first) so the stacked sub-figure lists the
    family's modes by popularity.
    """
    return [(label, count) for label, count in beam_mix if _subfamily_of(label) == family]


def _mix_font() -> str:
    """The best available serif font name for the mix panels (Times New Roman if present)."""
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in _MIX_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return "serif"


#: Default raster resolution (points-per-inch) for every inlined figure.
_DEFAULT_FIGURE_DPI: int = 160


def _safe_figure(name: str, draw: Callable[[], None], *, dpi: int = _DEFAULT_FIGURE_DPI) -> str:
    """Render ``draw()`` into an ``<img>`` tag, degrading to a note on any failure."""
    try:
        matplotlib.use("Agg", force=False)
        draw()
    except Exception as exc:  # noqa: BLE001 - a figure must never break the report
        logger.warning("Could not render figure %r for the report: %s", name, exc)
        return (
            f'<p class="empty">Figure “{_e(name)}” could not be rendered '
            f"({html.escape(type(exc).__name__)}).</p>\n"
        )
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close("all")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img alt="{_e(name)}" class="figure" src="data:image/png;base64,{data}">\n'


def _per_day_beam(frame: pd.DataFrame, days: list[dt.date]) -> dict[dt.date, dict[str, int]] | None:
    """Record counts per (day, beam/mode), from the catalog frame's ``start_time``/``beam_mode``.

    Returns a ``{day: {beam: count}}`` mapping covering every day in ``days`` (absent beams are
    simply missing from the inner dict). Returns ``None`` when the frame has no usable columns.
    """
    if frame is None or frame.empty:
        return None
    if "start_time" not in frame.columns or "beam_mode" not in frame.columns:
        return None
    day = frame["start_time"].dt.date
    beam = frame["beam_mode"].fillna("(empty)").astype(str).str.strip()
    beam = beam.replace("", "(empty)")
    grouped = pd.DataFrame({"_d": day, "_b": beam}).groupby(["_d", "_b"]).size()
    result: dict[dt.date, dict[str, int]] = {d: {} for d in days}
    for (d, b), n in grouped.items():
        if d in result:
            result[d][str(b)] = int(n)
    return result


def _fig_volume(stats: CatalogStats, frame: pd.DataFrame | None = None) -> None:
    """Daily record counts (stacked by beam/mode when a catalog frame is given) plus a
    cumulative-records twin axis (figure 1).

    When ``frame`` is provided each day's bar is split into a stack, one segment per beam/mode,
    coloured with the same palette as the footprint map so a mode reads consistently everywhere.
    Without a frame the bars fall back to a single neutral colour.
    """
    days = [d for d, _ in stats.per_day]
    counts = [int(n) for _, n in stats.per_day]
    if not days:
        fig, ax = plt.subplots(figsize=(11, 2.8))
        ax.text(0.5, 0.5, "No records to plot", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    day_beam = _per_day_beam(frame, days) if frame is not None else None
    fig, ax = plt.subplots(figsize=(11, 3.4 if day_beam is not None else 2.8))
    x = np.arange(len(days))
    if day_beam is None:
        ax.bar(x, counts, width=1.0, color="#2f6fed")
    else:
        colors = _beam_color_map(stats.beam_mix)
        beam_order = [label for label, _ in stats.beam_mix]
        bottom = np.zeros(len(days))
        for beam in beam_order:
            vals = np.array([day_beam[d].get(beam, 0) for d in days], dtype=float)
            if vals.max() == 0:
                continue
            ax.bar(x, vals, width=1.0, bottom=bottom, color=colors.get(beam, _BEAM_UNKNOWN))
            bottom += vals
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markersize=8,
                color=colors.get(label, _BEAM_UNKNOWN),
                label=_beam_legend_label(label),
            )
            for label in beam_order[:12]
        ]
        if handles:
            ax.legend(
                handles=handles,
                title="Beam / mode",
                loc="upper left",
                fontsize=7,
                title_fontsize=8,
                framealpha=0.9,
                ncol=2,
            )
    ax.set_ylabel("Records / day")
    ax.set_title("Daily records (by beam/mode) and cumulative records")
    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(counts), color="#c0392b", linewidth=1.4)
    ax2.set_ylabel("Cumulative records")
    ax.set_xticks(x[:: max(1, len(days) // 8)])
    ax.set_xticklabels(
        [f"{d:%Y-%m-%d}" for d in days[:: max(1, len(days) // 8)]], rotation=30, ha="right"
    )
    fig.tight_layout()


def _fig_subfamily(stats: CatalogStats, frame: pd.DataFrame | None, family: str) -> None:
    """Daily records for one frequency sub-family, stacked by its individual beam/modes.

    The family (5/20/40/77 MHz) is the title; within it each day's bar is split into a stack —
    one segment per beam/mode of that family (e.g. "5 MHz" → single-pol VV + dual-pol VV/VH) —
    coloured with the same beam/mode palette as everywhere else. A cumulative-records twin axis
    shows the family's running total.
    """
    days = [d for d, _ in stats.per_day]
    if not days:
        fig, ax = plt.subplots(figsize=(11, 2.6))
        ax.text(0.5, 0.5, "No records to plot", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    family_beams = [label for label, _ in _beams_for_family(stats.beam_mix, family)]
    day_beam = _per_day_beam(frame, days) if frame is not None else None
    fig, ax = plt.subplots(figsize=(11, 3.0))
    x = np.arange(len(days))
    if day_beam is None:
        # No beam detail: a single bar per day in the family's colour.
        color = _SUBFAMILY_COLORS.get(family, "#2f6fed")
        family_per_day = np.array([0] * len(days), dtype=float)
        ax.bar(x, family_per_day, width=1.0, color=color)
    else:
        colors = _beam_color_map(stats.beam_mix)
        family_per_day = np.array(
            [sum(day_beam[d].get(b, 0) for b in family_beams) for d in days], dtype=float
        )
        bottom = np.zeros(len(days))
        for beam in family_beams:
            vals = np.array([day_beam[d].get(beam, 0) for d in days], dtype=float)
            if vals.max() == 0:
                continue
            ax.bar(x, vals, width=1.0, bottom=bottom, color=colors.get(beam, _BEAM_UNKNOWN))
            bottom += vals
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markersize=8,
                color=colors.get(label, _BEAM_UNKNOWN),
                label=_beam_legend_label(label),
            )
            for label in family_beams
        ]
        if handles:
            ax.legend(
                handles=handles,
                title="Beam / mode",
                loc="upper left",
                fontsize=7,
                title_fontsize=8,
                framealpha=0.9,
                ncol=2,
            )
    ax.set_ylabel("Records / day")
    ax.set_title(f"Daily records — {family} (by beam/mode)")
    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(family_per_day), color="#c0392b", linewidth=1.4)
    ax2.set_ylabel("Cumulative records")
    step = max(1, len(days) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f"{d:%Y-%m-%d}" for d in days[::step]], rotation=30, ha="right")
    fig.tight_layout()


def _subfamily_figure(
    stats: CatalogStats, frame: pd.DataFrame | None, family: str
) -> Callable[[], None]:
    """A zero-argument callable drawing the ``family`` daily-records sub-figure."""

    def _draw() -> None:
        _fig_subfamily(stats, frame, family)

    return _draw


def _mix_panel_draw(
    title: str, items: list[tuple[str, int]], colors: dict[str, str] | None
) -> Callable[[], None]:
    """A zero-argument callable that draws one mix panel (used to dodge late-binding closures)."""

    def _draw() -> None:
        _fig_mix_panel(title, items, colors)

    return _draw


def _fig_mix_panel(title: str, items: list[tuple[str, int]], colors: dict[str, str] | None) -> None:
    """A single horizontal mix panel rendered in the serif (Times) font.

    One such figure is produced per mix (product type, processing level, beam/mode, polarization)
    so each reads cleanly at high resolution. The beam/mode panel is colour-coded with the same
    palette used by the footprint map so a mode's colour reads consistently across the figure,
    the beam table and the map.
    """
    fig, ax = plt.subplots(figsize=(6.6, max(2.2, 0.42 * len(items) + 1.2)))
    with matplotlib.rc_context({"font.family": "serif", "font.serif": [_mix_font()]}):
        items = items[:15]
        if not items:
            ax.axis("off")
            ax.set_title(title)
        else:
            labels = [label for label, _ in items][::-1]
            values = [n for _, n in items][::-1]
            bar_colors = [(colors or {}).get(label, "#2f6fed") for label in labels]
            bars = ax.barh(labels, values, color=bar_colors)
            total = sum(values)
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:,} ({100.0 * value / total:.1f}%)",
                    va="center",
                    fontsize=9,
                )
            ax.set_title(title, fontsize=12, pad=10)
            ax.margins(x=0.45)
    fig.tight_layout()


#: One land polygon: ``(outer_ring, [hole, ...])`` where each ring is a list of (lon, lat) points.
_LandRing = list[tuple[float, float]]
_LandPoly = tuple[_LandRing, list[_LandRing]]

#: Bundled Natural Earth 110m land polygons, loaded once and cached (offline coastline).
_LAND_POLYS: list[_LandPoly] | None = None


def _land_polygons() -> list[_LandPoly]:
    """Land polygons from the bundled Natural Earth file: ``(outer_ring, [hole, ...])``.

    Reads ``assets/land_110m.geojson`` (a GeoJSON FeatureCollection of simple
    ``Polygon`` features) once and caches the result. Returns an empty list if the asset is
    missing or malformed, in which case the map simply shows an ocean background.
    """
    global _LAND_POLYS
    if _LAND_POLYS is not None:
        return _LAND_POLYS

    polygons: list[_LandPoly] = []
    path = Path(__file__).parent / "assets" / "land_110m.geojson"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        for feature in document.get("features", []):
            geometry = feature.get("geometry", {})
            if geometry.get("type") != "Polygon":
                continue
            rings = geometry.get("coordinates", [])
            if not rings:
                continue
            outer = [(float(x), float(y)) for x, y in rings[0]]
            holes = [[(float(x), float(y)) for x, y in ring] for ring in rings[1:]]
            polygons.append((outer, holes))
    except (OSError, ValueError) as exc:
        logger.debug("Could not load land polygons for the report map: %s", exc)

    _LAND_POLYS = polygons
    return polygons


#: A single merged land geometry (or ``None`` when unavailable), cached for point-in-land tests.
_LAND_UNION: shapely.geometry.Polygon | shapely.geometry.MultiPolygon | None = None


def _land_union() -> shapely.geometry.Polygon | shapely.geometry.MultiPolygon | None:
    """The bundled land polygons merged into one geometry, cached on first use.

    Returns ``None`` when the asset is missing/empty so callers can degrade to "no land/ocean
    split" rather than failing the report.
    """
    global _LAND_UNION
    if _LAND_UNION is None:
        polygons = _land_polygons()
        if polygons:
            parts = [shapely.geometry.Polygon(outer, holes) for outer, holes in polygons]
            _LAND_UNION = shapely.unary_union(parts)
    return _LAND_UNION


def _draw_land(ax: matplotlib.axes.Axes, ocean: str, land: str, coast: str) -> None:
    """Fill land polygons and stroke their coastlines on ``ax`` (ocean is the background)."""
    for outer, holes in _land_polygons():
        if not outer:
            continue
        xs = [pt[0] for pt in outer]
        ys = [pt[1] for pt in outer]
        ax.fill(xs, ys, color=land)
        ax.plot(xs, ys, color=coast, linewidth=0.5)
        for hole in holes:
            hx = [pt[0] for pt in hole]
            hy = [pt[1] for pt in hole]
            ax.fill(hx, hy, color=ocean)
            ax.plot(hx, hy, color=coast, linewidth=0.5)


def _graticule(ax: matplotlib.axes.Axes, coast: str) -> None:
    """A 30° lon/lat graticule plus the equator and prime meridian on ``ax``."""
    for x in range(-180, 181, 30):
        ax.axvline(x, color="white", linewidth=0.4, alpha=0.5)
    for y in range(-60, 61, 30):
        ax.axhline(y, color="white", linewidth=0.4, alpha=0.5)
    ax.axhline(0.0, color=coast, linewidth=0.6, alpha=0.6)
    ax.axvline(0.0, color=coast, linewidth=0.6, alpha=0.6)


def _footprint_basemap(ax: matplotlib.axes.Axes) -> tuple[str, str, str]:
    """Ocean background + land fill + graticule shared by the two footprint maps."""
    ocean, land, coast = "#cfe0f0", "#eef2f6", "#8aa0b8"
    ax.set_facecolor(ocean)
    _draw_land(ax, ocean, land, coast)
    _graticule(ax, coast)
    return ocean, land, coast


def _exterior_coords(geom: shapely.geometry.BaseGeometry | None) -> NDArray[np.float64] | None:
    """The (N, 2) longitude/latitude ring of a footprint's exterior, else ``None``.

    NISAR footprints are simple polygons; a ``MultiPolygon`` falls back to its largest part so a
    single representative ring is still drawn.
    """
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, shapely.geometry.Polygon):
        return np.asarray(geom.exterior.coords)
    if isinstance(geom, shapely.geometry.MultiPolygon):
        part = max(geom.geoms, key=lambda p: p.area)
        return np.asarray(part.exterior.coords)
    return None


def _draw_footprints(
    ax: matplotlib.axes.Axes,
    geoms: NDArray[Any],  # object array of shapely geometries
    idx: NDArray[np.intp],
    colors: list[str],
    *,
    linewidth: float = 0.6,
    alpha: float = 0.7,
) -> int:
    """Outline the footprints selected by ``idx`` in a single ``PolyCollection``.

    ``colors`` is aligned to ``idx``. Returns the number of footprints actually drawn.
    """
    polys: list[NDArray[np.float64]] = []
    ring_colors: list[str] = []
    for i, color in zip(idx, colors, strict=True):
        coords = _exterior_coords(geoms[int(i)])
        if coords is None:
            continue
        polys.append(coords)
        ring_colors.append(color)
    if polys:
        collection = PolyCollection(
            polys,
            facecolors="none",
            edgecolors=ring_colors,
            linewidths=linewidth,
            alpha=alpha,
        )
        ax.add_collection(collection)
    return len(polys)


def _fig_map(catalog: pd.DataFrame, *, max_plots: int = 12000) -> None:
    """World map of NISAR footprint outlines (coloured by beam/mode) over ocean + land.

    Each footprint is drawn as its actual polygon outline — not a centroid dot — so the true
    swath geometry (shape, tilt, near-polar wrapping) is visible. Outlines are coloured with the
    same beam/mode palette as everywhere else in the report.
    """
    geom = catalog["geometry"]
    valid = shapely.from_wkt(np.asarray(geom, dtype=object), on_invalid="ignore")
    keep = shapely.is_valid(valid)
    minx, miny, maxx, maxy = shapely.bounds(valid).T
    mask = keep & np.isfinite(np.column_stack([minx, miny, maxx, maxy])).all(axis=1)
    if int(mask.sum()) == 0:
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.text(0.5, 0.5, "No plottable footprints", ha="center", va="center")
        ax.axis("off")
        return
    idx = np.flatnonzero(mask)
    if len(idx) > max_plots:
        idx = idx[:: len(idx) // max_plots + 1]

    beam_mix = _value_counts(catalog, "beam_mode")
    beam_colors = _beam_color_map(beam_mix)
    beam_values = (
        catalog["beam_mode"].to_numpy()
        if "beam_mode" in catalog.columns
        else np.array([None] * len(catalog))
    )
    outline_colors = [_beam_color_for(beam_values[i], beam_colors) for i in idx]

    fig, ax = plt.subplots(figsize=(13, 7))
    _footprint_basemap(ax)
    drawn = _draw_footprints(ax, valid, idx, outline_colors)
    _beam_legend(ax, beam_mix, beam_colors)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title(
        "NISAR footprints coloured by beam/mode "
        f"({drawn:,} of {int(mask.sum()):,} parsed footprints outlined)"
    )
    fig.tight_layout()


def _fig_map_beam(
    catalog: pd.DataFrame,
    beam_label: str,
    beam_values: NDArray[np.str_],
    valid: NDArray[Any],  # object array of shapely geometries
    keep: NDArray[np.bool_],
    *,
    max_plots: int = 6000,
) -> None:
    """World map of the footprints belonging to a single beam/mode.

    The subset of the catalog whose ``beam_mode`` equals ``beam_label`` is outlined (in that
    beam/mode's palette colour) over the ocean + land basemap, so the spatial pattern of each
    mode can be inspected in isolation.
    """
    target = beam_values == beam_label
    mask = keep & target
    if int(mask.sum()) == 0:
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.text(0.5, 0.5, "No footprints for this beam/mode", ha="center", va="center")
        ax.axis("off")
        return
    idx = np.flatnonzero(mask)
    if len(idx) > max_plots:
        idx = idx[:: len(idx) // max_plots + 1]
    beam_mix = _value_counts(catalog, "beam_mode")
    color = _beam_color_map(beam_mix).get(beam_label, _BEAM_UNKNOWN)
    fig, ax = plt.subplots(figsize=(13, 7))
    _footprint_basemap(ax)
    drawn = _draw_footprints(ax, valid, idx, [color] * len(idx))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title(
        f"Footprints — beam/mode: {beam_label} " f"({drawn:,} of {int(mask.sum()):,} outlined)"
    )
    fig.tight_layout()


def _beam_maps(catalog: pd.DataFrame, *, max_plots: int = 3000) -> dict[str, str]:
    """A footprint map for every populated beam/mode, keyed by its label (in mix order).

    Geometry is parsed once and reused across all sub-maps. Beam/modes with no record (and the
    empty “no beam” bucket) are skipped. Each map is inlined at a lighter DPI than the main maps.
    """
    if catalog.empty or "beam_mode" not in catalog.columns or "geometry" not in catalog.columns:
        return {}
    beam_values = (
        catalog["beam_mode"].fillna("(empty)").astype(str).str.strip().replace("", "(empty)")
    ).to_numpy()
    valid = shapely.from_wkt(np.asarray(catalog["geometry"], dtype=object), on_invalid="ignore")
    keep = shapely.is_valid(valid)
    result: dict[str, str] = {}
    for label, count in _value_counts(catalog, "beam_mode"):
        if count == 0 or label == "(empty)":
            continue

        def _draw(lab: str = label) -> None:
            _fig_map_beam(catalog, lab, beam_values, valid, keep, max_plots=max_plots)

        result[label] = _safe_figure(f"beam:{label}", _draw, dpi=120)
    return result


#: Centroid colours for the land/ocean split map.
_OCEAN_POINT: str = "#1f6fb2"
_LAND_POINT: str = "#c0392b"


def _fig_land_ocean(catalog: pd.DataFrame, *, max_plots: int = 12000) -> None:
    """World map splitting NISAR footprints into those over ocean vs those over land.

    Each footprint is drawn as its actual polygon outline and classified by its centroid against
    the bundled Natural Earth land mask (the same mask used by the geometry counts). Ocean
    footprints are outlined in blue, land footprints in red.
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    land_union = _land_union()
    if land_union is None:
        ax.text(0.5, 0.5, "Land/ocean split unavailable (no land mask)", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return

    valid = shapely.from_wkt(np.asarray(catalog["geometry"], dtype=object), on_invalid="ignore")
    keep = shapely.is_valid(valid)
    cx = np.asarray(shapely.get_x(shapely.centroid(valid)))
    cy = np.asarray(shapely.get_y(shapely.centroid(valid)))
    usable = keep & np.isfinite(cx) & np.isfinite(cy)
    if int(usable.sum()) == 0:
        ax.text(0.5, 0.5, "No plottable footprints", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return

    on_land = np.asarray(shapely.contains(land_union, shapely.centroid(valid)), dtype=bool) & usable
    n_land = int(on_land.sum())
    n_ocean = int((usable & ~on_land).sum())

    idx = np.flatnonzero(usable)
    if len(idx) > max_plots:
        idx = idx[:: len(idx) // max_plots + 1]
    outline_colors = [_LAND_POINT if bool(on_land[i]) else _OCEAN_POINT for i in idx]

    _footprint_basemap(ax)
    _draw_footprints(ax, valid, idx, outline_colors)
    handles = [
        Patch(facecolor="none", edgecolor=_OCEAN_POINT, label=f"Over ocean ({n_ocean:,})"),
        Patch(facecolor="none", edgecolor=_LAND_POINT, label=f"Over land ({n_land:,})"),
    ]
    ax.legend(
        handles=handles,
        title="Centroid location",
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.9,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title(
        "NISAR footprints over ocean vs land (centroids classified: " f"{n_ocean + n_land:,})"
    )
    fig.tight_layout()


def _beam_color_for(value: object, colors: dict[str, str]) -> str:
    """Hex colour for a raw ``beam_mode`` value (unknown/empty → the neutral colour)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return _BEAM_UNKNOWN
    return colors.get(str(value).strip(), _BEAM_UNKNOWN)


def _beam_legend(
    ax: matplotlib.axes.Axes, beam_mix: list[tuple[str, int]], colors: dict[str, str]
) -> None:
    """Dot legend mapping each beam/mode colour used on the map to its label.

    A trailing grey entry is always shown: it stands for footprints whose beam/mode is empty
    or could not be decoded, which are plotted in the neutral unknown colour.
    """
    handles = []
    for label, _count in beam_mix[:12]:
        if label == "(empty)":
            continue  # represented by the always-present unknown entry below
        color = colors.get(label)
        if color is None:
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=color,
                markersize=6,
                label=label,
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=_BEAM_UNKNOWN,
            markersize=6,
            label="Unknown / no beam/mode",
        )
    )
    if handles:
        ax.legend(
            handles=handles,
            title="Beam / mode",
            loc="upper left",
            fontsize=7,
            title_fontsize=8,
            framealpha=0.9,
            handlelength=1.0,
            borderaxespad=0.6,
            ncol=2,
        )


def _fig_geometry_issues(stats: CatalogStats) -> None:
    """Horizontal bar chart of the geometry-issue counts (figure 4)."""
    labels = [
        "Invalid as delivered",
        "Cross the antimeridian",
        f"Near-polar (|lat| > {_NEAR_POLAR_LAT:.0f}°)",
        f"Within 1° of a pole (|lat| > {_POLE_1DEG_LAT:.0f}°)",
    ]
    values = [
        stats.geom_invalid,
        stats.geom_antimeridian,
        stats.geom_near_polar,
        stats.geom_pole_1deg,
    ]
    total = stats.geom_total or stats.total_records or 1
    fig, ax = plt.subplots(figsize=(11, 3.0))
    bars = ax.barh(labels[::-1], values[::-1], color="#e67e22")
    for bar, value in zip(bars, values[::-1], strict=True):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"{value:,} ({100.0 * value / total:.2f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_title("Footprint geometry issues over the whole archive")
    ax.set_xlabel("Number of footprints")
    ax.margins(x=0.3)
    fig.tight_layout()


# --- HTML rendering ---------------------------------------------------------


def _doctype() -> str:
    return '<!DOCTYPE html>\n<html lang="en">\n'


def _head() -> str:
    return (
        '<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>metadata-asf catalog report</title>\n"
        "<style>\n"
        "  :root { --ink:#1a2230; --muted:#5b6675; --line:#e3e8ef; --bg:#f6f8fa; --card:#ffffff; "
        "--accent:#2f6fed; }"
        "  * { box-sizing:border-box; }"
        "  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
        "Helvetica,Arial,sans-serif; color:var(--ink); background:var(--bg); line-height:1.5; }"
        "  .wrap { max-width:1040px; margin:0 auto; padding:24px 20px 64px; }"
        "  h1 { font-size:22px; margin:0 0 4px; }"
        "  h2 { font-size:16px; margin:36px 0 12px; padding-bottom:6px; "
        "border-bottom:2px solid var(--line); }"
        "  .meta { color:var(--muted); font-size:13px; margin:0; }"
        "  .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); "
        "gap:12px; margin:20px 0; }"
        "  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; "
        "padding:14px 16px; }"
        "  .card .n { font-size:24px; font-weight:700; }"
        "  .card .l { color:var(--muted); font-size:12px; margin-top:2px; }"
        "  .card .sub { color:var(--muted); font-size:11px; margin-top:6px; padding-top:6px; "
        "border-top:1px dashed var(--line); white-space:nowrap; overflow:hidden; text-overflow:"
        "ellipsis; }"
        "  .card .sub .k { color:var(--ink); font-weight:600; }"
        "  table { border-collapse:collapse; width:100%; background:var(--card); "
        "border:1px solid var(--line); border-radius:8px; overflow:hidden; }"
        "  th,td { text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); "
        "font-size:13px; }"
        "  th { background:#eef2f7; color:var(--muted); font-weight:600; }"
        "  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }"
        "  .bar { background:var(--accent); height:14px; border-radius:3px; display:inline-block; "
        "vertical-align:middle; }"
        "  .barwrap { display:flex; align-items:center; gap:8px; }"
        "  .note { font-size:13px; color:var(--muted); background:var(--card); border:1px solid "
        "var(--line); border-left:4px solid var(--accent); border-radius:6px; padding:10px 14px; "
        "margin:12px 0; }"
        "  .disclaimer { font-size:13px; color:#5a4210; background:#fdf4e0; border:1px solid "
        "#f0d9a8; border-left:4px solid #e6a700; border-radius:6px; padding:10px 14px; "
        "margin:12px 0; }"
        "  .pill { display:inline-block; background:#eef2f7; border-radius:20px; padding:2px 10px; "
        "font-size:12px; color:var(--ink); margin:2px 4px 2px 0; }"
        "  footer { margin-top:40px; padding-top:16px; border-top:1px solid var(--line); "
        "color:var(--muted); font-size:12px; }"
        "  .empty { color:var(--muted); font-style:italic; }"
        "  .figure { width:100%; height:auto; background:var(--card); border:1px solid "
        "var(--line); border-radius:8px; margin:14px 0; }"
        "  .logo { height:48px; width:auto; display:block; margin:0 0 8px; }"
        "  details.beammap { margin:10px 0; background:var(--card); border:1px solid "
        "var(--line); border-radius:8px; }"
        "  details.beammap summary { cursor:pointer; padding:10px 14px; font-size:14px; "
        "font-weight:600; list-style:none; }"
        "  details.beammap summary::before { content:'▸ '; color:var(--muted); }"
        "  details.beammap[open] summary::before { content:'▾ '; }"
        "  details.beammap summary::-webkit-details-marker { display:none; }"
        "  details.beammap .count { color:var(--muted); font-weight:400; font-size:12px; }"
        "  details.beammap .figure { margin:4px 14px 14px; }"
        '</style>\n</head>\n<body>\n<div class="wrap">\n'
    )


def _header(stats: CatalogStats) -> str:
    logo = ""
    if stats.mission == "NISAR":
        data_uri = _nisar_logo_data_uri()
        if data_uri:
            logo = f'<img class="logo" alt="NISAR mission logo" src="{data_uri}">\n'
    return (
        logo
        + f"<h1>{_e(stats.mission)} — daily catalog report</h1>\n"
        + f'<p class="meta">metadata-asf · generated {_fmt_dt(stats.generated_at)} UTC</p>\n'
    )


def _link(label: str, url: str) -> str:
    """An ``<a>`` tag for a source link (escaped, no target so it works offline too)."""
    return f'<a href="{_e(url)}">{_e(label)}</a>'


def _about_section(stats: CatalogStats) -> str:
    """A source-links block (provider + tool) and a scope disclaimer."""
    provider = (
        f"<tr><td>{_e(_PROVIDER_NAME)}</td>"
        f"<td>{_link('Website', _PROVIDER_SITE)}</td>"
        f"<td>{_link('API', _PROVIDER_API)}</td>"
        f"<td>{_link('Documentation', _PROVIDER_DOCS)}</td></tr>"
    )
    tool = (
        f"<tr><td>{_e(_REPO_NAME)}</td>"
        f"<td>{_link('GitHub', _REPO_GITHUB)}</td>"
        f"<td>{_link('Documentation', _REPO_DOCS)}</td><td></td></tr>"
    )
    if stats.mission == "NISAR":
        nisar = (
            f"<tr><td>{_e(_NISAR_NAME)}</td>"
            f"<td>{_link('Documentation', _NISAR_DOCS)}</td>"
            f"<td>{_link('Products', _NISAR_PRODUCTS)}</td>"
            f"<td>{_link('Sensor', _NISAR_SENSOR)}</td></tr>"
        )
    else:
        nisar = ""
    table = (
        "<table><thead><tr><th>Source</th><th>Website</th><th>API</th><th>Documentation</th>"
        f"</tr></thead><tbody>{provider}{tool}{nisar}</tbody></table>"
    )
    note = (
        '<p class="meta">The metadata in this catalog is harvested read-only from the '
        "<code>asf_search</code> client, which queries NASA CMR (the “API” link) with "
        "<code>provider=ASF</code>; the “Documentation” link is the ASF documentation site."
    )
    disclaimer = (
        '<p class="disclaimer"><strong>Scope disclaimer.</strong> This report only describes the '
        f"{_e(stats.mission)} metadata that has been collected and stored locally at Ifremer — it "
        "is <strong>not</strong> representative of the complete ASF collection. The record counts, "
        "dates and mixes below reflect what was harvested here, not what is available from ASF "
        "as a whole."
    )
    return (
        '<h2 style="margin-top:20px">About the data sources</h2>\n'
        + disclaimer
        + table
        + note
        + "</p>\n"
    )


def _kpi_cards(stats: CatalogStats) -> str:
    mission = stats.mission
    span = _span_text(stats)
    missing_label = f"Days without {mission} data" + (
        f" in {span}" if stats.span_start and stats.span_end else ""
    )
    cards = [
        (f"Total {mission} records at Ifremer", _fmt_int(stats.total_records)),
        ("Daily Parquet files", _fmt_int(stats.total_files)),
        ("Span", span),
        (missing_label, _fmt_int(len(stats.missing_days))),
        ("Empty days (a daily file with 0 records)", _fmt_int(len(stats.empty_days))),
        (f"Full size of the {mission} metadata", _fmt_bytes(stats.total_size_bytes)),
    ]
    body = "".join(
        f'<div class="card"><div class="n">{_e(value)}</div>'
        f'<div class="l">{_e(label)}</div></div>'
        for label, value in cards
    )
    return f'<div class="cards">{body}</div>\n'


def _volume_section(
    stats: CatalogStats,
    volume_fig: str,
    subfamily_figs: dict[str, str] | None = None,
) -> str:
    rows: list[str] = []
    for month, count in stats.per_month:
        label = _e(f"{month:%Y-%m}")
        rows.append(f'<tr><td>{label}</td><td class="num">{_fmt_int(count)}</td></tr>')
    table = (
        f'<table><thead><tr><th>Month</th><th class="num">Records</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows
        else '<p class="empty">No records to summarise.</p>'
    )
    missing = _day_pills(stats.missing_days)
    empty = _day_pills(stats.empty_days)
    busiest_label = "Busiest day" + (
        f" ({stats.busiest_day_date})" if stats.busiest_day_date else ""
    )
    note = (
        '<p class="note">Every daily file that exists counts as a day the harvest completed; '
        "a <em>missing</em> day is a calendar day between the first and last file with no file at "
        "all (never harvested), while an <em>empty</em> day has a file but zero records (a quiet "
        "day, an outage, or products that were never published).</p>"
    )
    cards = (
        '<div class="cards">'
        + _mini_card("Mean records / day", _fmt_int(int(stats.mean_per_day)))
        + _mini_card("Median records / day", _fmt_int(int(stats.median_per_day)))
        + _mini_card(busiest_label, _fmt_int(stats.busiest_day))
        + _day_product_card("First day with data", stats.first_day_with_data, stats.first_product)
        + _day_product_card("Last day with data", stats.last_day_with_data, stats.last_product)
        + "</div>\n"
    )
    missing_block = (
        f"<p><strong>Missing days</strong> ({len(stats.missing_days)}):</p>{missing}\n"
        if missing
        else ""
    )
    empty_block = (
        f"<p><strong>Empty days</strong> ({len(stats.empty_days)}):</p>{empty}\n" if empty else ""
    )
    subfamily_block = _subfamily_block(stats, subfamily_figs or {})
    return (
        "<h2>1 · Volume and completeness</h2>\n"
        + note
        + cards
        + volume_fig
        + subfamily_block
        + missing_block
        + empty_block
        + table
        + "\n"
    )


def _subfamily_block(stats: CatalogStats, subfamily_figs: dict[str, str]) -> str:
    """Collapsible per-sub-family daily-record figures, one ``<details>`` per populated family.

    Families are ordered 5/20/40/77 MHz; a family with no record in this catalog is skipped.
    """
    if not subfamily_figs:
        return ""
    counts = dict(stats.beam_mix)

    def _family_total(family: str) -> int:
        return sum(n for label, n in counts.items() if _subfamily_of(label) == family)

    intro = (
        '<p class="note">Daily records by frequency sub-family — expand a panel to see the '
        "temporal pattern of one NISAR band (5/20/40/77 MHz). Families with no record in this "
        "catalog are omitted.</p>\n"
    )
    parts: list[str] = [intro]
    for family in _SUBFAMILY_ORDER:
        fig = subfamily_figs.get(family)
        if fig is None:
            continue
        parts.append(
            f'<details class="beammap"><summary>{_e(family)} '
            f'<span class="count">({_fmt_int(_family_total(family))})</span></summary>'
            + fig
            + "</details>\n"
        )
    return "".join(parts)


def _mix_section(stats: CatalogStats, mix_figs: dict[str, str]) -> str:
    """Section 2: a figure per mix followed by the matching data table.

    Each mix (product type, processing level, beam/mode, polarization) gets its own high-DPI PNG
    (serif font) above its table, so every panel is legible at full width.
    """
    beam_colors = _beam_color_map(stats.beam_mix)
    # Platform is table-only (every record shares one platform here), the other four mixes get
    # a high-DPI serif figure above their table.
    panels = [
        ("Platform", stats.platform_mix, None),
        ("Product type", stats.product_mix, None),
        ("Processing level", stats.level_mix, None),
        ("Beam / mode", stats.beam_mix, beam_colors),
        ("Polarization", stats.polarization_mix, None),
    ]
    inner = "".join(
        (
            f'<h3 style="font-size:14px;margin:18px 0 6px">{_e(title)}</h3>\n'
            + mix_figs.get(title, "")
            + _mix_table(items, colors)
        )
        for title, items, colors in panels
    )
    return "<h2>2 · Product and instrument mix</h2>\n" + inner


def _geometry_section(
    stats: CatalogStats,
    map_fig: str,
    land_ocean_fig: str,
    geo_fig: str,
    beam_maps: dict[str, str] | None = None,
) -> str:
    total = stats.geom_total or stats.total_records
    rows = [
        ("Footprints parsed", stats.geom_total),
        ("Missing / unparsable", stats.geom_missing),
        ("Invalid as delivered (self-intersecting)", stats.geom_invalid),
        ("Cross the antimeridian (±180°)", stats.geom_antimeridian),
        (f"Near-polar (|lat| > {_NEAR_POLAR_LAT:.0f}°)", stats.geom_near_polar),
        (f"Within 1° of a pole (|lat| > {_POLE_1DEG_LAT:.0f}°)", stats.geom_pole_1deg),
        ("Centroid over land", stats.geom_land),
        ("Centroid over ocean", stats.geom_ocean),
    ]
    body = "".join(
        f'<tr><td>{_e(label)}</td><td class="num">{_fmt_int(n)}</td>'
        f'<td class="num">{_pct(n, total)}</td></tr>'
        for label, n in rows
    )
    note = (
        '<p class="note">Geometry is reported exactly as stored in the catalog — nothing is '
        "silently repaired. A footprint that crosses the antimeridian arrives as a single ring "
        "whose longitudes jump from about +179 to −179 and must be cut into an eastern and a "
        "western part before any spatial join. Above |lat| 85° a four-corner lon/lat ring stops "
        "describing the swath at all (longitudes converge) and should be reprojected rather than "
        "used in lon/lat.</p>"
    )
    lo_note = (
        '<p class="note">A footprint is <em>over land</em> when its centroid falls inside the '
        "bundled Natural Earth 110&nbsp;m land mask, and <em>over ocean</em> otherwise. Centroids "
        "are used (not the full polygon) so a swath that skims a coastline is classified by its "
        "centre — a deliberate, conservative split, not a per-pixel land/ocean mask.</p>\n"
    )
    beam_maps_block = _beam_maps_block(stats, beam_maps or {})
    return (
        "<h2>3 · Footprint geometry quality</h2>\n"
        + note
        + map_fig
        + beam_maps_block
        + land_ocean_fig
        + lo_note
        + geo_fig
        + '<table><thead><tr><th>Property</th><th class="num">Count</th>'
        '<th class="num">Share</th></tr></thead><tbody>' + body + "</tbody></table>\n"
    )


def _beam_maps_block(stats: CatalogStats, beam_maps: dict[str, str]) -> str:
    """Collapsible per-beam/mode maps, one ``<details>`` per populated beam/mode.

    The counts come from ``stats.beam_mix`` (the canonical order). Beam/modes with no record are
    absent from ``beam_maps`` and therefore skipped.
    """
    if not beam_maps:
        return ""
    counts = dict(stats.beam_mix)
    intro = (
        '<p class="note">Footprints by beam/mode — expand each panel to see the spatial pattern '
        "of a single mode. Modes with no record in this catalog are omitted.</p>\n"
    )
    parts: list[str] = [intro]
    for label, _count in stats.beam_mix:
        fig = beam_maps.get(label)
        if fig is None:
            continue
        n = counts.get(label, 0)
        label_html = f"{_e(label)} <span class='count'>({_fmt_int(n)})</span>"
        parts.append(f'<details class="beammap"><summary>{label_html}</summary>{fig}</details>\n')
    return "".join(parts)


def _footer(stats: CatalogStats) -> str:
    return (
        f"<footer>Generated by <code>metadata-asf report</code> on "
        f"{_fmt_dt(stats.generated_at)} UTC · {stats.total_files} daily file(s), "
        f"{_fmt_int(stats.total_records)} record(s).</footer>\n</div>\n"
    )


def _mix_table(items: list[tuple[str, int]], colors: dict[str, str] | None = None) -> str:
    if not items:
        return '<p class="empty">Nothing to show.</p>\n'
    top = max(n for _, n in items) or 1
    colors = colors or {}
    total = sum(x for _, x in items)
    body = "".join(
        "<tr>"
        f"<td>{_e(label)}</td>"
        f'<td class="num">{_fmt_int(n)}</td>'
        f'<td class="num">{_pct(n, total)}</td>'
        '<td><span class="barwrap"><span class="bar" '
        f'style="width:{int(120 * n / top)}px; background:{colors.get(label, "#2f6fed")}">'
        "</span></span></td>"
        "</tr>"
        for label, n in items
    )
    return (
        '<table><thead><tr><th>Value</th><th class="num">Count</th>'
        '<th class="num">Share</th><th style="width:140px"></th></tr></thead>'
        f"<tbody>{body}</tbody></table>\n"
    )


def _mini_card(label: str, value: str) -> str:
    """A single KPI card with a large value and a small label."""
    return (
        f'<div class="card"><div class="n">{_e(value)}</div>'
        f'<div class="l">{_e(label)}</div></div>'
    )


def _day_product_card(label: str, day: dt.date | None, product: str | None) -> str:
    """A KPI card for a first/last data day, optionally naming the product collected on it.

    The product file name (``granule_id``) is long, so it is truncated with an ellipsis and the
    full name is kept in a ``title`` tooltip on hover.
    """
    date = _e(str(day)) if day else "—"
    sub = ""
    if product:
        shown = product if len(product) <= 34 else product[:33] + "…"
        sub = f'<div class="sub" title="{_e(product)}">{_e(shown)}</div>'
    return (
        f'<div class="card"><div class="n">{date}</div>'
        f'<div class="l">{_e(label)}</div>{sub}</div>'
    )


def _day_pills(days: list[dt.date]) -> str:
    if not days:
        return ""
    if len(days) > 60:
        head = "".join(f'<span class="pill">{_e(str(d))}</span>' for d in days[:60])
        return head + f'<span class="pill">… and {len(days) - 60} more</span>'
    return "".join(f'<span class="pill">{_e(str(d))}</span>' for d in days)


def _span_text(stats: CatalogStats) -> str:
    if stats.span_start and stats.span_end:
        return f"{stats.span_start} → {stats.span_end}"
    return "—"


def _e(value: str) -> str:
    """Escape text for safe inclusion in HTML."""
    return html.escape(value, quote=True)


def _fmt_int(value: int) -> str:
    """Format an integer with thousands separators."""
    return f"{value:,}"


def _pct(part: int, whole: int) -> str:
    """Percentage of ``part`` over ``whole`` (``—`` when the denominator is 0)."""
    if whole <= 0:
        return "—"
    return f"{100.0 * part / whole:.2f}%"


def _fmt_bytes(num_bytes: int) -> str:
    """Human-readable size (KiB/MiB/GiB) from a byte count."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _fmt_dt(value: dt.datetime) -> str:
    """``YYYY-MM-DD HH:MM`` rendering of a (UTC) datetime."""
    return f"{value:%Y-%m-%d %H:%M}"


__all__ = ["CatalogStats", "analyze_catalog", "render_report_html", "write_report"]
