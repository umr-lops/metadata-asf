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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

from metadata_asf.profiles import UnknownMissionError, get_profile

logger = logging.getLogger(__name__)

#: Trailing ``YYYYMMDD`` in a daily file name, e.g. ``NISAR_ocean_20250101.parquet``.
_DAY_IN_NAME: re.Pattern[str] = re.compile(r"(\d{8})\.parquet$")

#: |latitude| above which a lon/lat four-corner ring stops describing a swath
#: (longitudes converge); such scenes are flagged, not repaired.
_NEAR_POLAR_LAT: float = 85.0
#: |latitude| above which a footprint is "within 1 degree of a pole".
_POLE_1DEG_LAT: float = 89.0


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
    mean_per_day: float
    median_per_day: float
    busiest_day: int
    per_day: list[tuple[dt.date, int]]
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
    parts: list[str] = [_doctype(), _head(), _header(stats), _kpi_cards(stats)]
    parts.append(_volume_section(stats, _safe_figure("volume", lambda: _fig_volume(stats))))
    parts.append(_mix_section(stats, _safe_figure("mix", lambda: _fig_mix(stats))))
    map_html = (
        _safe_figure("map", lambda: _fig_map(catalog))
        if catalog is not None and not catalog.empty and "geometry" in catalog.columns
        else ""
    )
    geo_fig = _safe_figure("geometry", lambda: _fig_geometry_issues(stats))
    parts.append(_geometry_section(stats, map_html, geo_fig))
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
    geom = _geometry_quality(combined)
    completeness = _completeness(file_days, file_rows)
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
        mean_per_day=completeness.mean_per_day,
        median_per_day=completeness.median_per_day,
        busiest_day=completeness.busiest_day,
        per_day=per_day,
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
    return _Completeness(
        span_start=span_start,
        span_end=span_end,
        missing_days=missing_days,
        empty_days=empty_days,
        first_day_with_data=_first_day_with_data(file_days, file_rows),
        mean_per_day=(sum(record_counts) / len(record_counts)) if record_counts else 0.0,
        median_per_day=float(pd.Series(record_counts).median()) if record_counts else 0.0,
        busiest_day=max(record_counts) if record_counts else 0,
    )


def _first_day_with_data(
    file_days: list[dt.date | None],
    file_rows: list[int],
) -> dt.date | None:
    """Earliest day whose file actually carries at least one record."""
    days = [d for d, n in zip(file_days, file_rows, strict=False) if d is not None and n > 0]
    return min(days) if days else None


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


def _geometry_quality(frame: pd.DataFrame) -> dict[str, int]:
    """Vectorized footprint quality counts over the ``geometry`` WKT column."""
    if frame.empty or "geometry" not in frame.columns:
        return {
            "geom_total": 0,
            "geom_missing": int(len(frame)),
            "geom_invalid": 0,
            "geom_antimeridian": 0,
            "geom_near_polar": 0,
            "geom_pole_1deg": 0,
        }

    present = np.asarray(frame["geometry"].notna(), dtype=bool)
    try:
        geoms = shapely.from_wkt(np.asarray(frame["geometry"], dtype=object), on_invalid="ignore")
    except (ValueError, TypeError) as exc:
        logger.debug("Could not parse footprints for the report: %s", exc)
        return {
            "geom_total": 0,
            "geom_missing": int(len(frame)),
            "geom_invalid": 0,
            "geom_antimeridian": 0,
            "geom_near_polar": 0,
            "geom_pole_1deg": 0,
        }

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

    return {
        "geom_total": geom_total,
        "geom_missing": geom_missing,
        "geom_invalid": geom_invalid,
        "geom_antimeridian": geom_antimeridian,
        "geom_near_polar": geom_near_polar,
        "geom_pole_1deg": geom_pole_1deg,
    }


@dataclasses.dataclass(frozen=True)
class _Completeness:
    span_start: dt.date | None
    span_end: dt.date | None
    missing_days: list[dt.date]
    empty_days: list[dt.date]
    first_day_with_data: dt.date | None
    mean_per_day: float
    median_per_day: float
    busiest_day: int


# --- Figure rendering (matplotlib, inlined as base64 PNG) --------------------


def _safe_figure(name: str, draw: Callable[[], None]) -> str:
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
    plt.gcf().savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close("all")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img alt="{_e(name)}" class="figure" src="data:image/png;base64,{data}">\n'


def _fig_volume(stats: CatalogStats) -> None:
    """Daily record counts with a cumulative-records twin axis (figure 1)."""
    days = [d for d, _ in stats.per_day]
    counts = [int(n) for _, n in stats.per_day]
    if not days:
        fig, ax = plt.subplots(figsize=(11, 2.8))
        ax.text(0.5, 0.5, "No records to plot", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    fig, ax = plt.subplots(figsize=(11, 2.8))
    x = np.arange(len(days))
    ax.bar(x, counts, width=1.0, color="#2f6fed")
    ax.set_ylabel("Records / day")
    ax.set_title("Daily records and cumulative records")
    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(counts), color="#c0392b", linewidth=1.4)
    ax2.set_ylabel("Cumulative records")
    ax.set_xticks(x[:: max(1, len(days) // 8)])
    ax.set_xticklabels(
        [f"{d:%Y-%m-%d}" for d in days[:: max(1, len(days) // 8)]], rotation=30, ha="right"
    )
    fig.tight_layout()


def _fig_mix(stats: CatalogStats) -> None:
    """Side-by-side horizontal mix panels: product type, level, beam, polarization."""
    panels = [
        ("Product type", stats.product_mix),
        ("Processing level", stats.level_mix),
        ("Beam / mode", stats.beam_mix),
        ("Polarization", stats.polarization_mix),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.6), sharey=False)
    for ax, (title, items) in zip(axes, panels, strict=True):
        items = items[:15]
        if not items:
            ax.axis("off")
            ax.set_title(title)
            continue
        labels = [label for label, _ in items][::-1]
        values = [n for _, n in items][::-1]
        bars = ax.barh(labels, values, color="#2f6fed")
        total = sum(values)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"{value:,} ({100.0 * value / total:.1f}%)",
                va="center",
                fontsize=8,
            )
        ax.set_title(title)
        ax.margins(x=0.35)
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


def _fig_map(catalog: pd.DataFrame, *, max_plots: int = 6000) -> None:
    """World map of sample footprint corners/centroids over an ocean + land basemap."""
    geom = catalog["geometry"]
    valid = shapely.from_wkt(np.asarray(geom, dtype=object), on_invalid="ignore")
    keep = shapely.is_valid(valid)
    minx, miny, maxx, maxy = shapely.bounds(valid).T
    mask = keep & np.isfinite(np.column_stack([minx, miny, maxx, maxy])).all(axis=1)
    if int(mask.sum()) == 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.text(0.5, 0.5, "No plottable footprints", ha="center", va="center")
        ax.axis("off")
        return
    idx = np.flatnonzero(mask)
    if len(idx) > max_plots:
        idx = idx[:: len(idx) // max_plots + 1]
    fig, ax = plt.subplots(figsize=(11, 5))
    ocean, land, coast = "#cfe0f0", "#eef2f6", "#8aa0b8"
    ax.set_facecolor(ocean)
    _draw_land(ax, ocean, land, coast)
    for x in range(-180, 181, 30):
        ax.axvline(x, color="white", linewidth=0.4, alpha=0.5)
    for y in range(-60, 61, 30):
        ax.axhline(y, color="white", linewidth=0.4, alpha=0.5)
    ax.axhline(0.0, color=coast, linewidth=0.6, alpha=0.6)
    ax.axvline(0.0, color=coast, linewidth=0.6, alpha=0.6)
    ax.plot(
        minx[idx],
        miny[idx],
        linestyle="none",
        marker=".",
        markersize=2.2,
        color="#2f6fed",
        alpha=0.55,
    )
    ax.plot(
        maxx[idx],
        maxy[idx],
        linestyle="none",
        marker=".",
        markersize=2.2,
        color="#2f6fed",
        alpha=0.55,
    )
    ax.plot(
        (minx + maxx)[idx] / 2.0,
        (miny + maxy)[idx] / 2.0,
        linestyle="none",
        marker=".",
        markersize=1.4,
        color="#c0392b",
        alpha=0.5,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title(
        "Sample of footprint corners and centroids"
        f" ({len(idx):,} of {int(mask.sum()):,} parsed footprints)"
    )
    fig.tight_layout()


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
        "  .pill { display:inline-block; background:#eef2f7; border-radius:20px; padding:2px 10px; "
        "font-size:12px; color:var(--ink); margin:2px 4px 2px 0; }"
        "  footer { margin-top:40px; padding-top:16px; border-top:1px solid var(--line); "
        "color:var(--muted); font-size:12px; }"
        "  .empty { color:var(--muted); font-style:italic; }"
        "  .figure { width:100%; height:auto; background:var(--card); border:1px solid "
        "var(--line); border-radius:8px; margin:14px 0; }"
        '</style>\n</head>\n<body>\n<div class="wrap">\n'
    )


def _header(stats: CatalogStats) -> str:
    return (
        f"<h1>{_e(stats.mission)} — daily catalog report</h1>\n"
        f'<p class="meta">metadata-asf · generated {_fmt_dt(stats.generated_at)} UTC</p>\n'
    )


def _kpi_cards(stats: CatalogStats) -> str:
    span = _span_text(stats)
    cards = [
        ("Records", _fmt_int(stats.total_records)),
        ("Daily files", _fmt_int(stats.total_files)),
        ("Span", span),
        ("Missing days", _fmt_int(len(stats.missing_days))),
        ("Empty days", _fmt_int(len(stats.empty_days))),
        ("Size", _fmt_bytes(stats.total_size_bytes)),
    ]
    body = "".join(
        f'<div class="card"><div class="n">{_e(value)}</div>'
        f'<div class="l">{_e(label)}</div></div>'
        for label, value in cards
    )
    return f'<div class="cards">{body}</div>\n'


def _volume_section(stats: CatalogStats, volume_fig: str) -> str:
    rows: list[str] = []
    for day, count in stats.per_day:
        rows.append(f'<tr><td>{_e(str(day))}</td><td class="num">{_fmt_int(count)}</td></tr>')
    table = (
        f'<table><thead><tr><th>Day</th><th class="num">Records</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows
        else '<p class="empty">No records to summarise.</p>'
    )
    missing = _day_pills(stats.missing_days)
    empty = _day_pills(stats.empty_days)
    first = _e(str(stats.first_day_with_data)) if stats.first_day_with_data else "—"
    note = (
        '<p class="note">Every daily file that exists counts as a day the harvest completed; '
        "a <em>missing</em> day is a calendar day between the first and last file with no file at "
        "all (never harvested), while an <em>empty</em> day has a file but zero records (a quiet "
        "day, an outage, or products that were never published).</p>"
    )
    cards = (
        '<div class="cards">'
        + _mini_card("Mean / day", _fmt_int(int(stats.mean_per_day)))
        + _mini_card("Median / day", _fmt_int(int(stats.median_per_day)))
        + _mini_card("Busiest day", _fmt_int(stats.busiest_day))
        + _mini_card("First day with data", first)
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
    return (
        "<h2>1 · Volume and completeness</h2>\n"
        + note
        + cards
        + volume_fig
        + missing_block
        + empty_block
        + table
        + "\n"
    )


def _mix_section(stats: CatalogStats, mix_fig: str) -> str:
    blocks = [
        ("Platform", stats.platform_mix),
        ("Product type", stats.product_mix),
        ("Processing level", stats.level_mix),
        ("Beam / mode", stats.beam_mix),
        ("Polarization", stats.polarization_mix),
    ]
    inner = "".join(
        (f'<h3 style="font-size:14px;margin:18px 0 6px">{_e(title)}</h3>\n' + _mix_table(items))
        for title, items in blocks
    )
    return "<h2>2 · Product and instrument mix</h2>\n" + mix_fig + inner


def _geometry_section(stats: CatalogStats, map_fig: str, geo_fig: str) -> str:
    total = stats.geom_total or stats.total_records
    rows = [
        ("Footprints parsed", stats.geom_total),
        ("Missing / unparsable", stats.geom_missing),
        ("Invalid as delivered (self-intersecting)", stats.geom_invalid),
        ("Cross the antimeridian (±180°)", stats.geom_antimeridian),
        (f"Near-polar (|lat| &gt; {_NEAR_POLAR_LAT:.0f}°)", stats.geom_near_polar),
        (f"Within 1° of a pole (|lat| &gt; {_POLE_1DEG_LAT:.0f}°)", stats.geom_pole_1deg),
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
    return (
        "<h2>3 · Footprint geometry quality</h2>\n"
        + note
        + map_fig
        + geo_fig
        + '<table><thead><tr><th>Property</th><th class="num">Count</th>'
        '<th class="num">Share</th></tr></thead><tbody>' + body + "</tbody></table>\n"
    )


def _footer(stats: CatalogStats) -> str:
    return (
        f"<footer>Generated by <code>metadata-asf report</code> on "
        f"{_fmt_dt(stats.generated_at)} UTC · {stats.total_files} daily file(s), "
        f"{_fmt_int(stats.total_records)} record(s).</footer>\n</div>\n"
    )


def _mix_table(items: list[tuple[str, int]]) -> str:
    if not items:
        return '<p class="empty">Nothing to show.</p>\n'
    top = max(n for _, n in items) or 1
    body = "".join(
        "<tr>"
        f"<td>{_e(label)}</td>"
        f'<td class="num">{_fmt_int(n)}</td>'
        f'<td class="num">{_pct(n, sum(x for _, x in items))}</td>'
        '<td><span class="barwrap"><span class="bar" '
        f'style="width:{int(120 * n / top)}px"></span></span></td>'
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
