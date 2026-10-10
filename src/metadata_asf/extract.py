"""Extraction: raw ASF products to a normalized pandas DataFrame."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any, cast

import asf_search as asf
import pandas as pd
import shapely.errors
import shapely.geometry

from metadata_asf.profiles import get_profile
from metadata_asf.utils import utc_datetime

logger = logging.getLogger(__name__)


#: Schema columns in their guaranteed order (AGENTS.md section 2).
COLUMNS: tuple[str, ...] = (
    "granule_id",
    "platform",
    "geometry",
    "start_time",
    "stop_time",
    "polarization",
    "beam_mode",
    "product_type",
    "processing_level",
)

#: Columns holding UTC timestamps, coerced to ``datetime64[ns, UTC]`` on the way out.
_DATETIME_COLUMNS: tuple[str, ...] = ("start_time", "stop_time")

#: Columns filled structurally or by derivation, never through the profile's field mapping.
_STRUCTURAL_COLUMNS: frozenset[str] = frozenset({"geometry", "processing_level"})


def to_dataframe(products: Sequence[asf.ASFProduct], mission: str) -> pd.DataFrame:
    """Normalize a list of ASF products into the project's tabular schema.

    One row per acquisition with exactly :data:`COLUMNS` in that order (see AGENTS.md
    section 2): ``granule_id``, ``platform``, WKT ``geometry``, UTC-aware
    ``start_time``/``stop_time``, a ``polarization`` list, ``beam_mode``,
    ``product_type`` and the derived ``processing_level``.

    Which raw ASF properties back a column comes from the mission profile's
    ``field_mapping``. The ``geometry`` column is always read structurally from
    ``product.geometry`` (GeoJSON, converted to WKT) and ``processing_level`` is
    derived from ``product_type`` through the profile's ``supported_products``.
    A column absent from the mapping (e.g. ``beam_mode`` on NISAR) is left empty
    without warning: the mission simply does not expose it.

    Args:
        products: raw objects returned by :func:`~metadata_asf.search.search`.
        mission: CLI identifier of the mission, selecting the field mapping.

    Returns:
        DataFrame with one row per product and exactly the guaranteed column
        order/types. A missing or unparseable value never aborts the run: that
        cell stays empty and at most one WARNING is logged for each such field
        across the whole call (AGENTS.md section 5).

    Raises:
        UnknownMissionError: if no profile is registered for ``mission``.
        ValueError: if the profile maps a column that is not part of the schema.
    """
    profile = get_profile(mission)
    for column in profile.field_mapping:
        if column not in COLUMNS or column in _STRUCTURAL_COLUMNS:
            raise ValueError(
                f"Profile {mission!r} maps {column!r}, which is not a mappable schema column; "
                f"allowed: {sorted(set(COLUMNS) - _STRUCTURAL_COLUMNS)}"
            )

    level_by_type = {ptype: level for level, ptype in profile.supported_products}
    rows: dict[str, list[object]] = {column: [] for column in COLUMNS}
    missing: dict[str, int] = {}

    def note_missing(column: str) -> None:
        missing[column] = missing.get(column, 0) + 1

    for product in products:
        properties = dict(product.properties) if product.properties else {}

        footprint, footprint_missing = _footprint_wkt(product)
        rows["geometry"].append(footprint)
        if footprint_missing:
            note_missing("geometry")

        product_type: str | None = None
        for column, sources in profile.field_mapping.items():
            if column == "polarization":
                merged = _merge_polarizations([properties.get(source) for source in sources])
                if not merged:
                    note_missing("polarization")
                rows["polarization"].append(merged)
                continue

            value = _first_present(properties, sources)
            if value is None:
                rows[column].append(None)
                note_missing(column)
                continue
            if column in _DATETIME_COLUMNS:
                try:
                    rows[column].append(utc_datetime(cast("dt.date | dt.datetime | str", value)))
                except ValueError:
                    logger.debug("Unparseable timestamp for %s: %r", column, value)
                    rows[column].append(None)
                    note_missing(column)
            else:
                text = str(value)
                rows[column].append(text)
                if column == "product_type":
                    product_type = text

        # Mappable columns absent from this mission's mapping (e.g. ``beam_mode`` on
        # NISAR) are left empty on purpose, so no warning is counted for them.
        for column in COLUMNS:
            if column not in profile.field_mapping and column not in _STRUCTURAL_COLUMNS:
                rows[column].append(None)

        level = level_by_type.get(product_type) if product_type is not None else None
        rows["processing_level"].append(level)
        if level is None:
            note_missing("processing_level")

    # ``beam_mode`` has no ASF property on some missions (e.g. NISAR); when the profile
    # provides a decoder it is derived from ``granule_id`` (NISAR encodes it in the
    # filename MODE/POLE fields). Rows already carrying a value are left untouched.
    if profile.decode_beam_mode is not None:
        for index, granule_id in enumerate(rows["granule_id"]):
            if rows["beam_mode"][index] is None and granule_id is not None:
                rows["beam_mode"][index] = profile.decode_beam_mode(str(granule_id))

    frame = pd.DataFrame(rows)
    for column in _DATETIME_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)

    for column in sorted(missing):
        logger.warning(
            "%d/%d product(s) have no usable value for %r; the cell is left empty",
            missing[column],
            len(products),
            column,
        )
    return frame


def _footprint_wkt(product: asf.ASFProduct) -> tuple[str | None, bool]:
    """Convert the product's GeoJSON footprint to WKT.

    Args:
        product: raw ASF product carrying a GeoJSON ``geometry`` attribute.

    Returns:
        ``(wkt, missing)``: the WKT string (``None`` when the footprint is
        absent, empty or malformed) and a flag telling whether a warning should
        be counted for this row.
    """
    geojson: object = getattr(product, "geometry", None)
    if not geojson:
        return None, True
    try:
        shape = shapely.geometry.shape(cast(Any, geojson))
    except (AttributeError, TypeError, ValueError, shapely.errors.GeometryError):
        logger.debug("Malformed footprint on %r", getattr(product, "sceneName", "?"))
        return None, True
    if shape.is_empty:
        return None, True
    return shape.wkt, False


def _first_present(properties: dict[str, object], sources: list[str]) -> object:
    """Return the first non-``None`` value among the listed source properties."""
    for source in sources:
        value = properties.get(source)
        if value is not None:
            return value
    return None


def _merge_polarizations(values: list[object]) -> list[str]:
    """Merge several raw polarization values into one sorted, de-duplicated list.

    ASF may carry a polarization as a plain string or a list of strings, and NISAR
    splits it across two properties (main band and side band), so every source
    value is flattened, normalized and de-duplicated case-insensitively.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            items: Sequence[object] = [value]
        elif isinstance(value, (list, tuple, set)):
            items = cast(Sequence[object], value)
        else:
            items = [value]
        for item in items:
            if item is None:
                continue
            text = str(item).strip().upper()
            if text and text not in seen:
                seen.add(text)
                merged.append(text)
    return sorted(merged)


__all__ = ["COLUMNS", "to_dataframe"]
