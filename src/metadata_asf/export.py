"""Export: normalized DataFrame to one Parquet file per acquisition day."""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def write_daily_parquet(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    mission: str | None = None,
) -> list[Path]:
    """Write ``df`` as daily Parquet files, named after the acquisition date.

    Rows are grouped by the calendar day of :obj:`start_time`; each group is
    written to ``{mission}_ocean_YYYYMMDD.parquet`` in ``output_dir`` with the
    PyArrow engine, Snappy compression and no index (see schema guarantee in
    ``AGENTS.md`` section 7).

    Args:
        df: DataFrame produced by :func:`~metadata_asf.extract.to_dataframe`;
            must provide a UTC-aware ``start_time`` column.
        output_dir: destination directory, created if missing (parents included);
            it is *not* cleared before writing, so this function can be re-run
            to top up a partial catalog day by day.
        mission: label used in the file names; falls back to the single
            ``platform`` value of the DataFrame, or ``"asf"`` when that is not
            available either.

    Returns:
        The list of written file paths, sorted chronologically. Empty when
        ``df`` has no row.

    Raises:
        ValueError: if ``start_time`` is missing or not a UTC-aware datetime column.
    """
    if "start_time" not in df.columns:
        raise ValueError("df has no 'start_time' column; nothing can be grouped by day")

    dtype = df["start_time"].dtype
    if not isinstance(dtype, pd.DatetimeTZDtype):
        raise ValueError(
            f"'start_time' must be a timezone-aware datetime column, got dtype {dtype!s}"
        )
    if dtype.tz != dt.timezone.utc:
        raise ValueError(f"'start_time' must be pinned to UTC, got timezone {dtype.tz!s}")

    label = _resolve_mission_label(df, mission)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return []

    groups = df.groupby(df["start_time"].dt.date, sort=True)
    written: list[Path] = []
    bar = tqdm(
        groups,
        desc="Écriture Parquet",
        total=len(groups),
        disable=not sys.stdout.isatty(),
    )
    for day, chunk in bar:
        path = output_dir / f"{label}_ocean_{day:%Y%m%d}.parquet"
        chunk.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        size_ko = path.stat().st_size / 1024
        logger.info("Écrit %s — %d acquisitions (%.1f Ko)", path, len(chunk), size_ko)
        written.append(path)

    return sorted(written)


def _resolve_mission_label(df: pd.DataFrame, mission: str | None) -> str:
    """Pick the ``{mission}`` part of the daily file names."""
    if mission:
        return mission
    if "platform" in df.columns and not df.empty and df["platform"].notna().any():
        platforms = df["platform"].dropna().unique()
        if len(platforms) == 1:
            return str(platforms[0])
    return "asf"


__all__ = ["write_daily_parquet"]
