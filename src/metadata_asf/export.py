"""Export: normalized DataFrame to one Parquet file per acquisition day."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_daily_parquet(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write ``df`` as daily Parquet files, named after the acquisition date.

    Rows are grouped by the calendar day of :obj:`start_time`; each group is written to
    ``{mission}_ocean_YYYYMMDD.parquet`` in ``output_dir`` with the PyArrow engine,
    Snappy compression and no index (see schema guarantee in ``AGENTS.md`` section 7).

    Args:
        df: DataFrame produced by :func:`~metadata_asf.extract.to_dataframe`; must
            provide at least a UTC-aware ``start_time`` column.
        output_dir: destination directory, created if missing (parents included); it is
            *not* cleared before writing, so this function can be re-run to top up a
            partial catalog day by day.

    Returns:
        The list of written file paths, sorted chronologically. Empty when ``df`` has no row.

    Raises:
        ValueError: if ``start_time`` is missing or not UTC-aware.
    """
    raise NotImplementedError("metadata_asf.export.write_daily_parquet")


__all__ = ["write_daily_parquet"]
