"""Extraction: raw ASF products to a normalized pandas DataFrame."""

from __future__ import annotations

import logging

import asf_search as asf
import pandas as pd

# noqa: F401 - re-exported for the public API surface documented in AGENTS.md section 7.

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


def to_dataframe(products: list[asf.ASFProduct], mission: str) -> pd.DataFrame:
    """Normalize a list of ASF products into the project's tabular schema.

    One row per acquisition with exactly :data:`COLUMNS` in that order (see AGENTS.md section 2):
    ``granule_id``, ``platform``, WKT ``geometry``, UTC-aware ``start_time``/``stop_time``,
    ``polarization`` list, ``beam_mode``, ``product_type`` and derived ``processing_level``.
    Which raw ASF property backs a column comes from the mission profile's ``field_mapping``
    (with sensible defaults for fields where every mission shares the same ASF alias).

    Args:
        products: raw objects returned by :func:`~metadata_asf.search.search`.
        mission: CLI identifier of the mission, selecting the field mapping.

    Returns:
        DataFrame with one row per product and exactly the guaranteed column order/types. A missing
        or unparseable value never aborts the run: that cell stays empty and at most one WARNING is
        logged for each such field across the whole call (AGENTS.md section 5).

    Raises:
        UnknownMissionError: if no profile is registered for ``mission``.
    """
    raise NotImplementedError("metadata_asf.extract.to_dataframe")


__all__ = ["COLUMNS", "to_dataframe"]
