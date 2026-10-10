"""End-to-end integration tests: real ASF API, light queries, full pipeline.

These tests hit the live ASF API (no credentials needed for search) and are
deselected by default (see the ``-m 'not integration'`` in ``addopts``).
Run them explicitly with::

    pytest -m integration

They are intentionally *light*: a short recent date window plus a small ocean
polygon, so a single run stays well under a few seconds of API work.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd
import pytest
import shapely.wkt

from metadata_asf import cli, export, extract
from metadata_asf.extract import COLUMNS
from metadata_asf.search import search

pytestmark = pytest.mark.integration

#: North Atlantic box (south of Ireland, off the UK coast) — a persistent NISAR
#: RSLC/GSLC ocean zone; the ~12-day repeat cycle means a 3-day recent window
#: reliably contains a handful of acquisitions here.
_NORTH_ATLANTIC_WKT = "POLYGON((-65.0 50.0, -45.0 50.0, -45.0 60.0, -65.0 60.0, -65.0 50.0))"

_FILE_NAME = re.compile(r"^NISAR_ocean_\d{8}\.parquet$")


def _window(days_back: int = 3) -> tuple[dt.date, dt.date]:
    today = dt.datetime.now(dt.timezone.utc).date()
    return today - dt.timedelta(days=days_back), today


def test_full_pipeline_on_recent_ocean_window(tmp_path: Path) -> None:
    """search -> to_dataframe -> write_daily_parquet against the live ASF API."""
    start, end = _window()
    products = search(
        mission="NISAR",
        start=start,
        end=end,
        intersects_with=_NORTH_ATLANTIC_WKT,
        max_results=500,
    )
    if not products:
        pytest.skip(
            f"No NISAR product in the North Atlantic box for {start}..{end}; nothing to assert"
        )

    frame = extract.to_dataframe(products, mission="NISAR")
    written = export.write_daily_parquet(frame, tmp_path, mission="NISAR")

    # --- DataFrame contract (AGENTS.md section 2 / 7) ---
    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == len(products)
    assert frame["granule_id"].notna().all()
    assert frame["platform"].dropna().unique().tolist() == ["NISAR"]
    assert set(frame["product_type"].dropna()) <= {"RSLC", "GSLC"}
    assert set(frame["processing_level"].dropna()) <= {"L1", "L2"}

    # --- Geometry column is valid WKT ---
    for wkt_value in frame["geometry"].dropna():
        shapely.wkt.loads(wkt_value)

    # --- Timestamps are UTC-aware and inside the requested window ---
    for column in ("start_time", "stop_time"):
        assert frame[column].notna().all()
        assert str(frame[column].dtype).endswith("UTC]")
    lower, upper = start, end + dt.timedelta(days=1)
    assert (frame["start_time"] >= pd.Timestamp(lower, tz="UTC")).all()
    assert (frame["start_time"] < pd.Timestamp(upper, tz="UTC")).all()
    assert (frame["stop_time"] >= frame["start_time"]).all()

    # --- Daily Parquet files ---
    assert written, "no Parquet file was written"
    for path in written:
        assert _FILE_NAME.match(path.name)
        reread = pd.read_parquet(path)
        assert list(reread.columns) == list(COLUMNS)
        assert reread["start_time"].notna().all()
    assert sum(len(pd.read_parquet(p)) for p in written) == len(frame)
    # Every written file corresponds to one acquisition day.
    assert len(written) == frame["start_time"].dt.date.nunique()


def test_cli_end_to_end(tmp_path: Path) -> None:
    """The console entry point drives the same pipeline with a light query."""
    start, end = _window()
    rc = cli.main(
        [
            "harvest",
            "--mission",
            "NISAR",
            "--outputdir",
            str(tmp_path),
            "--log-verbosity",
            "WARNING",
            "--start",
            start.isoformat(),
            "--stop",
            end.isoformat(),
        ]
    )
    # A 0 with files means success; a 0 with no files means the window was empty
    # (legitimate, the CLI only warns) — both are valid for a live-data test.
    assert rc == 0
    written = sorted(tmp_path.glob("NISAR_ocean_*.parquet"))
    if not written:
        pytest.skip(f"CLI run found no product for {start}..{end}; pipeline wiring still exercised")
    reread = pd.concat(pd.read_parquet(p) for p in written)
    assert list(reread.columns) == list(COLUMNS)
    assert reread["granule_id"].notna().all()
