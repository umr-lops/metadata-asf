"""Tests for :func:`metadata_asf.cli.main` with the three stages stubbed out."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from metadata_asf import cli


class FakeProduct:
    def __init__(self, scene: str = "SCENE") -> None:
        self.sceneName = scene


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Generator[None, None, None]:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers, root.level
    yield
    root.handlers, root.level = saved_handlers, saved_level


def _stub_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_search(**kwargs: Any) -> list[object]:
        calls["search"] = kwargs
        return [FakeProduct(), FakeProduct()]

    def fake_to_dataframe(products, mission):  # type: ignore[no-untyped-def]
        calls["extract"] = {"mission": mission, "n": len(products)}
        return pd.DataFrame(
            {
                "granule_id": ["A", "B"],
                "platform": ["NISAR", "NISAR"],
                "geometry": ["POLYGON((0 0,1 0,1 1,0 1,0 0))"] * 2,
                "start_time": pd.to_datetime(
                    ["2025-01-01T05:00:00", "2025-01-02T05:00:00"]
                ).tz_localize("UTC"),
                "stop_time": pd.to_datetime(
                    ["2025-01-01T05:00:10", "2025-01-02T05:00:10"]
                ).tz_localize("UTC"),
                "polarization": [["HH"], ["HH"]],
                "beam_mode": [None, None],
                "product_type": ["RSLC", "RSLC"],
                "processing_level": ["L1", "L1"],
            }
        )

    def fake_write(df, output_dir, **kwargs):  # type: ignore[no-untyped-def]
        calls["export"] = {"output_dir": output_dir, "mission": kwargs.get("mission")}
        (output_dir / "NISAR_ocean_20250101.parquet").write_text("x", encoding="utf-8")
        return [output_dir / "NISAR_ocean_20250101.parquet"]

    monkeypatch.setattr(cli.search_module, "search", fake_search)
    monkeypatch.setattr(cli.extract, "to_dataframe", fake_to_dataframe)
    monkeypatch.setattr(cli.export, "write_daily_parquet", fake_write)
    return calls


def test_success_returns_zero_and_runs_all_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_stages(monkeypatch)
    out = tmp_path / "out"
    rc = cli.main(
        ["--mission", "NISAR", "--outputdir", str(out), "--date", "2025-01-01:2025-01-31"]
    )
    assert rc == 0
    assert out.exists()
    assert calls["search"]["start"] == _d(2025, 1, 1)
    assert calls["search"]["end"] == _d(2025, 1, 31)
    assert calls["extract"]["mission"] == "NISAR"
    assert calls["export"]["output_dir"] == out


def test_no_result_returns_zero_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.search_module, "search", lambda **kw: [])

    def boom_to_dataframe(products, mission):  # type: ignore[no-untyped-def]
        raise AssertionError("extraction must not run on an empty search")

    monkeypatch.setattr(cli.extract, "to_dataframe", boom_to_dataframe)
    monkeypatch.setattr(cli.export, "write_daily_parquet", lambda *a, **k: [])
    rc = cli.main(["--outputdir", str(tmp_path), "--date", "2025-01-01"])
    assert rc == 0
    # setup_logging streams to stdout; the empty-result warning must be logged there.
    assert "No NISAR acquisition" in capsys.readouterr().out


def test_search_error_returns_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_search(**kw: object) -> list[object]:
        raise RuntimeError("cmr exploded")

    monkeypatch.setattr(cli.search_module, "search", failing_search)
    rc = cli.main(["--outputdir", str(tmp_path), "--date", "2025-01-01"])
    assert rc == 1


def test_extraction_error_returns_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.search_module, "search", lambda **kw: [FakeProduct()])

    def failing_extract(products, mission):  # type: ignore[no-untyped-def]
        raise ValueError("bad field")

    monkeypatch.setattr(cli.extract, "to_dataframe", failing_extract)
    rc = cli.main(["--outputdir", str(tmp_path), "--date", "2025-01-01"])
    assert rc == 1


def test_export_error_returns_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stages(monkeypatch)

    def failing_write(df, output_dir, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(cli.export, "write_daily_parquet", failing_write)
    rc = cli.main(["--outputdir", str(tmp_path), "--date", "2025-01-01"])
    assert rc == 1


def test_unknown_mission_returns_two_and_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_stages(monkeypatch)
    rc = cli.main(["--mission", "SENTINEL-1", "--outputdir", str(tmp_path), "--date", "2025-01-01"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "Unknown mission" in out
    assert "NISAR" in out


def test_bad_date_returns_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stages(monkeypatch)
    rc = cli.main(["--outputdir", str(tmp_path), "--date", "2025-01-31:2025-01-01"])
    assert rc == 2


def test_missing_date_returns_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stages(monkeypatch)
    rc = cli.main(["--outputdir", str(tmp_path)])
    assert rc == 2


def test_conf_file_provides_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_stages(monkeypatch)
    conf = tmp_path / "conf.yaml"
    conf.write_text(
        "output_dir: /ignored\n" 'date_range: ["2025-02-01", "2025-02-28"]\n' "max_results: 42\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    rc = cli.main(["--outputdir", str(out), "--conf", str(conf)])
    assert rc == 0
    assert calls["search"]["start"] == _d(2025, 2, 1)
    assert calls["search"]["end"] == _d(2025, 2, 28)
    assert calls["search"]["max_results"] == 42


def test_conf_file_beats_defaults_cli_beats_conf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_stages(monkeypatch)
    conf = tmp_path / "conf.yaml"
    conf.write_text('date_range: ["2025-02-01", "2025-02-28"]\nmax_results: 7\n', encoding="utf-8")
    out = tmp_path / "out"
    # CLI --date overrides the file window; the file max_results (7) survives.
    rc = cli.main(["--outputdir", str(out), "--conf", str(conf), "--date", "2025-03-01:2025-03-02"])
    assert rc == 0
    assert calls["search"]["start"] == _d(2025, 3, 1)
    assert calls["search"]["end"] == _d(2025, 3, 2)
    assert calls["search"].get("max_results") == 7


def test_single_date_becomes_one_day_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_stages(monkeypatch)
    rc = cli.main(["--outputdir", str(tmp_path), "--date", "2025-05-10"])
    assert rc == 0
    assert calls["search"]["start"] == _d(2025, 5, 10)
    assert calls["search"]["end"] == _d(2025, 5, 10)


def _d(year: int, month: int, day: int) -> dt.date:
    return dt.date(year, month, day)
