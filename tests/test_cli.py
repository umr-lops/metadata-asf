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
    search_windows: list[tuple[dt.date, dt.date]] = []
    calls["search_windows"] = search_windows

    def fake_search(**kwargs: Any) -> list[object]:
        search_windows.append((kwargs["start"], kwargs["end"]))
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
        [
            "harvest",
            "--mission",
            "NISAR",
            "--outputdir",
            str(out),
            "--start",
            "2025-01-01",
            "--stop",
            "2025-01-31",
        ]
    )
    assert rc == 0
    assert out.exists()
    # Sequential mode: one search per day, each a single-day [day, day] window.
    windows = calls["search_windows"]
    assert windows[0] == (_d(2025, 1, 1), _d(2025, 1, 1))
    assert windows[-1] == (_d(2025, 1, 31), _d(2025, 1, 31))
    assert len(windows) == 31
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
    rc = cli.main(["harvest", "--outputdir", str(tmp_path), "--start", "2025-01-01"])
    assert rc == 0
    # The per-day "no acquisition" warning is silenced; only the starting line + summary show.
    out = capsys.readouterr().out
    assert "=== Résumé ===" in out
    assert "No NISAR acquisition" not in out


def test_search_error_returns_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_search(**kw: object) -> list[object]:
        raise RuntimeError("cmr exploded")

    monkeypatch.setattr(cli.search_module, "search", failing_search)
    rc = cli.main(["harvest", "--outputdir", str(tmp_path), "--start", "2025-01-01"])
    assert rc == 1


def test_extraction_error_returns_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.search_module, "search", lambda **kw: [FakeProduct()])

    def failing_extract(products, mission):  # type: ignore[no-untyped-def]
        raise ValueError("bad field")

    monkeypatch.setattr(cli.extract, "to_dataframe", failing_extract)
    rc = cli.main(["harvest", "--outputdir", str(tmp_path), "--start", "2025-01-01"])
    assert rc == 1


def test_export_error_returns_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stages(monkeypatch)

    def failing_write(df, output_dir, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(cli.export, "write_daily_parquet", failing_write)
    rc = cli.main(["harvest", "--outputdir", str(tmp_path), "--start", "2025-01-01"])
    assert rc == 1


def test_unknown_mission_returns_two_and_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_stages(monkeypatch)
    rc = cli.main(
        [
            "harvest",
            "--mission",
            "SENTINEL-1",
            "--outputdir",
            str(tmp_path),
            "--start",
            "2025-01-01",
        ]
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "Unknown mission" in out
    assert "NISAR" in out


def test_bad_date_returns_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stages(monkeypatch)
    rc = cli.main(
        [
            "harvest",
            "--outputdir",
            str(tmp_path),
            "--start",
            "2025-01-31",
            "--stop",
            "2025-01-01",
        ]
    )
    assert rc == 2


def test_missing_start_returns_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stages(monkeypatch)
    rc = cli.main(["harvest", "--outputdir", str(tmp_path)])
    assert rc == 2


def test_stop_without_start_returns_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stages(monkeypatch)
    rc = cli.main(["harvest", "--outputdir", str(tmp_path), "--stop", "2025-01-05"])
    assert rc == 2


def test_conf_file_provides_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_stages(monkeypatch)
    conf = tmp_path / "conf.yaml"
    conf.write_text(
        "output_dir: /ignored\n" 'date_range: ["2025-02-01", "2025-02-03"]\n' "max_results: 42\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    rc = cli.main(["harvest", "--outputdir", str(out), "--conf", str(conf)])
    assert rc == 0
    # No --start on the CLI: the file window is harvested day by day.
    assert calls["search_windows"] == [
        (_d(2025, 2, 1), _d(2025, 2, 1)),
        (_d(2025, 2, 2), _d(2025, 2, 2)),
        (_d(2025, 2, 3), _d(2025, 2, 3)),
    ]
    assert calls["search"]["max_results"] == 42


def test_conf_file_beats_defaults_cli_beats_conf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_stages(monkeypatch)
    conf = tmp_path / "conf.yaml"
    conf.write_text('date_range: ["2025-02-01", "2025-02-03"]\nmax_results: 7\n', encoding="utf-8")
    out = tmp_path / "out"
    # CLI --start/--stop overrides the file window; the file max_results (7) survives.
    rc = cli.main(
        [
            "harvest",
            "--outputdir",
            str(out),
            "--conf",
            str(conf),
            "--start",
            "2025-03-01",
            "--stop",
            "2025-03-02",
        ]
    )
    assert rc == 0
    assert calls["search_windows"] == [
        (_d(2025, 3, 1), _d(2025, 3, 1)),
        (_d(2025, 3, 2), _d(2025, 3, 2)),
    ]
    assert calls["search"].get("max_results") == 7


def test_start_only_defaults_stop_to_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_stages(monkeypatch)
    rc = cli.main(["harvest", "--outputdir", str(tmp_path), "--start", "2025-05-10"])
    assert rc == 0
    assert calls["search_windows"] == [(_d(2025, 5, 10), _d(2025, 5, 10))]


def test_resume_skips_existing_day_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_stages(monkeypatch)
    out = tmp_path / "out"
    (out / "NISAR_ocean_20250301.parquet").parent.mkdir(parents=True)
    (out / "NISAR_ocean_20250301.parquet").write_text("done", encoding="utf-8")
    rc = cli.main(
        [
            "harvest",
            "--outputdir",
            str(out),
            "--start",
            "2025-03-01",
            "--stop",
            "2025-03-02",
        ]
    )
    assert rc == 0
    # The existing 03-01 file is skipped; only 03-02 is searched.
    assert calls["search_windows"] == [(_d(2025, 3, 2), _d(2025, 3, 2))]


def test_report_writes_html(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    _write_daily(catalog, days=["2025-01-01", "2025-01-02"])
    out = tmp_path / "sub" / "report.html"
    rc = cli.main(["report", "--catalogdir", str(catalog), "--outputfile", str(out)])
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "2" in text  # at least a couple of records reported


def test_report_missing_catalogdir_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["report", "--catalogdir", str(tmp_path / "nope"), "--outputfile", "x.html"])
    assert rc == 2
    assert "not found" in capsys.readouterr().out


def test_report_empty_dir_returns_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = cli.main(["report", "--catalogdir", str(empty), "--outputfile", "x.html"])
    assert rc == 2
    assert "No *.parquet" in capsys.readouterr().out


def _d(year: int, month: int, day: int) -> dt.date:
    return dt.date(year, month, day)


def _write_daily(catalog: Path, days: list[str]) -> None:
    """Write one tiny Parquet per day so the report has something to read."""
    catalog.mkdir(parents=True, exist_ok=True)
    for iso in days:
        date = dt.date.fromisoformat(iso)
        df = pd.DataFrame(
            {
                "granule_id": [f"g-{iso}"],
                "platform": ["NISAR"],
                "geometry": ["POLYGON((0 0,1 0,1 1,0 1,0 0))"],
                "start_time": pd.to_datetime([iso + "T05:00:00"]).tz_localize("UTC"),
                "stop_time": pd.to_datetime([iso + "T05:00:10"]).tz_localize("UTC"),
                "polarization": [["HH"]],
                "beam_mode": [None],
                "product_type": ["RSLC"],
                "processing_level": ["L1"],
            }
        )
        df.to_parquet(catalog / f"NISAR_ocean_{date:%Y%m%d}.parquet", index=False)


__all__ = ["FakeProduct"]
