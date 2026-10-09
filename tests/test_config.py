"""Tests for the Pydantic :class:`~metadata_asf.config.Config` model."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from metadata_asf import config


def test_config_defaults() -> None:
    """Only ``output_dir`` is required; everything else has documented defaults."""
    cfg = config.Config(output_dir=Path("./out"))
    assert cfg.mission == "NISAR"
    assert cfg.log_level == "INFO"
    assert cfg.date_range is None
    assert cfg.ocean_wkt is None
    assert cfg.processing_levels is None
    assert cfg.max_results == 10_000


def test_config_explicit_values() -> None:
    """All fields can be set explicitly and are validated."""
    cfg = config.Config(
        mission="NISAR",
        output_dir=Path("/data/nisar"),
        log_level="DEBUG",
        date_range=(dt.date(2025, 1, 1), dt.date(2025, 1, 31)),
        ocean_wkt="POLYGON((-180 -90, 180 90))",
        processing_levels=["RSLC"],
        max_results=42,
    )
    assert cfg.log_level == "DEBUG"
    assert cfg.date_range == (dt.date(2025, 1, 1), dt.date(2025, 1, 31))
    assert cfg.max_results == 42


@pytest.mark.parametrize("bad", ["TRACE", "info", ""])
def test_config_invalid_log_level(bad: str) -> None:
    """The log level is a closed set; anything else must be rejected."""
    with pytest.raises(ValidationError):
        # Negative test: `bad` is deliberately outside the closed literal set.
        config.Config(output_dir=Path("./out"), log_level=bad)  # type: ignore[arg-type]


def test_reference_yaml_keys_match_model() -> None:
    """A document mirroring ``config.example.yaml`` parses and matches the model fields.

    Placeholder for the real loader (plan step 3): it only asserts that the reference
    configuration uses exactly the keys :class:`~metadata_asf.config.Config` knows about.
    """
    document = r"""
mission: NISAR
output_dir: /data/nisar_ocean
log_level: INFO
date_range: ["2025-01-01", "2025-01-31"]
ocean_wkt: >
  POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))
processing_levels: ["RSLC", "GSLC"]
max_results: 10000
"""
    data = yaml.safe_load(document)
    assert set(data) == set(config.Config.model_fields)


def test_config_is_frozen() -> None:
    """A validated run parameter table has no use case after creation."""
    cfg = config.Config(output_dir=Path("./out"))
    with pytest.raises(ValidationError):  # frozen model rejects any mutation after validation.
        cfg.mission = "OTHER"


def test_config_from_yaml_round_trip(tmp_path: Path) -> None:
    """A reference-like document loads into a validated :class:`Config`."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "mission: NISAR\n"
        "output_dir: /data/nisar\n"
        "log_level: DEBUG\n"
        'date_range: ["2025-01-01", "2025-01-31"]\n'
        "max_results: 500\n",
        encoding="utf-8",
    )
    cfg = config.config_from_yaml(path)
    assert cfg.mission == "NISAR"
    assert cfg.output_dir == Path("/data/nisar")
    assert cfg.log_level == "DEBUG"
    assert cfg.date_range == (dt.date(2025, 1, 1), dt.date(2025, 1, 31))
    assert cfg.max_results == 500


def test_config_from_yaml_cli_override_wins(tmp_path: Path) -> None:
    """Non-``None`` overrides from the CLI beat the file value at a time."""
    path = tmp_path / "config.yaml"
    path.write_text("output_dir: /from/file\nmax_results: 100\n", encoding="utf-8")
    cfg = config.config_from_yaml(path, output_dir=Path("/from/cli"), max_results=None)
    assert cfg.output_dir == Path("/from/cli")
    assert cfg.max_results == 100  # None override leaves the file value untouched.


def test_config_from_yaml_missing_file(tmp_path: Path) -> None:
    """A missing file fails with ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        config.config_from_yaml(tmp_path / "absent.yaml")


def test_config_from_yaml_non_mapping_document(tmp_path: Path) -> None:
    """A top-level YAML scalar/list raises ``ValidationError`` (the ``dict_type`` branch)."""
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        config.config_from_yaml(path)


def test_config_from_yaml_unknown_key(tmp_path: Path) -> None:
    """An unrecognized top-level key raises ``ValidationError`` (the ``extra_forbidden`` branch)."""
    path = tmp_path / "bad.yaml"
    path.write_text("output_dir: /data\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError) as excinfo:
        config.config_from_yaml(path)
    assert "not_a_field" in str(excinfo.value)


def test_config_from_yaml_missing_required(tmp_path: Path) -> None:
    """A document without ``output_dir`` raises ``ValidationError`` (the model re-wrap branch)."""
    path = tmp_path / "bad.yaml"
    path.write_text("mission: NISAR\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        config.config_from_yaml(path)
