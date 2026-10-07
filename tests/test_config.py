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
        config.Config(output_dir=Path("./out"), log_level=bad)


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
