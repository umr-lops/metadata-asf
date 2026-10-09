"""Configuration: validated run parameters (Pydantic v2)."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticCustomError

logger = logging.getLogger(__name__)


class Config(BaseModel):
    """Validated configuration for one ``metadata-asf`` run.

    Field resolution order (highest priority first) is documented in
    ``AGENTS.md`` section 4: command-line options > values read from the ``--conf`` YAML file
    > defaults of the mission profile. Every field left unset falls back to those default
    values, except :attr:`output_dir` which has no sensible default and is always required.

    Attributes:
        mission: CLI identifier of the target mission (default ``"NISAR"``).
        output_dir: where daily Parquet files are written; created if missing.
        log_level: root logging level used across the run.
        date_range: ``(start, end)`` acquisition window in UTC calendar days; ``None`` means no
            explicit date filter from either source layer above it (CLI or file).
        ocean_wkt: WKT to restrict acquisitions (ocean filter); falls back to the mission
            profile's ``default_ocean_wkt`` when None.
        processing_levels: optional restriction on which product types are collected;
            falls back to every type supported by the mission profile.
        max_results: hard cap of products per ASF API call.
    """

    model_config = {"frozen": True}

    mission: str = "NISAR"
    output_dir: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    date_range: tuple[dt.date, dt.date] | None = None
    ocean_wkt: str | None = None
    processing_levels: list[str] | None = None
    max_results: int = 10_000

    @property
    def mission_profile(self) -> Any:  # noqa: ANN401 - avoids a hard import cycle at module load.
        """Return the registered :class:`~metadata_asf.profiles.base.MissionProfile`."""
        from metadata_asf.profiles import get_profile

        return get_profile(self.mission)


def config_from_yaml(path: Path, **overrides: Any) -> Config:
    """Build a :class:`Config` from a YAML file with keyword overrides on top.

    Args:
        path: YAML file to read; keys mirror the model's own field names, unknown keys fail fast.
        **overrides: explicit non-``None`` values (usually from the CLI) overriding one file
            entry at a time.

    Returns:
        The merged, validated :class:`Config`.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        pydantic.ValidationError: if required fields are missing or inconsistent, if any
            top-level key is unknown here, or if the YAML document itself is not a mapping
            (scalar or list etc.).
    """
    logger.debug("Building configuration from %s", path)

    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    if not isinstance(document, dict):
        raise ValidationError.from_exception_data(
            title=f"Config input ({path.name})",
            line_errors=[
                {
                    "type": PydanticCustomError(
                        "config_input_not_mapping",
                        "Top-level YAML document must be a mapping, got {document_type}",
                        {"document_type": type(document).__name__},
                    ),
                    "input": document,
                }
            ],
        )

    unknown_keys = set(document) - set(Config.model_fields)
    if unknown_keys:
        raise ValidationError.from_exception_data(
            title=f"Config input ({path.name})",
            line_errors=[
                {
                    "type": PydanticCustomError(
                        "config_input_unknown_key",
                        "Unrecognized top-level key(s) in {path}: {keys}; expected exactly one of: "
                        "{expected}",
                        {
                            "path": path.name,
                            "keys": ", ".join(sorted(unknown_keys)),
                            "expected": ", ".join(Config.model_fields),
                        },
                    ),
                    "input": document,
                }
            ],
        )

    merged_document = dict(document)
    valid_keys = set(Config.model_fields)
    # CLI wins over the file, but only genuine (non-None) explicit values override.
    for name, value in overrides.items():
        if name in valid_keys and value is not None:
            merged_document[name] = value

    # Field-level failures surface as the model's own ValidationError (title "Config"),
    # which already names the offending field and value.
    return Config(**merged_document)


__all__ = ["Config", "config_from_yaml"]
