"""Command-line interface for metadata-asf."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from metadata_asf import config as config_module
from metadata_asf import export, extract
from metadata_asf import search as search_module
from metadata_asf.profiles import UnknownMissionError, get_profile
from metadata_asf.utils import parse_date, setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        Parser implementing the documented flags (--mission, --outputdir, --log-verbosity,
        --date, --conf); see AGENTS.md section 4 for each option's meaning and default value.

    Note:
        Options that also exist in a ``--conf`` file default to ``None`` (not to their
        documented fallback) so that only the values the user actually typed override the
        file; the documented defaults then come from the file or the mission profile.
    """
    parser = argparse.ArgumentParser(
        prog="metadata-asf",
        description="Collect ASF SAR metadata as daily Parquet catalogs.",
    )
    parser.add_argument("--mission", default=None, help="Target mission (default: NISAR).")
    parser.add_argument(
        "--outputdir",
        type=Path,
        required=True,
        help="Directory the daily Parquet files are written to.",
    )
    parser.add_argument(
        "--log-verbosity",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Acquisition window: YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD.",
    )
    parser.add_argument("--conf", type=Path, default=None, help="YAML configuration file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``metadata-asf`` console script.

    Orchestrates one full run in this order (flow diagram: AGENTS.md section 11):
        - parse CLI plus conf, then resolve the mission profile;
        - configure logging;
        - run the ASF search over the requested window;
        - extract the results into a DataFrame;
        - write the daily Parquet files and log the final summary.

    Args:
        argv: command-line tokens (defaults to ``sys.argv[1:]``); injected in tests.

    Returns:
        Process exit code per AGENTS.md section 7: ``0`` on success (a run that matched no
        product at all is legitimate and only logs a warning); ``1`` when any error hits search,
        extraction or export; ``2`` for invalid usage such as bad date syntax or unknown mission,
        in which case the available missions are listed to help recover.

    Raises:
        SystemExit: never raised directly; argparse handles it itself for unrecognized flags.
    """
    args = build_parser().parse_args(argv)

    setup_logging(args.log_verbosity or "INFO")

    try:
        date_range = _parsed_date_range(args.date)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    try:
        config = _build_config(args, date_range)
    except FileNotFoundError:
        logger.error("Configuration file not found: %s", args.conf)
        return 2
    except ValidationError as exc:
        logger.error("Invalid configuration:\n%s", exc)
        return 2

    setup_logging(config.log_level)

    try:
        get_profile(config.mission)
    except UnknownMissionError as exc:
        logger.error("%s", exc)
        return 2

    if config.date_range is None:
        logger.error(
            "No acquisition window given: pass --date (YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD) "
            "or set date_range in the configuration file."
        )
        return 2

    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        products = search_module.search(
            mission=config.mission,
            start=config.date_range[0],
            end=config.date_range[1],
            intersects_with=config.ocean_wkt,
            max_results=config.max_results,
            product_types=config.processing_levels,
        )
    except UnknownMissionError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - the CLI boundary: any search failure -> exit 1.
        logger.error("ASF search failed: %s", exc)
        return 1

    if not products:
        logger.warning(
            "No %s acquisition matched the %s..%s window; nothing to write.",
            config.mission,
            config.date_range[0],
            config.date_range[1],
        )
        return 0

    try:
        frame: pd.DataFrame = extract.to_dataframe(products, mission=config.mission)
        written = export.write_daily_parquet(frame, config.output_dir, mission=config.mission)
    except Exception as exc:  # noqa: BLE001 - extraction/export failures -> exit 1.
        logger.error("Extraction or export failed: %s", exc)
        return 1

    _log_summary(frame, written)
    return 0


def _build_config(
    args: argparse.Namespace,
    date_range: tuple[dt.date, dt.date] | None,
) -> config_module.Config:
    """Merge the CLI values with an optional ``--conf`` file into a validated Config.

    Only the CLI values the user actually supplied override the file; the rest falls back to
    the file, then to the mission-profile/Config defaults (AGENTS.md section 4).
    """
    overrides: dict[str, Any] = {
        "mission": args.mission,
        "output_dir": args.outputdir,
        "log_level": args.log_verbosity,
        "date_range": date_range,
    }
    if args.conf is not None:
        return config_module.config_from_yaml(args.conf, **overrides)
    return config_module.Config(**{k: v for k, v in overrides.items() if v is not None})


def _parsed_date_range(value: str | None) -> tuple[dt.date, dt.date] | None:
    """Parse the ``--date`` token (a single date becomes a one-day window)."""
    if value is None:
        return None
    parsed = parse_date(value)
    if isinstance(parsed, tuple):
        return parsed
    return (parsed, parsed)


def _log_summary(frame: pd.DataFrame, written: list[Path]) -> None:
    """Log the final run summary (AGENTS.md section 6)."""
    platforms = frame["platform"].dropna().unique()
    mission = platforms[0] if len(platforms) == 1 else ", ".join(str(p) for p in platforms)
    covered_from = frame["start_time"].min()
    covered_to = frame["start_time"].max()
    print("=== Résumé ===")
    print(f"Mission                : {mission}")
    print(f"Acquisitions totales   : {len(frame)}")
    print(f"Fichiers écrits        : {len(written)}")
    print(f"Dates couvertes        : {covered_from:%Y-%m-%d} → {covered_to:%Y-%m-%d}")
    print("Erreurs                : 0")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "export", "extract", "main", "search_module"]
