"""Command-line interface for metadata-asf.

Two subcommands share the single ``metadata-asf`` console script:

* ``harvest`` — query the ASF API and write daily Parquet catalogs (the original behaviour);
* ``report`` — read a directory of daily Parquet catalogs and render a self-contained HTML
  report on the available metadata (no API calls, no downloads).

Exit codes follow AGENTS.md section 7: ``0`` success, ``1`` a runtime error, ``2`` invalid
usage (bad arguments, unknown mission, missing input, ...).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from metadata_asf import config as config_module
from metadata_asf import export, extract, report
from metadata_asf import search as search_module
from metadata_asf.profiles import UnknownMissionError, get_profile
from metadata_asf.utils import parse_date, setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with the ``harvest`` and ``report`` subcommands.

    Returns:
        The argument parser. ``harvest`` reproduces the pre-subcommand flags
        (``--mission``, ``--outputdir``, ``--log-verbosity``, ``--date``, ``--conf``);
        ``report`` adds ``--catalogdir`` (required) and ``--outputfile``.

    Note:
        Options that can also come from a ``--conf`` file default to ``None`` (not to their
        documented fallback) so that only the values the user actually typed override the
        file; the documented defaults then come from the file or the mission profile.
    """
    parser = argparse.ArgumentParser(
        prog="metadata-asf",
        description="Collect ASF SAR metadata as daily Parquet catalogs and report on them.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    harvest = subparsers.add_parser(
        "harvest",
        help="Query the ASF API and write daily Parquet catalogs.",
        description="Query the ASF API and write daily Parquet catalogs.",
    )
    _add_harvest_args(harvest)

    rep = subparsers.add_parser(
        "report",
        help="Render an HTML report on a directory of daily Parquet catalogs.",
        description="Render a self-contained HTML report on a directory of daily Parquet "
        "catalogs (read-only, no API calls).",
    )
    rep.add_argument(
        "--catalogdir",
        type=Path,
        required=True,
        help="Directory holding the daily Parquet files to report on.",
    )
    rep.add_argument(
        "--outputfile",
        type=Path,
        default=Path("catalog_report.html"),
        help="Path of the HTML report to write (default: catalog_report.html).",
    )
    rep.add_argument(
        "--mission",
        default=None,
        help="Mission label for the report header (default: inferred from the data).",
    )
    _add_log_verbosity(rep)

    return parser


def _add_harvest_args(sub: argparse.ArgumentParser) -> None:
    """Register the harvest options on a (sub)parser."""
    sub.add_argument("--mission", default=None, help="Target mission (default: NISAR).")
    sub.add_argument(
        "--outputdir",
        type=Path,
        required=True,
        help="Directory the daily Parquet files are written to.",
    )
    _add_log_verbosity(sub)
    sub.add_argument(
        "--date",
        default=None,
        help="Acquisition window: YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD.",
    )
    sub.add_argument("--conf", type=Path, default=None, help="YAML configuration file.")


def _add_log_verbosity(sub: argparse.ArgumentParser) -> None:
    """Register the shared ``--log-verbosity`` flag on a (sub)parser."""
    sub.add_argument(
        "--log-verbosity",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging verbosity (default: INFO).",
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``metadata-asf`` console script.

    Dispatches on the first positional token to :func:`run_harvest` or :func:`run_report`.

    Args:
        argv: command-line tokens (defaults to ``sys.argv[1:]``); injected in tests.

    Returns:
        Process exit code per AGENTS.md section 7: ``0`` on success (a harvest that matched
        no product at all is legitimate and only logs a warning); ``1`` on a runtime error in
        search, extraction, export or report generation; ``2`` for invalid usage such as bad
        date syntax, an unknown mission or a missing catalog directory.
    """
    args = build_parser().parse_args(argv)
    if args.command == "report":
        return run_report(args)
    return run_harvest(args)


def run_report(args: argparse.Namespace) -> int:
    """Generate the HTML report from a catalog directory.

    Args:
        args: the parsed ``report`` subcommand namespace (``catalogdir``, ``outputfile``,
            ``mission``, ``log_verbosity``).

    Returns:
        ``0`` on success; ``2`` if the catalog directory is missing or holds no Parquet file
        (a usage error — the user pointed at the wrong place); ``1`` on a genuine read error.
    """
    setup_logging(args.log_verbosity or "INFO")
    try:
        report.write_report(
            args.catalogdir,
            args.outputfile,
            mission=args.mission,
        )
    except (FileNotFoundError, ValueError) as exc:
        # Missing directory or no Parquet file: the caller pointed at the wrong place.
        logger.error("%s", exc)
        return 2
    except OSError as exc:
        logger.error("Report generation failed: %s", exc)
        return 1
    return 0


def run_harvest(args: argparse.Namespace) -> int:
    """Run a full harvest: search, extract and write the daily Parquet catalogs.

    Orchestrates one run in this order (flow diagram: AGENTS.md section 11):
        - parse CLI plus conf, then resolve the mission profile;
        - configure logging;
        - run the ASF search over the requested window;
        - extract the results into a DataFrame;
        - write the daily Parquet files and log the final summary.

    Args:
        args: the parsed ``harvest`` subcommand namespace (``mission``, ``outputdir``,
            ``log_verbosity``, ``date``, ``conf``).

    Returns:
        ``0`` on success (a run that matched no product at all is legitimate and only logs a
        warning); ``1`` when any error hits search, extraction or export; ``2`` for invalid
        usage such as bad date syntax or unknown mission, in which case the available
        missions are listed to help recover.
    """
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


__all__ = [
    "build_parser",
    "export",
    "extract",
    "main",
    "run_harvest",
    "run_report",
    "search_module",
]
