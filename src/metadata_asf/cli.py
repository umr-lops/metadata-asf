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
import contextlib
import datetime as dt
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from tqdm import tqdm

from metadata_asf import config as config_module
from metadata_asf import export, extract, report
from metadata_asf import search as search_module
from metadata_asf.profiles import UnknownMissionError, get_profile
from metadata_asf.utils import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with the ``harvest`` and ``report`` subcommands.

    Returns:
        The argument parser. ``harvest`` reproduces the pre-subcommand flags
        (``--mission``, ``--outputdir``, ``--log-verbosity``, ``--start``,
        ``--stop``, ``--conf``); ``report`` adds ``--catalogdir`` (required)
        and ``--outputfile``.

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
        "--start",
        default=None,
        help="First day of the acquisition window, YYYY-MM-DD "
        "(required unless date_range is set in --conf).",
    )
    sub.add_argument(
        "--stop",
        default=None,
        help="Last day of the window, YYYY-MM-DD (inclusive; default: --start).",
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

    The requested window is harvested **sequentially, one calendar day at a time**: each day
    is searched, extracted and written on its own so a long span stays resumable (a day whose
    Parquet already exists is skipped on a re-run) and a failure on one day does not lose the
    days that came before it (flow diagram: AGENTS.md section 11).

    Args:
        args: the parsed ``harvest`` subcommand namespace (``mission``, ``outputdir``,
            ``log_verbosity``, ``start``, ``stop``, ``conf``).

    Returns:
        ``0`` on success (a day that matched no product at all is legitimate and only logs a
        warning); ``1`` when any error hits search, extraction or export; ``2`` for invalid
        usage such as bad date syntax or unknown mission, in which case the available
        missions are listed to help recover.
    """
    setup_logging(args.log_verbosity or "INFO")

    try:
        start = _parse_cli_date(args.start, "--start")
        stop = _parse_cli_date(args.stop, "--stop")
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    if start is None and stop is not None:
        logger.error("--stop was given without --start; pass --start (YYYY-MM-DD).")
        return 2
    if stop is not None and start is not None and stop < start:
        logger.error("Invalid window: --start %s is after --stop %s.", start, stop)
        return 2

    try:
        config = _build_config(args, start, stop)
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
            "No acquisition window given: pass --start (YYYY-MM-DD, plus an optional "
            "--stop), or set date_range in the configuration file."
        )
        return 2

    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create output directory %s: %s", config.output_dir, exc)
        return 1
    return _run_sequential(config)


@contextlib.contextmanager
def _quiet(names: tuple[str, ...]) -> Iterator[None]:
    """Temporarily mute loggers (and propagate to their children) while a block runs.

    Used to keep the per-day harvest loop on screen to a single starting line and a tqdm bar
    rather than one line per day/file/empty-day. The previous level of each named logger is
    restored on exit (including on error).
    """
    loggers = [logging.getLogger(name) for name in names]
    saved = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        for lg, level in saved:
            lg.setLevel(level)


def _run_sequential(config: config_module.Config) -> int:
    """Harvest ``config.date_range`` one calendar day at a time.

    For each day the ASF API is searched, the result extracted to a DataFrame and written to
    its ``{mission}_ocean_YYYYMMDD.parquet`` file. A day whose file already exists is skipped so
    a partially finished span is resumable. A day that yields no product is a legitimate, quiet
    day; a day whose search/extraction/export raises is counted as an error and skipped so the
    rest of the window still runs.

    The run prints a single starting line, a tqdm progress bar, and a final summary — the
    per-day, per-file and ASF chatter is silenced for the duration of the loop.

    Returns:
        ``0`` when the window is processed (even if every day was empty); ``1`` when at least
        one day raised an error.
    """
    if config.date_range is None:  # defensive: run_harvest guards this already.
        logger.error("No acquisition window set; nothing to harvest.")
        return 1
    days = _iter_days(config.date_range[0], config.date_range[1])
    logger.info(
        "Harvesting %s: %s → %s (%d day(s)) → %s",
        config.mission,
        days[0],
        days[-1],
        len(days),
        config.output_dir,
    )
    total_records = 0
    written: list[Path] = []
    errors: list[str] = []

    bar = tqdm(days, desc="Harvest", unit=" day", disable=not sys.stdout.isatty())
    with _quiet(
        ("metadata_asf.search", "metadata_asf.export", "metadata_asf.extract", "asf_search")
    ):
        for day in bar:
            bar.set_description(f"Harvest {day:%Y-%m-%d}")
            target = config.output_dir / f"{config.mission}_ocean_{day:%Y%m%d}.parquet"
            if target.exists():
                continue

            try:
                products = search_module.search(
                    mission=config.mission,
                    start=day,
                    end=day,
                    intersects_with=config.ocean_wkt,
                    max_results=config.max_results,
                    product_types=config.processing_levels,
                )
                if not products:
                    continue
                frame = extract.to_dataframe(products, mission=config.mission)
                written.extend(
                    export.write_daily_parquet(frame, config.output_dir, mission=config.mission)
                )
                total_records += int(len(frame))
            except Exception as exc:  # noqa: BLE001 - one bad day must not sink the run.
                logger.error("Harvest failed for %s: %s", day, exc)
                errors.append(f"{day}: {exc}")

    _log_summary(config.mission, total_records, written, len(days), errors)

    if errors:
        logger.error("%d day(s) failed during the harvest: %s", len(errors), "; ".join(errors))
        return 1
    return 0


def _iter_days(start: dt.date, stop: dt.date) -> list[dt.date]:
    """Every calendar day in the inclusive ``[start, stop]`` range, oldest first."""
    days = [start]
    while days[-1] < stop:
        days.append(days[-1] + dt.timedelta(days=1))
    return days


def _build_config(
    args: argparse.Namespace,
    start: dt.date | None,
    stop: dt.date | None,
) -> config_module.Config:
    """Merge the CLI values with an optional ``--conf`` file into a validated Config.

    Only the CLI values the user actually supplied override the file; the rest falls back to
    the file, then to the mission-profile/Config defaults (AGENTS.md section 4). The acquisition
    window is derived from ``--start``/``--stop`` (a bare ``--start`` is a one-day window).
    """
    date_range: tuple[dt.date, dt.date] | None = None
    if start is not None:
        date_range = (start, stop if stop is not None else start)

    overrides: dict[str, Any] = {
        "mission": args.mission,
        "output_dir": args.outputdir,
        "log_level": args.log_verbosity,
        "date_range": date_range,
    }
    if args.conf is not None:
        return config_module.config_from_yaml(args.conf, **overrides)
    return config_module.Config(**{k: v for k, v in overrides.items() if v is not None})


def _parse_cli_date(value: str | None, option: str) -> dt.date | None:
    """Parse a single ``YYYY-MM-DD`` token from ``--start`` or ``--stop``.

    Args:
        value: raw token, or ``None`` when the option was not given.
        option: option name used only in error messages.

    Returns:
        The parsed calendar day, or ``None`` when the option was absent.

    Raises:
        ValueError: on a malformed token, quoting the offending value verbatim.
    """
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Cannot parse {option} {value!r}: expected YYYY-MM-DD.") from exc


def _log_summary(
    mission: str,
    total_records: int,
    written: list[Path],
    days: int,
    errors: list[str],
) -> None:
    """Print the final run summary (AGENTS.md section 6)."""
    day_tokens = sorted(_day_from_filename(p.name) for p in written)
    span = f"{day_tokens[0]} → {day_tokens[-1]}" if day_tokens else "—"
    print("=== Résumé ===")
    print(f"Mission                : {mission}")
    print(f"Acquisitions totales   : {total_records}")
    print(f"Fichiers écrits        : {len(written)}")
    print(f"Jours traités          : {days}")
    print(f"Dates couvertes        : {span}")
    print(f"Erreurs                : {len(errors)}")


def _day_from_filename(name: str) -> str:
    """The ``YYYYMMDD`` acquisition day embedded in a ``{mission}_ocean_YYYYMMDD.parquet`` name."""
    stem = name[: -len(".parquet")] if name.endswith(".parquet") else name
    return stem.rsplit("_", 1)[-1]


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
