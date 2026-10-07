"""Command-line interface for metadata-asf."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        Parser implementing the documented flags (--mission, --outputdir, --log-verbosity,
        --date, --conf); see AGENTS.md section 4 for each option's meaning and default value.
    """
    raise NotImplementedError("metadata_asf.cli.build_parser")


def main() -> int:
    """Entry point of the ``metadata-asf`` console script.

    Orchestrates one full run in this order (flow diagram: AGENTS.md section 11):
        - parse CLI plus conf, then resolve the mission profile;
        - configure logging;
        - loop over days calling the ASF search;
        - extract the results into a DataFrame and group rows by acquisition date;
        - write the daily Parquet files and log the final summary.

    Returns:
        Process exit code per AGENTS.md section 7: ``0`` on success (a run that matched no
        product at all is legitimate and only logs a warning); ``1`` when any error hits search,
        extraction or export; ``2`` for invalid usage such as bad date syntax or unknown mission,
        in which case the available missions are listed to help recover.

    Raises:
        SystemExit: never raised directly; argparse handles it itself for unrecognized flags.
    """
    raise NotImplementedError("metadata_asf.cli.main")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
