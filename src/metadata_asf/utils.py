"""Assorted helpers: date parsing, UTC coercion and logging setup."""

from __future__ import annotations

import datetime as dt
import logging
import sys


def parse_date(value: str) -> dt.date | tuple[dt.date, dt.date]:
    """Parse a ``YYYY-MM-DD`` date or an ``A:B`` range token accepted on the CLI.

    Args:
        value: the raw ``--date`` token (single date or inclusive range).

    Returns:
        For a single date, a :class:`datetime.date`; otherwise, an inclusive ``(start, end)`` pair of UTC days.

    Raises:
        ValueError: raised on malformed tokens; the message includes the offending token as-is so it can be
            pasted verbatim into a bug report.
    """
    parts = value.split(":")
    if len(parts) == 1:
        return _parse_single(parts[0])

    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse --date {value!r}: expected YYYY-MM-DD or "
            "YYYY-MM-DD:YYYY-MM-DD (got more than two ':'-separated parts)"
        )

    start, end = _parse_single(parts[0]), _parse_single(parts[1])
    if start > end:
        raise ValueError(
            f"Cannot parse --date {value!r}: range starts ({start}) after its end ({end})"
        )
    return (start, end)


def utc_datetime(value: dt.date | dt.datetime | str | None) -> dt.datetime | None:
    """Coerce a raw ASF timestamp into an aware :class:`datetime.datetime` in UTC.

    Accepts ISO 8601 strings such as ``"2025-01-15T04:30:00Z"``, naive or aware datetimes, and bare
    dates (taken as midnight UTC).

    Args:
        value: raw property value read from an ASF product; ``None`` means missing field.

    Returns:
        A timezone-aware datetime pinned to UTC, or ``None`` when the input was empty/``None``.

    Raises:
        ValueError: when the value cannot be parsed into anything meaningful. Callers are expected to degrade
            to an empty cell, never letting a single bad value abort a whole execution.
    """
    if value is None:
        return None

    candidate = value
    if isinstance(candidate, str):
        text = candidate.strip()
        if not text:
            return None

        # Normalise trailing Z so both strftime patterns below can accept the same string shape.
        normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        for style in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                candidate = dt.datetime.strptime(normalized, style)
                break
            except ValueError:
                continue

        if isinstance(candidate, str):  # what remains is delegated to fromisoformat.
            try:
                candidate = dt.datetime.fromisoformat(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse ASF timestamp {text!r} into an ISO 8601 datetime"
                ) from exc

    if isinstance(candidate, dt.date) and not isinstance(candidate, dt.datetime):
        candidate = dt.datetime.combine(candidate, dt.time(0))
    elif hasattr(candidate, "to_pydatetime"):
        # Structural duck-typing check that also covers pandas Timestamps.
        try:
            candidate = candidate.to_pydatetime()
        except (ValueError, TypeError) as exc:  # pragma: no cover - purely defensive.
            raise ValueError(
                f"Cannot interpret value {value!r} of type "
                f"{type(value).__name__} as an ASF timestamp"
            ) from exc

    if not isinstance(candidate, dt.datetime):
        # Guard so that a precise failure message wins over some buried TypeError.
        raise ValueError(
            f"Cannot interpret value {value!r} of type {type(value).__name__}"
            " as an ASF timestamp"
        )

    if candidate.tzinfo is None:
        # Naive values from the API are UTC by project convention (AGENTS.md section 13).
        candidate = candidate.replace(tzinfo=dt.timezone.utc)
    return candidate.astimezone(dt.timezone.utc)


def setup_logging(level: str) -> None:
    """Configure root logging for a metadata-asf run.

    Applies the format fixed in AGENTS.md section 4 (``%(asctime)s | %(levelname)-8s |
    %(name)s | %(message)s``), writes to stdout and sets the given level. Calling it twice is safe —
    existing handlers are removed first, never stacked up on top of each other.

    Args:
        level: one of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` — validated by the config layer above; an
            unknown value simply falls back to INFO instead of aborting execution.
    """
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    root = logging.getLogger()
    for handler in list(root.handlers):
        # Repeated calls must remain idempotent and never stack handlers up.
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(log_format))
    root.addHandler(stream_handler)

    name = str(level).strip().upper()
    candidate: object = getattr(logging, name, None)  # fallback is purely defensive.
    root.setLevel(candidate if isinstance(candidate, int) else logging.INFO)


def _parse_single(token: str) -> dt.date:
    """Parse one strict ``YYYY-MM-DD`` token into a :class:`datetime.date`.

    Args:
        token: raw CLI fragment, already split off from its sibling exactly once before this call.

    Returns:
        The parsed calendar date.

    Raises:
        ValueError: if the shape or any component is out of range; message quotes the offending
            token verbatim so it can be pasted into a bug report unchanged.
    """
    parts = token.strip().split("-")

    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(
            "Cannot parse --date component"
            f" {token!r}: expected three dash-separated digit groups (YYYY-MM-DD)"
        )

    try:
        return dt.date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise ValueError(f"Cannot parse --date component {token!r}: {exc}") from exc


__all__ = ["parse_date", "setup_logging", "utc_datetime"]
