"""Assorted helpers: UTC coercion and logging setup."""

from __future__ import annotations

import datetime as dt
import logging
import sys


def utc_datetime(value: dt.date | dt.datetime | str | None) -> dt.datetime | None:
    """Coerce a raw ASF timestamp into an aware :class:`datetime.datetime` in UTC.

    Accepts ISO 8601 strings such as ``"2025-01-15T04:30:00Z"``, naive or aware datetimes, and bare
    dates (taken as midnight UTC).

    Args:
        value: raw property value read from an ASF product; ``None`` means missing field.

    Returns:
        A timezone-aware datetime pinned to UTC, or ``None`` when the input was empty/``None``.

    Raises:
        ValueError: when the value cannot be parsed into anything meaningful. Callers are
            expected to degrade to an empty cell, never letting a single bad value abort a
            whole execution.
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
        level: one of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` — validated by the config
            layer above; an unknown value simply falls back to INFO instead of aborting
            execution.
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


__all__ = ["setup_logging", "utc_datetime"]
