"""Tests for :mod:`metadata_asf.utils` (date parsing, UTC coercion, logging)."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Generator

import pandas as pd
import pytest

from metadata_asf.utils import parse_date, setup_logging, utc_datetime

UTC = dt.timezone.utc


class TestParseDate:
    def test_single_date(self) -> None:
        assert parse_date("2025-01-15") == dt.date(2025, 1, 15)

    def test_inclusive_range(self) -> None:
        assert parse_date("2025-01-01:2025-01-31") == (dt.date(2025, 1, 1), dt.date(2025, 1, 31))

    def test_inverted_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="starts .* after"):
            parse_date("2025-01-31:2025-01-01")

    @pytest.mark.parametrize(
        "bad",
        [
            "2025-01",  # too short
            "2025-01-15:01:02",  # too many ':' parts
            "2025-13-01",  # month out of range
            "2025-01-32",  # day out of range
            "15-01-2025",  # wrong order
            "abc:def",  # not digits
            "",  # empty
        ],
    )
    def test_malformed_tokens_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_date(bad)


class TestUtcDatetime:
    def test_iso_with_zulu_suffix(self) -> None:
        assert utc_datetime("2025-01-15T04:30:00Z") == dt.datetime(2025, 1, 15, 4, 30, tzinfo=UTC)

    def test_iso_with_fractional_seconds(self) -> None:
        expected = dt.datetime(2025, 1, 15, 4, 30, 0, 500_000, tzinfo=UTC)
        assert utc_datetime("2025-01-15T04:30:00.500Z") == expected

    def test_iso_with_offset_converted_to_utc(self) -> None:
        assert utc_datetime("2025-01-15T06:30:00+02:00") == dt.datetime(
            2025, 1, 15, 4, 30, tzinfo=UTC
        )

    def test_naive_datetime_treated_as_utc(self) -> None:
        assert utc_datetime(dt.datetime(2025, 1, 15, 4, 30)) == dt.datetime(
            2025, 1, 15, 4, 30, tzinfo=UTC
        )

    def test_aware_datetime_converted_to_utc(self) -> None:
        plus_two = dt.timezone(dt.timedelta(hours=2))
        assert utc_datetime(dt.datetime(2025, 1, 15, 6, 30, tzinfo=plus_two)) == dt.datetime(
            2025, 1, 15, 4, 30, tzinfo=UTC
        )

    def test_bare_date_is_midnight_utc(self) -> None:
        assert utc_datetime(dt.date(2025, 1, 15)) == dt.datetime(2025, 1, 15, 0, 0, tzinfo=UTC)

    def test_pandas_timestamp_supported(self) -> None:
        expected = dt.datetime(2025, 1, 15, 4, 30, tzinfo=UTC)
        assert utc_datetime(pd.Timestamp("2025-01-15T04:30:00Z")) == expected

    def test_none_stays_none(self) -> None:
        assert utc_datetime(None) is None

    def test_blank_string_stays_none(self) -> None:
        assert utc_datetime("   ") is None

    def test_garbage_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse ASF timestamp"):
            utc_datetime("not-a-timestamp")

    def test_non_datetime_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot interpret value"):
            utc_datetime(12_345)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Generator[None, None, None]:
    """Save/restore the root logger state so setup_logging tests stay isolated."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers, root.level
    yield
    root.handlers, root.level = saved_handlers, saved_level


def _stream_handlers() -> int:
    root = logging.getLogger()
    return len([h for h in root.handlers if isinstance(h, logging.StreamHandler)])


class TestSetupLogging:
    def test_sets_level_and_single_stream_handler(self, capsys: pytest.CaptureFixture[str]) -> None:
        setup_logging("DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert _stream_handlers() == 1

        setup_logging("INFO")  # calling twice must not stack handlers up.
        assert root.level == logging.INFO
        assert _stream_handlers() == 1

        logging.getLogger("metadata_asf.utils.test").info("hello-log")
        out = capsys.readouterr().out
        assert "INFO" in out
        assert "hello-log" in out

    def test_unknown_level_falls_back_to_info(self) -> None:
        setup_logging("TRACE")
        assert logging.getLogger().level == logging.INFO
