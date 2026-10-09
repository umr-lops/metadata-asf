"""Tests for :func:`metadata_asf.search.search` (options, retry, streaming)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Generator

import pytest
from asf_search.exceptions import ASFSearchError

from metadata_asf.profiles import MISSIONS, UnknownMissionError
from metadata_asf.profiles.base import MissionProfile
from metadata_asf.search import search


def test_profile_options_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_search(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr("asf_search.search", fake_search)
    result = search("NISAR", dt.date(2025, 1, 1), dt.date(2025, 1, 31))
    assert result == []
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["dataset"] == "NISAR"
    assert kwargs["processingLevel"] == ["RSLC", "GSLC"]
    # Calendar days are sent as full UTC days: start at 00:00:00Z, end at 23:59:59Z.
    assert kwargs["start"] == "2025-01-01T00:00:00Z"
    assert kwargs["end"] == "2025-01-31T23:59:59Z"
    assert kwargs["maxResults"] == 10_000
    # The profile's default ocean WKT is used when no WKT is given explicitly.
    assert str(kwargs["intersectsWith"]).startswith("POLYGON")


def test_single_day_is_a_full_24h_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-day window must span the whole day, not collapse to a zero-width midnight.

    Regression: asf_search's dateparser reads a bare date as midnight, so sending
    start=end=midnight made a single ``--date`` match nothing (see the Aug-2026 incident).
    """
    calls: list[dict[str, object]] = []

    def fake_search(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr("asf_search.search", fake_search)
    day = dt.date(2026, 8, 31)
    search("NISAR", day, day)
    assert calls[0]["start"] == "2026-08-31T00:00:00Z"
    assert calls[0]["end"] == "2026-08-31T23:59:59Z"


def test_range_is_inclusive_on_both_calendar_days(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_search(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr("asf_search.search", fake_search)
    search("NISAR", dt.date(2025, 1, 1), dt.date(2025, 1, 31))
    assert calls[0]["start"] == "2025-01-01T00:00:00Z"
    assert calls[0]["end"] == "2025-01-31T23:59:59Z"


def test_explicit_wkt_overrides_profile_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_search(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr("asf_search.search", fake_search)
    search("NISAR", dt.date(2025, 1, 1), dt.date(2025, 1, 31), intersects_with="POINT(0 0)")
    assert calls[0]["intersectsWith"] == "POINT(0 0)"


def test_profile_without_ocean_wkt_sends_no_intersects(monkeypatch: pytest.MonkeyPatch) -> None:
    MISSIONS["NOWKT"] = MissionProfile(
        name="NOWKT",
        asf_dataset="NOWKT",
        supported_products=[("L1", "RSLC")],
    )
    calls: list[dict[str, object]] = []

    def fake_search(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        return []

    try:
        monkeypatch.setattr("asf_search.search", fake_search)
        search("NOWKT", dt.date(2025, 1, 1), dt.date(2025, 1, 31))
    finally:
        MISSIONS.pop("NOWKT")
    assert "intersectsWith" not in calls[0]


def test_product_types_restricts_processing_level(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_search(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr("asf_search.search", fake_search)
    search("NISAR", dt.date(2025, 1, 1), dt.date(2025, 1, 31), product_types=["RSLC"])
    assert calls[0]["processingLevel"] == ["RSLC"]


def test_unknown_mission_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("asf_search.search", lambda **kw: [])
    with pytest.raises(UnknownMissionError):
        search("SENTINEL-1", dt.date(2025, 1, 1), dt.date(2025, 1, 31))


def test_transient_error_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def flaky(**kwargs: object) -> list[object]:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ASFSearchError("boom")
        return ["P1"]

    monkeypatch.setattr("asf_search.search", flaky)
    result = search("NISAR", dt.date(2025, 1, 1), dt.date(2025, 1, 2), max_results=100)
    assert result == ["P1"]
    assert attempts["n"] == 3


def test_connection_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def flaky(**kwargs: object) -> list[object]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("dropped")
        return []

    monkeypatch.setattr("asf_search.search", flaky)
    assert search("NISAR", dt.date(2025, 1, 1), dt.date(2025, 1, 2), max_results=100) == []
    assert attempts["n"] == 2


def test_retries_exhausted_reraise_original_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(**kwargs: object) -> list[object]:
        raise ASFSearchError("boom")

    monkeypatch.setattr("asf_search.search", always_fail)
    with pytest.raises(ASFSearchError, match="boom"):
        search("NISAR", dt.date(2025, 1, 1), dt.date(2025, 1, 2), max_results=100)


def test_cap_reached_switches_to_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    capped_products: list[object] = [f"capped_{i}" for i in range(5)]
    streamed_pages: list[list[object]] = [
        [capped_products[0], capped_products[1]],
        ["capped_2"],
    ]
    generator_calls: list[dict[str, object]] = []

    def fake_search(**kwargs: object) -> list[object]:
        return capped_products

    def fake_generator(**kwargs: object) -> Generator[list[object], None, None]:
        generator_calls.append(kwargs)
        yield from streamed_pages

    monkeypatch.setattr("asf_search.search", fake_search)
    monkeypatch.setattr("asf_search.search_generator", fake_generator)
    result = search("NISAR", dt.date(2025, 1, 1), dt.date(2025, 1, 2), max_results=5)
    assert result == ["capped_0", "capped_1", "capped_2"]
    assert len(generator_calls) == 1
    # The uncapped path must not carry a maxResults key.
    assert "maxResults" not in generator_calls[0]


def test_below_cap_does_not_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("asf_search.search", lambda **kw: ["p0", "p1", "p2"])
    monkeypatch.setattr(
        "asf_search.search_generator",
        lambda **kw: pytest.fail("search_generator must not be called below the cap"),
    )
    assert search("NISAR", dt.date(2025, 1, 1), dt.date(2025, 1, 2), max_results=5) == [
        "p0",
        "p1",
        "p2",
    ]


def test_search_returns_a_plain_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capped result must be materialized so callers can take its length twice."""
    monkeypatch.setattr("asf_search.search", lambda **kw: ["p0"])
    result = search("NISAR", dt.date(2025, 1, 1), dt.date(2025, 1, 2), max_results=10)
    assert isinstance(result, list)
    assert len(result) == 1
