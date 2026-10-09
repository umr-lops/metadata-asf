"""ASF search wrapper: ``asf_search`` + exponential retry."""

from __future__ import annotations

import datetime as dt
import logging

import asf_search as asf
from asf_search.exceptions import ASFSearchError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from metadata_asf.profiles import get_profile

logger = logging.getLogger(__name__)


#: Error classes retried with exponential backoff before giving up (AGENTS.md section 4).
_RETRYABLE: tuple[type[Exception], ...] = (ASFSearchError, ConnectionError, TimeoutError)


def search(
    mission: str,
    start: dt.date,
    end: dt.date,
    intersects_with: str | None = None,
    max_results: int = 10_000,
) -> list[asf.ASFProduct]:
    """Search the ASF API for acquisitions of ``mission`` between two dates.

    Thin wrapper over :func:`asf_search.search`. All mission-specific data — dataset name,
    product levels, default ocean WKT — comes from the mission profile, and transient failures
    are retried with exponential backoff. If the cap is reached exactly, the capped answer may
    have been truncated server-side rather than genuinely complete: the same window is then
    re-run uncapped through :func:`asf_search.search_generator` so no match escapes the catalog
    (AGENTS.md, sections 4 and 13).

    Args:
        mission: CLI identifier of the mission, e.g. ``"NISAR"``, which must be registered in
            :data:`~metadata_asf.profiles.MISSIONS`.
        start: first day (UTC) of the acquisition window, inclusive.
        end: last day (UTC) of the acquisition window, inclusive.
        intersects_with: WKT restricting to acquisitions that intersect this geometry; falls back to
            ``MissionProfile.default_ocean_wkt`` when `None`.
        max_results: hard cap on the number of products returned for a single capped API call.

    Returns:
        Raw ASF product objects as built by ``asf_search``, not yet normalized. An empty list is
        a legitimate answer, not an error.

    Raises:
        UnknownMissionError: raised if ``mission`` has no registered profile. Once all retries
            are exhausted, one exception from :data:`_RETRYABLE` is re-raised against the failed
            original call as-is.
    """
    profile = get_profile(mission)
    options: dict[str, object] = {
        "dataset": profile.asf_dataset,
        "processingLevel": [product_type for _level, product_type in profile.supported_products],
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    wkt = intersects_with or profile.default_ocean_wkt
    if wkt is not None:
        options["intersectsWith"] = wkt

    products = _capped_search(options, max_results=max_results)

    if 0 < len(products) and len(products) >= max_results:
        # Cap reached exactly: the answer above may be incomplete; stream everything (section 13).
        logger.warning(
            "ASF search %s hit maxResults=%d; re-running uncapped through search_generator()",
            profile.asf_dataset,
            max_results,
        )
        products = _stream_all(options)

    logger.info("ASF search %s (%s..%s): %d product(s)", mission, start, end, len(products))
    return list(products)


def _capped_search(options: dict[str, object], *, max_results: int) -> list[asf.ASFProduct]:
    """Run a capped :func:`asf_search.search` call with retry on transient failures.

    Args:
        options: base query kwargs shared by both entrypoints of this module (no ``maxResults``).
        max_results: cap bound to asf_search's ``maxResults`` keyword for exactly one API call.

    Returns:
        Every product materialized from the result object, in a plain list (never a live iterator so
        callers can take its length twice — which :func:`search` relies on).
    """

    @retry(  # AGENTS.md section 4: "3 tentatives, backoff 2x", bounded to ~60 s per wait.
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=60),
        reraise=True,  # surface the original backend exception type up to the CLI layer unchanged.
    )
    def attempt() -> list[asf.ASFProduct]:
        # Fresh dict: we must never share/mutate the mapped options across retries or streaming.
        call_options = {**options, "maxResults": max_results}
        return list(asf.search(**call_options))

    try:
        return attempt()
    except _RETRYABLE as err:
        logger.error("ASF search failed after all retry attempts: %s", err)
        raise


def _stream_all(options: dict[str, object]) -> list[asf.ASFProduct]:
    """Stream the whole result set through :func:`asf_search.search_generator`, uncapped.

    Args:
        options: base query kwargs; must NOT contain ``maxResults`` here by construction — see
            :func:`search` above, which never stores that key into this dict.

    Returns:
        Every product yielded page by page across the full result set, materialized as a plain list.
    """
    # Guardrail so a future refactor cannot silently re-cap the uncapped path.
    assert "maxResults" not in options

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=60),
        reraise=True,
    )
    def attempt() -> list[asf.ASFProduct]:
        generated: list[asf.ASFProduct] = []
        # search_generator yields one ASFSearchResults per CMR page (default size 250).
        for page in asf.search_generator(**options):
            generated.extend(iter(page))
        return generated

    try:
        return attempt()
    except _RETRYABLE as err:
        logger.error("ASF streaming search failed after all retry attempts: %s", err)
        raise


__all__ = ["search"]
