"""NISAR mission profile: L1 RSLC + L2 GSLC products."""

from __future__ import annotations

from metadata_asf.profiles.base import MissionProfile

#: Coarse default geographic filter: the whole planet. A real ocean-only polygon is a
#: deliberate refinement (see ``AGENTS.md`` section 13 on land-vs-ocean filtering).
_FULL_EARTH_WKT = "POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))"

#: Mapping between the normalized Parquet schema columns and the raw property
#: name(s) of an ``ASFProduct.properties`` dict (asf_search).
#:
#: NISAR quirk (verified against the live ASF API, 2026-10): the ``polarization``
#: property is always ``None`` on NISAR RSLC/GSLC; the polarizations are carried
#: separately in ``mainBandPolarization`` and ``sideBandPolarization``. The mapping
#: therefore lists both, and :mod:`metadata_asf.extract` merges them. ``beamModeType``
#: is intentionally NOT mapped: it is a Sentinel-1 concept and is ``None`` on NISAR.
_NISAR_FIELD_MAPPING: dict[str, list[str]] = {
    "granule_id": ["sceneName"],
    "platform": ["platform"],
    "start_time": ["startTime"],
    "stop_time": ["stopTime"],
    "polarization": ["mainBandPolarization", "sideBandPolarization"],
    "product_type": ["processingLevel"],
}


#: NISAR profile: first target mission of this project (see ``AGENTS.md`` sections 3-4).
nisar_profile = MissionProfile(
    name="NISAR",
    asf_dataset="NISAR",
    supported_products=[("L1", "RSLC"), ("L2", "GSLC")],
    default_ocean_wkt=_FULL_EARTH_WKT,
    field_mapping=_NISAR_FIELD_MAPPING,
)


__all__ = ["nisar_profile"]
