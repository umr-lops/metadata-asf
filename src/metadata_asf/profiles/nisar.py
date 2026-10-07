"""NISAR mission profile: L1 RSLC + L2 GSLC products."""

from __future__ import annotations

from metadata_asf.profiles.base import MissionProfile

#: Coarse default geographic filter: the whole planet. A real ocean-only polygon is a
#: deliberate refinement (see ``AGENTS.md`` section 13 on land-vs-ocean filtering).
_FULL_EARTH_WKT = "POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))"

#: Mapping between the normalized Parquet schema columns and the raw property names of
#: an ``ASFProduct.properties`` dict (asf_search v14). The special case of the
#: ``geometry`` column is handled structurally in :mod:`extract` via ``product.geometry``,
#: not through this mapping.
_NISAR_FIELD_MAPPING: dict[str, str] = {
    "granule_id": "sceneName",
    "platform": "platform",
    "start_time": "startTime",
    "stop_time": "stopTime",
    "polarization": "polarization",
    "beam_mode": "beamModeType",
    "product_type": "processingLevel",  # ASF exposes e.g. RSLC / GSLC here for NISAR
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
