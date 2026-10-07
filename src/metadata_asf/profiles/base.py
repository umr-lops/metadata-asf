"""Mission profiles: per-mission static description of the ASF dataset."""

from __future__ import annotations


class MissionProfile:
    """Static description of a mission as exposed by the ASF API.

    A profile groups every piece of knowledge that is specific to one mission, so
    that adding a new mission never requires touching the core code (search,
    extraction, export). One module per mission lives in this package and is
    registered in :data:`metadata_asf.profiles.MISSIONS`.

    Args:
        name: CLI identifier (e.g. ``"NISAR"``).
        asf_dataset: value passed to ``asf.search(dataset=...)``; centralize it
            here so naming changes at mission launch are fixed in one place only.
        supported_products: list of ``(processing_level, product_type)`` pairs that
            the catalog collects for this mission (case-sensitive, e.g. ``("L1", "RSLC")``).
        default_ocean_wkt: WKT used to filter acquisitions towards ocean areas;
            ``None`` means "no geographic filtering".
        field_mapping: mapping between raw ASF product properties and the columns of
            the normalized Parquet schema (see ``AGENTS.md`` section 2).
    """

    def __init__(
        self,
        name: str,
        asf_dataset: str,
        supported_products: list[tuple[str, str]],
        default_ocean_wkt: str | None = None,
        field_mapping: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.asf_dataset = asf_dataset
        self.supported_products = list(supported_products)
        self.default_ocean_wkt = default_ocean_wkt
        self.field_mapping = field_mapping or {}


__all__ = ["MissionProfile"]
