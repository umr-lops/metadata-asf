"""Mission profiles: per-mission static description of the ASF dataset."""

from __future__ import annotations

from collections.abc import Callable


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
        field_mapping: mapping from normalized Parquet schema columns to the raw
            property name(s) of an ``ASFProduct.properties`` dict. A column may list
            several source properties (their values are merged in
            :mod:`metadata_asf.extract`, e.g. NISAR polarization spread over
            ``mainBandPolarization`` and ``sideBandPolarization``). The ``geometry``
            column is never mapped: it is read structurally from ``product.geometry``.
        decode_beam_mode: optional callable deriving the ``beam_mode`` column from a
            ``granule_id`` (e.g. NISAR has no ASF beam-mode property, so it is decoded
            from the product file name); ``None`` leaves the column empty.
    """

    def __init__(
        self,
        name: str,
        asf_dataset: str,
        supported_products: list[tuple[str, str]],
        default_ocean_wkt: str | None = None,
        field_mapping: dict[str, list[str]] | None = None,
        decode_beam_mode: Callable[[str], str | None] | None = None,
    ) -> None:
        self.name = name
        self.asf_dataset = asf_dataset
        self.supported_products = list(supported_products)
        self.default_ocean_wkt = default_ocean_wkt
        self.field_mapping = {
            column: list(properties) for column, properties in (field_mapping or {}).items()
        }
        self.decode_beam_mode = decode_beam_mode


__all__ = ["MissionProfile"]
