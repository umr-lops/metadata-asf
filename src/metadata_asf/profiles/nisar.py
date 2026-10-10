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


#: NISAR ``POLE`` mnemonic halves (2 chars each, L-band then S-band) → readable label.
#: ``NA`` means "not applicable" for that band and is therefore mapped to ``None``.
_POL_MODES: dict[str, str | None] = {
    "SH": "single-pol HH",
    "SV": "single-pol VV",
    "DH": "dual-pol HH/HV",
    "DV": "dual-pol VV/VH",
    "QP": "quad-pol HH/HV/VV/VH",
    "QD": "quad-pol HH/VV (dual-band)",
    "QQ": "quad-pol HH/HV + VV/VH (dual-band)",
    "NA": None,
}


def _stem_tokens(granule_id: str) -> list[str] | None:
    """Split a NISAR product file name into tokens, or ``None`` if it is not a NISAR name.

    Strips an optional ``.h5``/``.slc.h5`` style suffix and requires the canonical
    ``NISAR_<LVL>_...`` prefix and at least 10 underscore-separated tokens (the
    ``MODE`` and ``POLE`` fields sit at positions 8 and 9).
    """
    stem = granule_id.strip()
    for suffix in (".slc.h5", ".h5"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    tokens = stem.split("_")
    # MODE and POLE live at positions 8 and 9; require the NISAR prefix and enough tokens.
    if len(tokens) < 10 or tokens[0] != "NISAR":
        return None
    return tokens


def _decode_beam_mode(granule_id: str) -> str | None:
    """Decode a NISAR ``MODE``/``POLE`` code pair from an RSLC/GSLC product file name.

    NISAR carries no ASF ``beamModeType`` property, so the acquisition "mode" is only
    readable from the product name, which follows the CATALYST pattern
    ``NISAR_<LVL>_PR_<TYPE>_<...>_<MODE>_<POLE>_...`` (18 underscore-separated tokens).
    Two 4-character fields encode it:

    * ``MODE`` (token 8): the *bandwidth mode code*, i.e. the main-band bandwidth in MHz
      (e.g. ``2005`` → 20 MHz, ``4005`` → 40 MHz, ``7700`` → 77 MHz, ``0005`` → 5 MHz);
    * ``POLE`` (token 9): the *polarization/mode mnemonic* (e.g. ``SHNA`` → single-pol HH,
      ``DHDH`` → dual-pol HH/HV, ``QPDH`` → quad-pol).

    Args:
        granule_id: a NISAR RSLC/GSLC product file name (stem, without extension).

    Returns:
        A human-readable ``"<bandwidth> MHz, <mode>"`` string, e.g.
        ``"40 MHz, dual-pol HH/HV"``; ``None`` when the name does not match the expected
        shape (unknown mission format, truncated name, …) so the caller can leave the cell
        empty rather than show a bogus value.
    """
    tokens = _stem_tokens(granule_id)
    if tokens is None:
        return None
    mode_code, pole_code = tokens[8], tokens[9]
    bandwidth = _decode_bandwidth(mode_code)
    polarization = _decode_pole(pole_code)
    parts: list[str] = []
    if bandwidth is not None:
        parts.append(f"{bandwidth} MHz")
    if polarization is not None:
        parts.append(polarization)
    if not parts:
        return None
    return ", ".join(parts)


def _decode_bandwidth(code: str) -> int | None:
    """Main-band bandwidth in MHz from a NISAR 4-character ``MODE`` code.

    The code is ``XXXX``: the first two digits are the *primary* (L-band) bandwidth and the
    last two the secondary (S-band) bandwidth, in MHz — the CATALYST naming doc's concrete
    example ``2000`` → 20 MHz is read the same way. A ``00`` sub-code means "not applicable"
    for that band, so the primary value is used unless it is ``00``.
    """
    if len(code) != 4 or not code.isdigit():
        return None
    primary, secondary = int(code[0:2]), int(code[2:4])
    return primary if primary else (secondary if secondary else None)


def _decode_pole(code: str) -> str | None:
    """Human-readable polarization/mode from a NISAR 4-character ``POLE`` mnemonic.

    The mnemonic is two 2-character halves, the first for the L band and the second for the
    S band (e.g. ``SHNA`` = L-band single-pol HH, S-band not applicable). Each half is looked
    up in :data:`_POL_MODES`; the L-band (primary) half is preferred, falling back to the
    S-band half when the primary is ``NA``.
    """
    if len(code) != 4 or not code.isalpha():
        return None
    primary = _POL_MODES.get(code[0:2])
    if primary is not None:
        return primary
    return _POL_MODES.get(code[2:4])


#: NISAR profile: first target mission of this project (see ``AGENTS.md`` sections 3-4).
nisar_profile = MissionProfile(
    name="NISAR",
    asf_dataset="NISAR",
    supported_products=[("L1", "RSLC"), ("L2", "GSLC")],
    default_ocean_wkt=_FULL_EARTH_WKT,
    field_mapping=_NISAR_FIELD_MAPPING,
    decode_beam_mode=_decode_beam_mode,
)


__all__ = ["nisar_profile"]
