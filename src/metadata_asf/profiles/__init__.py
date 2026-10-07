"""Mission profile registry."""

from __future__ import annotations

import logging

from metadata_asf.profiles.base import MissionProfile
from metadata_asf.profiles.nisar import nisar_profile

logger = logging.getLogger(__name__)


class UnknownMissionError(KeyError):
    """Raised when a mission has no registered profile.

    Attributes:
        available: names of the missions currently defined in :data:`MISSIONS`.
    """

    def __init__(self, mission: str, available: list[str]) -> None:
        self.available = available
        super().__init__(
            f"Unknown mission {mission!r}. Available missions: {', '.join(available)}"
        )


#: Registry of all available mission profiles. Adding a mission amounts to adding
#: a module in this package and one entry here — nothing else.
MISSIONS: dict[str, MissionProfile] = {
    "NISAR": nisar_profile,
}


def get_profile(name: str) -> MissionProfile:
    """Return the profile registered for ``name``.

    Args:
        name: CLI identifier of the mission (case-sensitive), e.g. ``"NISAR"``.

    Returns:
        The corresponding :class:`~metadata_asf.profiles.base.MissionProfile`.

    Raises:
        UnknownMissionError: if no profile is registered for ``name``; the message
            lists every available mission so it can be relayed verbatim by the CLI.
    """
    try:
        return MISSIONS[name]
    except KeyError:
        logger.warning("Profile not found for mission %s", name)
        raise UnknownMissionError(name, [m.name for m in MISSIONS.values()] or []) from None


__all__ = ["MISSIONS", "UnknownMissionError", "get_profile"]
