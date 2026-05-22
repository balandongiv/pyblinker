"""Channel pick normalization and validation utilities."""

from __future__ import annotations

from typing import Iterable, Sequence

import mne
from mne.io import BaseRaw

from pyblinker._logging import get_logger

logger = get_logger(__name__)


def _is_ear_channel(name: str) -> bool:
    lower = name.lower()
    if "eye_aspect_ratio" in lower:
        return True
    if "ear" in lower and "a1" not in lower and "a2" not in lower:
        return True
    return False


def normalize_picks(picks: str | Iterable[str]) -> list[str]:
    """Normalize channel picks to a list.

    Parameters
    ----------
    picks : str or iterable of str
        Channel name or collection of channel names.

    Returns
    -------
    list of str
        Normalized list of channel names.
    """
    if isinstance(picks, str):
        return [picks]
    return list(picks)


def require_channels(
    data: mne.Epochs | BaseRaw | Sequence[str], picks: Sequence[str]
) -> None:
    """Validate that all requested channels exist in the provided data.

    Parameters
    ----------
    data : mne.Epochs or mne.io.BaseRaw or sequence of str
        Data structure whose channel names are checked.
    picks : sequence of str
        Channel names to validate.

    Raises
    ------
    ValueError
        If any channel in ``picks`` is missing from ``data``.
    """
    logger.debug("Validating channel picks: %s", picks)
    if isinstance(data, (mne.Epochs, BaseRaw)):
        ch_names = data.info["ch_names"]
    else:
        ch_names = list(data)
    missing = [p for p in picks if p not in ch_names]
    if missing:
        raise ValueError(f"Channels not found: {', '.join(missing)}")
    logger.debug("All channels present")


def pick_ear_channels_from_info(info: mne.Info) -> list[int]:
    """Return indices of EAR-style channels from an :class:`mne.Info`."""

    return [idx for idx, name in enumerate(info["ch_names"]) if _is_ear_channel(name)]


def pick_ear_channels_from_raw(raw: BaseRaw) -> list[int]:
    """Return indices of EAR-style channels from a :class:`mne.io.BaseRaw`."""

    return [idx for idx, name in enumerate(raw.ch_names) if _is_ear_channel(name)]


__all__ = [
    "normalize_picks",
    "require_channels",
    "pick_ear_channels_from_info",
    "pick_ear_channels_from_raw",
]
