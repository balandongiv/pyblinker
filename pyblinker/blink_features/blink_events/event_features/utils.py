"""Utility helpers for blink event features."""
from __future__ import annotations
from pyblinker.logging import get_logger

from typing import Iterable, Sequence, List

import mne

logger = get_logger(__name__)


def normalize_picks(picks: str | Iterable[str]) -> List[str]:
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
    data: mne.Epochs | mne.io.BaseRaw,
    picks: Sequence[str],
) -> None:
    """Validate that all requested channels exist in the provided MNE object.

    Parameters
    ----------
    data : mne.Epochs or mne.io.BaseRaw
        Object whose channel names are checked.
    picks : sequence of str
        Channel names to validate.

    Raises
    ------
    ValueError
        If any channel in ``picks`` is missing from ``data``.
    """
    logger.info("Validating channel picks: %s", picks)
    missing = [p for p in picks if p not in data.info["ch_names"]]
    if missing:
        raise ValueError(f"Channels not found: {', '.join(missing)}")
    logger.debug("All channels present")

