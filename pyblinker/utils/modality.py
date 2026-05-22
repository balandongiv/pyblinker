"""Utilities for inferring channel modalities."""

from __future__ import annotations

from pyblinker._logging import get_logger

logger = get_logger(__name__)


def infer_modality(channel_name: str) -> str:
    """Infer a modality label from a channel name.

    Parameters
    ----------
    channel_name : str
        Channel identifier from which the modality should be derived.

    Returns
    -------
    str
        Lowercase modality string. Known substrings such as ``"eeg"``,
        ``"eog"``, and ``"ear"`` take precedence. If none of these are found,
        the function falls back to the channel prefix before the first hyphen.
    """
    logger.debug("Inferring modality from channel name: %s", channel_name)

    lower_name = channel_name.lower().strip()
    if "-" in channel_name:
        prefix = lower_name.split("-", 1)[0]
        if prefix in {"eeg", "eog", "ear"}:
            logger.debug(
                "Detected modality '%s' based on channel prefix in '%s'",
                prefix,
                channel_name,
            )
            return prefix
    for candidate in ("ear", "eog", "eeg"):
        if candidate in lower_name:
            logger.debug(
                "Detected modality '%s' based on substring match in '%s'",
                candidate,
                channel_name,
            )
            return candidate
    if "-" in channel_name:
        prefix = lower_name.split("-", 1)[0]
        if prefix:
            logger.debug(
                "Falling back to prefix-based modality '%s' for channel '%s'",
                prefix,
                channel_name,
            )
            return prefix
    logger.debug(
        "No modality keyword found; defaulting to lowercase channel name '%s'",
        lower_name,
    )
    return lower_name


__all__ = ["infer_modality"]
