"""Utilities for inferring signal modality from channel names."""

from __future__ import annotations


def infer_modality(channel_name: str) -> str:
    """Infer the modality label for a recording channel.

    Parameters
    ----------
    channel_name : str
        Channel name whose modality should be determined.

    Returns
    -------
    str
        Lowercase modality label derived from the channel name. Known
        modality keywords (``"eeg"``, ``"eog"``, ``"ear"``) are prioritised.
        When the channel name contains a hyphen, the substring preceding the
        first hyphen is returned if present. If no keywords or separator are
        present the channel name itself (lowercased) is returned.
    """

    cleaned = channel_name.strip()
    if not cleaned:
        return ""

    if "-" in cleaned:
        prefix = cleaned.split("-", 1)[0].strip().lower()
        if prefix:
            return prefix

    lower_name = cleaned.lower()
    for keyword in ("eeg", "eog", "ear"):
        if keyword in lower_name:
            return keyword

    return lower_name
