"""Shared constants/helpers for blink feature extraction."""

from __future__ import annotations

import mne
import pandas as pd

STATS: tuple[str, ...] = ("mean", "std", "cv")

ENERGY_METRICS: tuple[str, ...] = (
    "blink_signal_energy",
    "teager_kaiser_energy",
    "blink_line_length",
    "blink_velocity_integral",
)


LEGACY_METRICS_BY_FAMILY: dict[str, dict[str, tuple[str, ...]]] = {
    "energy": {
        "default": ENERGY_METRICS,
    },
}


def infer_modality(channel_name: str, info: mne.Info) -> str:
    """Infer modality label (ear/eeg/eog) from channel metadata."""

    ch_type = info.get_channel_types(picks=[channel_name])[0]
    ch_lower = channel_name.lower()
    if "ear" in ch_lower:
        return "ear"
    if ch_type == "eog" or "eog" in ch_lower:
        return "eog"
    if ch_type == "eeg" or "eeg" in ch_lower:
        return "eeg"
    return ch_type.lower()


def cast_columns_to_object(df: pd.DataFrame) -> pd.DataFrame:
    """Force DataFrame column-index dtype to legacy object for compatibility."""

    df.columns = df.columns.astype(object)
    return df
