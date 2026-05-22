"""Blink count feature utilities."""

from typing import Dict, Iterable, List, Optional, Union

import mne
import numpy as np
import pandas as pd

from pyblinker._logging import get_logger
from pyblinker.utils import normalize_picks
from pyblinker.utils.metadata_utils import extract_blink_windows
from pyblinker.utils.modality import infer_modality

logger = get_logger(__name__)

_MODALITY_START_COLUMN = {
    "ear": "start__th_point__ear",
    "eeg": "start__refine__eeg",
    "eog": "start__refine__eog",
}


def _count_from_metadata_start_column(row: pd.Series, column: str) -> float:
    """Count blinks from a modality-specific start landmark metadata column."""

    if column not in row.index:
        return 0.0

    value = row.get(column)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, list):
        return float(sum(0 if pd.isna(v) else 1 for v in value))
    return float(0 if pd.isna(value) else 1)


def blink_count_epoch(
    blinks: Union[List[Dict[str, int]], "mne.io.BaseRaw", "mne.Epochs"],
    label: Optional[str] = None,
) -> int:
    """Return the number of blinks for a single epoch or from an MNE object."""
    logger.info("Calculating blink count for input of type %s", type(blinks))

    if mne and isinstance(blinks, mne.io.BaseRaw):
        logger.debug("Using MNE Raw logic for annotation counting")
        mask = np.ones(len(blinks.annotations), dtype=bool)
        if label is not None:
            mask &= blinks.annotations.description == label
        count = int(mask.sum())
        logger.debug("Found %d blink annotations matching label '%s'", count, label)
        return count

    if mne and isinstance(blinks, mne.Epochs):
        logger.warning("Blink count from MNE Epochs not implemented")
        raise NotImplementedError(
            "blink_count_epoch does not support MNE Epochs input."
        )

    if isinstance(blinks, list):
        logger.debug("Counting %s blinks from list of dicts", len(blinks))
        return len(blinks)

    logger.error("Unsupported type passed to blink_count_epoch: %s", type(blinks))
    raise TypeError(f"Unsupported input type: {type(blinks)}")


def blink_count(
    epochs: mne.Epochs,
    picks: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Count blinks for each epoch using metadata."""
    logger.info("Counting blinks across %d epochs", len(epochs))
    metadata = epochs.metadata
    if metadata is None:
        raise ValueError("Epochs.metadata must contain blink information")

    if isinstance(metadata, pd.DataFrame):
        metadata_df = metadata
        index = metadata_df.index
    else:
        metadata_df = pd.DataFrame(metadata)
        index = pd.RangeIndex(len(epochs))

    df = pd.DataFrame(index=index)
    df.insert(0, "epoch_id", index.to_numpy())

    picks_list = normalize_picks(picks) if picks is not None else []
    rows = [row for _, row in metadata_df.iterrows()]

    if not picks_list:
        if "n_blinks" in metadata_df.columns:
            counts = metadata_df["n_blinks"].fillna(0).astype(float).tolist()
        else:
            counts = [
                float(len(extract_blink_windows(row, None, epoch_idx)))
                for epoch_idx, row in enumerate(rows)
            ]
        df["blink_count"] = counts
        logger.debug("Blink counts per epoch: %s", counts)
        return df

    mod_to_channel: Dict[str, str] = {}
    for ch in picks_list:
        mod = infer_modality(ch)
        if mod not in mod_to_channel:
            mod_to_channel[mod] = ch

    for modality, channel in mod_to_channel.items():
        expected_column = f"{modality}__ncount__{channel}"
        start_column = _MODALITY_START_COLUMN.get(modality)

        if expected_column in metadata_df.columns:
            counts = metadata_df[expected_column].fillna(0).astype(float).tolist()
        elif start_column and start_column in metadata_df.columns:
            counts = [
                _count_from_metadata_start_column(row, start_column) for row in rows
            ]
        elif "n_blinks" in metadata_df.columns:
            counts = metadata_df["n_blinks"].fillna(0).astype(float).tolist()
        else:
            counts = [
                float(len(extract_blink_windows(row, modality, epoch_idx)))
                for epoch_idx, row in enumerate(rows)
            ]

        df[expected_column] = counts
        logger.debug("Blink counts for %s: %s", modality, counts)

    return df
