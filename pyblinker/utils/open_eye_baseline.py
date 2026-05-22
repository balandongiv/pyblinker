"""Utilities for computing open-eye baseline features."""

from __future__ import annotations
from pyblinker._logging import get_logger

from typing import Sequence

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.open_eye import (
    baseline_drift_epoch,
    baseline_mad_epoch,
    baseline_mean_epoch,
    baseline_std_epoch,
    eye_opening_rms_epoch,
    micropause_count_epoch,
    perclos_epoch,
    zero_crossing_rate_epoch,
)

logger = get_logger(__name__)


def _blinks_from_metadata(meta: pd.Series, sfreq: float) -> list[dict[str, int]]:
    """Convert blink onset/duration metadata to frame spans.

    Parameters
    ----------
    meta: pd.Series
        Metadata for a single epoch containing ``blink_onset`` and
        ``blink_duration`` in seconds.
    sfreq: float
        Sampling frequency of the signal.

    Returns
    -------
    list[dict[str, int]]
        Blink spans with start and end frames.
    """
    onset = meta.get("blink_onset")
    duration = meta.get("blink_duration")
    blinks: list[dict[str, int]] = []
    if onset is None or (isinstance(onset, float) and pd.isna(onset)):
        return blinks
    onsets = np.atleast_1d(onset)
    durs = np.atleast_1d(duration if duration is not None else 0.0)
    if durs.size < onsets.size:
        durs = np.pad(durs, (0, onsets.size - durs.size), constant_values=durs[-1])
    for o, d in zip(onsets, durs):
        if pd.isna(o):
            continue
        start = int(float(o) * sfreq)
        end = int((float(o) + float(d or 0.0)) * sfreq)
        blinks.append({"refined_start_frame": start, "refined_end_frame": end})
    return blinks


def _compute_features(
    signal: np.ndarray, blinks: list[dict[str, int]], sfreq: float
) -> pd.Series:
    """Compute baseline features for a single-channel epoch.

    Parameters
    ----------
    signal: np.ndarray
        Signal values for an epoch and channel.
    blinks: list[dict[str, int]]
        Blink spans within the epoch.
    sfreq: float
        Sampling frequency of the signal.

    Returns
    -------
    pd.Series
        Baseline feature values.
    """
    return pd.Series(
        {
            "baseline_mean": baseline_mean_epoch(signal, blinks),
            "baseline_drift": baseline_drift_epoch(signal, blinks, sfreq),
            "baseline_std": baseline_std_epoch(signal, blinks),
            "baseline_mad": baseline_mad_epoch(signal, blinks),
            "perclos": perclos_epoch(signal, blinks),
            "eye_opening_rms": eye_opening_rms_epoch(signal, blinks),
            "micropause_count": micropause_count_epoch(signal, blinks, sfreq),
            "zero_crossing_rate": zero_crossing_rate_epoch(signal, blinks),
        }
    )


def compute_open_eye_baseline_features(
    epochs: mne.Epochs, picks: Sequence[str], indices: Sequence[int]
) -> pd.DataFrame:
    """Compute averaged baseline features across selected epochs.

    Parameters
    ----------
    epochs: mne.Epochs
        Epochs containing the signals and metadata.
    picks: Sequence[str]
        Channel names to include in the computation.
    indices: Sequence[int]
        Epoch indices considered blink-free and used for aggregation.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by channel name with baseline metrics averaged
        across the specified epochs.
    """
    logger.info(
        "Computing open-eye baseline features for %d epochs and %d channels",
        len(indices),
        len(picks),
    )
    sfreq = epochs.info["sfreq"]
    channel_features: dict[str, list[pd.Series]] = {ch: [] for ch in picks}
    data = epochs.get_data()
    for idx in indices:
        meta = epochs.metadata.iloc[idx]
        blinks = _blinks_from_metadata(meta, sfreq)
        for ch in picks:
            ch_idx = epochs.ch_names.index(ch)
            signal = data[idx, ch_idx, :]
            feats = _compute_features(signal, blinks, sfreq)
            channel_features[ch].append(feats)
    aggregated = {
        ch: pd.concat(feats, axis=1).mean(axis=1)
        for ch, feats in channel_features.items()
    }
    df = pd.DataFrame.from_dict(aggregated, orient="index")
    logger.info("Computed baseline features for %d channels", len(df))
    return df
