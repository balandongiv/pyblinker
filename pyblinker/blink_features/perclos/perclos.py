"""Interval-based epoch PERCLOS calculation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import List, Tuple

import mne
import pandas as pd

from pyblinker.logging import get_logger

from ...utils.iter_utils import ensure_list
from ..constants import cast_columns_to_object, infer_modality
from .labeling import assign_fatigue_labels

logger = get_logger(__name__)

_EAR_INTERVAL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("start__th_interpolation__ear", "end__th_interpolation__ear"),
    ("start__th_point__ear", "end__th_point__ear"),
)


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _coerce_numeric_values(value: object) -> List[float]:
    if _is_missing(value):
        return []

    out: List[float] = []
    for item in ensure_list(value):
        if _is_missing(item):
            continue
        numeric = float(item)
        if math.isfinite(numeric):
            out.append(numeric)
    return out


def _has_ear_channel(epochs: mne.Epochs) -> bool:
    return any(infer_modality(ch, epochs.info) == "ear" for ch in epochs.ch_names)


def _epoch_index(epochs: mne.Epochs) -> pd.Index:
    if isinstance(epochs.metadata, pd.DataFrame):
        return epochs.metadata.index
    return pd.RangeIndex(len(epochs))





def _select_interval_columns(
    metadata: pd.DataFrame,
    *,
    allow_empty_blinks: bool,
) -> tuple[str, str] | None:
    for start_col, end_col in _EAR_INTERVAL_COLUMNS:
        if start_col in metadata.columns and end_col in metadata.columns:
            return start_col, end_col

    if allow_empty_blinks:
        return None

    expected = ", ".join(
        f"{start_col}/{end_col}" for start_col, end_col in _EAR_INTERVAL_COLUMNS
    )
    raise ValueError(
        "Epochs.metadata missing EAR closed-eye interval columns. "
        f"Expected one of: {expected}"
    )


def extract_closed_eye_intervals(
    metadata_row: Mapping[str, object],
    *,
    interval_columns: tuple[str, str] | None = None,
) -> List[Tuple[float, float]]:
    """Return closed-eye intervals stored in one epoch metadata row."""

    if not isinstance(metadata_row, (Mapping, pd.Series)):
        raise TypeError("metadata_row must be a mapping")

    if interval_columns is None:
        for candidate in _EAR_INTERVAL_COLUMNS:
            if candidate[0] in metadata_row and candidate[1] in metadata_row:
                interval_columns = candidate
                break

    if interval_columns is None:
        return []

    start_col, end_col = interval_columns
    starts = _coerce_numeric_values(metadata_row.get(start_col))
    ends = _coerce_numeric_values(metadata_row.get(end_col))

    intervals: List[Tuple[float, float]] = []
    for start, end in zip(starts, ends):
        if end <= start:
            continue
        intervals.append((start, end))

    return intervals


def clip_intervals_to_epoch(
    intervals: Iterable[tuple[float, float]],
    *,
    epoch_start: float,
    epoch_end: float,
) -> List[Tuple[float, float]]:
    """Clip intervals to an epoch window."""

    start = float(epoch_start)
    end = float(epoch_end)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("epoch_start and epoch_end must define a positive window")

    clipped: List[Tuple[float, float]] = []
    for interval_start, interval_end in intervals:
        left = max(float(interval_start), start)
        right = min(float(interval_end), end)
        if right > left:
            clipped.append((left, right))
    return clipped


def sum_closed_eye_duration(
    intervals: Iterable[tuple[float, float]],
    *,
    epoch_start: float,
    epoch_end: float,
) -> float:
    """Return the total clipped closed-eye duration inside an epoch."""

    clipped_intervals = clip_intervals_to_epoch(
        intervals,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        )

    total_duration = 0.0

    for interval_start, interval_end in clipped_intervals:
        duration = interval_end - interval_start
        total_duration += duration

    return total_duration


def compute_epoch_perclos(
    intervals: Iterable[tuple[float, float]],
    *,
    epoch_length: float,
    epoch_start: float = 0.0,
) -> float:
    """Compute ratio-form PERCLOS for one epoch window."""

    closed_eye_duration = sum_closed_eye_duration(
        intervals,
        epoch_start=epoch_start,
        epoch_end=epoch_start + epoch_length,
    )
    perclos = closed_eye_duration / epoch_length
    return perclos


def compute_perclos_features(
    epochs: mne.Epochs,
    *,
    perclos_cutoff: float = 0.80,
) -> pd.DataFrame:
    """Compute interval-based PERCLOS and fatigue labels for refined epochs."""

    if epochs is None:
        raise ValueError("epochs is required")
    if len(epochs) == 0:
        raise ValueError("epochs must contain at least one epoch")
    if not _has_ear_channel(epochs):
        raise ValueError("epochs must include an EAR channel for PERCLOS computation")
    if not isinstance(epochs.metadata, pd.DataFrame):
        raise ValueError("epochs.metadata must contain EAR refinement metadata")

    metadata = epochs.metadata.reset_index(drop=True)
    sfreq = float(epochs.info["sfreq"])
    epoch_length = float(epochs.tmax - epochs.tmin)
    allow_empty_blinks = (
        "n_blinks" in metadata.columns and float(metadata["n_blinks"].fillna(0).sum()) == 0.0
    )
    interval_columns = _select_interval_columns(
        metadata, allow_empty_blinks=allow_empty_blinks
    )

    records: List[dict[str, float | int]] = []
    for epoch_idx in range(len(epochs)):
        row = metadata.iloc[epoch_idx]
        epoch_start = float(epochs.events[epoch_idx, 0] / sfreq)
        epoch_end = epoch_start + epoch_length

        intervals_samples = extract_closed_eye_intervals(
            row,
            interval_columns=interval_columns,
        )
        intervals_seconds = [
            (start_sample / sfreq, end_sample / sfreq)
            for start_sample, end_sample in intervals_samples
        ]
        perclos = compute_epoch_perclos(
            intervals_seconds,
            epoch_length=epoch_length,
        )
        records.append(
            {
                "epoch_start": epoch_start,
                "epoch_end": epoch_end,
                "perclos": perclos,
            }
        )

    df = pd.DataFrame.from_records(records, index=_epoch_index(epochs))
    df["fatigue_label"] = assign_fatigue_labels(
        df["perclos"].tolist(),
        perclos_cutoff=perclos_cutoff,
    )
    logger.debug("Computed PERCLOS DataFrame shape: %s", df.shape)
    return cast_columns_to_object(df)


__all__ = [
    "clip_intervals_to_epoch",
    "compute_epoch_perclos",
    "compute_perclos_features",
    "extract_closed_eye_intervals",
    "sum_closed_eye_duration",
]
