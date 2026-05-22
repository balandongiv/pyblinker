"""Annotation-related helper utilities."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import mne
import pandas as pd

from pyblinker._logging import get_logger

logger = get_logger(__name__)

DIFF_EVENT_LABEL_MATCH = "blink"
DIFF_EVENT_LABEL_DETECTED = "blink detect"
DIFF_EVENT_LABEL_GROUND_TRUTH = "blink ground truth"

ANN_DESCRIPTION_MATCH = "B"
ANN_DESCRIPTION_DETECTED = "BD"
ANN_DESCRIPTION_GROUND_TRUTH = "BG"


def _to_seconds(
    sample_index: Optional[float], sampling_rate_hz: float
) -> Optional[float]:
    if sample_index is None:
        return None
    return (float(sample_index) - 1.0) / sampling_rate_hz


def _duration_seconds(
    start_sample: int, end_sample: int, sampling_rate_hz: float
) -> float:
    return (float(end_sample) - float(start_sample) + 1.0) / sampling_rate_hz


def create_annotation(
    sblink: pd.DataFrame,
    sfreq: float,
    label: str,
) -> mne.Annotations:
    """Convert blink spans into an :class:`mne.Annotations` object.

    Parameters
    ----------
    sblink
        DataFrame containing ``start_blink`` and ``end_blink`` columns with
        blink sample indices.
    sfreq
        Sampling frequency of the signal in Hertz. Must be positive.
    label
        Annotation label applied to each blink.

    Returns
    -------
    mne.Annotations
        Annotation object describing the blinks.

    Raises
    ------
    TypeError
        If ``sblink`` is not a DataFrame or ``label`` is not a string.
    ValueError
        If required columns are missing or ``sfreq`` is non-positive.
    """

    logger.debug("Entering create_annotation")
    if not isinstance(sblink, pd.DataFrame):
        raise TypeError("sblink must be a pandas.DataFrame")

    required_cols = {"start_blink", "end_blink"}
    missing = required_cols - set(sblink.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(
            f"sblink is missing required columns: {missing_cols}",
        )

    if not isinstance(sfreq, (int, float)) or sfreq <= 0:
        raise ValueError("sfreq must be a positive number")

    if not isinstance(label, str) or not label:
        raise TypeError("label must be a non-empty string")

    onset = (sblink["start_blink"] / sfreq).to_list()
    duration = ((sblink["end_blink"] - sblink["start_blink"]) / sfreq).to_list()
    descriptions: Iterable[str] = [label] * len(onset)

    annot = mne.Annotations(
        onset=onset,
        duration=duration,
        description=list(descriptions),
    )
    logger.debug("Exiting create_annotation")
    return annot


def annotations_from_diff_table(
    diff_table: pd.DataFrame, sampling_rate_hz: float
) -> mne.Annotations | None:
    """Create :class:`mne.Annotations` from diff table rows.

    The diff table is expected to contain ``event_label`` columns matching the
    ``DIFF_EVENT_LABEL_*`` constants along with ``ground_truth_start``/
    ``ground_truth_end`` and ``detected_start``/``detected_end`` values. When a
    ``onset`` column is available it is used directly as the annotation onset;
    otherwise onsets are derived from the earliest available sample index.
    """

    if diff_table.empty:
        return None

    label_to_description = {
        DIFF_EVENT_LABEL_MATCH: ANN_DESCRIPTION_MATCH,
        DIFF_EVENT_LABEL_DETECTED: ANN_DESCRIPTION_DETECTED,
        DIFF_EVENT_LABEL_GROUND_TRUTH: ANN_DESCRIPTION_GROUND_TRUTH,
    }

    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []

    for row in diff_table.itertuples(index=False):
        description = label_to_description.get(row.event_label)
        if description is None:
            continue

        start_candidates = [
            value
            for value in (row.ground_truth_start, row.detected_start)
            if pd.notna(value)
        ]
        end_candidates = [
            value
            for value in (row.ground_truth_end, row.detected_end)
            if pd.notna(value)
        ]

        if not start_candidates or not end_candidates:
            continue

        start_sample = int(min(start_candidates))
        end_sample = int(max(end_candidates))
        if end_sample < start_sample:
            continue

        onset = (
            float(row.onset)
            if hasattr(row, "onset") and pd.notna(row.onset)
            else float(_to_seconds(start_sample, sampling_rate_hz))
        )
        duration = float(_duration_seconds(start_sample, end_sample, sampling_rate_hz))

        onsets.append(onset)
        durations.append(duration)
        descriptions.append(description)

    if not onsets:
        return None

    order = np.argsort(onsets)
    ordered_onsets = [onsets[idx] for idx in order]
    ordered_durations = [durations[idx] for idx in order]
    ordered_descriptions = [descriptions[idx] for idx in order]

    return mne.Annotations(
        onset=ordered_onsets,
        duration=ordered_durations,
        description=ordered_descriptions,
    )


__all__ = [
    "ANN_DESCRIPTION_DETECTED",
    "ANN_DESCRIPTION_GROUND_TRUTH",
    "ANN_DESCRIPTION_MATCH",
    "DIFF_EVENT_LABEL_DETECTED",
    "DIFF_EVENT_LABEL_GROUND_TRUTH",
    "DIFF_EVENT_LABEL_MATCH",
    "annotations_from_diff_table",
    "create_annotation",
]
