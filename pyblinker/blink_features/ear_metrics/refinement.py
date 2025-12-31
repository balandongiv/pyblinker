"""Threshold-based EAR blink boundary refinement utilities.

This module refines coarse blink annotations using a user-specified EAR threshold.
Blink existence comes from the CSV; the threshold is used only to sharpen onset
and offset. When threshold crossings are not available inside the coarse window,
the search is deterministically expanded outward up to a configurable limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from pyblinker.logging import get_logger

logger = get_logger(__name__)


AnnotationUnit = Literal["seconds", "samples"]


@dataclass
class EARRefinementConfig:
    """Configuration for threshold-based blink refinement.

    Parameters
    ----------
    threshold : float
        EAR threshold (raw units) used to detect downward/upward crossings.
    annotation_time_unit : {"seconds", "samples"}, default "seconds"
        Unit of the ``onset`` and ``duration`` columns in the coarse annotations.
    max_extension : float, default 0.35
        Maximum outward expansion in seconds to search for crossings when they
        do not exist inside the coarse window.
    extension_step : float, default 0.05
        Step size in seconds for each outward expansion attempt.
    padding : float, default 0.0
        Symmetric padding in seconds added around the coarse window before
        expansion attempts.
    extend_before : bool, default True
        Whether to allow expansion before the coarse onset.
    extend_after : bool, default True
        Whether to allow expansion after the coarse offset.
    """

    threshold: float
    annotation_time_unit: AnnotationUnit = "seconds"
    max_extension: float = 0.35
    extension_step: float = 0.05
    padding: float = 0.0
    extend_before: bool = True
    extend_after: bool = True

    def to_samples(self, value: float, sfreq: float) -> int:
        """Convert a time/duration value to samples based on the configured unit."""

        if self.annotation_time_unit == "seconds":
            return int(np.round(value * sfreq))
        return int(np.round(value))

    def duration_seconds(self, value: float, sfreq: float) -> float:
        """Convert a time/duration value to seconds based on the configured unit."""

        if self.annotation_time_unit == "seconds":
            return float(value)
        return float(value / sfreq)


def _compute_crossings(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Locate downward and upward threshold crossings from a boolean mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array where ``True`` marks samples under the threshold.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Indices of downward (False->True) and upward (True->False) crossings.
    """

    downward = np.flatnonzero(~mask[:-1] & mask[1:]) + 1
    upward = np.flatnonzero(mask[:-1] & ~mask[1:]) + 1
    return downward, upward


def _select_crossing_pair(
    mask: np.ndarray, start: int, end: int
) -> Optional[Tuple[int, int]]:
    """Select the first downward + subsequent upward crossing inside a window.

    Parameters
    ----------
    mask : np.ndarray
        Boolean under-threshold mask.
    start : int
        Inclusive search start sample.
    end : int
        Inclusive search end sample.

    Returns
    -------
    tuple[int, int] | None
        Downward and upward crossing sample indices if found; otherwise ``None``.
    """

    downward, upward = _compute_crossings(mask)
    downward = downward[(downward >= start) & (downward <= end)]
    upward = upward[(upward >= start) & (upward <= end)]

    if mask[start]:
        up_after = upward[upward >= start]
        if up_after.size:
            return start, int(up_after[0])

    for down in downward:
        up_after = upward[upward > down]
        if up_after.size:
            return int(down), int(up_after[0])

    return None


def _progressive_search(
    signal: np.ndarray,
    threshold: float,
    coarse_start: int,
    coarse_end: int,
    sfreq: float,
    config: EARRefinementConfig,
) -> Dict[str, float | int | bool]:
    """Search for threshold crossings, extending the window outward if needed.

    Parameters
    ----------
    signal : np.ndarray
        Full EAR signal (raw units).
    threshold : float
        EAR threshold for defining closed periods.
    coarse_start : int
        Coarse onset sample.
    coarse_end : int
        Coarse offset sample.
    sfreq : float
        Sampling frequency in Hertz.
    config : EARRefinementConfig
        Search configuration controlling padding and expansion.

    Returns
    -------
    dict
        Refined onset/offset samples, search window bounds (samples and seconds),
        whether refinement succeeded, and search diagnostics (extensions, attempts).
    """

    n_samples = signal.shape[0]
    padding_samples = int(round(config.padding * sfreq))
    window_start = max(0, coarse_start - padding_samples)
    window_end = min(n_samples - 1, coarse_end + padding_samples)

    max_ext_samples = int(round(config.max_extension * sfreq))
    step_samples = int(max(1, round(config.extension_step * sfreq)))

    attempts = 0
    mask = signal < threshold
    pair = _select_crossing_pair(mask, window_start, window_end)
    extension_used = 0

    while pair is None and extension_used < max_ext_samples:
        attempts += 1
        new_start = window_start
        new_end = window_end

        if config.extend_before:
            new_start = max(0, window_start - step_samples)
        if config.extend_after:
            new_end = min(n_samples - 1, window_end + step_samples)

        if new_start == window_start and new_end == window_end:
            break

        window_start, window_end = new_start, new_end
        extension_used = max(coarse_start - window_start, window_end - coarse_end)
        pair = _select_crossing_pair(mask, window_start, window_end)

    search_exhausted = pair is None
    refinement_success = pair is not None
    if pair is None:
        pair = (coarse_start, coarse_end)

    refined_start, refined_end = pair
    extension_seconds = extension_used / sfreq
    return {
        "refined_start_sample": int(refined_start),
        "refined_end_sample": int(refined_end),
        "refined_left_zero": int(refined_start),
        "refined_right_zero": int(refined_end),
        "search_window_start_sample": int(window_start),
        "search_window_end_sample": int(window_end),
        "search_window_start_time": float(window_start / sfreq),
        "search_window_end_time": float(window_end / sfreq),
        "refinement_succeeded": bool(refinement_success),
        "search_exhausted": bool(search_exhausted),
        "extension_seconds_used": float(extension_seconds),
        "extension_attempts": int(attempts),
    }


class EARThresholdBlinkRefiner:
    """Refine coarse blink annotations using EAR threshold crossings."""

    def __init__(
        self,
        signal: np.ndarray,
        sfreq: float,
        config: EARRefinementConfig,
    ):
        """Create a blink refiner.

        Parameters
        ----------
        signal : np.ndarray
            Full EAR signal (raw units).
        sfreq : float
            Sampling frequency in Hertz.
        config : EARRefinementConfig
            Threshold and search settings controlling refinement.
        """
        self.signal = np.asarray(signal, dtype=float)
        self.sfreq = float(sfreq)
        self.config = config

    def refine_annotation_row(
        self, row: Dict[str, float | str], candidate_id: int
    ) -> Dict[str, float | int | str | bool]:
        """Refine a single blink annotation row.

        Parameters
        ----------
        row : dict
            Annotation row containing ``onset`` and ``duration`` plus optional labels.
        candidate_id : int
            Unique identifier for the annotation row.

        Returns
        -------
        dict
            Refined timing, offsets, window metadata, and search diagnostics.
        """

        coarse_onset = float(row["onset"])
        duration = float(row["duration"])
        blink_label = row.get("blink_type") or row.get("description")

        coarse_start_sample = self.config.to_samples(coarse_onset, self.sfreq)
        coarse_duration_samples = self.config.to_samples(duration, self.sfreq)
        coarse_end_sample = max(coarse_start_sample, coarse_start_sample + coarse_duration_samples)

        coarse_end_sample = min(coarse_end_sample, self.signal.shape[0] - 1)
        coarse_duration_seconds = self.config.duration_seconds(duration, self.sfreq)

        search_result = _progressive_search(
            self.signal,
            self.config.threshold,
            coarse_start_sample,
            coarse_end_sample,
            self.sfreq,
            self.config,
        )

        refined_start_sample = int(search_result["refined_start_sample"])
        refined_end_sample = int(search_result["refined_end_sample"])

        refined_onset_time = refined_start_sample / self.sfreq
        refined_offset_time = refined_end_sample / self.sfreq

        coarse_offset_time = coarse_onset + coarse_duration_seconds

        return {
            "candidate_id": int(candidate_id),
            "blink_type": blink_label,
            "coarse_onset_time": float(coarse_onset),
            "coarse_offset_time": float(coarse_offset_time),
            "coarse_duration": float(coarse_duration_seconds),
            "refined_onset_time": float(refined_onset_time),
            "refined_offset_time": float(refined_offset_time),
            "refined_duration": float(refined_offset_time - refined_onset_time),
            "onset_offset_seconds": float(refined_onset_time - coarse_onset),
            "offset_offset_seconds": float(refined_offset_time - coarse_offset_time),
            "refined_start_sample": refined_start_sample,
            "refined_end_sample": refined_end_sample,
            "coarse_start_sample": int(coarse_start_sample),
            "coarse_end_sample": int(coarse_end_sample),
            "zero_crossing_found": bool(search_result["refinement_succeeded"]),
            **search_result,
        }

    def refine_annotations(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """Refine all blink annotations in a DataFrame.

        Parameters
        ----------
        annotations : pd.DataFrame
            Annotation table with ``onset`` and ``duration`` columns (seconds or samples).

        Returns
        -------
        pd.DataFrame
            Table containing refined timing, offsets, search metadata, and flags.
        """

        missing_cols = {"onset", "duration"} - set(annotations.columns)
        if missing_cols:
            raise ValueError(
                f"Annotation file is missing required columns: {sorted(missing_cols)}"
            )

        records: List[Dict[str, float | int | str | bool]] = []
        for idx, row in enumerate(annotations.itertuples(index=False)):
            record = self.refine_annotation_row(row._asdict(), idx)
            records.append(record)

        refined = pd.DataFrame.from_records(records)
        refined["refinement_used_outward_extension"] = (
            refined["extension_seconds_used"] > 0
        )
        refined["refinement_fallback_to_coarse"] = ~refined["refinement_succeeded"]

        logger.info(
            "Refined %s coarse blink annotations (threshold=%s)",
            len(refined),
            self.config.threshold,
        )
        return refined
