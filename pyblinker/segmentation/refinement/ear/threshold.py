"""Threshold-based EAR blink boundary refinement utilities.

This module refines coarse blink annotations using a user-specified EAR threshold.
Blink existence comes from the CSV; the threshold is used only to sharpen onset
and offset. When threshold crossings are not available inside the coarse window,
the search is deterministically expanded outward up to a configurable limit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pyblinker._logging import get_logger

logger = get_logger(__name__)


AnnotationUnit = Literal["seconds", "samples"]
Number = Union[int, float]


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


def _progressive_threshold_search(
    signal: np.ndarray,
    threshold: float,
    coarse_start: int,
    coarse_end: int,
    sfreq: float,
    config: EARRefinementConfig,
) -> Dict[str, float | int | bool]:
    """Use this
    Search for threshold crossings, extending the window outward if needed.

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
        By refined, we mean the first valid threshold crossing pair found within the search windows
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
        "start__th_point__ear": int(refined_start),
        "end__th_point__ear": int(refined_end),
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

    def _compute_lowest_point_sample(self, start: int, end: int) -> float:
        """Return the lowest EAR sample index within the refined interval.

        The search is inclusive of ``start`` and ``end``. When the interval is invalid,
        empty, or contains no finite values, ``nan`` is returned.
        """

        n_samples = self.signal.shape[0]
        if start is None or end is None:
            return float("nan")
        try:
            start_idx = int(start)
            end_idx = int(end)
        except (TypeError, ValueError):
            return float("nan")

        if start_idx < 0 or end_idx >= n_samples or start_idx > end_idx:
            return float("nan")

        window = self.signal[start_idx : end_idx + 1]
        if window.size == 0:
            return float("nan")

        finite_mask = np.isfinite(window)
        if not finite_mask.any():
            return float("nan")

        local_min = int(np.argmin(np.where(finite_mask, window, np.inf)))
        return float(start_idx + local_min)

    def _compute_interpolated_threshold_crossings(
        self,
        refined_start_sample: int,
        refined_end_sample: int,
        lowest_point_sample: float,
    ) -> Dict[str, float | int | bool]:
        """
        Check by rpb on 5/1/25
        Return linearly interpolated threshold crossings around a refined blink.

        Interpolated crossings are searched within a padded window surrounding the refined
        start/end samples. Crossings must occur before and after the blink minimum,
        respectively, and are computed using linear interpolation between adjacent samples.
        Missing or invalid inputs produce per-side ``nan`` outputs without raising.
        """

        result: Dict[str, float | int | bool] = {
            "left_interpolated_threshold": float("nan"),
            "right_interpolated_threshold": float("nan"),
            "left_interpolated_threshold_sample": float("nan"),
            "right_interpolated_threshold_sample": float("nan"),
            "left_interpolated_threshold_found": False,
            "right_interpolated_threshold_found": False,
            "interpolated_thresholds_found": False,
        }

        n_samples = self.signal.shape[0]
        if not np.isfinite(lowest_point_sample):
            return result

        try:
            refined_start = int(refined_start_sample)
            refined_end = int(refined_end_sample)
            min_sample = int(lowest_point_sample)
        except (TypeError, ValueError):
            return result

        padding_samples = int(round(self.config.padding * self.sfreq))
        search_start = max(0, refined_start - padding_samples)
        search_end = min(n_samples - 1, refined_end + padding_samples)
        if search_start >= search_end:
            return result
        if min_sample < search_start or min_sample > search_end:
            return result

        window = self.signal[search_start : search_end + 1]
        if window.size < 2:
            return result

        distances = window - self.config.threshold
        downward = np.flatnonzero((distances[:-1] > 0) & (distances[1:] <= 0))
        upward = np.flatnonzero((distances[:-1] < 0) & (distances[1:] >= 0))

        left_candidates = downward + search_start
        right_candidates = upward + search_start

        left_index = None
        if left_candidates.size:
            before_min = left_candidates[left_candidates <= min_sample]
            if before_min.size:
                left_index = int(before_min[-1])

        right_index = None
        if right_candidates.size:
            after_min = right_candidates[right_candidates >= min_sample]
            if after_min.size:
                right_index = int(after_min[0])

        if left_index is None or right_index is None:
            return result

        def interpolate(crossing_sample: int) -> Optional[float]:
            local_idx = crossing_sample - search_start
            denom = distances[local_idx] - distances[local_idx + 1]
            if denom == 0:
                return None
            return crossing_sample + distances[local_idx] / denom

        left_cross = interpolate(
            left_index
        )  # Since this interpolation,the output should be a float sample index representing the estimated crossing point, The same applies to the right_cross.
        right_cross = interpolate(right_index)
        if left_cross is None or right_cross is None:
            return result

        left_time = left_cross / self.sfreq
        right_time = right_cross / self.sfreq
        result.update(
            {
                "start_interpolated_threshold": float(left_cross),
                "end_interpolated_threshold": float(right_cross),
                # "left_interpolated_threshold_sample": int(left_sample_int),
                # "right_interpolated_threshold_sample": int(right_sample_int),
                "left_interpolated_threshold_found": True,
                "right_interpolated_threshold_found": True,
                "interpolated_thresholds_found": True,
                "onset__th_interpolation__ear": float(
                    left_time
                ),  # To be removed as we will use only start_interpolated_threshold
                "duration__th_interpolation__ear": float(
                    right_time - left_time
                ),  # To be removed as we will use only start_interpolated_threshold and end_interpolated_threshold
            }
        )
        return result

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
            Refined timing, offsets, lowest-point sample index, window metadata, and
            search diagnostics.
        """

        coarse_onset = float(row["onset"])
        duration = float(row["duration"])
        blink_label = row.get("blink_type") or row.get("description")

        coarse_start_sample = self.config.to_samples(coarse_onset, self.sfreq)
        coarse_duration_samples = self.config.to_samples(duration, self.sfreq)
        coarse_end_sample = max(
            coarse_start_sample, coarse_start_sample + coarse_duration_samples
        )

        coarse_end_sample = min(coarse_end_sample, self.signal.shape[0] - 1)
        coarse_duration_seconds = self.config.duration_seconds(duration, self.sfreq)

        search_result = _progressive_threshold_search(
            self.signal,
            self.config.threshold,
            coarse_start_sample,
            coarse_end_sample,
            self.sfreq,
            self.config,
        )
        # Here, we focus based on the threshold search result to get the lowest point within the threshold refined window
        refined_start_sample = int(search_result["start__th_point__ear"])
        refined_end_sample = int(search_result["end__th_point__ear"])

        # The output is the lowest point within the refined window define with threshold crossings
        refined_lowest_point_sample = self._compute_lowest_point_sample(
            refined_start_sample, refined_end_sample
        )

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
            "refined_lowest_point_sample": refined_lowest_point_sample,
            "coarse_start_sample": int(coarse_start_sample),
            "coarse_end_sample": int(coarse_end_sample),
            "threshold_crossing_found": bool(search_result["refinement_succeeded"]),
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
        # why not use refinements = _refine_ear_blinks_for_epoch(
        #         seg,
        #         blink_starts,
        #         blink_ends,
        #         sfreq,
        #         segment_config,
        #     )

        records: List[Dict[str, float | int | str | bool]] = []
        for idx, row in enumerate(annotations.itertuples(index=False)):
            record = self.refine_annotation_row(row._asdict(), idx)
            record.update(
                self._compute_interpolated_threshold_crossings(
                    refined_start_sample=record["refined_start_sample"],
                    refined_end_sample=record["refined_end_sample"],
                    lowest_point_sample=record["refined_lowest_point_sample"],
                )
            )
            records.append(record)

        refined = pd.DataFrame.from_records(records)
        refined["refinement_used_outward_extension"] = (
            refined["extension_seconds_used"] > 0
        )
        refined["refinement_fallback_to_coarse"] = ~refined["refinement_succeeded"]
        refined["threshold_value"] = float(self.config.threshold)

        logger.info(
            "Refined %s coarse blink annotations (threshold=%s)",
            len(refined),
            self.config.threshold,
        )
        return refined


def refine_annotations_for_threshold(
    signal: np.ndarray,
    sfreq: float,
    annotations: pd.DataFrame,
    base_config: EARRefinementConfig,
    candidate_threshold: Number,
    threshold_index: Optional[int] = None,
) -> pd.DataFrame:
    """Refine annotations for a single threshold value.

    Parameters
    ----------
    signal : np.ndarray
        Full EAR signal (raw units).
    sfreq : float
        Sampling frequency in Hertz.
    annotations : pd.DataFrame
        Coarse blink annotations with ``onset`` and ``duration`` columns.
    base_config : EARRefinementConfig
        Baseline configuration that will be copied for the provided threshold.
    candidate_threshold : float | int
        Threshold value to evaluate.
    threshold_index : int | None, optional
        Optional index to preserve caller-provided ordering across thresholds.

    Returns
    -------
    pd.DataFrame
        Refinement table containing the ``threshold_value`` and ``threshold_index`` columns
        identifying the threshold used for each row.
    """

    theta = float(candidate_threshold)
    threshold_config = replace(base_config, threshold=theta)
    refiner = EARThresholdBlinkRefiner(signal, sfreq, threshold_config)
    refined = refiner.refine_annotations(annotations)
    refined["threshold_value"] = theta
    refined["threshold_index"] = (
        int(threshold_index) if threshold_index is not None else 0
    )

    logger.info(
        "Refined annotations for threshold=%s; resulting rows=%s",
        theta,
        len(refined),
    )
    return refined


__all__ = [
    "EARRefinementConfig",
    "EARThresholdBlinkRefiner",
    "_progressive_threshold_search",
    "refine_annotations_for_threshold",
]
