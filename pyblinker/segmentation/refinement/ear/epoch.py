"""EAR-specific refinement helpers used by segmentation pipelines."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pyblinker._logging import get_logger

from ..eeg import compute_outer_bounds
from .threshold import (
    EARRefinementConfig,
    EARThresholdBlinkRefiner,
    _progressive_threshold_search,
)

logger = get_logger(__name__)


def _select_seg_type(seg_type_value: Any) -> Optional[str]:
    """Normalize segmentation type configuration."""

    if isinstance(seg_type_value, str):
        return seg_type_value
    if isinstance(seg_type_value, Sequence):
        seg_types = [str(item) for item in seg_type_value]
        if not seg_types:
            return None
        if "threshold_interpolation" in seg_types:
            return "threshold_interpolation"
        return seg_types[0]
    return None


def _locate_trough(signal: np.ndarray, start: int, end: int) -> Optional[int]:
    """Return the index of the minimum value between ``start`` and ``end``."""

    # if signal.size == 0:
    #     return None
    n = len(signal)
    start_idx = int(np.clip(start, 0, n - 1))
    end_idx = int(np.clip(end, 0, n - 1))
    if end_idx < start_idx:
        start_idx = end_idx
    window = signal[start_idx : end_idx + 1]
    if window.size == 0:
        return start_idx
    finite_mask = np.isfinite(window)
    if not finite_mask.any():
        return start_idx
    local_min = int(np.nanargmin(window))
    return start_idx + local_min


def _empty_interpolated_thresholds() -> Dict[str, float | int | bool]:
    """Return default interpolated threshold metadata with NaN/False values."""

    return {
        "left_interpolated_threshold": float("nan"),
        "right_interpolated_threshold": float("nan"),
        "left_interpolated_threshold_sample": float("nan"),
        "right_interpolated_threshold_sample": float("nan"),
        "left_interpolated_threshold_found": False,
        "right_interpolated_threshold_found": False,
        "interpolated_thresholds_found": False,
    }


def _append_outer_bounds_from_peaks(
    row_data: Dict[str, Any],
    peaks: Sequence[int],
    key_prefix: str,
    n_samp_epoch: int,
) -> None:
    if not peaks:
        return
    bounds = compute_outer_bounds(peaks, n_samp_epoch)
    row_data[f"blink_outer_start_{key_prefix}"] = [
        outer_start for outer_start, _ in bounds
    ]
    row_data[f"blink_outer_end_{key_prefix}"] = [outer_end for _, outer_end in bounds]


def _fallback_refinement(
    coarse_start: int, coarse_end: int, segment: np.ndarray, sfreq: float
) -> Dict[str, Any]:
    n = len(segment)
    if n == 0:
        start = end = 0
    else:
        start = int(np.clip(coarse_start, 0, n - 1))
        end = int(np.clip(coarse_end, 0, n - 1))
        if end < start:
            end = start
    return {
        "refined_start_sample": int(start),
        "refined_end_sample": int(end),
        "start__th_point__ear": int(start),
        "end__th_point__ear": int(end),
        "refined_left_threshold": int(start),
        "refined_right_threshold": int(end),
        "search_window_start_sample": int(start),
        "search_window_end_sample": int(end),
        "search_window_start_time": float(start / sfreq),
        "search_window_end_time": float(end / sfreq),
        "refinement_succeeded": False,
        "search_exhausted": True,
        "extension_seconds_used": 0.0,
        "extension_attempts": 0,
    }


def _refine_ear_blinks_for_epoch(
    segment: np.ndarray,
    blink_starts: Sequence[int],
    blink_ends: Sequence[int],
    sfreq: float,
    segmentation_config: Optional[dict],
) -> List[Dict[str, Any]]:
    """Refine EAR blinks for a single epoch based on segmentation settings.
    The function name should be change to ear_landmark_refinement_for_epoch
    This function first find the timepoints that near the selected threshold,

    Then, it proceed to find the interpolated threshold crossings
    """

    if segment.size == 0 or not blink_starts:
        return []

    ear_config = (
        (segmentation_config or {}).get("ear", {})
        if segmentation_config is not None
        else {}
    )
    seg_type = _select_seg_type(ear_config.get("seg_type"))
    # use_threshold_interpolation = seg_type

    threshold = ear_config.get("threshold")
    # if use_threshold_interpolation and threshold is None:
    #     logger.warning("EAR threshold missing for threshold_interpolation; falling back to coarse bounds.")
    #     use_threshold_interpolation = False

    config: EARRefinementConfig | None = None
    refiner: EARThresholdBlinkRefiner | None = None

    if threshold is not None:
        config = EARRefinementConfig(threshold=threshold)
        override_fields = {
            k: v
            for k, v in ear_config.items()
            if k
            in {
                "annotation_time_unit",
                "max_extension",
                "extension_step",
                "padding",
                "extend_before",
                "extend_after",
            }
        }
        if override_fields:
            config = replace(config, **override_fields)
        refiner = EARThresholdBlinkRefiner(segment, sfreq, config)

    refinements: List[Dict[str, Any]] = []
    for coarse_start, coarse_end in zip(blink_starts, blink_ends):
        if refiner is None or config is None:
            refinement = _fallback_refinement(coarse_start, coarse_end, segment, sfreq)
        else:
            refinement = _progressive_threshold_search(
                segment,
                threshold,
                coarse_start,
                coarse_end,
                sfreq,
                config,
            )

        trough_sample = _locate_trough(
            segment,
            refinement["start__th_point__ear"],
            refinement["end__th_point__ear"],
        )
        refinement["trough__th_point__ear"] = trough_sample

        # Step 2: compute interpolated threshold crossings if applicable
        if seg_type == "threshold_interpolation" and refiner is not None:
            interp = refiner._compute_interpolated_threshold_crossings(
                refined_start_sample=refinement["start__th_point__ear"],
                refined_end_sample=refinement["end__th_point__ear"],
                lowest_point_sample=refinement["trough__th_point__ear"]
                if trough_sample is not None
                else float("nan"),
            )
        else:
            interp = _empty_interpolated_thresholds()

        # Step 3: Calculate the outer bounds based on the peaks and n_samp_epoch

        # bounds = compute_outer_bounds(peaks, n_samp_epoch)
        refinement.update(interp)
        refinements.append(refinement)

    return refinements


def _append_ear_refinements(
    row_data: Dict[str, Any],
    refinements: Sequence[Dict[str, Any]],
    sfreq: float,
    n_samp_epoch: int,
) -> None:
    if not refinements:
        return

    transposed: Dict[str, List[Any]] = {}
    initialized = False

    for refinement in refinements:
        start__th_point__ear = refinement[
            "start__th_point__ear"
        ]  # ?????? This is the refined start sample based on threshold search, not necessarily the original coarse start
        end__th_point__ear = refinement[
            "end__th_point__ear"
        ]  # ?????? This is the refined end sample based on threshold search, not necessarily the original coarse end
        trough__th_point__ear = refinement[
            "trough__th_point__ear"
        ]  # ?????? This is the located trough sample between the refined start and end, not necessarily related to the original coarse bounds

        # onset__th__ear = start_th
        # duration__th__ear = (end_th - start_th) / sfreq

        start__th_interpolation__ear = refinement.get(
            "start_interpolated_threshold", float("nan")
        )
        end__th_interpolation__ear = refinement.get(
            "end_interpolated_threshold", float("nan")
        )

        # start = refinement.get("refined_start_sample", start_th)
        # end = refinement.get("refined_end_sample", end_th)
        # trough = trough_th

        # onset_time = float(start_th) / sfreq
        # duration_time = max(0.0, float(end_th - start_th) / sfreq)

        # has_interp = np.isfinite(onset__th_interpolation__ear) and np.isfinite(
        # 	duration__th_interpolation__ear
        # 	)
        # left_time = float(onset__th_interpolation__ear) if has_interp else float("nan")
        # right_time = (
        # 		float(onset__th_interpolation__ear + duration__th_interpolation__ear)
        # 		if has_interp
        # 		else float("nan")
        # )
        #
        # left_sample = (
        # 		int(np.clip(round(left_time * sfreq), 0, n_samp_epoch - 1))
        # 		if np.isfinite(left_time)
        # 		else None
        # )
        # right_sample = (
        # 		int(np.clip(round(right_time * sfreq), 0, n_samp_epoch - 1))
        # 		if np.isfinite(right_time)
        # 		else None
        # )

        entry: Dict[str, Any] = {
            "start__th_point__ear": start__th_point__ear,
            "end__th_point__ear": end__th_point__ear,
            "trough__th_point__ear": trough__th_point__ear,
            "start__th_interpolation__ear": start__th_interpolation__ear,
            "end__th_interpolation__ear": end__th_interpolation__ear,
            # "blink_onset_ear": onset_time,
            # "blink_duration_ear": duration_time,
            # "blink_onset_extremum_ear": (
            # 		float(trough) / sfreq
            # 		if trough is not None and np.isfinite(trough)
            # 		else float(onset_time)
            # ),
            # "onset__refine__ear": onset_time,
            # "duration__refine__ear": duration_time,
            # "refined_start_sample": int(start),
            # "refined_end_sample": int(end),
            # "refined_lowest_point_sample": (
            # 		int(trough) if trough is not None and np.isfinite(trough) else np.nan
            # ),
            # "refined_left_threshold": int(refinement["refined_left_threshold"]),
            # "refined_right_threshold": int(refinement["refined_right_threshold"]),
            # "search_window_start_sample": int(refinement["search_window_start_sample"]),
            # "search_window_end_sample": int(refinement["search_window_end_sample"]),
            # "search_window_start_time": float(refinement["search_window_start_time"]),
            # "search_window_end_time": float(refinement["search_window_end_time"]),
            # "refinement_succeeded": bool(refinement["refinement_succeeded"]),
            # "search_exhausted": bool(refinement["search_exhausted"]),
            # "extension_seconds_used": float(refinement["extension_seconds_used"]),
            # "extension_attempts": int(refinement["extension_attempts"]),
            # "onset__th_interpolation__ear": float(
            # 	refinement.get("onset__th_interpolation__ear", float("nan"))
            # 	),
            # "duration__th_interpolation__ear": float(
            # 	refinement.get("duration__th_interpolation__ear", float("nan"))
            # 	),
            # "left_interpolated_threshold": float(left_time),
            # "right_interpolated_threshold": float(right_time),
            # "left_interpolated_threshold_sample": (
            # 		float(left_sample) if left_sample is not None else float("nan")
            # ),
            # "right_interpolated_threshold_sample": (
            # 		float(right_sample) if right_sample is not None else float("nan")
            # ),
            # "left_interpolated_threshold_found": bool(
            # 	refinement.get("left_interpolated_threshold_found", False)
            # 	),
            # "right_interpolated_threshold_found": bool(
            # 	refinement.get("right_interpolated_threshold_found", False)
            # 	),
            # "interpolated_thresholds_found": bool(
            # 	refinement.get("interpolated_thresholds_found", False)
            # 	),
            # "onset__th_sample__ear": float(
            # 	refinement.get("onset__th_sample__ear", float("nan"))
            # 	),
            # "duration__th_sample__ear": float(
            # 	refinement.get("duration__th_sample__ear", float("nan"))
            # 	),
            # "start__th_point__ear": float(start_th),
            # "end__th_point__ear": float(end_th),
            # "trough__th_point__ear": (
            # 		float(trough_th)
            # 		if trough_th is not None and np.isfinite(trough_th)
            # 		else float("nan")
            # ),
            # "onset__th__ear": onset__th__ear,
            # "duration__th__ear": duration__th__ear,
        }

        # Initialize transposed dict once (preserves key order from first entry)
        if not initialized:
            transposed = {k: [] for k in entry.keys()}
            initialized = True

        # Append values directly into the transposed structure
        for k, v in entry.items():
            transposed[k].append(v)

    if not initialized:
        return

    row_data.update(transposed)


__all__ = [
    "_append_ear_refinements",
    "_append_outer_bounds_from_peaks",
    "_empty_interpolated_thresholds",
    "_locate_trough",
    "_refine_ear_blinks_for_epoch",
    "_select_seg_type",
]
