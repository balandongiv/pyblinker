"""Common routines for comparing blink detections against ground truth tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import mne
import numpy as np
import pandas as pd

from pyblinker._logging import get_logger
from pyblinker.utils.annotation_utils import annotations_from_diff_table

logger = get_logger(__name__)


@dataclass(slots=True)
class ComparisonResult:
    """Bundle containing comparison artifacts (metrics, diff table, alignments)."""

    annotations: mne.Annotations | None
    alignments: list | None
    metrics: dict[str, float]
    diff_table: pd.DataFrame


def build_indicator_signal(n_samples: int, events: pd.DataFrame) -> np.ndarray:
    """Create a binary signal marking blink intervals."""

    signal = np.zeros(int(n_samples), dtype=float)
    starts = events["start_blink"].to_numpy(dtype=int)
    ends = events["end_blink"].to_numpy(dtype=int)

    for start, end in zip(starts, ends, strict=False):
        start_idx = max(start - 1, 0)
        end_idx = min(end - 1, signal.size - 1)
        if end_idx < start_idx:
            continue
        signal[start_idx : end_idx + 1] = 1.0
    return signal


def _max_amplitude_within_events(
    events: pd.DataFrame, signal: np.ndarray | None
) -> pd.Series:
    """Return the maximum amplitude of ``signal`` within each blink interval."""

    if signal is None or signal.size == 0:
        return pd.Series(np.nan, index=events.index, dtype=float)

    max_values = np.full(len(events), np.nan, dtype=float)
    starts = events["start_blink"].to_numpy(dtype=int)
    ends = events["end_blink"].to_numpy(dtype=int)

    for idx, (start, end) in enumerate(zip(starts, ends, strict=False)):
        start_idx = max(int(start), 0)
        end_idx = min(int(end), signal.size - 1)
        if end_idx < start_idx:
            continue
        max_values[idx] = float(np.max(signal[start_idx : end_idx + 1]))

    return pd.Series(max_values, index=events.index, dtype=float, name="max_amplitude")


def _print_indicator_diagnostics(
    ground_truth_signal: np.ndarray,
    detected_signal: np.ndarray,
) -> dict[str, float]:
    """Return indicator signal diagnostics and log the summary."""

    if ground_truth_signal.size == 0 or detected_signal.size == 0:
        logger.info(
            "Indicator signal comparison skipped because one or more signals are empty."
        )
        return {}

    if ground_truth_signal.shape != detected_signal.shape:
        min_len = min(ground_truth_signal.size, detected_signal.size)
        if min_len == 0:
            logger.info(
                "Indicator signal comparison skipped because aligned signal length is zero."
            )
            return {}
        logger.warning(
            "Indicator signals have mismatched lengths (gt=%d, detected=%d); truncating to %d samples for diagnostics.",
            ground_truth_signal.size,
            detected_signal.size,
            min_len,
        )
        ground_truth_signal = ground_truth_signal[:min_len]
        detected_signal = detected_signal[:min_len]

    diff_signal = ground_truth_signal - detected_signal
    mean_abs_diff = float(np.mean(np.abs(diff_signal)))
    max_abs_diff = float(np.max(np.abs(diff_signal)))
    rms_diff = float(np.sqrt(np.mean(diff_signal**2)))
    if np.std(ground_truth_signal) > 0 and np.std(detected_signal) > 0:
        corr = float(np.corrcoef(ground_truth_signal, detected_signal)[0, 1])
    else:
        corr = float("nan")

    logger.info(
        "Indicator signal comparison:\n"
        "  • Mean absolute difference: %.6f\n"
        "  • RMS difference: %.6f\n"
        "  • Max absolute difference: %.6f\n"
        "  • Pearson correlation: %.6f",
        mean_abs_diff,
        rms_diff,
        max_abs_diff,
        corr,
    )

    return {
        "mean_abs_diff": mean_abs_diff,
        "rms_diff": rms_diff,
        "max_abs_diff": max_abs_diff,
        "pearson_corr": corr,
    }


def _filter_events_to_sample_window(
    events: pd.DataFrame, *, min_sample: int, max_sample: int
) -> pd.DataFrame:
    """Return a copy of ``events`` limited to blinks overlapping the sample window."""

    mask = (events["end_blink"] >= min_sample) & (events["start_blink"] <= max_sample)
    return events.loc[mask].copy()


def compute_alignments(
    detected_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    tolerance_samples: int,
    *,
    amplitude_rtol: float | None = None,
    amplitude_atol: float | None = None,
    require_both_conditions: bool | None = None,
):
    """Return alignments and metrics for two blink event tables."""

    from . import similarity

    if amplitude_rtol is None:
        amplitude_rtol = similarity.DEFAULT_AMPLITUDE_RTOL
    if amplitude_atol is None:
        amplitude_atol = similarity.DEFAULT_AMPLITUDE_ATOL
    if require_both_conditions is None:
        require_both_conditions = similarity.DEFAULT_REQUIRE_BOTH_CONDITIONS

    alignments = similarity.align_events(
        detected_df=detected_df,
        ground_truth_df=ground_truth_df,
        tolerance_samples=tolerance_samples,
        amplitude_rtol=amplitude_rtol,
        amplitude_atol=amplitude_atol,
        require_both_conditions=require_both_conditions,
    )

    return alignments


def build_comparison_annotations(
    *,
    ground_truth_starts: Sequence[int],
    ground_truth_ends: Sequence[int],
    detected_starts: Sequence[int],
    detected_ends: Sequence[int],
    sampling_rate_hz: float,
    tolerance_samples: int,
    alignments: Iterable | None = None,
):
    """Construct :class:`mne.Annotations` describing blink comparisons."""

    from . import reporting

    ground_truth_df = pd.DataFrame(
        {"start_blink": ground_truth_starts, "end_blink": ground_truth_ends}
    )
    detected_df = pd.DataFrame(
        {"start_blink": detected_starts, "end_blink": detected_ends}
    )

    ground_truth_df["max_amplitude"] = np.nan
    detected_df["max_amplitude"] = np.nan

    if alignments is None:
        from . import similarity

        alignments = similarity.align_events(
            detected_df=detected_df,
            ground_truth_df=ground_truth_df,
            tolerance_samples=tolerance_samples,
        )

    alignments_list = list(alignments)

    diff_table = reporting.make_diff_table(
        detected_df,
        ground_truth_df,
        alignments_list,
        tolerance_samples,
        max_rows=len(ground_truth_df) + len(detected_df),
        sampling_rate_hz=sampling_rate_hz,
    )

    return annotations_from_diff_table(diff_table, sampling_rate_hz)


def compare_detected_vs_ground_truth(
    detected: pd.DataFrame,
    ground_truth_events: pd.DataFrame,
    sampling_rate_hz: float,
    *,
    tolerance_samples: int,
    amplitude_rtol: float | None = None,
    amplitude_atol: float | None = None,
    require_both_conditions: bool | None = None,
    n_preview_rows: int,
    n_diff_rows: int,
    ground_truth_signal: np.ndarray | None = None,
    detected_signal: np.ndarray | None = None,
) -> ComparisonResult:
    """Compare detected events with ground truth and build diagnostic visuals."""

    from . import reporting, similarity

    detected_df = detected.copy()
    ground_truth_df = ground_truth_events.copy()

    similarity.validate_event_table(detected_df)
    similarity.validate_event_table(ground_truth_df)

    sample_window: tuple[int, int] | None = None
    if detected_signal is not None and len(detected_signal) > 0:
        sample_window = (1, int(len(detected_signal)))
    elif ground_truth_signal is not None and len(ground_truth_signal) > 0:
        sample_window = (1, int(len(ground_truth_signal)))

    if sample_window is not None:
        min_sample, max_sample = sample_window
        detected_before = len(detected_df)
        ground_truth_before = len(ground_truth_df)
        detected_df = _filter_events_to_sample_window(
            detected_df, min_sample=min_sample, max_sample=max_sample
        )
        ground_truth_df = _filter_events_to_sample_window(
            ground_truth_df, min_sample=min_sample, max_sample=max_sample
        )
        removed_detected = detected_before - len(detected_df)
        removed_ground_truth = ground_truth_before - len(ground_truth_df)
        if removed_detected or removed_ground_truth:
            logger.info(
                "Ignoring blink events outside the available sample window [%d, %d] -> "
                "removed %d detected / %d ground truth",
                min_sample,
                max_sample,
                removed_detected,
                removed_ground_truth,
            )

    nprev = min(n_preview_rows, len(detected_df), len(ground_truth_df))

    detected_df["max_amplitude"] = _max_amplitude_within_events(
        detected_df, detected_signal
    )
    ground_truth_df["max_amplitude"] = _max_amplitude_within_events(
        ground_truth_df, detected_signal
    )

    alignments = compute_alignments(
        detected_df=detected_df,
        ground_truth_df=ground_truth_df,
        tolerance_samples=tolerance_samples,
        amplitude_rtol=amplitude_rtol,
        amplitude_atol=amplitude_atol,
        require_both_conditions=require_both_conditions,
    )

    start_diff, end_diff = similarity.compute_pairwise_differences(
        detected_df, ground_truth_df
    )
    ok_start = (
        start_diff <= tolerance_samples if start_diff.size else np.array([], dtype=bool)
    )
    ok_end = (
        end_diff <= tolerance_samples if end_diff.size else np.array([], dtype=bool)
    )
    all_ok = (
        (ok_start.size == 0 or np.all(ok_start))
        and (ok_end.size == 0 or np.all(ok_end))
        and (len(detected_df) == len(ground_truth_df))
    )

    diff_table = reporting.make_diff_table(
        detected_df,
        ground_truth_df,
        alignments,
        tolerance_samples,
        n_diff_rows,
        sampling_rate_hz,
    )

    match_category = diff_table.get("match_category")
    within_tolerance = diff_table.get("within_tolerance")
    within_tolerance_series = (
        pd.Series(within_tolerance, copy=False).astype("boolean")
        if within_tolerance is not None
        else pd.Series(False, index=diff_table.index, dtype="boolean")
    )
    within_tolerance_mask = within_tolerance_series.fillna(False)

    mismatch_mask = (diff_table["event_label"] != reporting.DIFF_EVENT_LABEL_MATCH) | (
        match_category.isin(["pairs_outside_tolerance", "share_within_tolerance"])
        & ~within_tolerance_mask
    )

    metrics = similarity.compute_alignment_metrics(diff_table)
    reporting.print_comparison_summary(metrics, tolerance_samples)

    indicator_metrics: dict[str, float] = {}
    if ground_truth_signal is not None and detected_signal is not None:
        indicator_metrics = _print_indicator_diagnostics(
            ground_truth_signal, detected_signal
        )
    else:
        logger.debug(
            "Skipping indicator diagnostics because one or both signals were not provided."
        )

    if indicator_metrics:
        metrics.update(
            {f"indicator_{key}": value for key, value in indicator_metrics.items()}
        )

    annotations = annotations_from_diff_table(diff_table, sampling_rate_hz)

    metrics.update(
        {
            "input_tolerance_samples": float(tolerance_samples),
            "input_detected_rows": float(len(detected_df)),
            "input_ground_truth_rows": float(len(ground_truth_df)),
            "input_row_count_matches": float(len(detected_df) == len(ground_truth_df)),
            "input_row_count_delta": float(len(detected_df) - len(ground_truth_df)),
            "input_preview_rows": float(nprev),
        }
    )

    return ComparisonResult(
        annotations=annotations,
        alignments=alignments,
        metrics=metrics,
        diff_table=diff_table,
    )
