"""Common routines for comparing blink detections against ground truth tables."""

from __future__ import annotations

from typing import Iterable, Sequence

import mne
import numpy as np
import pandas as pd

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
) -> None:
    """Print quick diagnostics comparing binary indicator signals."""

    if ground_truth_signal.size == 0:
        print("\n[metrics] Indicator signal comparison skipped (no samples).")
        return

    diff_signal = ground_truth_signal - detected_signal
    mean_abs_diff = float(np.mean(np.abs(diff_signal)))
    max_abs_diff = float(np.max(np.abs(diff_signal)))
    rms_diff = float(np.sqrt(np.mean(diff_signal**2)))
    if np.std(ground_truth_signal) > 0 and np.std(detected_signal) > 0:
        corr = float(np.corrcoef(ground_truth_signal, detected_signal)[0, 1])
    else:
        corr = float("nan")
    print("\n[metrics] Indicator signal comparison:")
    print(f"  • Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"  • RMS difference: {rms_diff:.6f}")
    print(f"  • Max absolute difference: {max_abs_diff:.6f}")
    print(f"  • Pearson correlation: {corr:.6f}")


def compute_alignments_and_metrics(
    detected_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    tolerance_samples: int,
):
    """Return alignments and metrics for two blink event tables."""

    from . import similarity

    alignments = similarity.align_events(
        detected_df=detected_df,
        ground_truth_df=ground_truth_df,
        tolerance_samples=tolerance_samples,
    )

    amplitude_column = "max_amplitude"
    if amplitude_column in detected_df.columns and amplitude_column in ground_truth_df.columns:
        detected_amplitudes = detected_df[amplitude_column].to_numpy(dtype=float, copy=False)
        ground_truth_amplitudes = ground_truth_df[amplitude_column].to_numpy(dtype=float, copy=False)

        amplitude_rtol = 1e-3
        amplitude_atol = 5e-10

        def _is_comparable(value: float) -> bool:
            return np.isfinite(value)

        refined_alignments: list[similarity.Alignment] = []
        for alignment in alignments:
            det_idx = alignment.detected_idx
            gt_idx = alignment.ground_truth_idx

            if det_idx is None or gt_idx is None:
                refined_alignments.append(alignment)
                continue

            det_amp = detected_amplitudes[int(det_idx)] if int(det_idx) < detected_amplitudes.size else np.nan
            gt_amp = ground_truth_amplitudes[int(gt_idx)] if int(gt_idx) < ground_truth_amplitudes.size else np.nan

            if not (_is_comparable(det_amp) and _is_comparable(gt_amp)):
                refined_alignments.append(alignment)
                continue

            amplitudes_are_close = np.isclose(
                det_amp,
                gt_amp,
                rtol=amplitude_rtol,
                atol=amplitude_atol,
            )

            if amplitudes_are_close and alignment.is_match(tolerance_samples):
                refined_alignments.append(alignment)
                continue

            refined_alignments.append(
                similarity.Alignment(
                    ground_truth_idx=gt_idx,
                    detected_idx=None,
                    start_diff=None,
                    end_diff=None,
                )
            )
            refined_alignments.append(
                similarity.Alignment(
                    ground_truth_idx=None,
                    detected_idx=det_idx,
                    start_diff=None,
                    end_diff=None,
                )
            )

        alignments = refined_alignments

    metrics = similarity.compute_alignment_metrics(alignments, tolerance_samples)
    return alignments, metrics


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

    _, onsets, durations, descriptions = reporting._build_alignment_annotation_payload(
        ground_truth_starts,
        ground_truth_ends,
        detected_starts,
        detected_ends,
        sampling_rate_hz,
        tolerance_samples,
        alignments,
    )

    if onsets:
        return mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    return None


def compare_detected_vs_ground_truth(
    detected: pd.DataFrame,
    ground_truth_events: pd.DataFrame,
    sampling_rate_hz: float,
    *,
    tolerance_samples: int,
    n_preview_rows: int,
    n_diff_rows: int,
    ground_truth_signal: np.ndarray | None = None,
    detected_signal: np.ndarray | None = None,
) -> mne.io.RawArray:
    """Compare detected events with ground truth and build diagnostic visuals."""

    from . import reporting, similarity

    detected_df = detected.copy()
    ground_truth_df = ground_truth_events.copy()

    similarity.validate_event_table(detected_df)
    similarity.validate_event_table(ground_truth_df)

    print("\n================ COMPARISON: Detected vs Ground Truth ================")
    print(f"Tolerance allowed: ±{tolerance_samples} samples")
    print(f"Detected rows    : {len(detected_df)}")
    print(f"Ground truth rows: {len(ground_truth_df)}")
    print("Row count matches? ->", len(detected_df) == len(ground_truth_df))

    nprev = min(n_preview_rows, len(detected_df), len(ground_truth_df))
    preview = reporting.preview_side_by_side(detected_df, ground_truth_df, nprev)
    print(f"\nFirst {nprev} rows (ground truth vs detected):")
    print(preview)

    detected_df["max_amplitude"] = _max_amplitude_within_events(
        detected_df, detected_signal
        )
    ground_truth_df["max_amplitude"] = _max_amplitude_within_events(
        ground_truth_df, detected_signal
        )

    alignments, metrics = compute_alignments_and_metrics(
        detected_df=detected_df,
        ground_truth_df=ground_truth_df,
        tolerance_samples=tolerance_samples,
    )



    start_diff, end_diff = similarity.compute_pairwise_differences(detected_df, ground_truth_df)
    ok_start = start_diff <= tolerance_samples if start_diff.size else np.array([], dtype=bool)
    ok_end = end_diff <= tolerance_samples if end_diff.size else np.array([], dtype=bool)
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

    if all_ok and diff_table.empty:
        print(f"\n✅ PASSED: all blink intervals match within ±{tolerance_samples} samples.")
    else:
        print(f"\n[diff] mismatches beyond ±{tolerance_samples} samples (showing {n_diff_rows}):")
        if diff_table.empty:
            print("No mismatches found despite metric discrepancies.")
        else:
            print(diff_table)

    reporting.print_comparison_summary(metrics, tolerance_samples)

    n_samples = int(len(detected_signal))
    gt_signal = (
        ground_truth_signal
        if ground_truth_signal is not None
        else build_indicator_signal(n_samples, ground_truth_df)
    )
    det_signal = (
        detected_signal
        if detected_signal is not None
        else build_indicator_signal(n_samples, detected_df)
    )

    _print_indicator_diagnostics(gt_signal, det_signal)

    diagnostic_raw = reporting.build_diagnostic_raw(
        ground_truth_signal=gt_signal,
        detected_signal=det_signal,
        ground_truth_starts=ground_truth_df["start_blink"].to_numpy(),
        ground_truth_ends=ground_truth_df["end_blink"].to_numpy(),
        detected_starts=detected_df["start_blink"].to_numpy(),
        detected_ends=detected_df["end_blink"].to_numpy(),
        sampling_rate_hz=sampling_rate_hz,
        tolerance_samples=tolerance_samples,
        alignments=alignments,
    )

    return diagnostic_raw
