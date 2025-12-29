"""Tutorial: Evaluate multiple EAR thresholds end-to-end.

This example mirrors ``tutorial/ear_threshold_blink_refinement.py`` but demonstrates
evaluating several EAR thresholds by *refining and extracting features per threshold*.
Each refined blink row is tied to a single threshold so threshold-dependent metrics
appear once per row without any internal per-blink threshold loops.
"""

from __future__ import annotations

from pathlib import Path
import sys
import os

import mne
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyblinker.blink_features.ear_metrics import (  # noqa: E402
    EARBlinkFeatureExtractor,
    EARFeatureConfig,
    EARRefinementConfig,
    load_coarse_blinks,
    load_ear_channel,
    refine_annotations_for_thresholds,
)
from pyblinker.outside_annotation import build_refined_blink_report  # noqa: E402


def main() -> None:
    save_reports = os.environ.get("PYBLINKER_SAVE_REPORTS", "1") != "0"
    data_dir = PROJECT_ROOT / "manual_annotation_feature_calculation_data"
    output_dir = PROJECT_ROOT / "tutorial_outputs"
    if save_reports:
        output_dir.mkdir(parents=True, exist_ok=True)

    annotation_csv = data_dir / "ear_eog.csv"
    fif_path = data_dir / "ear_eog.fif"

    # Evaluate several EAR thresholds per blink.
    candidate_thresholds = [0.18, 0.2, 0.22, 0.24, 0.26]

    refinement_config = EARRefinementConfig(
        threshold=0.23,  # baseline; replaced per candidate during refinement
        annotation_time_unit="seconds",
        max_extension=0.5,
        extension_step=0.05,
        padding=0.05,
        extend_before=True,
        extend_after=True,
    )
    feature_config = EARFeatureConfig(
        baseline_window=0.25,
        classification_threshold=None,  # classification uses each threshold value
        context_window=0.1,
    )

    print("Loading coarse blink annotations from:", annotation_csv)
    annotations = load_coarse_blinks(annotation_csv)
    print(f"{len(annotations)} coarse blinks loaded.")

    print("Loading EAR channel from FIF:", fif_path)
    ear_signal, sfreq = load_ear_channel(fif_path, channel="EAR-avg_ear")
    print(f"Sampling rate: {sfreq} Hz; signal length: {len(ear_signal)} samples")

    refined = refine_annotations_for_thresholds(
        signal=ear_signal,
        sfreq=sfreq,
        annotations=annotations,
        candidate_thresholds=candidate_thresholds,
        base_config=refinement_config,
    )

    # Extract features for every threshold-specific refined row.
    extractor = EARBlinkFeatureExtractor(
        ear_signal,
        sfreq,
        threshold=None,  # threshold is supplied per-row via refinement
        feature_config=feature_config,
    )
    features = extractor.build_feature_table(refined)

    if save_reports:
        output_path = output_dir / "ear_multi_threshold_refined_blinks.csv"
        features.to_csv(output_path, index=False)

    print("Example refined rows (first five across thresholds):")
    print(features.loc[:4, ["candidate_id", "threshold_value", "refined_onset_time", "refined_offset_time"]])

    # Build visual reports:
    # 1) User-specified plot threshold.
    # 2) Auto-selected threshold with rationale surfaced in the HTML summary.
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    eeg_overlay = raw.get_data(picks="EEG-E8")[0]
    overlay_sfreq = float(raw.info["sfreq"])

    # Use a single threshold slice for reporting/visualization.
    report_threshold = candidate_thresholds[2]
    report_df = features.loc[features["threshold_value"] == report_threshold].copy()
    left_time = pd.to_numeric(report_df["ear_threshold_left_time"], errors="coerce")
    right_time = pd.to_numeric(report_df["ear_threshold_right_time"], errors="coerce")
    min_time = pd.to_numeric(report_df["ear_threshold_min_time"], errors="coerce")

    report_df["ear_threshold_left_sample"] = (left_time * sfreq).round()
    report_df["ear_threshold_right_sample"] = (right_time * sfreq).round()
    report_df["ear_threshold_min_sample"] = (min_time * sfreq).round()

    missing_left = report_df["refined_left_zero"].isna()
    missing_right = report_df["refined_right_zero"].isna()
    missing_min = report_df["ear_threshold_min_sample"].isna()

    report_df.loc[missing_left, "ear_threshold_left_sample"] = report_df.loc[
        missing_left, "refined_start_sample"
    ]
    report_df.loc[missing_right, "ear_threshold_right_sample"] = report_df.loc[
        missing_right, "refined_end_sample"
    ]
    report_df.loc[missing_min, "ear_threshold_min_sample"] = report_df.loc[
        missing_min, "refined_start_sample"
    ]

    report_df["ear_threshold_left_sample"] = (
        report_df["ear_threshold_left_sample"].fillna(report_df["refined_start_sample"]).astype(int)
    )
    report_df["ear_threshold_right_sample"] = (
        report_df["ear_threshold_right_sample"].fillna(report_df["refined_end_sample"]).astype(int)
    )
    report_df["ear_threshold_min_sample"] = (
        report_df["ear_threshold_min_sample"].fillna(report_df["refined_start_sample"]).astype(int)
    )
    report_df["zero_crossing_found"] = report_df["ear_threshold_status"].eq("ok")

    user_plot_threshold = report_threshold
    if save_reports:
        user_report_path = output_dir / "ear_multi_threshold_refined_blink_report_user.html"
        build_refined_blink_report(
            results=report_df,
            signal=ear_signal,
            sfreq=sfreq,
            channel_name="EAR-avg_ear",
            plot_overlay=True,
            plot_signal_as_scatter=True,
            mark_threshold_crossings=True,
            threshold_value=user_plot_threshold,
            overlay_signal=eeg_overlay,
            overlay_sfreq=overlay_sfreq,
            overlay_label="EEG-E8",
            output_path=user_report_path,
        )

    n_success = int(features["refinement_succeeded"].sum())
    print(f"Refined {len(features)} blink-threshold pairs; {n_success} used threshold crossings.")
    print(f"Report threshold used for visualization: {report_threshold}")
    print("Average onset shift (s):", features["onset_offset_seconds"].mean())
    print("Average offset shift (s):", features["offset_offset_seconds"].mean())

    preview_cols = [
        "candidate_id",
        "threshold_value",
        "blink_type",
        "refined_onset_time",
        "refined_offset_time",
        "refined_duration",
        "refinement_succeeded",
        "ear_min",
        "ear_blink_depth",
        "closed_duration_seconds",
        "auc_below_threshold",
        "blink_classification",
    ]
    print("\nExample rows with multi-threshold EAR features:")
    print(features.loc[:, preview_cols].head())

    if save_reports:
        print("\nSaved refined blink table to:", output_path)
        print("Blink validation report with user threshold saved to:", user_report_path)
    else:
        print("\nSaving is disabled (set PYBLINKER_SAVE_REPORTS=1 to enable).")
    print(
        "You can adjust the candidate thresholds to compare how crossings and derived\n"
        "metrics change without rerunning annotation refinement."
    )


if __name__ == "__main__":
    main()
