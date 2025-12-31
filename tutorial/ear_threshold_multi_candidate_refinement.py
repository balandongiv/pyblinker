"""Tutorial: Evaluate multiple EAR thresholds and auto-select the plot threshold.

This example mirrors ``tutorial/ear_threshold_blink_refinement.py`` but demonstrates
evaluating several EAR thresholds per blink window. Threshold-dependent metrics are
stored per candidate, and the plotting threshold is auto-selected when none is
explicitly provided.
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
    EARThresholdBlinkRefiner,
    apply_flat_threshold_selection,
    load_coarse_blinks,
    load_ear_channel,
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
        threshold=0.23,  # still used for refining boundaries
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

    refiner = EARThresholdBlinkRefiner(ear_signal, sfreq, refinement_config)
    refined = refiner.refine_annotations(annotations)

    # Extract features for every candidate threshold; plotting will auto-select.
    extractor = EARBlinkFeatureExtractor(
        ear_signal,
        sfreq,
        threshold=candidate_thresholds,
        feature_config=feature_config,
        plot_threshold=None,
    )
    features = extractor.build_feature_table(refined)
    best_threshold = apply_flat_threshold_selection(features, extractor.threshold_store)

    if save_reports:
        output_path = output_dir / "ear_multi_threshold_refined_blinks.csv"
        features.to_csv(output_path, index=False)

    print("Example selected thresholds (first five rows):")
    print(features.loc[:4, ["candidate_id", "selected_threshold_value", "threshold_selection_mode"]])

    print("Available per-threshold metrics for the first blink:")
    threshold_cols = [c for c in features.columns if c.startswith("threshold_")]
    print(features.loc[0, threshold_cols].to_frame().T)

    # Build visual reports:
    # 1) User-specified plot threshold.
    # 2) Auto-selected threshold with rationale surfaced in the HTML summary.
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    eeg_overlay = raw.get_data(picks="EEG-E8")[0]
    overlay_sfreq = float(raw.info["sfreq"])

    report_df = features.copy()
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

    user_plot_threshold = 0.22
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

        auto_report_path = output_dir / "ear_multi_threshold_refined_blink_report_auto.html"
        build_refined_blink_report(
            results=report_df,
            signal=ear_signal,
            sfreq=sfreq,
            channel_name="EAR-avg_ear",
            plot_overlay=True,
            plot_signal_as_scatter=True,
            mark_threshold_crossings=True,
            threshold_value=best_threshold,
            overlay_signal=eeg_overlay,
            overlay_sfreq=overlay_sfreq,
            overlay_label="EEG-E8",
            output_path=auto_report_path,
        )

    n_success = int(features["refinement_succeeded"].sum())
    print(f"Refined {len(features)} blinks; {n_success} used threshold crossings.")
    print(f"Representative threshold used for plotting: {best_threshold}")
    print("Average onset shift (s):", features["onset_offset_seconds"].mean())
    print("Average offset shift (s):", features["offset_offset_seconds"].mean())

    preview_cols = [
        "candidate_id",
        "blink_type",
        "refined_onset_time",
        "refined_offset_time",
        "refined_duration",
        "refinement_succeeded",
        "ear_min",
        "ear_blink_depth",
        "selected_threshold_value",
        "threshold_selection_mode",
        "closed_duration_seconds",
        "auc_below_threshold",
        "blink_classification",
    ]
    print("\nExample rows with multi-threshold EAR features:")
    print(features.loc[:, preview_cols].head())

    if save_reports:
        print("\nSaved refined blink table to:", output_path)
        print("Blink validation report with user threshold saved to:", user_report_path)
        print("Blink validation report with auto-selected threshold saved to:", auto_report_path)
    else:
        print("\nSaving is disabled (set PYBLINKER_SAVE_REPORTS=1 to enable).")
    print(
        "You can adjust the candidate thresholds to compare how crossings and derived\n"
        "metrics change without rerunning annotation refinement."
    )


if __name__ == "__main__":
    main()
