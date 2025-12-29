"""Tutorial: EAR-threshold blink refinement anchored by CSV annotations.

What this tutorial demonstrates
-------------------------------
1. Load coarse blink annotations (onset, duration, blink_type) from CSV.
2. Load the EAR channel from a FIF recording.
3. Use the CSV to define *where* a blink happens and the EAR threshold to define
   *when* it truly starts and ends via downward/upward threshold crossings.
4. Progressively extend the search window when crossings are outside the coarse
   window (up to a configurable maximum).
5. Compute rich EAR-based blink features using the refined timing.

Key takeaways (mirroring the problem statement)
-----------------------------------------------
- CSV annotations provide existence + approximate timing; they are not forced to
  align with threshold crossings.
- Threshold crossings can legitimately fall outside the coarse window, so the
  algorithm deterministically expands the search region when needed.
- Refined onset/offset reshape every derived metric (duration, depth, slopes,
  time-under-threshold, etc.).
- Tunable knobs: EAR threshold, maximum extension, extension step, padding, and
  feature parameters (baseline window, classification threshold).
"""

from __future__ import annotations

from pathlib import Path
import sys

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
    project_root = PROJECT_ROOT
    data_dir = project_root / "manual_annotation_feature_calculation_data"
    output_dir = project_root / "tutorial_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_csv = data_dir / "ear_eog.csv"
    fif_path = data_dir / "ear_eog.fif"

    # User-tunable parameters
    ear_threshold = 0.23
    refinement_config = EARRefinementConfig(
        threshold=ear_threshold,
        annotation_time_unit="seconds",
        max_extension=0.5,  # seconds allowed outside the coarse window
        extension_step=0.05,  # grow the search window in 50 ms steps
        padding=0.05,  # include a fixed buffer around each coarse window
        extend_before=True,
        extend_after=True,
    )
    feature_config = EARFeatureConfig(
        baseline_window=0.25,  # seconds before refined onset used for baseline
        classification_threshold=ear_threshold,  # partial vs full classification
        context_window=0.1,  # optional stats window around the blink
    )

    print("Loading coarse blink annotations from:", annotation_csv)
    annotations = load_coarse_blinks(annotation_csv)
    print(f"{len(annotations)} coarse blinks loaded with columns: {list(annotations.columns)}")

    print("Loading EAR channel from FIF:", fif_path)
    ear_signal, sfreq = load_ear_channel(fif_path, channel="EAR-avg_ear")
    print(f"Sampling rate: {sfreq} Hz; signal length: {len(ear_signal)} samples")

    refiner = EARThresholdBlinkRefiner(ear_signal, sfreq, refinement_config)
    refined = refiner.refine_annotations(annotations)

    extractor = EARBlinkFeatureExtractor(
        ear_signal, sfreq, threshold=ear_threshold, feature_config=feature_config
    )
    features = extractor.build_feature_table(refined)
    features, best_threshold = apply_flat_threshold_selection(
        features, extractor.threshold_store
    )

    output_path = output_dir / "ear_threshold_refined_blinks.csv"
    features.to_csv(output_path, index=False)

    # Build visual report with EEG overlay by default to mirror the CSV + threshold story
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    eeg_overlay = raw.get_data(picks="EEG-E8")[0]
    overlay_sfreq = float(raw.info["sfreq"])

    report_df = features.copy()
    left_time = pd.to_numeric(report_df["ear_threshold_left_time"], errors="coerce")
    right_time = pd.to_numeric(report_df["ear_threshold_right_time"], errors="coerce")
    min_time = pd.to_numeric(report_df["ear_threshold_min_time"], errors="coerce")

    report_df["refined_left_zero"] = (left_time * sfreq).round()
    report_df["refined_right_zero"] = (right_time * sfreq).round()
    report_df["refined_min_zero"] = (min_time * sfreq).round()

    missing_left = report_df["refined_left_zero"].isna()
    missing_right = report_df["refined_right_zero"].isna()
    missing_min = report_df["refined_min_zero"].isna()

    report_df.loc[missing_left, "refined_left_zero"] = report_df.loc[
        missing_left, "refined_start_sample"
    ]
    report_df.loc[missing_right, "refined_right_zero"] = report_df.loc[
        missing_right, "refined_end_sample"
    ]
    report_df.loc[missing_min, "refined_min_zero"] = report_df.loc[
        missing_min, "refined_start_sample"
    ]

    report_df["refined_left_zero"] = report_df["refined_left_zero"].astype(int)
    report_df["refined_right_zero"] = report_df["refined_right_zero"].astype(int)
    report_df["refined_min_zero"] = report_df["refined_min_zero"].astype(int)
    report_df["zero_crossing_found"] = report_df["ear_threshold_status"].eq("ok")

    report_path = output_dir / "ear_threshold_refined_blink_report.html"
    build_refined_blink_report(
        results=report_df,
        signal=ear_signal,
        sfreq=sfreq,
        channel_name="EAR-avg_ear",
        plot_overlay=True,
        plot_signal_as_scatter=True,
        mark_threshold_crossings=True,
        threshold_value=ear_threshold,
        overlay_signal=eeg_overlay,
        overlay_sfreq=overlay_sfreq,
        overlay_label="EEG-E8",
        output_path=report_path,
    )

    n_success = int(features["refinement_succeeded"].sum())
    print(f"Refined {len(features)} blinks; {n_success} used threshold crossings.")
    print("Average onset shift (s):", features["onset_offset_seconds"].mean())
    print("Average offset shift (s):", features["offset_offset_seconds"].mean())

    preview_cols = [
        "candidate_id",
        "blink_type",
        "coarse_onset_time",
        "coarse_offset_time",
        "refined_onset_time",
        "refined_offset_time",
        "refined_duration",
        "refinement_used_outward_extension",
        "refinement_succeeded",
        "ear_min",
        "ear_baseline",
        "ear_blink_depth",
        "max_closing_speed",
        "max_opening_speed",
        "closed_duration_seconds",
        "auc_below_threshold",
        "blink_classification",
    ]
    print("\nExample rows with refined timing and EAR features:")
    print(features.loc[:, preview_cols].head())

    print("\nSaved refined blink table to:", output_path)
    print("Blink validation report saved to:", report_path)
    print(
        "You can tune `ear_threshold`, `max_extension`, `extension_step`, and "
        "`baseline_window` to suit different sensors or annotation granularity."
    )


if __name__ == "__main__":
    main()
