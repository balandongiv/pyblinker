from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyblinker.blink_features.ear_metrics import (
    EARBlinkFeatureExtractor,
    EARFeatureConfig,
    EARRefinementConfig,
    EARThresholdBlinkRefiner,
    load_coarse_blinks,
    load_ear_channel,
    refine_annotations_for_thresholds,
)


@pytest.fixture(scope="module")
def ear_data() -> tuple[np.ndarray, float, pd.DataFrame]:
    data_dir = Path(__file__).resolve().parents[1] / "manual_annotation_feature_calculation_data"
    annotation_csv = data_dir / "ear_eog.csv"
    fif_path = data_dir / "ear_eog.fif"

    if not annotation_csv.exists() or not fif_path.exists():
        pytest.skip("EAR tutorial inputs are missing")

    signal, sfreq = load_ear_channel(fif_path, channel="EAR-avg_ear")
    annotations = load_coarse_blinks(annotation_csv)
    return signal, sfreq, annotations


def test_refinement_detects_crossings(ear_data: tuple[np.ndarray, float, pd.DataFrame]) -> None:
    signal, sfreq, annotations = ear_data
    config = EARRefinementConfig(
        threshold=0.23,
        annotation_time_unit="seconds",
        max_extension=0.4,
        extension_step=0.05,
        padding=0.05,
    )
    refiner = EARThresholdBlinkRefiner(signal, sfreq, config)
    refined = refiner.refine_annotations(annotations.head(5))

    assert not refined.empty
    assert refined["refinement_succeeded"].any()
    assert set(refined.columns).issuperset(
        {
            "refined_start_sample",
            "refined_end_sample",
            "onset_offset_seconds",
            "refined_left_zero",
            "refined_right_zero",
        }
    )
    assert (refined["refined_end_sample"] >= refined["refined_start_sample"]).all()


def test_refinement_falls_back_without_crossing(
    ear_data: tuple[np.ndarray, float, pd.DataFrame]
) -> None:
    signal, sfreq, annotations = ear_data
    # Set an unrealistically low threshold so no crossings are found.
    config = EARRefinementConfig(threshold=0.01, annotation_time_unit="seconds")
    refiner = EARThresholdBlinkRefiner(signal, sfreq, config)
    refined = refiner.refine_annotations(annotations.head(3))

    assert not refined["refinement_succeeded"].any()
    assert (refined["refined_start_sample"] == refined["coarse_start_sample"]).all()
    assert (refined["refined_end_sample"] == refined["coarse_end_sample"]).all()


def test_feature_extraction_outputs_expected_columns(
    ear_data: tuple[np.ndarray, float, pd.DataFrame]
) -> None:
    signal, sfreq, annotations = ear_data
    refinement = EARThresholdBlinkRefiner(
        signal,
        sfreq,
        EARRefinementConfig(threshold=0.23, annotation_time_unit="seconds"),
    ).refine_annotations(annotations.head(4))

    extractor = EARBlinkFeatureExtractor(
        signal,
        sfreq,
        threshold=0.23,
        feature_config=EARFeatureConfig(baseline_window=0.1, context_window=0.05),
    )
    features = extractor.build_feature_table(refinement)

    required = {
        "ear_min",
        "ear_baseline",
        "ear_blink_depth",
        "blink_classification",
        "max_closing_speed",
        "max_opening_speed",
        "closed_duration_seconds",
        "auc_below_threshold",
        "time_to_close",
        "time_to_reopen",
    }
    assert set(features.columns).issuperset(required)
    assert (features["closed_duration_seconds"] >= 0).all()
    assert (features["refined_duration"] >= 0).all()


def test_feature_extraction_handles_multiple_thresholds(
    ear_data: tuple[np.ndarray, float, pd.DataFrame]
) -> None:
    signal, sfreq, annotations = ear_data
    thresholds = [0.18, 0.2, 0.22, 0.24, 0.26]
    refinement = refine_annotations_for_thresholds(
        signal=signal,
        sfreq=sfreq,
        annotations=annotations.head(2),
        candidate_thresholds=thresholds,
        base_config=EARRefinementConfig(threshold=0.23, annotation_time_unit="seconds"),
    )
    extractor = EARBlinkFeatureExtractor(
        signal,
        sfreq,
        threshold=None,
        feature_config=EARFeatureConfig(baseline_window=0.1, context_window=0.05),
    )
    features = extractor.build_feature_table(refinement)

    assert (features["threshold_value"].isin(thresholds)).all()
    # One row per (blink, threshold)
    assert len(features) == len(thresholds) * 2

    # No nested dictionaries should be present.
    dict_in_columns = features.apply(lambda col: col.map(lambda x: isinstance(x, dict)).any()).any()
    assert not dict_in_columns
