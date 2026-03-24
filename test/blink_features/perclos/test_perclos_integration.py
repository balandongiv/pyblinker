from __future__ import annotations

from pathlib import Path
import unittest
import mne


from pyblinker.blink_features.perclos import (
    compute_perclos_features,
    # resolve_subject_specific_ear_threshold,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_compute_perclos_features_from_refined_ear_epochs() -> None:
    raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    threshold = 0.13772966774319917 # The value is extracted from experiement. If the EAR value is 70% (baseline from EC and EO)
    segmentation_config = build_segment_config(
        raw,
        eeg_channel=None,
        eog_channel=None,
        base_config={
            "ear": {
                "seg_type": "threshold_interpolation",
                "threshold": threshold,
                "annotation_time_unit": "seconds",
                "max_extension": 0.35,
                "extension_step": 0.05,
                "padding": 0.05,
                "extend_before": True,
                "extend_after": True,
            }
        },
    )

    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw,
        epoch_len=30.0,
        blink_label=None,
        progress_bar=False,
        segmentation_type=segmentation_config,
    )
    df = compute_perclos_features(epochs)

    assert len(df) == len(epochs)
    assert df["perclos"].between(0.0, 1.0).all()
    assert set(df["fatigue_label"].unique()).issubset({0, 1})
    assert df["epoch_start"].iloc[0] == 0.0


if __name__ == "__main__":
    unittest.main()