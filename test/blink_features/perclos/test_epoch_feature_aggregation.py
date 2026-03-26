from __future__ import annotations

from pathlib import Path

import mne

from pyblinker.blink_features.aggregate_epoch_features import (
    compute_epoch_feature_families,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_compute_epoch_feature_families_includes_perclos_columns() -> None:
    raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw,
        epoch_len=30.0,
        blink_label=None,
        progress_bar=False,
        segmentation_type={
            "ear": {
                "channel": "EAR-avg_ear",
                "seg_type": "threshold_interpolation",
                "threshold": 0.13772966774319917,
                "annotation_time_unit": "seconds",
                "max_extension": 0.35,
                "extension_step": 0.05,
                "padding": 0.05,
                "extend_before": True,
                "extend_after": True,
            }
        },
    )

    df = compute_epoch_feature_families(
        epochs,
        picks=["EAR-avg_ear"],
        feature_families=("energy", "perclos"),
        progress_bar=False,
    )
    perclos_column = "ear__th_interpolation__perclos__EAR-AVG_EAR"
    fatigue_column = "ear__th_interpolation__fatigue_label__EAR-AVG_EAR"

    assert len(df) == len(epochs)
    assert perclos_column in df.columns
    assert fatigue_column in df.columns
    assert "epoch_start" not in df.columns
    assert "epoch_end" not in df.columns
    assert df[perclos_column].between(0.0, 1.0).all()
