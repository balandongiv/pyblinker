"""Scenario B: EEG-only kinematic pipeline coverage."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.kinematics.kinematic_features import (
    KinematicBlinkFeatureExtractor,
    _available_styles,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"

_REQUIRED_KINEMATIC_METRICS = (
    "amp_vel_ratio_base",
    "amp_vel_ratio_tent",
    "amp_vel_ratio_zero_to_max",
    "blink_velocity",
    "inter_blink_max_vel",
    "inter_blink_max_vel_base",
    "inter_blink_max_vel_zero",
    "aver_left_velocity",
    "aver_right_velocity",
    "neg_amp_vel_ratio_base",
    "pos_amp_vel_ratio_base",
    "neg_amp_vel_ratio_zero",
    "pos_amp_vel_ratio_zero",
    "neg_amp_vel_ratio_tent",
    "pos_amp_vel_ratio_tent",
)


class TestEegOnlyKinematicPipeline(unittest.TestCase):
    """EEG-only kinematic pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"

    def test_eeg_only_runs_without_ear_channel(self) -> None:
        """EEG-only config (with optional EOG) runs and yields EEG outputs."""

        raw = mne.io.read_raw_fif(self.raw_path, preload=True, verbose=False)

        segment_config = {
            "eeg": {"channel": EEG_CHANNEL, "seg_type": "base"},
            "eog": {"channel": EOG_CHANNEL, "seg_type": "base"},
        }

        epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segment_config,
        )

        extractor = KinematicBlinkFeatureExtractor(epochs=epochs)
        df = extractor.compute(picks=EEG_CHANNEL)

        self.assertNotIn("blink_onset_ear", epochs.metadata.columns)
        self.assertIn("blink_onset_eeg", epochs.metadata.columns)
        self.assertTrue(all(col.endswith(f"__{EEG_CHANNEL}") for col in df.columns))

        styles = _available_styles(tuple(epochs.metadata.columns), "eeg")
        self.assertTrue(styles)
        for style in styles:
            for metric in _REQUIRED_KINEMATIC_METRICS:
                for stat in ("mean", "std", "cv"):
                    expected = f"eeg__{style}__kinematic__{metric}_{stat}__{EEG_CHANNEL}"
                    self.assertIn(expected, df.columns)

        self.assertGreater(df.notna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
