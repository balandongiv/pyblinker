"""EOG-only kinematic pipeline coverage."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.kinematics.kinematic_features import (
    KinematicBlinkFeatureExtractor,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EOG_CHANNEL = "EOG-EEG-eog_vert_left"


class TestEogOnlyKinematicPipeline(unittest.TestCase):
    """EOG-only kinematic pipeline coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"

    def test_eog_only_runs_without_ear_or_eeg(self) -> None:
        """EOG-only config runs and returns EOG columns without other modalities."""

        raw = mne.io.read_raw_fif(self.raw_path, preload=True, verbose=False)

        segment_config = {
            "eog": {
                "channel": EOG_CHANNEL,
                "seg_type": "base",
            }
        }

        epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segment_config,
        )

        extractor = KinematicBlinkFeatureExtractor(epochs=epochs)
        df = extractor.compute(picks=EOG_CHANNEL)

        self.assertNotIn("blink_onset_ear", epochs.metadata.columns)
        self.assertNotIn("blink_onset_eeg", epochs.metadata.columns)
        self.assertIn("blink_onset_eog", epochs.metadata.columns)
        self.assertTrue(all(col.endswith(f"__{EOG_CHANNEL}") for col in df.columns))
        self.assertGreater(df.notna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
