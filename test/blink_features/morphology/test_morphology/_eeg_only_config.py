"""EEG-only morphology extraction tests."""
from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.morphology import MorphologyBlinkFeatureExtractor
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import DEFAULT_EEG_CHANNEL

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class TestMorphologyEegOnlyConfig(unittest.TestCase):
    """Validate EEG-only morphology outputs for duration columns."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        self.raw_path = raw_path

    def test_eeg_only_runs_(self) -> None:
        """EEG-only config runs and yields EEG outputs."""
        raw = mne.io.read_raw_fif(self.raw_path, preload=True, verbose=False)

        segment_config = {
            "eeg": {
                "channel": DEFAULT_EEG_CHANNEL,
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

        extractor = MorphologyBlinkFeatureExtractor(epochs=epochs)
        df = extractor.compute(picks=[DEFAULT_EEG_CHANNEL])

        self.assertIsNotNone(df)
        expected_cols = [
            f"eeg__base__morphology__duration_mean__{DEFAULT_EEG_CHANNEL}",
            f"eeg__zero__morphology__duration_mean__{DEFAULT_EEG_CHANNEL}",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
