"""Tests for aggregated open-eye baseline features."""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.open_eye_baseline import compute_open_eye_baseline_features
from test.segment_config import build_segment_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestOpenEyeBaselineFeatures(unittest.TestCase):
    """Validate aggregated baseline metrics over blink-free epochs."""

    def setUp(self) -> None:  # noqa: D401
        self.raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(self.raw_path, preload=True, verbose=False)
        segmentation_config = build_segment_config(raw)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )
        self.ear_channel = "EAR-avg_ear"
        self.eeg_channel = "EEG-E8"
        self.eog_channel = "EOG-EEG-eog_vert_left"

    def _make_epochs(
        self,
        *,
        include_ear: bool = True,
        include_eeg: bool = True,
        include_eog: bool = True,
        require_ear: bool | None = None,
    ) -> mne.Epochs:
        raw = mne.io.read_raw_fif(self.raw_path, preload=True, verbose=False)
        segmentation_config = build_segment_config(
            raw,
            include_ear=include_ear,
            include_eeg=include_eeg,
            include_eog=include_eog,
            require_ear=True if require_ear is None else require_ear,
        )
        return slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )

    def test_aggregated_baseline_features(self) -> None:
        """Baseline features averaged across selected blink-free epochs."""
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left", "EAR-avg_ear"]
        baseline_idx = (
            self.epochs.metadata.index[self.epochs.metadata["n_blinks"] == 0][:4]
            .tolist()
        )

        for idx in baseline_idx:
            self.assertEqual(self.epochs.metadata.loc[idx, "n_blinks"], 0)

        df = compute_open_eye_baseline_features(self.epochs, picks, baseline_idx)

        expected_cols = [
            "baseline_mean",
            "baseline_drift",
            "baseline_std",
            "baseline_mad",
            "perclos",
            "eye_opening_rms",
            "micropause_count",
            "zero_crossing_rate",
        ]
        self.assertEqual(len(df), len(picks))
        self.assertListEqual(list(df.index), picks)
        for col in expected_cols:
            self.assertIn(col, df.columns)
        self.assertTrue(np.isfinite(df.to_numpy()).all())
        # verify not all channels yield identical features
        self.assertTrue(df.nunique(axis=0).gt(1).any())

    def test_ear_only_configuration(self) -> None:
        """Open-eye baseline runs when only EAR modality is configured."""
        epochs = self._make_epochs(include_eeg=False, include_eog=False)
        baseline_idx = epochs.metadata.index[epochs.metadata["n_blinks"] == 0][:2].tolist()
        df = compute_open_eye_baseline_features(epochs, [self.ear_channel], baseline_idx)
        self.assertEqual(len(df), 1)
        self.assertIn(self.ear_channel, df.index)

    def test_eeg_only_missing_ear_key(self) -> None:
        """Baseline computation works when EAR key is absent from SEGMENT_CONFIG."""
        epochs = self._make_epochs(include_ear=False, include_eeg=True, include_eog=False, require_ear=False)
        baseline_idx = epochs.metadata.index[epochs.metadata["n_blinks"] == 0][:2].tolist()
        df = compute_open_eye_baseline_features(epochs, [self.eeg_channel], baseline_idx)
        self.assertEqual(len(df), 1)
        self.assertIn(self.eeg_channel, df.index)


if __name__ == "__main__":
    unittest.main()
