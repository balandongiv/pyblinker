"""Tests for aggregated open-eye baseline features."""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np

from refine_annotation.util import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.open_eye_baseline import compute_open_eye_baseline_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestOpenEyeBaselineFeatures(unittest.TestCase):
    """Validate aggregated baseline metrics over blink-free epochs."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
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


if __name__ == "__main__":
    unittest.main()
