"""Aggregation tests for waveform features.s"""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.waveform_features.aggregate import (
    compute_epoch_waveform_features,
)
from pyblinker.utils import slice_raw_into_mne_epochs

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestWaveformFeaturesAggregation(unittest.TestCase):
    """Verify aggregation behavior when joined with blink counts."""

    def setUp(self) -> None:
        """Load epochs with blink metadata."""
        raw_path = (
            PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def test_waveform_features_aggregation(self) -> None:
        """Join waveform features with blink counts and validate NaN policy."""
        feats = compute_epoch_waveform_features(self.epochs)
        csv_path = (
            PROJECT_ROOT
            / "unit_test"
            / "test_files"
            / "ear_eog_blink_count_epoch.csv"
        )
        counts = pd.read_csv(csv_path)
        index_col = "epoch_index" if "epoch_index" in counts.columns else "epoch_id"
        counts = counts.set_index(index_col)
        merged = feats.join(counts, how="left").rename(
            columns={"blink_count": "blink_count_epoch"}
        )
        cols = [
            "duration_base_mean",
            "duration_zero_mean",
            "neg_amp_vel_ratio_zero_mean",
        ]
        zero = merged["blink_count_epoch"] == 0
        self.assertTrue(merged.loc[zero, cols].isna().all(axis=None))
        positive = merged["blink_count_epoch"] > 0
        self.assertTrue(
            merged.loc[positive, cols]
            .apply(lambda r: np.isfinite(r).any(), axis=1)
            .all()
        )
        self.assertFalse(np.isinf(merged[cols].to_numpy()).any())


if __name__ == "__main__":
    unittest.main()
