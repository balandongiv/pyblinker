"""Tests for epoch-level waveform feature extraction.s"""
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
from unit_test.blink_features.utils.helpers import (
    assert_df_has_columns,
    assert_numeric_or_nan,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestEpochWaveformFeatures(unittest.TestCase):
    """Validate waveform feature calculations on :class:`mne.Epochs`."""

    def setUp(self) -> None:
        """Load test epochs with blink metadata."""
        raw_path = (
            PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def _has_blink(self, idx: int) -> bool:
        """Return ``True`` if the epoch contains at least one blink."""
        row = self.epochs.metadata.loc[idx]
        onset = row.get("blink_onset")
        duration = row.get("blink_duration")
        if onset is None or duration is None:
            return False
        if isinstance(onset, float) and np.isnan(onset):
            return False
        if isinstance(duration, float) and np.isnan(duration):
            return False
        if isinstance(onset, (list, tuple, np.ndarray, pd.Series)):
            return len(onset) > 0
        return True

    def test_schema_and_alignment(self) -> None:
        """Output DataFrame matches epoch index and schema."""
        df = compute_epoch_waveform_features(self.epochs)
        expected = [
            "duration_base_mean",
            "duration_zero_mean",
            "neg_amp_vel_ratio_zero_mean",
        ]
        assert_df_has_columns(self, df, expected)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.epochs))
        self.assertTrue(df.index.equals(self.epochs.metadata.index))

    def test_epoch_values_mixed_blinks(self) -> None:
        """Epochs with and without blinks yield appropriate values."""
        df = compute_epoch_waveform_features(self.epochs)
        cols = [
            "duration_base_mean",
            "duration_zero_mean",
            "neg_amp_vel_ratio_zero_mean",
        ]
        for ei in range(4):
            idx = self.epochs.metadata.index[ei]
            has_blink = self._has_blink(idx)
            values = df.loc[idx, cols]
            if not has_blink:
                self.assertTrue(values.isna().all())
            else:
                assert_numeric_or_nan(self, values)
                self.assertFalse(np.isinf(values.to_numpy()).any())

    def test_channel_picks(self) -> None:
        """Channel selection yields suffixed columns per channel."""
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left", "EAR-avg_ear"]
        df = compute_epoch_waveform_features(self.epochs, picks=picks)
        base_cols = [
            "duration_base_mean",
            "duration_zero_mean",
            "neg_amp_vel_ratio_zero_mean",
        ]
        assert_df_has_columns(self, df, base_cols)
        for ch in picks:
            suffixed = [f"{c}_{ch}" for c in base_cols]
            assert_df_has_columns(self, df, suffixed)
            for ei in range(4):
                idx = self.epochs.metadata.index[ei]
                has_blink = self._has_blink(idx)
                values = df.loc[idx, suffixed]
                if not has_blink:
                    self.assertTrue(values.isna().all())
                else:
                    assert_numeric_or_nan(self, values)
                    self.assertFalse(np.isinf(values.to_numpy()).any())

    def test_empty_epochs(self) -> None:
        """Empty input returns an empty DataFrame with proper columns."""
        df_full = compute_epoch_waveform_features(self.epochs)
        df_empty = compute_epoch_waveform_features(self.epochs[:0])
        self.assertEqual(len(df_empty), 0)
        self.assertListEqual(list(df_full.columns), list(df_empty.columns))

    def test_blink_metadata_robustness(self) -> None:
        """Missing blink metadata in blink-free epochs still yields NaNs."""
        epochs = self.epochs.copy()
        no_blink_idx = epochs.metadata.index[epochs.metadata["blink_onset"].isna()][0]
        epochs.metadata.loc[no_blink_idx, ["blink_onset", "blink_duration"]] = np.nan
        df = compute_epoch_waveform_features(epochs)
        cols = [
            "duration_base_mean",
            "duration_zero_mean",
            "neg_amp_vel_ratio_zero_mean",
        ]
        self.assertTrue(df.loc[no_blink_idx, cols].isna().all())


if __name__ == "__main__":
    unittest.main()
