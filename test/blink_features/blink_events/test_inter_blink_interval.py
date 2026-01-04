"""Tests for channel-aware inter-blink interval (IBI) features.

Epochs are constructed using ``slice_raw_into_mne_epochs_refine_annot`` to
match the canonical pipeline. This test does not rely on any CSV ground truth
files.
"""
from __future__ import annotations

import unittest
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.blink_events.event_features.inter_blink_interval import (
    inter_blink_interval_epochs,
)
from pyblinker.blink_features.blink_events.event_features.blink_count import blink_count
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config
from test.blink_features.utils.helpers import assert_df_has_columns

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestInterBlinkInterval(unittest.TestCase):
    """Validate IBI computation using epoch metadata."""

    def setUp(self) -> None:
        self.raw_path = (
            PROJECT_ROOT
            / "test"
            / "test_files"
            / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(self.raw_path, preload=True, verbose=False)
        segmentation_config = build_segment_config(raw)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )

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

    def test_channel_ibi(self) -> None:
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left", "EAR-avg_ear"]
        df = inter_blink_interval_epochs(self.epochs, picks=picks)
        expected_cols = ["ep"] + [f"ibi_{p}" for p in picks]
        assert_df_has_columns(self, df, expected_cols)
        self.assertEqual(len(df), len(self.epochs))
        pd.testing.assert_series_equal(
            df["ep"], pd.Series(self.epochs.metadata.index, name="ep"), check_names=False
        )

        # epoch-wise checks across all epochs using blink counts
        counts = blink_count(self.epochs)["blink_count"]
        for idx in range(len(self.epochs)):
            for p in picks:
                val = df.loc[idx, f"ibi_{p}"]
                if counts.loc[idx] >= 2:
                    self.assertTrue(np.isfinite(val))
                else:
                    self.assertTrue(np.isnan(val))

        # explicit checks for first four epochs
        self.assertTrue(np.isfinite(df.loc[0, "ibi_EEG-E8"]))
        for idx in [1, 2, 3]:
            self.assertTrue(np.isnan(df.loc[idx, "ibi_EEG-E8"]))
        for col in [f"ibi_{p}" for p in picks]:
            self.assertTrue(np.issubdtype(df[col].dtype, np.number))

    def test_missing_channel(self) -> None:
        with self.assertRaises(ValueError):
            inter_blink_interval_epochs(self.epochs, picks=["BAD-CHAN"])

    def test_empty_epochs(self) -> None:
        empty = self.epochs[:0]
        df = inter_blink_interval_epochs(empty, picks="EEG-E8")
        assert_df_has_columns(self, df, ["ep", "ibi_EEG-E8"])
        self.assertEqual(len(df), 0)

    def test_generic_columns_fallback(self) -> None:
        """IBI computation falls back to generic blink columns."""
        epochs = self.epochs.copy()
        df = inter_blink_interval_epochs(epochs)
        assert_df_has_columns(self, df, ["ep", "ibi"])
        self.assertEqual(len(df), len(epochs))
        counts = blink_count(epochs)["blink_count"]
        for idx in range(len(epochs)):
            val = df.loc[idx, "ibi"]
            if counts.loc[idx] >= 2:
                self.assertTrue(np.isfinite(val))
            else:
                self.assertTrue(np.isnan(val))

    def test_ear_only_configuration(self) -> None:
        """EAR-only segmentation still yields valid IBI columns."""
        epochs = self._make_epochs(include_eeg=False, include_eog=False)
        df = inter_blink_interval_epochs(epochs, picks=["EAR-avg_ear"])
        assert_df_has_columns(self, df, ["ep", "ibi_EAR-avg_ear"])
        self.assertEqual(len(df), len(epochs))

    def test_eeg_only_missing_ear_key(self) -> None:
        """IBI computation works when EAR config is omitted."""
        epochs = self._make_epochs(include_ear=False, include_eeg=True, include_eog=False, require_ear=False)
        df = inter_blink_interval_epochs(epochs, picks=["EEG-E8"])
        assert_df_has_columns(self, df, ["ep", "ibi_EEG-E8"])
        self.assertEqual(len(df), len(epochs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
