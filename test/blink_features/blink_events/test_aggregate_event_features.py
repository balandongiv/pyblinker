"""Tests for aggregating blink event features with refined annotations.

Epochs are produced by ``slice_raw_into_mne_epochs_refine_annot`` and blink
totals are validated against ``ear_eog_blink_count_epoch.csv``. Rows 31 and 55
are known mismatches and are excluded from comparisons; see
``tutorial/epoching_and_blink_validation_report.py`` for background.
"""
from __future__ import annotations

import unittest
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.blink_events.event_features import (
    aggregate_blink_event_features,
)
from pyblinker.utils.refine_util import slice_raw_into_mne_epochs_refine_annot
from test.blink_features.utils.helpers import assert_df_has_columns

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestAggregateBlinkFeatures(unittest.TestCase):
    """Validate aggregation of blink features from epochs.

    Blink totals are checked against the ground-truth CSV with rows 31 and 55
    excluded. See ``tutorial/epoching_and_blink_validation_report.py`` for
    context on these exceptions.
    """

    def setUp(self) -> None:
        raw_path = (
            PROJECT_ROOT
            / "test"
            / "test_files"
            / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )
        csv_path = (
            PROJECT_ROOT / "test" / "test_files" / "ear_eog_blink_count_epoch.csv"
        )
        self.assertTrue(
            csv_path.is_file(),
            f"Missing ground truth CSV at {csv_path}"
        )
        expected_full = (
            pd.read_csv(csv_path).set_index("epoch_id")["blink_count"].astype(float)
        )
        self.expected_counts = expected_full.loc[self.epochs.metadata.index]
        self.allowed_exception_rows = {31, 55}

        # metadata sanity checks
        self.assertIsInstance(self.epochs.metadata, pd.DataFrame)
        for col in ("blink_onset", "blink_duration"):
            self.assertIn(col, self.epochs.metadata.columns)

        self.epoch_len = (
            self.epochs.tmax - self.epochs.tmin + 1.0 / self.epochs.info["sfreq"]
        )

    def test_aggregate_all_features(self) -> None:
        """Compare aggregated features to CSV, ignoring rows 31 and 55."""
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left", "EAR-avg_ear"]
        df = aggregate_blink_event_features(self.epochs, picks=picks)
        expected_cols = ["ep", "blink_total", "blink_rate"] + [f"ibi_{p}" for p in picks]
        assert_df_has_columns(self, df, expected_cols)
        self.assertEqual(len(df), len(self.epochs))
        pd.testing.assert_series_equal(
            df["ep"], pd.Series(self.epochs.metadata.index, name="ep"), check_names=False
        )

        # Compare blink totals against CSV, excluding mismatched rows
        expected = self.expected_counts.drop(
            self.allowed_exception_rows, errors="ignore"
        )
        computed = df["blink_total"].drop(
            self.allowed_exception_rows, errors="ignore"
        )
        self.assertEqual(
            len(expected),
            len(computed),
            "Length mismatch after dropping rows 31 and 55; see "
            "tutorial/epoching_and_blink_validation_report.py.",
        )
        self.assertTrue(
            expected.index.equals(computed.index),
            "Index mismatch after dropping rows 31 and 55; see "
            "tutorial/epoching_and_blink_validation_report.py.",
        )
        pd.testing.assert_series_equal(computed, expected, check_names=False)

        for idx in range(4):
            expected_rate = self.expected_counts.iloc[idx] / self.epoch_len * 60.0
            self.assertAlmostEqual(df.loc[idx, "blink_rate"], expected_rate)

        for col in expected_cols:
            self.assertTrue(np.issubdtype(df[col].dtype, np.number))
        for ch in picks:
            vals = df[f"ibi_{ch}"].iloc[:4]
            self.assertTrue(vals.apply(lambda v: np.isfinite(v) or np.isnan(v)).all())

    def test_missing_channel(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_blink_event_features(self.epochs, picks=["BAD-CHAN"])

    def test_select_subset(self) -> None:
        df = aggregate_blink_event_features(
            self.epochs, picks=["EEG-E8"], features=["blink_total"]
        )
        assert_df_has_columns(self, df, ["blink_total"])
        self.assertEqual(list(df.columns), ["blink_total"])
        self.assertEqual(len(df), len(self.epochs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

