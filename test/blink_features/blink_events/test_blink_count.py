"""Unit tests for :mod:`blink_count` feature extraction.

Epochs are generated with ``slice_raw_into_mne_epochs_refine_annot`` and blink
counts are strictly compared against the ground-truth CSV
``ear_eog_blink_count_epoch.csv``. Rows 31 and 55 in the CSV are known
discrepancies and are excluded from comparisons. See
``tutorial/epoching_and_blink_validation_report.py`` for details.
"""

import logging
from pathlib import Path
import unittest

import mne
import numpy as np
import pandas as pd

from pyblinker.blink_features.blink_events.event_features.blink_count import (
    blink_count,
)
from pyblinker.utils.refine_util import slice_raw_into_mne_epochs_refine_annot
from test.blink_features.utils.helpers import assert_df_has_columns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class TestBlinkCount(unittest.TestCase):
    """Unit tests for blink counting from ``mne.Epochs`` metadata.

    Blink counts are validated against the CSV ground truth with rows 31 and 55
    excluded. See ``tutorial/epoching_and_blink_validation_report.py`` for
    context on these exceptions.
    """

    def setUp(self) -> None:
        """Load raw data and slice into epochs for blink counting."""
        logger.info("Setting up epochs for blink count tests...")
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
        # Load ground truth blink counts for cross-verification
        csv_path = (
            PROJECT_ROOT
            / "test"
            / "test_files"
            / "ear_eog_blink_count_epoch.csv"
        )
        self.assertTrue(
            csv_path.is_file(),
            f"Missing ground truth CSV at {csv_path}"
        )
        expected_full = (
            pd.read_csv(csv_path).set_index("epoch_id")["blink_count"].astype(float)
        )
        # Align ground truth with available epochs
        self.expected_counts = expected_full.loc[self.epochs.metadata.index]
        self.allowed_exception_rows = {31, 55}

        # metadata sanity checks
        self.assertIsInstance(self.epochs.metadata, pd.DataFrame)
        for col in ("blink_onset", "blink_duration"):
            self.assertIn(col, self.epochs.metadata.columns)

        logger.info("Epoch setup complete.")

    def test_counts(self) -> None:
        """Verify blink counts against CSV, ignoring rows 31 and 55."""
        df = blink_count(self.epochs)
        assert_df_has_columns(self, df, ["ep", "blink_count"])
        self.assertEqual(len(df), len(self.epochs))
        pd.testing.assert_series_equal(
            df["ep"], pd.Series(self.epochs.metadata.index, name="ep"), check_names=False
        )

        # Align and drop known mismatched rows
        expected = self.expected_counts.drop(
            self.allowed_exception_rows, errors="ignore"
        )
        computed = df["blink_count"].drop(
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
        self.assertTrue(np.issubdtype(df["blink_count"].dtype, np.number))
        for idx, expected_val in self.expected_counts.items():
            if idx in self.allowed_exception_rows:
                continue
            self.assertEqual(df.loc[idx, "blink_count"], expected_val)
            self.assertTrue(np.isfinite(df.loc[idx, "blink_count"]))

    def test_modality_specific_columns(self) -> None:
        """Blink counting with modality-specific metadata columns."""
        epochs = self.epochs.copy()
        epochs.metadata = epochs.metadata.drop(
            columns=["blink_onset_eeg", "blink_duration_eeg"], errors="ignore"
        ).rename(
            columns={"blink_onset": "blink_onset_eeg", "blink_duration": "blink_duration_eeg"}
        )
        df = blink_count(epochs)
        assert_df_has_columns(self, df, ["ep", "blink_count"])
        pd.testing.assert_series_equal(
            df["ep"], pd.Series(epochs.metadata.index, name="ep"), check_names=False
        )
        expected = self.expected_counts.drop(self.allowed_exception_rows, errors="ignore")
        computed = df["blink_count"].drop(self.allowed_exception_rows, errors="ignore")
        pd.testing.assert_series_equal(computed, expected, check_names=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
