"""Unit tests for :mod:`blink_count` feature extraction.

Epochs are generated with ``slice_raw_into_mne_epochs_refine_annot`` and blink
counts are strictly compared against the ground-truth CSV
``ear_eog_blink_count_epoch.csv``. Rows 31 and 55 in the CSV are known
discrepancies and are excluded from comparisons. See
``tutorial/04_epoching_and_blink_validation_report.py`` for details.
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
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config
from test.blink_features.utils.helpers import assert_df_has_columns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class TestBlinkCount(unittest.TestCase):
    """Unit tests for blink counting from ``mne.Epochs`` metadata.

    Blink counts are validated against the CSV ground truth with rows 31 and 55
    excluded. See ``tutorial/04_epoching_and_blink_validation_report.py`` for
    context on these exceptions.
    """

    def setUp(self) -> None:
        """Load raw data and slice into epochs for blink counting."""
        logger.info("Setting up epochs for blink count tests...")
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

    def test_counts(self) -> None:
        """Verify blink counts against CSV, ignoring rows 31 and 55."""
        df = blink_count(self.epochs, picks="EEG-E8")
        assert_df_has_columns(self, df, ["ep", "blink_count_eeg"])
        self.assertEqual(len(df), len(self.epochs))
        pd.testing.assert_series_equal(
            df["ep"], pd.Series(self.epochs.metadata.index, name="ep"), check_names=False
        )

        expected = self.expected_counts.drop(
            self.allowed_exception_rows, errors="ignore"
        )
        computed = df["blink_count_eeg"].drop(
            self.allowed_exception_rows, errors="ignore"
        )
        self.assertEqual(
            len(expected),
            len(computed),
            "Length mismatch after dropping rows 31 and 55; see "
            "tutorial/04_epoching_and_blink_validation_report.py.",
        )
        self.assertTrue(
            expected.index.equals(computed.index),
            "Index mismatch after dropping rows 31 and 55; see "
            "tutorial/04_epoching_and_blink_validation_report.py.",
        )
        pd.testing.assert_series_equal(computed, expected, check_names=False)
        self.assertTrue(np.issubdtype(df["blink_count_eeg"].dtype, np.number))
        for idx, expected_val in self.expected_counts.items():
            if idx in self.allowed_exception_rows:
                continue
            self.assertEqual(df.loc[idx, "blink_count_eeg"], expected_val)
            self.assertTrue(np.isfinite(df.loc[idx, "blink_count_eeg"]))

    def test_modality_specific_columns(self) -> None:
        """Blink counting with modality-specific metadata columns."""
        epochs = self.epochs.copy()
        epochs.metadata = epochs.metadata.drop(
            columns=["blink_onset_eeg", "blink_duration_eeg"], errors="ignore"
        ).rename(
            columns={"blink_onset": "blink_onset_eeg", "blink_duration": "blink_duration_eeg"}
        )
        df = blink_count(epochs, picks="EEG-E8")
        assert_df_has_columns(self, df, ["ep", "blink_count_eeg"])
        pd.testing.assert_series_equal(
            df["ep"], pd.Series(epochs.metadata.index, name="ep"), check_names=False
        )
        expected = self.expected_counts.drop(self.allowed_exception_rows, errors="ignore")
        computed = df["blink_count_eeg"].drop(self.allowed_exception_rows, errors="ignore")
        pd.testing.assert_series_equal(computed, expected, check_names=False)

    def test_generic_columns_without_picks(self) -> None:
        """Use generic blink metadata when no channel picks are provided."""
        df = blink_count(self.epochs)
        assert_df_has_columns(self, df, ["ep", "blink_count"])
        expected = self.expected_counts.drop(self.allowed_exception_rows, errors="ignore")
        computed = df["blink_count"].drop(self.allowed_exception_rows, errors="ignore")
        pd.testing.assert_series_equal(computed, expected, check_names=False)

    def test_multi_pick_modality_lookup(self) -> None:
        """Return separate blink counts for each modality when multiple picks."""
        epochs = self.epochs[:2].copy()
        epochs.metadata["blink_onset"] = pd.Series([[0.1], [0.2, 0.4]])
        epochs.metadata["blink_duration"] = pd.Series([[0.1], [0.1, 0.1]])
        epochs.metadata["blink_onset_eog"] = pd.Series([[0.1, 0.3], [0.2]])
        epochs.metadata["blink_duration_eog"] = pd.Series([[0.1, 0.1], [0.1]])
        epochs.metadata = epochs.metadata.drop(
            columns=["blink_onset_eeg", "blink_duration_eeg"], errors="ignore"
        )
        df = blink_count(epochs, picks=["EEG-E8", "EOG-EEG-eog_vert_left"])
        assert_df_has_columns(self, df, ["ep", "blink_count_eeg", "blink_count_eog"])
        self.assertListEqual(df["blink_count_eeg"].tolist(), [1.0, 2.0])
        self.assertListEqual(df["blink_count_eog"].tolist(), [2.0, 1.0])

    def test_ear_only_configuration(self) -> None:
        """EAR-only segmentation still matches expected counts."""
        epochs = self._make_epochs(include_eeg=False, include_eog=False)
        df = blink_count(epochs, picks="EAR-avg_ear")
        assert_df_has_columns(self, df, ["ep", "blink_count_ear"])
        self.assertEqual(len(df), len(epochs))
        expected = self.expected_counts.drop(self.allowed_exception_rows, errors="ignore")
        computed = df["blink_count_ear"].drop(self.allowed_exception_rows, errors="ignore")
        pd.testing.assert_series_equal(computed, expected, check_names=False)

    def test_eeg_only_missing_ear_key(self) -> None:
        """EEG-only segmentation omits EAR modality without errors."""
        epochs = self._make_epochs(include_ear=False, include_eeg=True, include_eog=False, require_ear=False)
        df = blink_count(epochs, picks="EEG-E8")
        assert_df_has_columns(self, df, ["ep", "blink_count_eeg"])
        self.assertEqual(len(df), len(epochs))
        expected = self.expected_counts.drop(self.allowed_exception_rows, errors="ignore")
        computed = df["blink_count_eeg"].drop(self.allowed_exception_rows, errors="ignore")
        pd.testing.assert_series_equal(computed, expected, check_names=False)

    def test_partial_segment_config_missing_eog(self) -> None:
        """SEGMENT_CONFIG missing EOG key does not block blink counting."""
        epochs = self._make_epochs(include_ear=True, include_eeg=True, include_eog=False)
        df = blink_count(epochs, picks=["EEG-E8"])
        assert_df_has_columns(self, df, ["ep", "blink_count_eeg"])
        self.assertEqual(len(df), len(epochs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
