"""Tests for aggregating blink event features with refined annotations.

Epochs are produced by ``slice_raw_into_mne_epochs_refine_annot`` and blink
totals are validated against ``ear_eog_blink_count_epoch.csv``. Rows 31 and 55
are known mismatches and are excluded from comparisons; see
``tutorial/04_epoching_and_blink_validation_report.py`` for background.
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
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config
from test.blink_features.utils.helpers import assert_df_has_columns

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestAggregateBlinkFeatures(unittest.TestCase):
    """Validate aggregation of blink features from epochs.

    Blink totals are checked against the ground-truth CSV with rows 31 and 55
    excluded. See ``tutorial/04_epoching_and_blink_validation_report.py`` for
    context on these exceptions.
    """

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

    def test_aggregate_all_features(self) -> None:
        """Compare aggregated features to CSV, ignoring rows 31 and 55."""
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left", "EAR-avg_ear"]
        df = aggregate_blink_event_features(self.epochs, picks=picks)
        expected_cols = [
            "ep",
            "blink_total_eeg",
            "blink_rate_eeg",
            "blink_total_eog",
            "blink_rate_eog",
            "blink_total_ear",
            "blink_rate_ear",
        ] + [f"ibi_{p}" for p in picks]
        assert_df_has_columns(self, df, expected_cols)
        self.assertEqual(len(df), len(self.epochs))
        pd.testing.assert_series_equal(
            df["ep"], pd.Series(self.epochs.metadata.index, name="ep"), check_names=False
        )

        # Compare blink totals against CSV, excluding mismatched rows
        expected = self.expected_counts.drop(
            self.allowed_exception_rows, errors="ignore"
        )
        computed = df["blink_total_eeg"].drop(
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

        for idx in range(4):
            expected_rate = self.expected_counts.iloc[idx] / self.epoch_len * 60.0
            self.assertAlmostEqual(df.loc[idx, "blink_rate_eeg"], expected_rate)

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
        assert_df_has_columns(self, df, ["blink_total_eeg"])
        self.assertEqual(list(df.columns), ["blink_total_eeg"])
        self.assertEqual(len(df), len(self.epochs))

    def test_modality_specific_precedence(self) -> None:
        """Modality-specific blink columns override generic ones."""
        epochs = self.epochs[:2].copy()
        epochs.metadata["blink_onset"] = pd.Series([[0.1], [0.2, 0.4]])
        epochs.metadata["blink_duration"] = pd.Series([[0.1], [0.1, 0.1]])
        epochs.metadata["blink_onset_eeg"] = pd.Series([[0.1, 0.3], [0.2]])
        epochs.metadata["blink_duration_eeg"] = pd.Series([[0.1, 0.1], [0.1]])

        df = aggregate_blink_event_features(
            epochs, picks=["EEG-E8"], features=["blink_total"]
        )
        assert_df_has_columns(self, df, ["blink_total_eeg"])
        self.assertListEqual(df["blink_total_eeg"].tolist(), [2.0, 1.0])

        epochs_g = epochs.copy()
        epochs_g.metadata = epochs_g.metadata.drop(
            columns=["blink_onset_eeg", "blink_duration_eeg"]
        )
        df_g = aggregate_blink_event_features(
            epochs_g, picks=["EEG-E8"], features=["blink_total"]
        )
        self.assertListEqual(df_g["blink_total_eeg"].tolist(), [1.0, 2.0])

    def test_multi_pick_modality_lookup(self) -> None:
        """Blink totals reported separately for each modality."""
        epochs = self.epochs[:2].copy()
        epochs.metadata["blink_onset"] = pd.Series([[0.1], [0.2, 0.4]])
        epochs.metadata["blink_duration"] = pd.Series([[0.1], [0.1, 0.1]])
        epochs.metadata["blink_onset_eog"] = pd.Series([[0.1, 0.3], [0.2]])
        epochs.metadata["blink_duration_eog"] = pd.Series([[0.1, 0.1], [0.1]])
        epochs.metadata = epochs.metadata.drop(
            columns=["blink_onset_eeg", "blink_duration_eeg"], errors="ignore"
        )
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left"]
        df = aggregate_blink_event_features(epochs, picks=picks, features=["blink_total"])
        assert_df_has_columns(self, df, ["blink_total_eeg", "blink_total_eog"])
        self.assertListEqual(df["blink_total_eeg"].tolist(), [1.0, 2.0])
        self.assertListEqual(df["blink_total_eog"].tolist(), [2.0, 1.0])

    def test_ear_only_configuration(self) -> None:
        """Aggregation succeeds when only EAR modality is configured."""
        epochs = self._make_epochs(include_eeg=False, include_eog=False)
        df = aggregate_blink_event_features(epochs, picks=["EAR-avg_ear"])
        assert_df_has_columns(self, df, ["ep", "blink_total_ear", "blink_rate_ear"])
        self.assertEqual(len(df), len(epochs))
        expected = self.expected_counts.drop(self.allowed_exception_rows, errors="ignore")
        computed = df["blink_total_ear"].drop(self.allowed_exception_rows, errors="ignore")
        pd.testing.assert_series_equal(computed, expected, check_names=False)

    def test_eeg_only_missing_ear_key(self) -> None:
        """EEG-only configuration omits EAR key without validation errors."""
        epochs = self._make_epochs(include_ear=False, include_eeg=True, include_eog=False, require_ear=False)
        df = aggregate_blink_event_features(epochs, picks=["EEG-E8"])
        assert_df_has_columns(self, df, ["ep", "blink_total_eeg", "blink_rate_eeg"])
        self.assertEqual(len(df), len(epochs))

    def test_partial_segment_config_missing_eog_key(self) -> None:
        """Missing EOG config key is treated as a skipped modality."""
        epochs = self._make_epochs(include_ear=True, include_eeg=True, include_eog=False)
        picks = ["EEG-E8", "EAR-avg_ear"]
        df = aggregate_blink_event_features(epochs, picks=picks)
        assert_df_has_columns(self, df, ["blink_total_eeg", "blink_total_ear"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
