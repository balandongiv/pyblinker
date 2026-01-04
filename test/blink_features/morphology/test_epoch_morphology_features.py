"""Unit tests for epoch-level blink morphology features."""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np

from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from pyblinker.blink_features.morphology.per_blink import compute_blink_waveform_metrics
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config

from ..utils.helpers import (
    assert_df_has_columns,
    assert_numeric_or_nan,
    morphology_column_names,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestEpochMorphologyFeatures(unittest.TestCase):
    """Validate morphology feature extraction from epochs."""

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

    def test_schema_and_alignment(self) -> None:
        """DataFrame has expected columns and indexing for first epochs."""
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left", "EAR-avg_ear"]
        df = compute_epoch_morphology_features(self.epochs, picks=picks)
        assert_df_has_columns(self, df, morphology_column_names(picks))
        self.assertEqual(len(df), len(self.epochs))
        self.assertTrue(df.index.equals(self.epochs.metadata.index))
        for idx in range(4):
            self.assertIn(idx, df.index)
            assert_numeric_or_nan(self, df.iloc[idx])

        zero_idx = self.epochs.metadata.index[self.epochs.metadata["n_blinks"] == 0][0]
        self.assertTrue(df.loc[zero_idx].isna().all())
    #
    def test_channel_suffixes(self) -> None:
        """Verify per-channel suffixes are present in column names."""
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left", "EAR-avg_ear"]
        df = compute_epoch_morphology_features(self.epochs, picks=picks)
        for ch in picks:
            cols = [c for c in df.columns if c.endswith(f"_{ch}")]
            self.assertTrue(cols, msg=f"Missing columns for {ch}")
    #
    def test_missing_channel(self) -> None:
        """Unknown channel names should raise a ValueError."""
        with self.assertRaises(ValueError):
            compute_epoch_morphology_features(self.epochs, picks="not-a-channel")
    #

    def test_empty_epochs(self) -> None:
        """Empty epoch objects yield empty feature DataFrames with schema."""
        picks = ["EAR-avg_ear"]
        df = compute_epoch_morphology_features(self.epochs[:0], picks=picks)
        assert_df_has_columns(self, df, morphology_column_names(picks))
        self.assertEqual(len(df), 0)

    def test_waveform_metric_suffixes(self) -> None:
        """Waveform metrics expose method suffixes and modality-specific NaNs."""
        segment = np.array([0.0, 0.5, 0.1, 0.0])
        metrics = compute_blink_waveform_metrics(
            {"base": segment, "tent": segment}, 100.0, modality="eeg"
        )
        self.assertIn("area_abs_total_rect_base", metrics)
        self.assertIn("area_abs_total_rect_tent", metrics)

        ear_metrics = compute_blink_waveform_metrics(
            segment, 100.0, methods=("zero",), modality="ear"
        )
        self.assertTrue(
            np.isnan(ear_metrics["area_abs_total_rect_zero"]),
            msg="EAR modality should yield NaN for zero-crossing metrics",
        )

    def test_ear_only_configuration(self) -> None:
        """Morphology features compute when only EAR modality is configured."""
        epochs = self._make_epochs(include_eeg=False, include_eog=False)
        picks = [self.ear_channel]
        df = compute_epoch_morphology_features(epochs, picks=picks)
        assert_df_has_columns(self, df, morphology_column_names(picks))
        zero_idx = epochs.metadata.index[epochs.metadata["n_blinks"] == 0][0]
        self.assertTrue(df.loc[zero_idx].isna().all())

    def test_eeg_only_missing_ear_key(self) -> None:
        """EEG-only segmentation omits EAR without affecting EEG morphology."""
        epochs = self._make_epochs(include_ear=False, include_eeg=True, include_eog=False, require_ear=False)
        picks = [self.eeg_channel]
        df = compute_epoch_morphology_features(epochs, picks=picks)
        assert_df_has_columns(self, df, morphology_column_names(picks))
        self.assertEqual(len(df), len(epochs))

    def test_partial_segment_config_missing_eog(self) -> None:
        """SEGMENT_CONFIG without EOG still supports EAR/EEG morphology."""
        epochs = self._make_epochs(include_ear=True, include_eeg=True, include_eog=False)
        picks = [self.ear_channel, self.eeg_channel]
        df = compute_epoch_morphology_features(epochs, picks=picks)
        assert_df_has_columns(self, df, morphology_column_names(picks))


if __name__ == "__main__":
    unittest.main()
