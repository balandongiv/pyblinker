"""Integration of blink counts with morphology features."""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np

from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config

from ..utils.helpers import assert_df_has_columns, assert_numeric_or_nan, morphology_column_names

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestMorphologyAggregation(unittest.TestCase):
    """Test aggregation of morphology features with blink counts."""

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

    def test_merge_blink_counts(self) -> None:
        """Joined DataFrame exposes why certain rows are NaN."""
        picks = ["EAR-avg_ear"]
        feats = compute_epoch_morphology_features(self.epochs, picks=picks)
        merged = feats.join(self.epochs.metadata["n_blinks"])
        expected_cols = morphology_column_names(picks) + ["n_blinks"]
        assert_df_has_columns(self, merged, expected_cols)
        assert_numeric_or_nan(self, merged.iloc[0])

        feature_cols = morphology_column_names(picks)
        for idx, row in merged.iterrows():
            if row["n_blinks"] == 0:
                self.assertTrue(row[feature_cols].isna().all())
            else:
                self.assertTrue(np.isfinite(row[feature_cols]).any())

    def test_eeg_only_missing_ear_key(self) -> None:
        """Aggregation and blink counts still align without EAR config."""
        epochs = self._make_epochs(include_ear=False, include_eeg=True, include_eog=False, require_ear=False)
        picks = ["EEG-E8"]
        feats = compute_epoch_morphology_features(epochs, picks=picks)
        merged = feats.join(epochs.metadata["n_blinks"])
        expected_cols = morphology_column_names(picks) + ["n_blinks"]
        assert_df_has_columns(self, merged, expected_cols)


if __name__ == "__main__":
    unittest.main()
