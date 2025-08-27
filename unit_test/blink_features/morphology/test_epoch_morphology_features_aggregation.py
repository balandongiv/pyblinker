"""Integration of blink counts with morphology features."""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np

from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from refine_annotation.util import slice_raw_into_mne_epochs_refine_annot

from ..utils.helpers import assert_df_has_columns, assert_numeric_or_nan, morphology_column_names

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestMorphologyAggregation(unittest.TestCase):
    """Test aggregation of morphology features with blink counts."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
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


if __name__ == "__main__":
    unittest.main()
