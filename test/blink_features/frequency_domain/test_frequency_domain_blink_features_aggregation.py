"""Integration of blink counts with frequency-domain features."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
import pandas as pd

from pyblinker.blink_features.frequency_domain import aggregate_frequency_domain_features
from pyblinker.utils.refine_util import slice_raw_into_mne_epochs_refine_annot

from ..utils.helpers import assert_df_has_columns, assert_numeric_or_nan


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestFrequencyDomainAggregation(unittest.TestCase):
    """Test aggregation with external blink counts."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = (
            PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def test_merge_blink_counts(self) -> None:
        """Joined DataFrame includes blink counts and energies."""
        df = aggregate_frequency_domain_features(
            self.epochs, picks="EAR-avg_ear", progress_bar=False
        )
        blink_counts_path = (
            PROJECT_ROOT / "test" / "test_files" / "ear_eog_blink_count_epoch.csv"
        )
        blink_counts = pd.read_csv(blink_counts_path, index_col="epoch_id")
        df = df.join(blink_counts)
        assert_df_has_columns(
            self,
            df,
            ["ep"]
            + [f"wavelet_energy_d{i}" for i in range(1, 5)]
            + ["blink_count"],
        )
        assert_numeric_or_nan(self, df.drop(columns=["ep"]).iloc[0])
        zero_idx = self.epochs.metadata.index[
            self.epochs.metadata["blink_onset"].isna()
        ][0]
        self.assertTrue(
            df.drop(columns=["ep", "blink_count"]).loc[zero_idx].isna().all()
        )


if __name__ == "__main__":
    unittest.main()

