"""Unit tests for wavelet-based blink frequency features."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.frequency_domain import (
    FrequencyDomainBlinkFeatureExtractor,
    aggregate_frequency_domain_features,
)
from refine_annotation.util import slice_raw_into_mne_epochs_refine_annot

from ..utils.helpers import (
    assert_df_has_columns,
    assert_numeric_or_nan,
    with_userwarning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestFrequencyDomainBlinkFeatures(unittest.TestCase):
    """Validate DWT energy features per epoch."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = (
            PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )

    def test_schema_and_rows(self) -> None:
        """DataFrame has expected columns and indexing for first epochs."""
        df = aggregate_frequency_domain_features(self.epochs, picks="EAR-avg_ear")
        assert_df_has_columns(
            self,
            df,
            [f"wavelet_energy_d{i}" for i in range(1, 5)],
        )
        self.assertEqual(len(df), len(self.epochs))
        for idx in range(4):
            self.assertIn(idx, df.index)
            assert_numeric_or_nan(self, df.iloc[idx])

    def test_requires_mne_object(self) -> None:
        """Extractor must have epochs or raw defined."""
        extractor = FrequencyDomainBlinkFeatureExtractor()
        with self.assertRaises(ValueError):
            extractor.compute()

    def test_low_sampling_frequency_warning(self) -> None:
        """Warn and drop Nyquist-touching levels when fs < 30 Hz."""
        epochs = self.epochs.copy().resample(20.0, npad="auto")
        with with_userwarning(self):
            df = aggregate_frequency_domain_features(epochs, picks="EAR-avg_ear")
        self.assertTrue(df["wavelet_energy_d1"].isna().all())
        assert_df_has_columns(
            self, df, [f"wavelet_energy_d{i}" for i in range(2, 5)]
        )

    def test_no_blink_epochs(self) -> None:
        """Epochs without blinks yield NaN energies."""
        df = aggregate_frequency_domain_features(self.epochs, picks="EAR-avg_ear")
        no_blink_idx = self.epochs.metadata.index[
            self.epochs.metadata["blink_onset"].isna()
        ][0]
        self.assertTrue(
            df.loc[no_blink_idx, [f"wavelet_energy_d{i}" for i in range(1, 5)]].isna().all()
        )


if __name__ == "__main__":
    unittest.main()

