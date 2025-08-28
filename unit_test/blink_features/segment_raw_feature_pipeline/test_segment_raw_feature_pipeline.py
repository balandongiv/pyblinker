"""Integration test for full segment-level feature pipeline.

This test combines frequency-domain metrics, time-domain energy features,
and averaged blink properties into a single DataFrame for each raw segment.
It mirrors the canonical setup by building epochs with
``slice_raw_into_mne_epochs_refine_annot`` and calling
``compute_segment_blink_properties`` with identical arguments to ensure input
parity.
"""
import logging
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from refine_annotation.util import slice_raw_into_mne_epochs_refine_annot
from pyblinker.blink_features.frequency_domain.segment_features import compute_frequency_domain_features
from pyblinker.blink_features.energy.segment_features import compute_time_domain_features
from pyblinker.segment_blink_properties import compute_segment_blink_properties

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestSegmentRawFeaturePipeline(unittest.TestCase):
    """Validate feature aggregation across processing stages."""

    def setUp(self) -> None:
        """Load raw data and construct refined epochs."""
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        )
        self.sfreq = raw.info["sfreq"]
        self.params = {
            "base_fraction": 0.5,
            "shut_amp_fraction": 0.9,
            "p_avr_threshold": 3,
            "z_thresholds": np.array([[0.9, 0.98], [2.0, 5.0]]),
        }

    def _build_dataframe(self) -> pd.DataFrame:
        """Construct a combined feature table.

        Returns
        -------
        pandas.DataFrame
            Table indexed by ``seg_id`` containing spectral metrics, time-domain
            metrics, blink counts and averaged blink properties.
        """
        freq_rows: list[dict[str, float]] = []
        energy_rows: list[dict[str, float]] = []
        data = self.epochs.get_data(picks="EEG-E8")[:, 0]
        for seg_id, signal in enumerate(data):
            fd_feats = compute_frequency_domain_features([], signal, self.sfreq)
            td_feats = compute_time_domain_features(signal, self.sfreq)
            freq_rows.append({"seg_id": seg_id, **fd_feats})
            energy_rows.append({"seg_id": seg_id, **td_feats})

        df_freq = pd.DataFrame(freq_rows)
        df_energy = pd.DataFrame(energy_rows)

        blink_epochs = compute_segment_blink_properties(
            self.epochs, self.params, channel="EEG-E8", progress_bar=False
        )
        blink_md = blink_epochs.metadata
        blink_counts = (
            blink_md.groupby("seg_id").size().rename("blink_count").reset_index()
        )
        blink_averages = (
            blink_md.groupby("seg_id").mean(numeric_only=True).reset_index()
        )

        df = df_freq.merge(df_energy, on="seg_id")
        df = df.merge(blink_counts, on="seg_id", how="left")
        df = df.merge(blink_averages, on="seg_id", how="left")
        df["blink_count"] = df["blink_count"].fillna(0).astype(int)
        return df

    def test_pipeline(self) -> None:
        """End-to-end feature extraction with refined epochs."""
        df = self._build_dataframe()
        logger.debug("Combined feature DataFrame:\n%s", df.head())
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.epochs))
        self.assertIn("blink_count", df.columns)
        expected_fd = {f"wavelet_energy_d{i}" for i in range(1, 5)}
        expected_td = {"energy", "teager", "line_length", "velocity_integral"}
        self.assertTrue(expected_fd.issubset(df.columns))
        self.assertTrue(expected_td.issubset(df.columns))
        self.assertFalse(df[list(expected_fd | expected_td)].isna().any().any())
        self.assertTrue((df["blink_count"] >= 0).all())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
