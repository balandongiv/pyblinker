"""Tests for blink energy feature extraction.
For EEG and EOG data,we assume the blink start and blink end is start from the interpolated zero crossing point.
Wheres, for the EAR data, we assume the blink start and blink end is start from the threshold crossing point.
However, these assumptions may not hold true for all datasets, so users should verify the results accordingly.

But since all feature calculation is based on blink onsert and duration in the metadata,
this mean, user have the flexibility to define blink onset and duration as they see fit.
For example, user can define blink onset based on
	- the interpolated zero crossing point for EAR data as well.
	- the threshold crossing point for EEG/EOG data as well.
	- or any other custom definition.
	        Segmentation strategy name (``"base"``, ``"zero"``, ``"tent"``,
        ``"half_base"``, or ``"half_zero"``).

This mean, we need to make sure the slice_raw_into_mne_epochs_refine_annot to support different way to define blink onset and duration.
f
By this doing this, the feature calculation functions can remain agnostic to how blink onset and duration are defined,
"""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.utils.refinement_utils import slice_raw_into_mne_epochs_refine_annot
from test.blink_features.utils.helpers import assert_df_has_columns

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestEnergyFeatures(unittest.TestCase):
    """Verify energy metrics computed from :class:`mne.Epochs`."""

    def setUp(self) -> None:
        """Load test epochs with blink metadata."""
        raw_path = PROJECT_ROOT / "manual_annotation_feature_calculation_data" / "ear_eog.fif"
        csv_path = PROJECT_ROOT / "manual_annotation_feature_calculation_data" / "ear_eog.csv"

        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

        # Attach manual CSV annotations to the raw recording.
        # CSV columns: onset (sec), duration (sec), description (label)
        from pyblinker.utils.evaluation import mat_data

        raw.set_annotations(mat_data.read_annotations_as_mne(csv_path))

        # self.epochs = slice_raw_into_mne_epochs_refine_annot(
        #     raw, epoch_len=30.0, blink_label=None, progress_bar=False
        # )

        channel = "EAR-avg_ear"
        if channel not in raw.ch_names:
            raise ValueError(f"Required channel '{channel}' not found in raw data.")
        raw.pick([channel, "EEG-E8", "EOG-EEG-eog_vert_left"])
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False, ear_threshold=0.22
        )


    def test_single_channel_columns(self) -> None:
        """Returned DataFrame has expected columns for one channel."""
        ch = "EAR-avg_ear"
        df = compute_energy_features(self.epochs, picks=ch)
        expected = [
            f"blink_signal_energy_mean_{ch}",
            f"blink_signal_energy_std_{ch}",
            f"blink_signal_energy_cv_{ch}",
            f"teager_kaiser_energy_mean_{ch}",
            f"teager_kaiser_energy_std_{ch}",
            f"teager_kaiser_energy_cv_{ch}",
            f"blink_line_length_mean_{ch}",
            f"blink_line_length_std_{ch}",
            f"blink_line_length_cv_{ch}",
            f"blink_velocity_integral_mean_{ch}",
            f"blink_velocity_integral_std_{ch}",
            f"blink_velocity_integral_cv_{ch}",
        ]
        assert_df_has_columns(self, df, expected)
        self.assertEqual(len(df), len(self.epochs))

    def test_epoch_without_blinks_is_nan(self) -> None:
        """Epochs lacking blinks yield NaNs for all metrics."""
        df = compute_energy_features(self.epochs, picks="EAR-avg_ear")
        no_blink_idx = self.epochs.metadata.index[
            self.epochs.metadata["blink_onset"].isna()
        ][0]
        self.assertTrue(df.loc[no_blink_idx].isna().all())

    def test_multiple_channels(self) -> None:
        """Processing multiple channels produces suffixed columns."""
        picks = ["EEG-E8", "EOG-EEG-eog_vert_left"]
        df = compute_energy_features(self.epochs, picks=picks)
        for ch in picks:
            prefix = [
                f"blink_signal_energy_mean_{ch}",
                f"teager_kaiser_energy_mean_{ch}",
                f"blink_line_length_mean_{ch}",
                f"blink_velocity_integral_mean_{ch}",
            ]
            assert_df_has_columns(self, df, prefix)
    #
    def test_missing_channel_raises(self) -> None:
        """Requesting an unknown channel results in ``ValueError``."""
        with self.assertRaises(ValueError):
            compute_energy_features(self.epochs, picks="bogus")
    #


if __name__ == "__main__":
    unittest.main()
