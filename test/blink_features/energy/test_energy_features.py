"""Tests for blink energy feature extraction."""
from __future__ import annotations

import unittest
from pathlib import Path

import mne
from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config
from test.blink_features.utils.helpers import assert_df_has_columns

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestEnergyFeatures(unittest.TestCase):
    """Verify energy metrics computed from :class:`mne.Epochs`."""

    def setUp(self) -> None:
        """Load test epochs with blink metadata."""
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
        self.ear_channel = "EAR-avg_ear"
        self.eeg_channel = "EEG-E8"
        self.eog_channel = "EOG-EEG-eog_vert_left"

    def _make_epochs(
        self,
        *,
        include_ear: bool = True,
        include_eeg: bool = True,
        include_eog: bool = True,
        ear_channel: str | None = None,
        eeg_channel: str | None = None,
        eog_channel: str | None = None,
        require_ear: bool | None = None,
    ) -> mne.Epochs:
        raw = mne.io.read_raw_fif(self.raw_path, preload=True, verbose=False)
        segmentation_config = build_segment_config(
            raw,
            ear_channel=ear_channel if ear_channel is not None else self.ear_channel,
            eeg_channel=eeg_channel if eeg_channel is not None else self.eeg_channel,
            eog_channel=eog_channel if eog_channel is not None else self.eog_channel,
            include_ear=include_ear,
            include_eeg=include_eeg,
            include_eog=include_eog,
            require_ear=self.ear_channel is not None if require_ear is None else require_ear,
        )
        return slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )


    def test_single_channel_columns(self) -> None:
        """Returned DataFrame has expected columns for one channel."""
        ch = self.eeg_channel
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
        df = compute_energy_features(self.epochs, picks=self.ear_channel)
        no_blink_idx = self.epochs.metadata.index[
            self.epochs.metadata["blink_onset"].isna()
        ][0]
        self.assertTrue(df.loc[no_blink_idx].isna().all())

    def test_multiple_channels(self) -> None:
        """Processing multiple channels produces suffixed columns."""
        picks = [self.eeg_channel, self.eog_channel]
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

    def test_ear_only_configuration(self) -> None:
        """Energy metrics compute when only EAR modality is configured."""
        epochs = self._make_epochs(include_eeg=False, include_eog=False)
        df = compute_energy_features(epochs, picks=self.ear_channel)
        self.assertEqual(len(df), len(epochs))
        expected = [
            f"blink_signal_energy_mean_{self.ear_channel}",
            f"blink_signal_energy_std_{self.ear_channel}",
            f"blink_velocity_integral_cv_{self.ear_channel}",
        ]
        assert_df_has_columns(self, df, expected)
        no_blink_idx = epochs.metadata.index[epochs.metadata["blink_onset"].isna()][0]
        self.assertTrue(df.loc[no_blink_idx].isna().all())

    def test_eeg_only_missing_ear_key(self) -> None:
        """EEG energy features run when EAR config is omitted entirely."""
        epochs = self._make_epochs(
            include_ear=False,
            include_eeg=True,
            include_eog=False,
            require_ear=False,
        )
        df = compute_energy_features(epochs, picks=self.eeg_channel)
        self.assertEqual(len(df), len(epochs))
        assert_df_has_columns(
            self,
            df,
            [
                f"blink_signal_energy_mean_{self.eeg_channel}",
                f"teager_kaiser_energy_std_{self.eeg_channel}",
            ],
        )

    def test_eeg_and_eog_partial_config(self) -> None:
        """Partial SEGMENT_CONFIG covering EEG+EOG still produces per-channel columns."""
        epochs = self._make_epochs(
            include_ear=False,
            include_eeg=True,
            include_eog=True,
            require_ear=False,
        )
        picks = [self.eeg_channel, self.eog_channel]
        df = compute_energy_features(epochs, picks=picks)
        for ch in picks:
            assert_df_has_columns(
                self,
                df,
                [
                    f"blink_signal_energy_mean_{ch}",
                    f"blink_velocity_integral_std_{ch}",
                ],
            )


if __name__ == "__main__":
    unittest.main()
