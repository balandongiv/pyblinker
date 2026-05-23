"""Unit tests for prepare_epoch_detection_input."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np

from pyblinker.epoch_detection import PreparedEpochDetectionInput, prepare_epoch_detection_input

TEST_FILES_DIR = Path(__file__).resolve().parents[1] / "test_files"
RAW_FIF = TEST_FILES_DIR / "ear_eog_raw.fif"


def _make_epochs(raw: mne.io.BaseRaw, duration_s: float = 10.0) -> mne.Epochs:
    return mne.make_fixed_length_epochs(raw, duration=duration_s, preload=True, verbose="ERROR")


class TestPrepareEpochDetectionInputBasic(unittest.TestCase):
    """Smoke tests using the bundled ear_eog_raw.fif test fixture."""

    @classmethod
    def setUpClass(cls) -> None:
        if not RAW_FIF.exists():
            raise unittest.SkipTest(f"Test fixture not found: {RAW_FIF}")
        cls.raw = mne.io.read_raw_fif(str(RAW_FIF), preload=True, verbose="ERROR")

    def test_returns_prepared_input_type(self) -> None:
        epochs = _make_epochs(self.raw)
        result = prepare_epoch_detection_input(epochs, pick_types_options={"eeg": True})
        self.assertIsInstance(result, PreparedEpochDetectionInput)

    def test_data_shape_matches_epochs(self) -> None:
        epochs = _make_epochs(self.raw)
        picks = mne.pick_types(epochs.info, eeg=True)
        result = prepare_epoch_detection_input(epochs, pick_types_options={"eeg": True})
        n_epochs = len(epochs)
        n_ch = len(picks)
        self.assertEqual(result.data.shape[0], n_epochs)
        self.assertEqual(result.data.shape[1], n_ch)

    def test_data_is_float64(self) -> None:
        epochs = _make_epochs(self.raw)
        result = prepare_epoch_detection_input(epochs, pick_types_options={"eeg": True})
        self.assertEqual(result.data.dtype, np.float64)

    def test_channel_names_match_picks(self) -> None:
        epochs = _make_epochs(self.raw)
        picks = mne.pick_types(epochs.info, eeg=True)
        expected = tuple(epochs.ch_names[p] for p in picks)
        result = prepare_epoch_detection_input(epochs, pick_types_options={"eeg": True})
        self.assertEqual(result.channel_names, expected)

    def test_sfreq_matches_raw(self) -> None:
        epochs = _make_epochs(self.raw)
        result = prepare_epoch_detection_input(epochs, pick_types_options={"eeg": True})
        self.assertAlmostEqual(result.sfreq, float(epochs.info["sfreq"]), places=4)

    def test_epoch_length_samples_consistent(self) -> None:
        epochs = _make_epochs(self.raw)
        result = prepare_epoch_detection_input(epochs, pick_types_options={"eeg": True})
        self.assertEqual(result.epoch_length_samples, result.data.shape[-1])

    def test_selection_array_length(self) -> None:
        epochs = _make_epochs(self.raw)
        result = prepare_epoch_detection_input(epochs, pick_types_options={"eeg": True})
        self.assertEqual(len(result.selection), len(epochs))


class TestPrepareEpochDetectionInputResampling(unittest.TestCase):
    """Verify that resampling produces the expected output sample rate."""

    @classmethod
    def setUpClass(cls) -> None:
        if not RAW_FIF.exists():
            raise unittest.SkipTest(f"Test fixture not found: {RAW_FIF}")
        cls.raw = mne.io.read_raw_fif(str(RAW_FIF), preload=True, verbose="ERROR")

    def test_resample_to_100hz(self) -> None:
        epochs = _make_epochs(self.raw)
        result = prepare_epoch_detection_input(
            epochs,
            pick_types_options={"eeg": True},
            resample_rate=100.0,
        )
        self.assertAlmostEqual(result.sfreq, 100.0, places=2)

    def test_resample_none_preserves_original_rate(self) -> None:
        epochs = _make_epochs(self.raw)
        orig_sfreq = float(epochs.info["sfreq"])
        result = prepare_epoch_detection_input(
            epochs,
            pick_types_options={"eeg": True},
            resample_rate=None,
        )
        self.assertAlmostEqual(result.sfreq, orig_sfreq, places=4)


class TestPrepareEpochDetectionInputInvalidPicks(unittest.TestCase):
    """Verify that requesting non-existent channel types raises an error."""

    @classmethod
    def setUpClass(cls) -> None:
        if not RAW_FIF.exists():
            raise unittest.SkipTest(f"Test fixture not found: {RAW_FIF}")
        cls.raw = mne.io.read_raw_fif(str(RAW_FIF), preload=True, verbose="ERROR")

    def test_no_matching_channels_raises(self) -> None:
        epochs = _make_epochs(self.raw)
        with self.assertRaises(ValueError):
            prepare_epoch_detection_input(
                epochs,
                pick_types_options={"meg": True},
            )


if __name__ == "__main__":
    unittest.main()
