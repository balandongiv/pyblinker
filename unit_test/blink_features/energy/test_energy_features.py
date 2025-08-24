"""
Blink feature extraction tests.

This module contains unit tests for per-blink feature computation using
``mne.Epochs`` derived from the real ``ear_eog_raw.fif`` recording. Using
recorded data ensures the feature functions operate consistently on realistic
blink annotations.

The tested feature set includes:

- **Energy-based features**
  - Total energy (Δt * Σ x²)
  - Duration-invariant measures such as average power and RMS amplitude
  - Area under the curve (Δt * Σ |x|)

- **Amplitude and slope features**
  - Peak amplitude
  - Maximum slope / peak velocity via finite differences

- **Temporal / morphological features**
  - Blink duration (samples, seconds)
  - Rise and decay times, symmetry measures, and time-to-peak

Together, these features provide a comprehensive representation of each blink’s
strength, duration, and shape, while separating duration-driven effects from
amplitude-driven ones.  Complexity metrics (entropy, fractal dimension, etc.)
may be added later but are not yet stable at 30 Hz per-blink sampling.
"""

import logging
import math
from pathlib import Path
import unittest

import mne

from pyblinker.blink_features.energy.energy_complexity_features import (
    compute_energy_complexity_features,
)
from pyblinker.utils import slice_raw_into_mne_epochs

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestEnergyComplexityFeatures(unittest.TestCase):
    """Tests for energy and complexity metric calculations on epochs."""

    def setUp(self) -> None:
        raw_path = (
            PROJECT_ROOT
            / "unit_test"
            / "test_files"
            / "ear_eog_raw.fif"
        )
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.sfreq = raw.info["sfreq"]
        self.epochs = slice_raw_into_mne_epochs(
            raw, epoch_len=30.0, blink_label=None, progress_bar=False
        ).copy().pick("EAR-avg_ear")

    def _blinks_from_epoch(self, epoch_index: int) -> list[dict]:
        """Extract blink dictionaries from a single epoch."""
        epoch = self.epochs[epoch_index]
        signal = epoch.get_data().squeeze()
        meta = epoch.metadata.iloc[0]
        onsets = meta["blink_onset"]
        durations = meta["blink_duration"]
        onsets = (
            onsets
            if isinstance(onsets, list)
            else ([] if onsets is None else [onsets])
        )
        durations = (
            durations
            if isinstance(durations, list)
            else ([] if durations is None else [durations])
        )
        blinks: list[dict] = []
        for o, d in zip(onsets, durations):
            start = int(float(o) * self.sfreq)
            end = int((float(o) + float(d)) * self.sfreq)
            blinks.append(
                {
                    "epoch_index": epoch_index,
                    "epoch_signal": signal,
                    "refined_start_frame": start,
                    "refined_end_frame": end,
                }
            )
        return blinks

    def test_first_epoch_features(self) -> None:
        """Verify energy metrics for the first epoch."""
        blinks = self._blinks_from_epoch(0)
        feats = compute_energy_complexity_features(blinks, self.sfreq)
        logger.debug(f"Energy features epoch 0: {feats}")
        self.assertAlmostEqual(
            feats["blink_signal_energy_mean"], 0.00999, places=5
        )
        self.assertAlmostEqual(
            feats["blink_line_length_mean"], 0.31655, places=5
        )
        self.assertAlmostEqual(
            feats["blink_velocity_integral_mean"], 0.30872, places=5
        )

    def test_nan_with_no_blinks(self) -> None:
        """Epoch without blinks should yield NaN for energy mean."""
        blinks = self._blinks_from_epoch(2)
        feats = compute_energy_complexity_features(blinks, self.sfreq)
        logger.debug(f"Energy features epoch 2: {feats}")
        self.assertTrue(math.isnan(feats["blink_signal_energy_mean"]))
        self.assertTrue(math.isnan(feats["blink_line_length_mean"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
