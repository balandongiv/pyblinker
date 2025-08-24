"""Blink feature extraction tests.

This module contains unit tests for per-blink feature computation using
blink annotations extracted from the real ``ear_eog_raw.fif`` recording.
Using recorded data ensures the feature functions operate consistently on
realistic blink annotations.

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
        self.n_epochs = len(self.epochs)

        data = self.epochs.get_data(picks=[0]).squeeze()
        durations = self.epochs.metadata["blink_duration"]
        self.blinks = []
        self.empty_epoch = None
        for idx, onset in enumerate(self.epochs.metadata["blink_onset"]):
            signal = data[idx]
            duration = durations[idx]
            if onset is None:
                if self.empty_epoch is None:
                    self.empty_epoch = idx
                continue
            onset_list = onset if isinstance(onset, list) else [onset]
            duration_list = duration if isinstance(duration, list) else [duration]
            for o, d in zip(onset_list, duration_list):
                start = int(float(o) * self.sfreq)
                end = int((float(o) + float(d)) * self.sfreq)
                self.blinks.append(
                    {
                        "epoch_index": idx,
                        "epoch_signal": signal,
                        "refined_start_frame": start,
                        "refined_end_frame": end,
                    }
                )
        if self.empty_epoch is None:
            self.empty_epoch = 0

    def test_first_epoch_features(self) -> None:
        """Verify energy metrics for the first epoch."""
        blinks0 = [b for b in self.blinks if b["epoch_index"] == 0]
        feats = compute_energy_complexity_features(blinks0, self.sfreq)
        logger.debug("Energy features epoch 0: %s", feats)
        self.assertAlmostEqual(
            feats["blink_signal_energy_mean"], 0.0099913, places=5
        )
        self.assertAlmostEqual(
            feats["blink_line_length_mean"], 0.316545, places=5
        )
        self.assertAlmostEqual(
            feats["blink_velocity_integral_mean"], 0.308717, places=5
        )

    def test_nan_with_no_blinks(self) -> None:
        """Epoch without blinks should yield NaN for energy mean."""
        blinks = [b for b in self.blinks if b["epoch_index"] == self.empty_epoch]
        feats = compute_energy_complexity_features(blinks, self.sfreq)
        logger.debug("Energy features epoch %d: %s", self.empty_epoch, feats)
        self.assertTrue(math.isnan(feats["blink_signal_energy_mean"]))
        self.assertTrue(math.isnan(feats["blink_line_length_mean"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
