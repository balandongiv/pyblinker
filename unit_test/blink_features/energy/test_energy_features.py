"""
Blink feature extraction tests.

This module contains unit tests for per-blink feature computation using
refined blink annotations extracted from the real ``ear_eog_raw.fif``
recording. Using recorded data ensures the feature functions operate
consistently on realistic blink annotations.

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
from pyblinker.utils import prepare_refined_segments

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
        segments, refined = prepare_refined_segments(
            raw_path,
            "EAR-avg_ear",
            epoch_len=30.0,
            keep_epoch_signal=True,
            progress_bar=False,
        )
        self.sfreq = segments[0].info["sfreq"]
        self.refined = refined
        self.n_epochs = len(segments)

    def test_first_epoch_features(self) -> None:
        """Verify energy metrics for the first epoch."""
        blinks0 = [b for b in self.refined if b["epoch_index"] == 0]
        feats = compute_energy_complexity_features(blinks0, self.sfreq)
        logger.debug("Energy features epoch 0: %s", feats)
        self.assertAlmostEqual(
            feats["blink_signal_energy_mean"], 0.0102538, places=5
        )
        self.assertAlmostEqual(
            feats["blink_line_length_mean"], 0.326137, places=5
        )
        self.assertAlmostEqual(
            feats["blink_velocity_integral_mean"], 0.318309, places=5
        )

    def test_nan_with_no_blinks(self) -> None:
        """Epoch without blinks should yield NaN for energy mean."""
        present = {b["epoch_index"] for b in self.refined}
        empty_epoch = next((i for i in range(self.n_epochs) if i not in present), 0)
        blinks = [b for b in self.refined if b["epoch_index"] == empty_epoch]
        feats = compute_energy_complexity_features(blinks, self.sfreq)
        logger.debug("Energy features epoch %d: %s", empty_epoch, feats)
        self.assertTrue(math.isnan(feats["blink_signal_energy_mean"]))
        self.assertTrue(math.isnan(feats["blink_line_length_mean"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
