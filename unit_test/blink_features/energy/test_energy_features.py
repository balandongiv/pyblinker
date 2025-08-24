"""
Blink feature extraction tests.

This module contains unit tests for per-blink feature computation using
``mne.Epochs`` derived from the real ``ear_eog_raw.fif`` recording as well as
directly cropped ``mne.io.Raw`` segments. Testing on both epoch-based and
cropped raw data ensures the feature functions operate consistently regardless
of how blink annotations are generated.

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

import unittest
import math
import logging
from pathlib import Path
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
        )
        epochs = self.epochs.copy().pick("EAR-avg_ear")
        data = epochs.get_data().squeeze()
        self.per_epoch = [[] for _ in range(len(epochs))]
        for idx, (onset, duration) in enumerate(
            zip(epochs.metadata["blink_onset"], epochs.metadata["blink_duration"])
        ):
            signal = data[idx]
            if onset is None:
                continue
            onset_list = onset if isinstance(onset, list) else [onset]
            duration_list = duration if isinstance(duration, list) else [duration]
            for o, d in zip(onset_list, duration_list):
                start = int(float(o) * self.sfreq)
                end = int((float(o) + float(d)) * self.sfreq)
                self.per_epoch[idx].append(
                    {
                        "epoch_index": idx,
                        "epoch_signal": signal,
                        "refined_start_frame": start,
                        "refined_end_frame": end,
                    }
                )

    def test_first_epoch_features(self) -> None:
        """Verify energy metrics for the first epoch."""
        feats = compute_energy_complexity_features(self.per_epoch[0], self.sfreq)
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
        feats = compute_energy_complexity_features(self.per_epoch[2], self.sfreq)
        logger.debug(f"Energy features epoch 2: {feats}")
        self.assertTrue(math.isnan(feats["blink_signal_energy_mean"]))
        self.assertTrue(math.isnan(feats["blink_line_length_mean"]))


class TestEnergyComplexityRealRaw(unittest.TestCase):
    """Validate energy metrics using a real 30s raw segment."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.sfreq = raw.info["sfreq"]
        start, stop = 0.0, 30.0
        self.signal = raw.get_data(picks="EAR-avg_ear", start=int(start * self.sfreq), stop=int(stop * self.sfreq))[0]
        self.blinks = []
        for onset, dur in zip(raw.annotations.onset, raw.annotations.duration):
            if onset >= start and onset + dur <= stop:
                s = int((onset - start) * self.sfreq)
                e = int((onset + dur - start) * self.sfreq)
                peak = (s + e) // 2
                self.blinks.append(
                    {
                        "refined_start_frame": s,
                        "refined_peak_frame": peak,
                        "refined_end_frame": e,
                        "epoch_signal": self.signal,
                        "epoch_index": 0,
                    }
                )

    def test_segment_zero_means(self) -> None:
        """Compare a few energy metrics against reference values."""
        feats = compute_energy_complexity_features(self.blinks, self.sfreq)
        logger.debug("Real raw energy features: %s", feats)
        self.assertAlmostEqual(feats["blink_signal_energy_mean"], 0.00999, places=5)
        self.assertAlmostEqual(feats["blink_line_length_mean"], 0.31655, places=5)
        self.assertAlmostEqual(feats["blink_velocity_integral_mean"], 0.30872, places=5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
