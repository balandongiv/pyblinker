"""Aggregate energy feature tests for real EAR data."""
import logging
import math
import unittest
from pathlib import Path

from pyblinker.blink_features.energy.aggregate import (
    aggregate_energy_complexity_features,
)
from pyblinker.utils import prepare_refined_segments

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestEnergyComplexityFeaturesAggregate(unittest.TestCase):
    """Validate aggregated energy metrics across epochs."""

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

    def test_aggregate_energy_features(self) -> None:
        """Aggregate per-epoch energy metrics and verify output."""
        df = aggregate_energy_complexity_features(
            self.refined, self.sfreq, self.n_epochs
        )
        logger.debug("Aggregated energy DataFrame: %s", df.to_dict("index"))
        present = {b["epoch_index"] for b in self.refined}
        empty_epoch = next((i for i in range(self.n_epochs) if i not in present), 0)
        self.assertEqual(df.shape, (self.n_epochs, 12))
        self.assertTrue(math.isnan(df.loc[empty_epoch, "blink_signal_energy_mean"]))
        if self.refined:
            idx = self.refined[0]["epoch_index"]
            feats = df.loc[idx]
            self.assertGreater(feats["blink_signal_energy_mean"], 0.0)
            self.assertGreater(feats["blink_line_length_mean"], 0.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
