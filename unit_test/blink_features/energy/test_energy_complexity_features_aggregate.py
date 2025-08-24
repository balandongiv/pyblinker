"""Aggregate energy feature tests for real EAR data."""
import logging
import math
import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.energy.aggregate import (
    aggregate_energy_complexity_features,
)
from pyblinker.utils import slice_raw_into_mne_epochs

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
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        self.sfreq = raw.info["sfreq"]
        self.epochs = (
            slice_raw_into_mne_epochs(
                raw, epoch_len=30.0, blink_label=None, progress_bar=False
            )
            .copy()
            .pick("EAR-avg_ear")
        )
        self.n_epochs = len(self.epochs)

    def _collect_blinks(self) -> tuple[list[dict], int]:
        """Convert epoch metadata to blink dicts.

        Returns
        -------
        tuple[list[dict], int]
            Blink annotations and an example empty epoch index.
        """
        blinks: list[dict] = []
        data = self.epochs.get_data(picks=[0]).squeeze()
        empty_epoch = None
        durations = self.epochs.metadata["blink_duration"]
        for idx, onset in enumerate(self.epochs.metadata["blink_onset"]):
            signal = data[idx]
            duration = durations[idx]
            if onset is None:
                if empty_epoch is None:
                    empty_epoch = idx
                continue
            onset_list = onset if isinstance(onset, list) else [onset]
            duration_list = duration if isinstance(duration, list) else [duration]
            for o, d in zip(onset_list, duration_list):
                start = int(float(o) * self.sfreq)
                end = int((float(o) + float(d)) * self.sfreq)
                blinks.append(
                    {
                        "epoch_index": idx,
                        "epoch_signal": signal,
                        "refined_start_frame": start,
                        "refined_end_frame": end,
                    }
                )
        if empty_epoch is None:
            empty_epoch = 0
        return blinks, empty_epoch

    def test_aggregate_energy_features(self) -> None:
        """Aggregate per-epoch energy metrics and verify output."""
        blinks, empty_epoch = self._collect_blinks()
        df = aggregate_energy_complexity_features(
            blinks, self.sfreq, self.n_epochs
        )
        logger.debug("Aggregated energy DataFrame: %s", df.to_dict("index"))
        self.assertEqual(df.shape, (self.n_epochs, 12))
        self.assertTrue(math.isnan(df.loc[empty_epoch, "blink_signal_energy_mean"]))
        if blinks:
            idx = blinks[0]["epoch_index"]
            feats = df.loc[idx]
            self.assertGreater(feats["blink_signal_energy_mean"], 0.0)
            self.assertGreater(feats["blink_line_length_mean"], 0.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
