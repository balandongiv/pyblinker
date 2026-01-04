"""Tutorial: Blink event features with optional modalities.

This script mirrors the structure of the energy tutorials and demonstrates how
to run the blink event pipeline with EAR-only, EEG-only, and combined
configurations. It also shows how to omit modality keys from ``SEGMENT_CONFIG``
without resorting to placeholder channels.
"""

from pathlib import Path
import sys

# ruff: noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mne

from pyblinker.blink_features.blink_events.event_features import (
    aggregate_blink_event_features,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.evaluation import mat_data

# -----------------------------------------------------------------------------
# Load raw data and annotations
# -----------------------------------------------------------------------------
raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
csv_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog.csv"

raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
raw.set_annotations(mat_data.read_annotations_as_mne(csv_path))

# -----------------------------------------------------------------------------
# Channel selection
# -----------------------------------------------------------------------------
ear_channel = "EAR-avg_ear"
eeg_channel = "EEG-E8"
eog_channel = "EOG-EEG-eog_vert_left"

# Restrict to channels that exist in the recording
picks = [ch for ch in (ear_channel, eeg_channel, eog_channel) if ch in raw.ch_names]
raw.pick(picks)


def run_example(name: str, segment_config: dict, picks_arg: list[str]) -> None:
    """Slice epochs, aggregate event features, and print the result."""

    print(f"\n=== {name} ===")
    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw.copy(),
        epoch_len=30.0,
        blink_label=None,
        segmentation_type=segment_config,
    )
    df = aggregate_blink_event_features(epochs, picks=picks_arg)
    print(df.head())


# -----------------------------------------------------------------------------
# Base segmentation settings
# -----------------------------------------------------------------------------
ear_seg = {
    "channel": ear_channel,
    "seg_type": "threshold_interpolation",
    "threshold": 0.260,
    "annotation_time_unit": "seconds",
    "max_extension": 0.35,
    "extension_step": 0.05,
    "padding": 0.05,
    "extend_before": True,
    "extend_after": True,
}


# -----------------------------------------------------------------------------
# Variations
# -----------------------------------------------------------------------------
# EAR-only (EEG/EOG keys omitted)
ear_only_config = {"ear": ear_seg}
run_example("EAR-only", ear_only_config, [ear_channel])

# EEG-only (EAR key omitted, optional EOG still excluded)
eeg_only_config = {"eeg": {"channel": eeg_channel}}
run_example("EEG-only", eeg_only_config, [eeg_channel])

# Combined EAR + EEG with partial SEGMENT_CONFIG (no EOG key)
combined_config = {
    "ear": ear_seg,
    "eeg": {"channel": eeg_channel},
}
run_example("EAR + EEG", combined_config, [ear_channel, eeg_channel])
