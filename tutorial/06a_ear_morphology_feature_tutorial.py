"""EAR morphology feature tutorial with optional channels."""

from pathlib import Path
import sys

# ruff: noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mne

from pyblinker.blink_features.morphology import compute_epoch_morphology_features
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

raw.pick([ch for ch in (ear_channel, eeg_channel) if ch in raw.ch_names])


def run_morphology_example(name: str, segment_config: dict, picks_arg: list[str]) -> None:
    print(f"\n=== {name} ===")
    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw.copy(),
        epoch_len=30.0,
        blink_label=None,
        segmentation_type=segment_config,
    )
    df = compute_epoch_morphology_features(epochs, picks=picks_arg)
    print(df.head())


# -----------------------------------------------------------------------------
# Variations
# -----------------------------------------------------------------------------
# EAR-only configuration
ear_only_config = {
    "ear": {
        "channel": ear_channel,
        "seg_type": "threshold_interpolation",
        "threshold": 0.260,
        "annotation_time_unit": "seconds",
    }
}
run_morphology_example("EAR-only", ear_only_config, [ear_channel])

# EAR + EEG without EOG key
ear_eeg_config = {
    "ear": ear_only_config["ear"],
    "eeg": {"channel": eeg_channel},
}
run_morphology_example("EAR + EEG", ear_eeg_config, [ear_channel, eeg_channel])
