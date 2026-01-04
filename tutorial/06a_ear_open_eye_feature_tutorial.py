"""EAR open-eye baseline tutorial with optional modalities."""

from pathlib import Path
import sys

# ruff: noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mne

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.evaluation import mat_data
from pyblinker.utils.open_eye_baseline import compute_open_eye_baseline_features

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


def run_open_eye_example(name: str, segment_config: dict, picks_arg: list[str]) -> None:
    print(f"\n=== {name} ===")
    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw.copy(),
        epoch_len=30.0,
        blink_label=None,
        segmentation_type=segment_config,
    )
    baseline_idx = epochs.metadata.index[epochs.metadata["n_blinks"] == 0][:4].tolist()
    df = compute_open_eye_baseline_features(epochs, picks_arg, baseline_idx)
    print(df)


# -----------------------------------------------------------------------------
# Variations
# -----------------------------------------------------------------------------
# EAR-only baseline (SEGMENT_CONFIG only contains EAR)
ear_only_config = {
    "ear": {
        "channel": ear_channel,
        "seg_type": "threshold_interpolation",
        "threshold": 0.260,
        "annotation_time_unit": "seconds",
    }
}
run_open_eye_example("EAR-only baseline", ear_only_config, [ear_channel])

# EAR + EEG (no EOG key)
ear_eeg_config = {
    "ear": ear_only_config["ear"],
    "eeg": {"channel": eeg_channel},
}
run_open_eye_example("EAR + EEG baseline", ear_eeg_config, [ear_channel, eeg_channel])
