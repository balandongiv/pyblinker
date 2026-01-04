"""EEG open-eye baseline tutorial showing partial SEGMENT_CONFIG shapes."""

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
eog_channel = "EOG-EEG-eog_vert_left"

raw.pick([ch for ch in (ear_channel, eeg_channel, eog_channel) if ch in raw.ch_names])


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
# EEG-only
eeg_only_config = {"eeg": {"channel": eeg_channel}}
run_open_eye_example("EEG-only baseline", eeg_only_config, [eeg_channel])

# EEG + EOG (EAR omitted)
eeg_eog_config = {
    "eeg": {"channel": eeg_channel},
    "eog": {"channel": eog_channel},
}
run_open_eye_example("EEG + EOG baseline", eeg_eog_config, [eeg_channel, eog_channel])

# EEG with EAR key present but disabled
eeg_with_disabled_ear = {
    "ear": {"channel": None},
    "eeg": {"channel": eeg_channel},
}
run_open_eye_example("EEG with EAR disabled", eeg_with_disabled_ear, [eeg_channel])
