"""EEG energy feature tutorial with optional channels and partial config."""

from pathlib import Path
import sys

# ruff: noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mne

from pyblinker.blink_features.energy.energy_features import compute_energy_features
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

available = [ch for ch in (ear_channel, eeg_channel, eog_channel) if ch in raw.ch_names]
raw.pick(available)


def run_energy_example(name: str, segment_config: dict, picks_arg: list[str]) -> None:
    print(f"\n=== {name} ===")
    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw.copy(),
        epoch_len=30.0,
        blink_label=None,
        segmentation_type=segment_config,
    )
    df = compute_energy_features(epochs, picks=picks_arg)
    print(df.head())


# -----------------------------------------------------------------------------
# Variations
# -----------------------------------------------------------------------------
# 1) EEG-only (SEGMENT_CONFIG contains only EEG)
eeg_only_config = {"eeg": {"channel": eeg_channel}}
run_energy_example("EEG-only", eeg_only_config, [eeg_channel])

# 2) EEG + EOG (EAR key omitted entirely)
eeg_eog_config = {
    "eeg": {"channel": eeg_channel},
    "eog": {"channel": eog_channel},
}
run_energy_example("EEG + EOG", eeg_eog_config, [eeg_channel, eog_channel])

# 3) EEG with EAR channel set to None (present but disabled)
eeg_with_disabled_ear = {
    "ear": {"channel": None},
    "eeg": {"channel": eeg_channel},
}
run_energy_example("EEG with EAR disabled", eeg_with_disabled_ear, [eeg_channel])
