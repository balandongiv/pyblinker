"""EAR energy feature tutorial with optional modalities.

This example mirrors ``05a_ear_energy_feature_tutorial.py`` but highlights how
to omit modality keys or channels from ``SEGMENT_CONFIG`` while still running
the pipeline.
"""
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

picks = [ch for ch in (ear_channel, eeg_channel, eog_channel) if ch in raw.ch_names]
raw.pick(picks)


def run_energy_example(name: str, segment_config: dict, picks_arg: list[str]) -> None:
    """Compute energy features for a given modality selection."""

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
# 1) EAR-only configuration (SEGMENT_CONFIG only contains EAR)
ear_only_config = {
    "ear": {
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
}
run_energy_example("EAR-only", ear_only_config, [ear_channel])

# 2) EAR + EEG with partial config (EOG key omitted)
ear_eeg_config = {
    "ear": ear_only_config["ear"],
    "eeg": {"channel": eeg_channel},
}
run_energy_example("EAR + EEG", ear_eeg_config, [ear_channel, eeg_channel])

# 3) EAR with EEG channel set to None (present but explicitly skipped)
ear_skip_eeg_config = {
    "ear": ear_only_config["ear"],
    "eeg": {"channel": None},
}
run_energy_example("EAR with EEG disabled", ear_skip_eeg_config, [ear_channel])
