from pathlib import Path

import mne
from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.utils.evaluation import mat_data
from pyblinker.utils.refinement_utils import (
	slice_raw_into_mne_epochs_refine_annot,
	)

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# -----------------------------------------------------------------------------
# Load raw data and annotations
# -----------------------------------------------------------------------------
raw_path = (
		PROJECT_ROOT
		/ "manual_annotation_feature_calculation_data"
		/ "ear_eog.fif"
)
csv_path = (
		PROJECT_ROOT
		/ "manual_annotation_feature_calculation_data"
		/ "ear_eog.csv"
)

# Load raw FIF file
raw = mne.io.read_raw_fif(
	raw_path,
	preload=True,
	verbose=False,
	)

# Attach manual CSV annotations
# CSV columns: onset (sec), duration (sec), description (label)
raw.set_annotations(
	mat_data.read_annotations_as_mne(csv_path)
	)

# -----------------------------------------------------------------------------
# Select EAR channel
# -----------------------------------------------------------------------------
channel = "EAR-avg_ear"
if channel not in raw.ch_names:
	raise ValueError(
		f"Required channel '{channel}' not found in raw data."
		)

raw.pick(channel)
SEGMENT_CONFIG = {
		"ear": {
				"seg_type": [
						"base",
						"zero",
						"tent",
						"half_base",
						"threshold_interpolation",
						],
				"threshold": 0.22,
				},
		"eeg": {
				"seg_type": [],
				"threshold": None,
				},
		"eog": {
				"seg_type": [],
				"threshold": None,
				},
		}
# Slice raw data into epochs
epochs = slice_raw_into_mne_epochs_refine_annot(
	raw,
	epoch_len=30.0,
	blink_label=None,
	segmentation_type=SEGMENT_CONFIG# if none, we want fallback to some specific segmentation strategies (To think more about this)
	)

# Compute energy features
df = compute_energy_features(
	epochs,
	picks=channel,
	)
