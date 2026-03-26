"""Epoch-level PERCLOS and fatigue labeling helpers."""

from .aggregate import aggregate_perclos_features
from .labeling import assign_fatigue_label, assign_fatigue_labels
from .perclos import (
    clip_intervals_to_epoch,
    compute_epoch_perclos,
    compute_perclos_features,
    sum_closed_eye_duration,
)
# from .thresholds import resolve_subject_specific_ear_threshold

__all__ = [
    "aggregate_perclos_features",
    "assign_fatigue_label",
    "assign_fatigue_labels",
    "clip_intervals_to_epoch",
    "compute_epoch_perclos",
    "compute_perclos_features",
    "sum_closed_eye_duration",
]
