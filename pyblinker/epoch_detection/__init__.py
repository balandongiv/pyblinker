"""Epoch-level blink detection utilities."""

from .bad_epochs import get_valid_epoch_indices, simulate_bad_epochs
from .epoch_channel import map_concatenated_blinks_to_epochs
from .epoch_health import assign_epoch_health, get_valid_epoch_indices_by_health
from .epoch_input import PreparedEpochDetectionInput, prepare_epoch_detection_input
from .pipeline_utils import build_epoch_boundaries, build_signal_by_epoch

__all__ = [
    "get_valid_epoch_indices",
    "simulate_bad_epochs",
    "map_concatenated_blinks_to_epochs",
    "assign_epoch_health",
    "get_valid_epoch_indices_by_health",
    "PreparedEpochDetectionInput",
    "prepare_epoch_detection_input",
    "build_epoch_boundaries",
    "build_signal_by_epoch",
]
