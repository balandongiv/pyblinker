"""EAR baseline, refinement, and feature aggregation."""

from .aggregate import aggregate_ear_features
from .feature_extraction import (
    EARBlinkFeatureExtractor,
    EARFeatureConfig,
    apply_flat_threshold_selection,
    compute_blink_features,
)
from .features import ear_before_blink_avg_epoch, ear_extrema_epoch
from .io import load_coarse_blinks, load_ear_channel
from .refinement import EARRefinementConfig, EARThresholdBlinkRefiner

__all__ = [
    "aggregate_ear_features",
    "EARBlinkFeatureExtractor",
    "EARFeatureConfig",
    "apply_flat_threshold_selection",
    "EARRefinementConfig",
    "EARThresholdBlinkRefiner",
    "compute_blink_features",
    "ear_before_blink_avg_epoch",
    "ear_extrema_epoch",
    "load_coarse_blinks",
    "load_ear_channel",
]
