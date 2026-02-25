"""Helper classes and functions for blink analysis."""

import pandas as pd

try:
    pd.set_option("future.infer_string", False)
except Exception:
    pass

from .blink_features.waveform_features.extract_blink_properties import BlinkProperties
from .blinker.fit_blink import FitBlinks
from .blinker.pyblinker import BlinkDetector
from .segment_blink_properties import compute_segment_blink_properties

__all__ = [
    "BlinkProperties",
    "FitBlinks",
    "BlinkDetector",
    "compute_segment_blink_properties",
]
