"""Helper classes and functions for blink analysis."""

import pandas as pd

from .blink_features.waveform_features.extract_blink_properties import BlinkProperties
from .blinker.fit_blink import FitBlinks
from .blinker.pyblinker import BlinkDetector
from .segment_blink_properties import compute_segment_blink_properties

# Keep string-only Index construction backward compatible with historical
# object-dtype column indexes used by baseline fixtures.
pd.options.future.infer_string = False

__all__ = [
    "BlinkProperties",
    "FitBlinks",
    "BlinkDetector",
    "compute_segment_blink_properties",
]
