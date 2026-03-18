"""Legacy MATLAB blink detection algorithms.

This subpackage houses the original port of the MATLAB *Blinker* methods.
It retains the historic logic for reference and compatibility."""

from .fit_blink import FitBlinks
from ..blink_features.waveform_features.extract_blink_properties import BlinkProperties
from .pyblinker import BlinkDetector
from .default_setting import DEFAULT_PARAMS, SCALING_FACTOR, build_blink_params

__all__ = [
    "FitBlinks",
    "BlinkProperties",
    "BlinkDetector",
    "DEFAULT_PARAMS",
    "SCALING_FACTOR",
    "build_blink_params",
]
