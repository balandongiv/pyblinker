"""Double-thresholding blink detection.

A two-stage detector that combines AutoReject epoch-level screening (Stage A)
with a robust ``center + k * MAD`` sample-level threshold (Stage B) before
locating blink regions via threshold crossings (Stage C).

The Stage A screener depends on the optional ``autoreject`` package; install it
with ``pip install pyblinker[double-thresholding]``.
"""

from ..blinker.get_blink_positions import compute_robust_threshold
from .autoreject_epoch_screener import screen_epochs_with_autoreject
from .blink_threshold import compute_flagged_epoch_threshold
from .core import blink_position_strategy_dbo

__all__ = [
    "blink_position_strategy_dbo",
    "compute_flagged_epoch_threshold",
    "compute_robust_threshold",
    "screen_epochs_with_autoreject",
]
