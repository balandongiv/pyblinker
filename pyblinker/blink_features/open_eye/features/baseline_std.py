"""Standard deviation of open-eye baseline.

Captures variability of the eyelid aperture when it should
remain steady. Increased variability may indicate unstable
oculomotor control due to fatigue.
"""

from typing import List, Dict
import numpy as np
from pyblinker._logging import get_logger

logger = get_logger(__name__)


def baseline_std_epoch(epoch_signal: np.ndarray, blinks: List[Dict[str, int]]) -> float:
    """Compute baseline standard deviation for an epoch."""
    mask = np.ones(len(epoch_signal), dtype=bool)
    for blink in blinks:
        mask[
            int(blink["refined_start_frame"]) : int(blink["refined_end_frame"]) + 1
        ] = False
    open_signal = epoch_signal[mask]
    if open_signal.size < 2:
        return float("nan")
    val = float(np.std(open_signal, ddof=1))
    logger.debug("Baseline std: %s", val)
    return val
