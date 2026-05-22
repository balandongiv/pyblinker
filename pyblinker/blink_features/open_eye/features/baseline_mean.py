"""Mean eyelid position between blinks.

This feature measures the average eyelid aperture during
periods when no blink is detected within an epoch. A lower
baseline mean suggests partial eye closure or reduced vigilance,
which are common signs of drowsiness.
"""

from typing import List, Dict
import numpy as np
from pyblinker._logging import get_logger

logger = get_logger(__name__)


def baseline_mean_epoch(
    epoch_signal: np.ndarray, blinks: List[Dict[str, int]]
) -> float:
    """Compute baseline mean for one epoch.

    Parameters
    ----------
    epoch_signal : numpy.ndarray
        Eyelid aperture samples for the epoch.
    blinks : list of dict
        Blink annotations with ``refined_start_frame`` and ``refined_end_frame``.

    Returns
    -------
    float
        Mean eyelid aperture outside blink periods. ``NaN`` if no open segments
        are available.
    """
    mask = np.ones(len(epoch_signal), dtype=bool)
    for blink in blinks:
        start = int(blink["refined_start_frame"])
        end = int(blink["refined_end_frame"])
        mask[start : end + 1] = False
    open_signal = epoch_signal[mask]
    if open_signal.size == 0:
        return float("nan")
    mean_val = float(np.mean(open_signal))
    logger.debug("Baseline mean: %s", mean_val)
    return mean_val
