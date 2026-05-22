"""Percentage of eyelid closure during an epoch.

High PERCLOS values are a well established indicator of
fatigue, representing the fraction of time the eye is mostly
closed.
"""

from typing import List, Dict
import numpy as np
from pyblinker._logging import get_logger

logger = get_logger(__name__)


def perclos_epoch(
    epoch_signal: np.ndarray, blinks: List[Dict[str, int]], threshold_ratio: float = 0.8
) -> float:
    """Compute the PERCLOS metric for one epoch.

    Parameters
    ----------
    epoch_signal : numpy.ndarray
        Eyelid aperture samples for the epoch.
    blinks : list of dict
        Blink annotations for the epoch.
    threshold_ratio : float, optional
        Fraction of the open-eye baseline used as the closure threshold,
        by default ``0.8``.

    Returns
    -------
    float
        Percentage of samples below the closure threshold.
    """
    mask = np.ones(len(epoch_signal), dtype=bool)
    for blink in blinks:
        mask[
            int(blink["refined_start_frame"]) : int(blink["refined_end_frame"]) + 1
        ] = False
    open_signal = epoch_signal[mask]
    if open_signal.size == 0:
        return float("nan")
    baseline = np.mean(open_signal)
    thresh = baseline * (1 - threshold_ratio)
    closed = epoch_signal <= thresh
    perc = float(np.sum(closed) / len(epoch_signal))
    logger.debug("PERCLOS: %s", perc)
    return perc
