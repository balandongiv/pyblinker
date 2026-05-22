"""Slope of eyelid baseline during open-eye periods.

A negative slope in the inter-blink baseline indicates a gradual
drooping of the eyelid across the epoch, often preceding prolonged
closures observed in drowsiness.
"""

from typing import List, Dict
import numpy as np
from pyblinker._logging import get_logger

logger = get_logger(__name__)


def baseline_drift_epoch(
    epoch_signal: np.ndarray, blinks: List[Dict[str, int]], sfreq: float
) -> float:
    """Estimate baseline drift for a single epoch.

    Parameters
    ----------
    epoch_signal : numpy.ndarray
        Eyelid aperture samples for the epoch.
    blinks : list of dict
        Blink annotations with ``refined_start_frame`` and ``refined_end_frame``.
    sfreq : float
        Sampling frequency in Hertz.

    Returns
    -------
    float
        Slope of the linear regression line fitted to open-eye samples
        expressed in units per second. ``NaN`` if insufficient data.
    """
    mask = np.ones(len(epoch_signal), dtype=bool)
    for blink in blinks:
        start = int(blink["refined_start_frame"])
        end = int(blink["refined_end_frame"])
        mask[start : end + 1] = False
    open_signal = epoch_signal[mask]
    if open_signal.size < 2:
        return float("nan")
    times = np.arange(open_signal.size) / sfreq
    slope, _ = np.polyfit(times, open_signal, 1)
    logger.debug("Baseline drift slope: %s", slope)
    return float(slope)
