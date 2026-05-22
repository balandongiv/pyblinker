"""Pre-blink EAR baseline feature.

This module computes the average eye aspect ratio (EAR) in the
3-second window preceding the first blink of an epoch. The
resulting baseline quantifies eye openness prior to blinking and
may reflect alertness or fatigue level. A similar metric is
implemented in the `Jena Facial Palsy Tool <https://github.com/cvjena/JeFaPaTo>`_.
"""

from typing import List, Dict
import numpy as np
from pyblinker._logging import get_logger

logger = get_logger(__name__)


def ear_before_blink_avg_epoch(
    epoch_signal: np.ndarray,
    blinks: List[Dict[str, int]],
    sfreq: float,
    lookback: float = 3.0,
) -> float:
    """Compute average EAR before the first blink in an epoch.

    Parameters
    ----------
    epoch_signal : numpy.ndarray
        Sequence of EAR samples for the epoch.
    blinks : list of dict
        Blink annotations containing ``refined_start_frame``.
    sfreq : float
        Sampling frequency in Hertz.
    lookback : float, optional
        Time span in seconds to average before the first blink,
        by default ``3.0``.

    Returns
    -------
    float
        Mean EAR in the specified pre-blink window. If the epoch
        contains no blinks the mean of the entire epoch is returned.
    """
    logger.debug("Calculating pre-blink EAR average")
    if not blinks:
        return float(np.mean(epoch_signal))
    start = int(blinks[0]["refined_start_frame"])
    start_idx = max(0, start - int(lookback * sfreq))
    mean_val = float(np.mean(epoch_signal[start_idx:start]))
    logger.debug("EAR before blink average: %s", mean_val)
    return mean_val
