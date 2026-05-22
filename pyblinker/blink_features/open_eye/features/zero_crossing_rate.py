"""Zero-crossing rate of the eyelid velocity."""

from typing import List, Dict
import numpy as np
from pyblinker._logging import get_logger

logger = get_logger(__name__)


def zero_crossing_rate_epoch(
    epoch_signal: np.ndarray, blinks: List[Dict[str, int]]
) -> float:
    """Count zero crossings of first derivative during open-eye periods."""
    mask = np.ones(len(epoch_signal), dtype=bool)
    for blink in blinks:
        mask[
            int(blink["refined_start_frame"]) : int(blink["refined_end_frame"]) + 1
        ] = False
    open_signal = epoch_signal[mask]
    if open_signal.size < 2:
        return float("nan")
    velocity = np.diff(open_signal)
    crossings = np.where(np.diff(np.signbit(velocity)))[0]
    rate = float(len(crossings))
    logger.debug("Zero-crossing rate: %s", rate)
    return rate
