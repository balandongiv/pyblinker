"""Median absolute deviation of open-eye baseline."""

from typing import List, Dict
import numpy as np
from pyblinker.fitutils import mad
from pyblinker._logging import get_logger

logger = get_logger(__name__)


def baseline_mad_epoch(epoch_signal: np.ndarray, blinks: List[Dict[str, int]]) -> float:
    """Compute baseline median absolute deviation for an epoch."""
    mask = np.ones(len(epoch_signal), dtype=bool)
    for blink in blinks:
        mask[
            int(blink["refined_start_frame"]) : int(blink["refined_end_frame"]) + 1
        ] = False
    open_signal = epoch_signal[mask]
    if open_signal.size == 0:
        return float("nan")
    mad_value = float(mad(open_signal))
    logger.debug("Baseline MAD: %s", mad_value)
    return mad_value
