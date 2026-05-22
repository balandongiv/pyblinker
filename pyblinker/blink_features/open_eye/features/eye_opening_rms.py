"""RMS of eyelid aperture during open-eye periods."""

from typing import List, Dict
import numpy as np
from pyblinker._logging import get_logger

logger = get_logger(__name__)


def eye_opening_rms_epoch(
    epoch_signal: np.ndarray, blinks: List[Dict[str, int]]
) -> float:
    """Compute RMS of the aperture signal between blinks."""
    mask = np.ones(len(epoch_signal), dtype=bool)
    for blink in blinks:
        mask[
            int(blink["refined_start_frame"]) : int(blink["refined_end_frame"]) + 1
        ] = False
    open_signal = epoch_signal[mask]
    if open_signal.size == 0:
        return float("nan")
    rms = float(np.sqrt(np.mean(np.square(open_signal))))
    logger.debug("Eye opening RMS: %s", rms)
    return rms
