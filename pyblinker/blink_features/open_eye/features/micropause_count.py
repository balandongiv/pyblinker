"""Count of brief partial closures between blinks."""

from typing import List, Dict
import numpy as np
from pyblinker._logging import get_logger

logger = get_logger(__name__)


def micropause_count_epoch(
    epoch_signal: np.ndarray,
    blinks: List[Dict[str, int]],
    sfreq: float,
    threshold_ratio: float = 0.5,
    min_dur: float = 0.1,
    max_dur: float = 0.3,
) -> int:
    """Count micropause events within an epoch.

    Parameters
    ----------
    epoch_signal : numpy.ndarray
        Eyelid aperture samples for the epoch.
    blinks : list of dict
        Blink annotations with ``refined_start_frame`` and ``refined_end_frame``.
    sfreq : float
        Sampling frequency in Hertz.
    threshold_ratio : float, optional
        Fraction of baseline considered a partial closure, by default ``0.5``.
    min_dur : float, optional
        Minimum duration of a micropause in seconds, by default ``0.1``.
    max_dur : float, optional
        Maximum duration of a micropause in seconds, by default ``0.3``.

    Returns
    -------
    int
        Number of micropause events detected in the epoch.
    """
    mask = np.ones(len(epoch_signal), dtype=bool)
    for blink in blinks:
        mask[
            int(blink["refined_start_frame"]) : int(blink["refined_end_frame"]) + 1
        ] = False
    open_signal = epoch_signal[mask]
    if open_signal.size == 0:
        return 0
    baseline = np.mean(open_signal)
    threshold = baseline * threshold_ratio

    in_event = False
    count = 0
    event_len = 0
    for i, val in enumerate(epoch_signal):
        if val <= threshold and not any(
            blink["refined_start_frame"] <= i <= blink["refined_end_frame"]
            for blink in blinks
        ):
            if not in_event:
                in_event = True
                event_len = 1
            else:
                event_len += 1
        else:
            if in_event:
                duration = event_len / sfreq
                if min_dur <= duration <= max_dur:
                    count += 1
                in_event = False
                event_len = 0
    if in_event:
        duration = event_len / sfreq
        if min_dur <= duration <= max_dur:
            count += 1
    logger.debug("Micropause count: %s", count)
    return count
