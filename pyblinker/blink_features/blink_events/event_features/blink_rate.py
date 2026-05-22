"""Blink rate feature."""

from typing import List, Dict
from pyblinker._logging import get_logger


from .blink_count import blink_count_epoch

logger = get_logger(__name__)


def blink_rate_epoch(blinks: List[Dict[str, int]], epoch_len: float) -> float:
    """Compute the blink rate for a single epoch.

    Parameters
    ----------
    blinks : list of dict
        Blink annotations belonging to one epoch.
    epoch_len : float
        Epoch length in seconds.

    Returns
    -------
    float
        Number of blinks per minute observed in the epoch.
    """
    count = blink_count_epoch(blinks)
    rate = count / epoch_len * 60.0
    logger.debug("Blink rate calculated: %s blinks/min", rate)
    return rate
