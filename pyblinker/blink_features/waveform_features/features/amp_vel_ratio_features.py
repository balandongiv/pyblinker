"""Amplitude‑velocity ratio metrics.

This module adapts the amplitude‑velocity ratio concept from the
`BLINKER`_ project.  The functions quantify blink steepness by comparing
amplitude changes in the eye aspect ratio to velocity extremes.

Example
-------
# >>> from pyblinker.features.waveform_features.features.amp_vel_ratio_features import neg_amp_vel_ratio_zero
# >>> ratio = neg_amp_vel_ratio_zero(blink, sfreq=100.0)

.. _BLINKER: https://github.com/VisLab/EEG-Blinks
"""
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def neg_amp_vel_ratio_zero(blink: Dict[str, Any], sfreq: float) -> float:
    """Compute negative amplitude‑velocity ratio based on zero landmarks.
s
    The implementation mirrors the method used in the `BLINKER`_ project
    where the downward velocity peak is taken between the blink apex and
    the reopening phase.

    Parameters
    ----------
    blink : dict
        Blink annotation containing ``refined_start_frame``, ``refined_end_frame``
        and ``epoch_signal``.
    sfreq : float
        Sampling frequency in Hertz.

    Returns
    -------
    float
        Ratio of blink amplitude to maximum downward velocity. ``nan`` if
        velocity cannot be computed.
    """
    start = int(blink["refined_start_frame"])
    end = int(blink["refined_end_frame"])
    signal = np.asarray(blink["epoch_signal"], dtype=float)
    segment = signal[start : end + 1]
    baseline = signal[start]
    amplitude = baseline - np.min(segment)
    dt = 1.0 / sfreq
    if segment.size < 2:
        logger.warning("Segment too short in duration_zero. Returning NaN.")
        return float("nan")
    velocity = np.gradient(segment, dt)
    if velocity.size == 0:
        return float("nan")
    neg_vel = np.min(velocity)
    if neg_vel == 0:
        return float("nan")
    ratio = amplitude / abs(neg_vel)
    logger.debug("neg_amp_vel_ratio_zero=%s", ratio)
    return float(ratio)
